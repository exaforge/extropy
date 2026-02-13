"""Step 2a: Independent Attribute Hydration.

Research distributions for independent attributes using web search.
"""

from ....core.llm import agentic_research, RetryCallback
from ....core.models import (
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
    SamplingConfig,
    GroundingInfo,
)
from ..schemas import build_independent_schema
from ..parsers import parse_distribution, parse_constraints
from ...validator import validate_independent_response
from .prompts import make_validator, format_context_section


def hydrate_independent(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    model: str | None = None,
    reasoning_effort: str = "low",
    on_retry: RetryCallback | None = None,
) -> tuple[list[HydratedAttribute], list[str], list[str]]:
    """
    Research distributions for independent attributes (Step 2a).

    Uses GPT-5 with web search to find real-world distribution data.

    Args:
        attributes: List of DiscoveredAttribute with strategy=independent
        population: Population description (e.g., "German surgeons")
        geography: Geographic scope (e.g., "Germany")
        context: Existing attributes from base population (for extend mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (hydrated_attributes, source_urls, validation_errors)
    """
    if not attributes:
        return [], [], []

    independent_attrs = [a for a in attributes if a.strategy == "independent"]
    if not independent_attrs:
        return [], [], []

    geo_context = f" in {geography}" if geography else ""

    # Build context section for extend mode
    context_section = format_context_section(context)

    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}, {attr.category}): {attr.description}"
        for attr in independent_attrs
    )

    prompt = f"""{context_section}Research realistic distributions for these INDEPENDENT attributes of {population}{geo_context}:

{attr_list}

## Your Task

For EACH attribute, research and provide:

### 1. Distribution Parameters

Based on attribute type:

**int/float (numeric) - use normal, lognormal, uniform, or beta:**
```json
{{
  "type": "normal",
  "mean": 44,
  "std": 8,
  "min": 26,
  "max": 78
}}
```

**categorical:**
```json
{{
  "type": "categorical",
  "options": ["option_a", "option_b", "option_c"],
  "weights": [0.4, 0.35, 0.25]
}}
```
Note: weights must sum to 1.0

**boolean:**
```json
{{
  "type": "boolean",
  "probability_true": 0.65
}}
```

### 2. Constraints

Hard limits for sampling. IMPORTANT: Set constraints WIDER than observed data to preserve valid outliers.

**Constraint Types:**
- `hard_min` / `hard_max`: Static bounds for clamping sampled values
- `expression`: Agent-level constraints validated after sampling (e.g., `children_count <= household_size - 1`)
- `spec_expression`: Spec-level constraints that validate the YAML definition itself (e.g., `sum(weights)==1` for categorical weights)

**IMPORTANT:** Use `spec_expression` (NOT `expression`) for constraints like:
- `sum(weights)==1` — validates that categorical weights sum to 1
- `weights[0]+weights[1]==1` — validates weight array
- `len(options) > 0` — validates options exist

Use `expression` for constraints that involve agent attributes:
- `children_count <= household_size - 1`
- `years_experience <= age - 23`

### 3. Grounding Quality

For EACH attribute, honestly assess:
- **level**: "strong" (direct data found), "medium" (extrapolated), "low" (estimated)
- **method**: "researched", "extrapolated", or "estimated"
- **source**: URL or citation if available
- **note**: Any caveats

## Research Guidelines

- Use web search to find real statistics from official sources
- Prefer: government data, professional associations, academic studies
- Be honest about data quality — mark as "low" if estimating
- Use {geography or "appropriate"} units and categories

Return JSON with distribution, constraints, and grounding for each attribute."""

    # Build validator for fail-fast validation
    expected_names = [a.name for a in independent_attrs]
    validate_response = make_validator(validate_independent_response, expected_names)

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=build_independent_schema(),
        schema_name="independent_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
        validator=validate_response,
        on_retry=on_retry,
    )

    attr_lookup = {a.name: a for a in independent_attrs}
    hydrated = []

    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)
        if not original:
            continue

        distribution = parse_distribution(
            attr_data.get("distribution", {}), original.type
        )
        constraints = parse_constraints(attr_data.get("constraints", []))

        grounding_data = attr_data.get("grounding", {})
        grounding = GroundingInfo(
            level=grounding_data.get("level", "low"),
            method=grounding_data.get("method", "estimated"),
            source=grounding_data.get("source"),
            note=grounding_data.get("note"),
        )

        sampling = SamplingConfig(
            strategy="independent",
            distribution=distribution,
            formula=None,
            depends_on=[],
            modifiers=[],
        )

        hydrated.append(
            HydratedAttribute(
                name=original.name,
                type=original.type,
                category=original.category,
                description=original.description,
                strategy="independent",
                depends_on=[],
                sampling=sampling,
                grounding=grounding,
                constraints=constraints,
            )
        )

    return hydrated, sources, []
