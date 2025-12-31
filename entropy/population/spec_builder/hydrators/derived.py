"""Step 2b: Derived Attribute Hydration.

Specify formulas for derived attributes (no web search needed).
"""

from ....core.llm import reasoning_call, RetryCallback
from ....core.models import (
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
    SamplingConfig,
    GroundingInfo,
)
from ..schemas import build_derived_schema
from ..parsers import sanitize_formula
from ...validator import validate_derived_response
from .prompts import make_validator, FORMULA_SYNTAX_GUIDELINES


def hydrate_derived(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    on_retry: RetryCallback | None = None,
) -> tuple[list[HydratedAttribute], list[str], list[str]]:
    """
    Specify formulas for derived attributes (Step 2b).

    Uses GPT-5 WITHOUT web search (formulas are deterministic).

    Args:
        attributes: List of DiscoveredAttribute with strategy=derived
        population: Population description
        geography: Geographic scope
        independent_attrs: Already hydrated independent attributes for reference
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (hydrated_attributes, source_urls, validation_errors)
    """
    if not attributes:
        return [], [], []

    derived_attrs = [a for a in attributes if a.strategy == "derived"]
    if not derived_attrs:
        return [], [], []

    # Build context sections
    context_section = ""
    if context:
        context_section = "## READ-ONLY CONTEXT ATTRIBUTES (from base population)\n\n"
        context_section += (
            "These attributes already exist. You can reference them in formulas.\n\n"
        )
        for attr in context:
            context_section += f"- {attr.name} ({attr.type}): {attr.description}\n"
        context_section += "\n---\n\n"

    independent_summary = context_section
    if independent_attrs:
        independent_summary += "## Available Upstream Attributes (already hydrated)\n\n"
        for attr in independent_attrs:
            dist_info = ""
            if attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "mean") and dist.mean is not None:
                    dist_info = f" (mean={dist.mean})"
                elif hasattr(dist, "options"):
                    dist_info = f" (options: {', '.join(dist.options[:3])}...)"
            independent_summary += (
                f"- {attr.name} ({attr.type}): {attr.description}{dist_info}\n"
            )
        independent_summary += "\n---\n\n"

    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}): {attr.description} [depends on: {', '.join(attr.depends_on)}]"
        for attr in derived_attrs
    )

    prompt = f"""{independent_summary}Specify deterministic formulas for these DERIVED attributes of {population}:

{attr_list}
{FORMULA_SYNTAX_GUIDELINES}

## Your Task

For EACH derived attribute, provide a Python expression that computes its value.

### Rules

1. Formula must be valid Python expression
2. Can only reference attributes in depends_on
3. Formula must produce correct type (int, float, categorical string, or boolean)
4. NO VARIANCE — same inputs must always produce same output

### Examples

**Categorical binning:**
```json
{{
  "name": "age_bracket",
  "formula": "'18-24' if age < 25 else '25-34' if age < 35 else '35-44' if age < 45 else '45-54' if age < 55 else '55-64' if age < 65 else '65+'"
}}
```

**Boolean flag:**
```json
{{
  "name": "is_senior",
  "formula": "years_experience >= 15"
}}
```

Return JSON array with formula for each attribute."""

    # Build validator for fail-fast validation
    expected_names = [a.name for a in derived_attrs]
    validate_response = make_validator(validate_derived_response, expected_names)

    data = reasoning_call(
        prompt=prompt,
        response_schema=build_derived_schema(),
        schema_name="derived_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
        validator=validate_response,
        on_retry=on_retry,
    )

    attr_lookup = {a.name: a for a in derived_attrs}
    hydrated = []

    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)
        if not original:
            continue

        formula = sanitize_formula(attr_data.get("formula", "")) or ""

        grounding = GroundingInfo(
            level="strong",
            method="computed",
            source=None,
            note="Deterministic transformation",
        )

        sampling = SamplingConfig(
            strategy="derived",
            distribution=None,
            formula=formula,
            depends_on=original.depends_on,
            modifiers=[],
        )

        hydrated.append(
            HydratedAttribute(
                name=original.name,
                type=original.type,
                category=original.category,
                description=original.description,
                strategy="derived",
                depends_on=original.depends_on,
                sampling=sampling,
                grounding=grounding,
                constraints=[],
            )
        )

    # Derived attributes don't use web search, so no sources
    return hydrated, [], []
