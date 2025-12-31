"""Step 2c + 2d: Conditional Attribute Hydration.

Research base distributions and specify modifiers for conditional attributes.
"""

from ....core.llm import agentic_research, RetryCallback
from ....core.models import (
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
    SamplingConfig,
    GroundingInfo,
)
from ..schemas import build_conditional_base_schema, build_modifiers_schema
from ..parsers import parse_distribution, parse_constraints, parse_modifiers
from ...validator import (
    validate_conditional_base_response,
    validate_modifiers_response,
)
from .prompts import make_validator, FORMULA_SYNTAX_GUIDELINES


def hydrate_conditional_base(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    derived_attrs: list[HydratedAttribute] | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    on_retry: RetryCallback | None = None,
) -> tuple[list[HydratedAttribute], list[str], list[str]]:
    """
    Research BASE distributions for conditional attributes (Step 2c).

    Uses GPT-5 with web search. Does NOT include modifiers - those come in Step 2d.

    Args:
        attributes: List of DiscoveredAttribute with strategy=conditional
        population: Population description
        geography: Geographic scope
        independent_attrs: Already hydrated independent attributes
        derived_attrs: Already hydrated derived attributes
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (hydrated_attributes, source_urls, validation_errors)
    """
    if not attributes:
        return [], [], []

    conditional_attrs = [a for a in attributes if a.strategy == "conditional"]
    if not conditional_attrs:
        return [], [], []

    geo_context = f" in {geography}" if geography else ""

    # Build context sections
    context_section = ""
    if context:
        context_section = "## READ-ONLY CONTEXT ATTRIBUTES (from base population)\n\n"
        context_section += "These attributes already exist. You can reference them in mean_formula.\n\n"
        for attr in context:
            context_section += f"- {attr.name} ({attr.type}): {attr.description}\n"
        context_section += "\n---\n\n"

    context_summary = context_section
    all_hydrated = (independent_attrs or []) + (derived_attrs or [])
    if all_hydrated:
        context_summary += "## Context: Already Hydrated Attributes\n\n"
        for attr in all_hydrated:
            dist_info = ""
            if attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "mean") and dist.mean is not None:
                    dist_info = f" (mean={dist.mean})"
                elif hasattr(dist, "options"):
                    dist_info = f" (options: {', '.join(dist.options[:3])}...)"
            elif attr.sampling.formula:
                dist_info = f" (formula: {attr.sampling.formula[:30]}...)"
            context_summary += (
                f"- {attr.name} ({attr.type}): {attr.description}{dist_info}\n"
            )
        context_summary += "\n---\n\n"

    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}): {attr.description} [depends on: {', '.join(attr.depends_on)}]"
        for attr in conditional_attrs
    )

    prompt = f"""{context_summary}Research BASE distributions for these CONDITIONAL attributes of {population}{geo_context}:

{attr_list}
{FORMULA_SYNTAX_GUIDELINES}

## Your Task

For EACH conditional attribute, provide the BASE distribution — what you would sample from before applying any modifiers.

### For Continuous Dependencies (numeric depends on numeric)

Use `mean_formula` to express the relationship:

```json
{{
  "name": "years_experience",
  "distribution": {{
    "type": "normal",
    "mean_formula": "age - 28",
    "std": 3
  }}
}}
```

### For Categorical Dependencies

Use static base distribution (modifiers will adjust in next step):

```json
{{
  "name": "income",
  "distribution": {{
    "type": "normal",
    "mean": 150000,
    "std": 40000
  }}
}}
```

### Dynamic Bounds with min_formula / max_formula

When an attribute's valid range depends on another attribute, use formula-based bounds:

```json
{{
  "name": "children_count",
  "distribution": {{
    "type": "normal",
    "mean_formula": "max(0, household_size - 2)",
    "std": 0.9,
    "min": 0,
    "max_formula": "max(0, household_size - 1)"
  }}
}}
```

This ensures `children_count` never exceeds `household_size - 1` regardless of the sampled value.

**When to use formula bounds:**
- When the valid range depends on a previously-sampled attribute
- When you have an expression constraint like `attr <= other_attr - N`
- To guarantee zero constraint violations

**Formulas can reference:**
- Any attribute in `depends_on`
- Built-in functions: max(), min(), abs()

**IMPORTANT RULE:** If you create an expression constraint like `attr <= expr`, you MUST also add `max_formula: "expr"` to the distribution. Otherwise the constraint will be violated during sampling. The same applies for `attr >= expr` requiring `min_formula`.

### Constraints

Set hard constraints WIDER than observed data.

**Constraint Types:**
- `hard_min` / `hard_max`: Static bounds for clamping sampled values
- `expression`: Agent-level constraints validated after sampling (e.g., `children_count <= household_size - 1`)
- `spec_expression`: Spec-level constraints that validate the YAML definition itself (e.g., `sum(weights)==1`)

**REQUIRED:** When adding an `expression` constraint with an inequality (<=, >=, <, >), you MUST add the corresponding `max_formula` or `min_formula` to enforce it during sampling.

### Grounding

Be honest about data quality.

## Important

- Do NOT specify modifiers yet — that's the next step
- Focus on the BASE distribution

Return JSON with distribution, constraints, and grounding for each attribute."""

    # Build validator for fail-fast validation
    expected_names = [a.name for a in conditional_attrs]
    validate_response = make_validator(
        validate_conditional_base_response, expected_names
    )

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=build_conditional_base_schema(),
        schema_name="conditional_base_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
        validator=validate_response,
        on_retry=on_retry,
    )

    attr_lookup = {a.name: a for a in conditional_attrs}
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
            strategy="conditional",
            distribution=distribution,
            formula=None,
            depends_on=original.depends_on,
            modifiers=[],
        )

        hydrated.append(
            HydratedAttribute(
                name=original.name,
                type=original.type,
                category=original.category,
                description=original.description,
                strategy="conditional",
                depends_on=original.depends_on,
                sampling=sampling,
                grounding=grounding,
                constraints=constraints,
            )
        )

    return hydrated, sources, []


def hydrate_conditional_modifiers(
    conditional_attrs: list[HydratedAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    derived_attrs: list[HydratedAttribute] | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    on_retry: RetryCallback | None = None,
) -> tuple[list[HydratedAttribute], list[str], list[str]]:
    """
    Specify MODIFIERS for conditional attributes (Step 2d).

    Uses GPT-5 with web search to find how distributions vary by dependency values.

    Args:
        conditional_attrs: List of HydratedAttribute from Step 2c (with base distributions)
        population: Population description
        geography: Geographic scope
        independent_attrs: Already hydrated independent attributes
        derived_attrs: Already hydrated derived attributes
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (hydrated_attributes, source_urls, validation_errors)
    """
    if not conditional_attrs:
        return [], [], []

    geo_context = f" in {geography}" if geography else ""

    # Build context sections
    context_section = ""
    if context:
        context_section = "## READ-ONLY CONTEXT ATTRIBUTES (from base population)\n\n"
        context_section += "These attributes already exist. You can reference them in 'when' conditions.\n\n"
        for attr in context:
            opt_info = ""
            if attr.sampling.distribution and hasattr(
                attr.sampling.distribution, "options"
            ):
                opts = attr.sampling.distribution.options
                if opts:
                    opt_info = f"\n    VALID OPTIONS (use exactly): {opts}"
            context_section += (
                f"- {attr.name} ({attr.type}): {attr.description}{opt_info}\n"
            )
        context_section += "\n---\n\n"

    context_summary = context_section + "## Full Context\n\n"

    if independent_attrs:
        context_summary += "**Independent Attributes:**\n"
        for attr in independent_attrs:
            dist_info = ""
            if attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "options") and dist.options:
                    # List options explicitly for easy copy-paste
                    dist_info = f"\n    VALID OPTIONS (use exactly): {dist.options}"
                elif hasattr(dist, "mean") and dist.mean is not None:
                    dist_info = f" — mean={dist.mean}, std={getattr(dist, 'std', '?')}"
            context_summary += (
                f"- {attr.name} ({attr.type}): {attr.description}{dist_info}\n"
            )
        context_summary += "\n"

    if derived_attrs:
        context_summary += "**Derived Attributes:**\n"
        for attr in derived_attrs:
            formula_info = (
                f" — formula: {attr.sampling.formula}" if attr.sampling.formula else ""
            )
            context_summary += (
                f"- {attr.name} ({attr.type}): {attr.description}{formula_info}\n"
            )
        context_summary += "\n"

    context_summary += "**Conditional Attributes (with base distributions):**\n"
    for attr in conditional_attrs:
        dist_info = ""
        if attr.sampling.distribution:
            dist = attr.sampling.distribution
            if hasattr(dist, "mean_formula") and dist.mean_formula:
                dist_info = f" — mean_formula: {dist.mean_formula}"
            elif hasattr(dist, "mean") and dist.mean is not None:
                dist_info = f" — base mean={dist.mean}"
            elif hasattr(dist, "options"):
                dist_info = f" — options: {', '.join(dist.options)}"
        deps_info = f" [depends on: {', '.join(attr.depends_on)}]"
        context_summary += (
            f"- {attr.name} ({attr.type}): {attr.description}{dist_info}{deps_info}\n"
        )

    context_summary += "\n---\n\n"

    prompt = f"""{context_summary}Specify MODIFIERS for conditional attributes of {population}{geo_context}.
{FORMULA_SYNTAX_GUIDELINES}

## Your Task

For EACH conditional attribute, specify how its base distribution should be MODIFIED.

### CRITICAL: Type-Specific Modifier Rules

Different attribute types require different modifier fields. Using the wrong type causes sampling to fail.

| Distribution Type | ALLOWED Fields | FORBIDDEN Fields |
|-------------------|----------------|------------------|
| normal, lognormal, uniform, beta | multiply, add | weight_overrides, probability_override |
| categorical | weight_overrides | multiply, add, probability_override |
| boolean | probability_override | multiply, add, weight_overrides |

COMMON MISTAKE: Do NOT use multiply/add on categorical attributes.
Instead of `multiply: 1.2, add: 0.3` -> Use `weight_overrides: {{"option1": 0.4, "option2": 0.6}}`

COMMON MISTAKE: Do NOT use multiply/add on boolean attributes.
Instead of `multiply: 1.5` -> Use `probability_override: 0.75`

### Scale Warning for Beta Distributions

Beta distributions output values between 0 and 1.

If an attribute uses beta distribution (common for proportions, shares, rates):
- Base output is already 0-1 (e.g., 0.25 = 25%)
- Use small `add` values: +/-0.05 to +/-0.15
- NEVER use add > 0.5 or add < -0.5

WRONG: add: 5.0 (thinking "add 5 percentage points")
RIGHT: add: 0.05 (actually adds 5 percentage points to 0-1 scale)

Example for private_insurance_patient_share (beta, mean ~0.25):
- To increase by ~5 percentage points: multiply: 1.0, add: 0.05
- To increase by ~20%: multiply: 1.2, add: 0.0

### Categorical Weight Overrides

When using weight_overrides, you MUST:
1. Include ALL options from the base distribution
2. Weights MUST sum to exactly 1.0

Example - if base distribution has options [urban, suburban, rural]:

CORRECT:
weight_overrides:
  urban: 0.70
  suburban: 0.20
  rural: 0.10
  # Sum = 1.0, all options included

WRONG:
weight_overrides:
  urban: 0.70
  suburban: 0.30
  # Missing 'rural', sum = 1.0 but incomplete

### Condition Rules

The `when` clause can ONLY reference attributes listed in that attribute's depends_on.

If the attribute has `depends_on: [age, employer_type]`, then:
VALID: when: age > 50
VALID: when: employer_type == 'university_hospital'
VALID: when: age > 50 and employer_type == 'university_hospital'
INVALID: when: gender == 'female' (gender not in depends_on)

### Modifier Conditions Syntax

The `when` clause supports:
- Equality: `role == 'chief'`
- Inequality: `age > 50`
- Membership: `specialty in ['cardiac', 'neuro']`
- Compound: `role == 'chief' and employer_type == 'university'`

### CRITICAL: Use EXACT Option Names in Conditions

When referencing categorical attributes in `when` conditions, you MUST use the EXACT option names as defined in the attribute's distribution. Copy-paste them exactly.

Common naming mismatches that WILL FAIL:
- 'University hospital' should be 'University_hospital'
- 'Senior/Oberarzt' should be 'Senior_Oberarzt'
- 'lead/PI' should be 'lead_PI'
- 'Private hospital' should be 'Private_hospital'

The valid option values are listed in the "Full Context" section above for each categorical attribute. Use those EXACT strings in your conditions.

### Rules

1. Numeric uses multiply/add. Categorical uses weight_overrides. Boolean uses probability_override.
2. Only add modifiers where distribution meaningfully differs from base.
3. Don't include no-ops like {{"multiply": 1.0, "add": 0}}.
4. `when` can only reference attributes in depends_on.
5. ALWAYS use EXACT option names from the attribute definitions - do NOT paraphrase or reformat them.

Return JSON array with modifiers for each conditional attribute."""

    # Build distribution type lookup for fail-fast validation
    attr_dist_types: dict[str, str] = {}
    for attr in conditional_attrs:
        if attr.sampling.distribution:
            dist = attr.sampling.distribution
            # Determine distribution type from the actual object
            dist_class_name = type(dist).__name__.lower()
            if "normal" in dist_class_name:
                attr_dist_types[attr.name] = "normal"
            elif "lognormal" in dist_class_name:
                attr_dist_types[attr.name] = "lognormal"
            elif "beta" in dist_class_name:
                attr_dist_types[attr.name] = "beta"
            elif "uniform" in dist_class_name:
                attr_dist_types[attr.name] = "uniform"
            elif "categorical" in dist_class_name:
                attr_dist_types[attr.name] = "categorical"
            elif "boolean" in dist_class_name:
                attr_dist_types[attr.name] = "boolean"

    validate_response = make_validator(validate_modifiers_response, attr_dist_types)

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=build_modifiers_schema(),
        schema_name="conditional_modifiers_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
        validator=validate_response,
        on_retry=on_retry,
    )

    attr_lookup = {a.name: a for a in conditional_attrs}
    updated = []

    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)
        if not original:
            continue

        modifiers = parse_modifiers(attr_data.get("modifiers", []))

        new_sampling = SamplingConfig(
            strategy=original.sampling.strategy,
            distribution=original.sampling.distribution,
            formula=original.sampling.formula,
            depends_on=original.sampling.depends_on,
            modifiers=modifiers,
        )

        updated.append(
            HydratedAttribute(
                name=original.name,
                type=original.type,
                category=original.category,
                description=original.description,
                strategy=original.strategy,
                depends_on=original.depends_on,
                sampling=new_sampling,
                grounding=original.grounding,
                constraints=original.constraints,
            )
        )
        del attr_lookup[name]

    # Add unprocessed attributes (no modifiers returned)
    for original in attr_lookup.values():
        updated.append(original)

    return updated, sources, []
