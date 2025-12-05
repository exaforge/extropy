"""Step 2: Split Attribute Hydration (Distribution Research).

This module implements the split hydration approach:
- Step 2a: hydrate_independent() - Research distributions for independent attributes
- Step 2b: hydrate_derived() - Specify formulas for derived attributes
- Step 2c: hydrate_conditional_base() - Research base distributions for conditional
- Step 2d: hydrate_conditional_modifiers() - Specify modifiers for conditional

Each step is processed separately with specialized prompts and validation.
"""

from typing import Callable

from ..llm import agentic_research, reasoning_call
from ..models import (
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
    SamplingConfig,
    GroundingInfo,
)
from .hydrator_utils import (
    # Validation
    validate_independent_hydration,
    validate_derived_hydration,
    validate_conditional_base,
    validate_modifiers,
    # Schemas
    build_independent_schema,
    build_derived_schema,
    build_conditional_base_schema,
    build_modifiers_schema,
    # Parsers
    parse_distribution,
    parse_constraints,
    parse_modifiers,
)


# =============================================================================
# Step 2a: Independent Attribute Hydration
# =============================================================================


def hydrate_independent(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[list[HydratedAttribute], list[str], list[str]]:
    """
    Research distributions for independent attributes (Step 2a).

    Uses GPT-5 with web search to find real-world distribution data.

    Args:
        attributes: List of DiscoveredAttribute with strategy=independent
        population: Population description (e.g., "German surgeons")
        geography: Geographic scope (e.g., "Germany")
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (list of HydratedAttribute, list of source URLs, list of validation errors)
    """
    if not attributes:
        return [], [], []

    independent_attrs = [a for a in attributes if a.strategy == "independent"]
    if not independent_attrs:
        return [], [], []

    geo_context = f" in {geography}" if geography else ""

    # Build context section for overlay mode
    context_section = ""
    if context:
        context_section = "## READ-ONLY CONTEXT ATTRIBUTES (from base population)\n\n"
        context_section += "These attributes already exist. Do NOT redefine them, but you may reference them.\n\n"
        for attr in context:
            context_section += f"- {attr.name} ({attr.type}): {attr.description}\n"
        context_section += "\n---\n\n"

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

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=build_independent_schema(),
        schema_name="independent_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    attr_lookup = {a.name: a for a in independent_attrs}
    hydrated = []

    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)
        if not original:
            continue

        distribution = parse_distribution(attr_data.get("distribution", {}), original.type)
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

    errors = validate_independent_hydration(hydrated)
    return hydrated, sources, errors


# =============================================================================
# Step 2b: Derived Attribute Hydration
# =============================================================================


def hydrate_derived(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[list[HydratedAttribute], list[str]]:
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
        Tuple of (list of HydratedAttribute with formulas, list of validation errors)
    """
    if not attributes:
        return [], []

    derived_attrs = [a for a in attributes if a.strategy == "derived"]
    if not derived_attrs:
        return [], []

    # Build context sections
    context_section = ""
    if context:
        context_section = "## READ-ONLY CONTEXT ATTRIBUTES (from base population)\n\n"
        context_section += "These attributes already exist. You can reference them in formulas.\n\n"
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
            independent_summary += f"- {attr.name} ({attr.type}): {attr.description}{dist_info}\n"
        independent_summary += "\n---\n\n"

    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}): {attr.description} [depends on: {', '.join(attr.depends_on)}]"
        for attr in derived_attrs
    )

    prompt = f"""{independent_summary}Specify deterministic formulas for these DERIVED attributes of {population}:

{attr_list}

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

    data = reasoning_call(
        prompt=prompt,
        response_schema=build_derived_schema(),
        schema_name="derived_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    attr_lookup = {a.name: a for a in derived_attrs}
    hydrated = []

    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)
        if not original:
            continue

        formula = attr_data.get("formula", "")

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

    # Build set of all known attribute names (independent + derived + context)
    all_names = {a.name for a in (independent_attrs or [])} | {a.name for a in derived_attrs}
    if context:
        all_names |= {a.name for a in context}
    errors = validate_derived_hydration(hydrated, all_names)
    return hydrated, errors


# =============================================================================
# Step 2c: Conditional Base Distributions
# =============================================================================


def hydrate_conditional_base(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    derived_attrs: list[HydratedAttribute] | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
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
        Tuple of (list of HydratedAttribute, list of source URLs, list of validation errors)
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
            context_summary += f"- {attr.name} ({attr.type}): {attr.description}{dist_info}\n"
        context_summary += "\n---\n\n"

    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}): {attr.description} [depends on: {', '.join(attr.depends_on)}]"
        for attr in conditional_attrs
    )

    prompt = f"""{context_summary}Research BASE distributions for these CONDITIONAL attributes of {population}{geo_context}:

{attr_list}

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

### Constraints

Set hard constraints WIDER than observed data. Expression constraints can reference dependencies.

### Grounding

Be honest about data quality.

## Important

- Do NOT specify modifiers yet — that's the next step
- Focus on the BASE distribution

Return JSON with distribution, constraints, and grounding for each attribute."""

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=build_conditional_base_schema(),
        schema_name="conditional_base_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    attr_lookup = {a.name: a for a in conditional_attrs}
    hydrated = []

    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)
        if not original:
            continue

        distribution = parse_distribution(attr_data.get("distribution", {}), original.type)
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

    errors = validate_conditional_base(hydrated)
    return hydrated, sources, errors


# =============================================================================
# Step 2d: Conditional Modifiers
# =============================================================================


def hydrate_conditional_modifiers(
    conditional_attrs: list[HydratedAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    derived_attrs: list[HydratedAttribute] | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
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
        Tuple of (updated HydratedAttribute list, list of source URLs, list of validation errors)
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
            context_section += f"- {attr.name} ({attr.type}): {attr.description}\n"
        context_section += "\n---\n\n"

    context_summary = context_section + "## Full Context\n\n"

    if independent_attrs:
        context_summary += "**Independent Attributes:**\n"
        for attr in independent_attrs:
            dist_info = ""
            if attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "options"):
                    dist_info = f" — options: {', '.join(dist.options)}"
                elif hasattr(dist, "mean") and dist.mean is not None:
                    dist_info = f" — mean={dist.mean}, std={getattr(dist, 'std', '?')}"
            context_summary += f"- {attr.name} ({attr.type}): {attr.description}{dist_info}\n"
        context_summary += "\n"

    if derived_attrs:
        context_summary += "**Derived Attributes:**\n"
        for attr in derived_attrs:
            formula_info = f" — formula: {attr.sampling.formula}" if attr.sampling.formula else ""
            context_summary += f"- {attr.name} ({attr.type}): {attr.description}{formula_info}\n"
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
        context_summary += f"- {attr.name} ({attr.type}): {attr.description}{dist_info}{deps_info}\n"

    context_summary += "\n---\n\n"

    prompt = f"""{context_summary}Specify MODIFIERS for conditional attributes of {population}{geo_context}.

## Your Task

For EACH conditional attribute, specify how its base distribution should be MODIFIED.

### Type-Specific Modifiers

**For NUMERIC attributes (int/float) — use multiply/add:**
Multiple matching modifiers STACK.

**For CATEGORICAL attributes — use weight_overrides:**
The LAST matching modifier wins.

**For BOOLEAN attributes — use probability_override:**

### Modifier Conditions

The `when` clause supports:
- Equality: `role == 'chief'`
- Inequality: `age > 50`
- Membership: `specialty in ['cardiac', 'neuro']`
- Compound: `role == 'chief' and employer_type == 'university'`

### Rules

1. Numeric uses multiply/add. Categorical uses weight_overrides. Boolean uses probability_override.
2. Only add modifiers where distribution meaningfully differs from base.
3. Don't include no-ops like {{"multiply": 1.0, "add": 0}}.
4. `when` can only reference attributes in depends_on.

Return JSON array with modifiers for each conditional attribute."""

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=build_modifiers_schema(),
        schema_name="conditional_modifiers_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
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

    all_attrs = {a.name: a for a in (independent_attrs or []) + (derived_attrs or []) + updated}
    errors = validate_modifiers(updated, all_attrs)
    return updated, sources, errors


# =============================================================================
# Main Orchestrator
# =============================================================================

# Type alias for progress callback: (step: str, status: str, count: int | None) -> None
ProgressCallback = Callable[[str, str, int | None], None]


def hydrate_attributes(
    attributes: list[DiscoveredAttribute],
    description: str,
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    on_progress: ProgressCallback | None = None,
) -> tuple[list[HydratedAttribute], list[str], list[str]]:
    """
    Research distributions for discovered attributes using split hydration.

    This function orchestrates the 4-step split hydration process:
    - Step 2a: hydrate_independent() - Research distributions for independent attributes
    - Step 2b: hydrate_derived() - Specify formulas for derived attributes
    - Step 2c: hydrate_conditional_base() - Research base distributions for conditional
    - Step 2d: hydrate_conditional_modifiers() - Specify modifiers for conditional

    When context is provided (overlay mode), the model can reference
    context attributes in formulas and modifiers.

    Args:
        attributes: List of DiscoveredAttribute from selector
        description: Original population description
        geography: Geographic scope for research
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"
        on_progress: Optional callback for progress updates (step, status, count)

    Returns:
        Tuple of (list of HydratedAttribute, list of source URLs, list of validation warnings)
    """
    if not attributes:
        return [], [], []

    all_sources = []
    all_warnings = []
    population = description

    def report(step: str, status: str, count: int | None = None):
        """Report progress via callback or print."""
        if on_progress:
            on_progress(step, status, count)
        else:
            if count is not None:
                print(f"  {step}: {status} ({count})")
            else:
                print(f"  {step}: {status}")

    # Step 2a: Independent attributes
    report("2a", "Researching independent distributions...")
    independent_attrs, independent_sources, independent_errors = hydrate_independent(
        attributes=attributes,
        population=population,
        geography=geography,
        context=context,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    all_sources.extend(independent_sources)
    all_warnings.extend([f"[2a] {e}" for e in independent_errors])
    report("2a", f"Hydrated {len(independent_attrs)} independent", len(independent_sources))

    # Step 2b: Derived attributes
    report("2b", "Specifying derived formulas...")
    derived_attrs, derived_errors = hydrate_derived(
        attributes=attributes,
        population=population,
        geography=geography,
        independent_attrs=independent_attrs,
        context=context,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    all_warnings.extend([f"[2b] {e}" for e in derived_errors])
    report("2b", f"Hydrated {len(derived_attrs)} derived", 0)

    # Step 2c: Conditional base distributions
    report("2c", "Researching conditional distributions...")
    conditional_base_attrs, conditional_sources, conditional_errors = hydrate_conditional_base(
        attributes=attributes,
        population=population,
        geography=geography,
        independent_attrs=independent_attrs,
        derived_attrs=derived_attrs,
        context=context,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    all_sources.extend(conditional_sources)
    all_warnings.extend([f"[2c] {e}" for e in conditional_errors])
    report("2c", f"Hydrated {len(conditional_base_attrs)} conditional", len(conditional_sources))

    # Step 2d: Conditional modifiers
    report("2d", "Specifying conditional modifiers...")
    conditional_attrs, modifier_sources, modifier_errors = hydrate_conditional_modifiers(
        conditional_attrs=conditional_base_attrs,
        population=population,
        geography=geography,
        independent_attrs=independent_attrs,
        derived_attrs=derived_attrs,
        context=context,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    all_sources.extend(modifier_sources)
    all_warnings.extend([f"[2d] {e}" for e in modifier_errors])
    report("2d", f"Added modifiers to {len(conditional_attrs)}", len(modifier_sources))

    # Combine all hydrated attributes
    all_hydrated = independent_attrs + derived_attrs + conditional_attrs
    unique_sources = list(set(all_sources))

    return all_hydrated, unique_sources, all_warnings
