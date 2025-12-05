"""Step 2: Split Attribute Hydration (Distribution Research).

This module implements the split hydration approach:
- Step 2a: hydrate_independent() - Research distributions for independent attributes
- Step 2b: hydrate_derived() - Specify formulas for derived attributes
- Step 2c: hydrate_conditional_base() - Research base distributions for conditional
- Step 2d: hydrate_conditional_modifiers() - Specify modifiers for conditional

Each step is processed separately with specialized prompts and validation.
"""

import ast
import re
from typing import Any

from ..llm import agentic_research, reasoning_call
from ..spec import (
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
    SamplingConfig,
    GroundingInfo,
    Constraint,
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    Modifier,
    NumericModifier,
    CategoricalModifier,
    BooleanModifier,
)


# =============================================================================
# Validation Helpers
# =============================================================================


BUILTIN_NAMES = {'True', 'False', 'None', 'abs', 'min', 'max', 'round', 'int', 'float', 'str', 'len'}


def extract_names_from_formula(formula: str) -> set[str]:
    """Extract variable names from a Python expression."""
    try:
        tree = ast.parse(formula, mode='eval')
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
        return names
    except SyntaxError:
        # Fallback to regex extraction
        return extract_names_from_condition(formula)


def extract_names_from_condition(condition: str) -> set[str]:
    """Extract attribute names referenced in a when condition."""
    tokens = re.findall(r'\b([a-z_][a-z0-9_]*)\b', condition, re.IGNORECASE)
    keywords = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None', 'if', 'else'}
    return {t for t in tokens if t not in keywords and t not in BUILTIN_NAMES}


def validate_independent_hydration(attributes: list[HydratedAttribute]) -> list[str]:
    """Validate hydrated independent attributes."""
    errors = []

    for attr in attributes:
        dist = attr.sampling.distribution

        if dist is None:
            errors.append(f"{attr.name}: independent attribute missing distribution")
            continue

        if attr.type == "categorical":
            if not isinstance(dist, CategoricalDistribution):
                errors.append(f"{attr.name}: categorical attribute needs categorical distribution")
            elif abs(sum(dist.weights) - 1.0) > 0.02:
                errors.append(f"{attr.name}: weights sum to {sum(dist.weights)}, should be ~1.0")

        if attr.type in ("int", "float"):
            if isinstance(dist, (NormalDistribution, LognormalDistribution)):
                if dist.min is not None and dist.max is not None:
                    if dist.min >= dist.max:
                        errors.append(f"{attr.name}: min ({dist.min}) >= max ({dist.max})")

    return errors


def validate_derived_hydration(
    attributes: list[HydratedAttribute],
    all_attribute_names: set[str]
) -> list[str]:
    """Validate hydrated derived attributes."""
    errors = []

    for attr in attributes:
        # Check formula exists
        if not attr.sampling.formula:
            errors.append(f"{attr.name}: derived attribute missing formula")
            continue

        # Check formula syntax
        try:
            ast.parse(attr.sampling.formula, mode='eval')
        except SyntaxError as e:
            errors.append(f"{attr.name}: invalid formula syntax: {e}")
            continue

        # Check formula only references depends_on
        used_names = extract_names_from_formula(attr.sampling.formula)
        for name in used_names:
            if name not in attr.depends_on and name not in BUILTIN_NAMES:
                errors.append(f"{attr.name}: formula references '{name}' not in depends_on")

    return errors


def validate_conditional_base(attributes: list[HydratedAttribute]) -> list[str]:
    """Validate conditional attributes with base distributions."""
    errors = []

    for attr in attributes:
        dist = attr.sampling.distribution

        if dist is None:
            errors.append(f"{attr.name}: conditional attribute missing distribution")
            continue

        # Check mean_formula references
        if isinstance(dist, (NormalDistribution, LognormalDistribution)) and dist.mean_formula:
            used_names = extract_names_from_formula(dist.mean_formula)
            for name in used_names:
                if name not in attr.depends_on and name not in BUILTIN_NAMES:
                    errors.append(f"{attr.name}: mean_formula references '{name}' not in depends_on")

        # Check std exists for formula-based distributions
        if isinstance(dist, (NormalDistribution, LognormalDistribution)):
            if dist.mean_formula and dist.std is None:
                errors.append(f"{attr.name}: has mean_formula but no std — this makes it derived, not conditional")

    return errors


def validate_modifiers(
    attributes: list[HydratedAttribute],
    all_attributes: dict[str, HydratedAttribute]
) -> list[str]:
    """Validate modifiers for conditional attributes."""
    errors = []

    for attr in attributes:
        for i, mod in enumerate(attr.sampling.modifiers):
            # Check when clause references valid attributes
            referenced = extract_names_from_condition(mod.when)
            for name in referenced:
                if name not in attr.depends_on:
                    errors.append(
                        f"{attr.name} modifier {i}: 'when' references '{name}' not in depends_on"
                    )

    return errors


# =============================================================================
# JSON Schema Builders
# =============================================================================


def _build_independent_schema() -> dict:
    """Build JSON schema for independent attribute hydration."""
    return {
        "type": "object",
        "properties": {
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "distribution": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["normal", "lognormal", "uniform", "beta", "categorical", "boolean"],
                                },
                                "mean": {"type": ["number", "null"]},
                                "std": {"type": ["number", "null"]},
                                "min": {"type": ["number", "null"]},
                                "max": {"type": ["number", "null"]},
                                "alpha": {"type": ["number", "null"]},
                                "beta": {"type": ["number", "null"]},
                                "options": {"type": ["array", "null"], "items": {"type": "string"}},
                                "weights": {"type": ["array", "null"], "items": {"type": "number"}},
                                "probability_true": {"type": ["number", "null"]},
                            },
                            "required": ["type"],
                            "additionalProperties": False,
                        },
                        "constraints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["hard_min", "hard_max", "expression"]},
                                    "value": {"type": ["number", "null"]},
                                    "expression": {"type": ["string", "null"]},
                                    "reason": {"type": ["string", "null"]},
                                },
                                "required": ["type"],
                                "additionalProperties": False,
                            },
                        },
                        "grounding": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "string", "enum": ["strong", "medium", "low"]},
                                "method": {"type": "string", "enum": ["researched", "extrapolated", "estimated"]},
                                "source": {"type": ["string", "null"]},
                                "note": {"type": ["string", "null"]},
                            },
                            "required": ["level", "method"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["name", "distribution", "constraints", "grounding"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["attributes"],
        "additionalProperties": False,
    }


def _build_derived_schema() -> dict:
    """Build JSON schema for derived attribute hydration."""
    return {
        "type": "object",
        "properties": {
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "formula": {"type": "string"},
                    },
                    "required": ["name", "formula"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["attributes"],
        "additionalProperties": False,
    }


def _build_conditional_base_schema() -> dict:
    """Build JSON schema for conditional base distribution hydration."""
    return {
        "type": "object",
        "properties": {
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "distribution": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["normal", "lognormal", "uniform", "beta", "categorical", "boolean"],
                                },
                                "mean": {"type": ["number", "null"]},
                                "std": {"type": ["number", "null"]},
                                "mean_formula": {"type": ["string", "null"]},
                                "std_formula": {"type": ["string", "null"]},
                                "min": {"type": ["number", "null"]},
                                "max": {"type": ["number", "null"]},
                                "alpha": {"type": ["number", "null"]},
                                "beta": {"type": ["number", "null"]},
                                "options": {"type": ["array", "null"], "items": {"type": "string"}},
                                "weights": {"type": ["array", "null"], "items": {"type": "number"}},
                                "probability_true": {"type": ["number", "null"]},
                            },
                            "required": ["type"],
                            "additionalProperties": False,
                        },
                        "constraints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["hard_min", "hard_max", "expression"]},
                                    "value": {"type": ["number", "null"]},
                                    "expression": {"type": ["string", "null"]},
                                    "reason": {"type": ["string", "null"]},
                                },
                                "required": ["type"],
                                "additionalProperties": False,
                            },
                        },
                        "grounding": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "string", "enum": ["strong", "medium", "low"]},
                                "method": {"type": "string", "enum": ["researched", "extrapolated", "estimated"]},
                                "source": {"type": ["string", "null"]},
                                "note": {"type": ["string", "null"]},
                            },
                            "required": ["level", "method"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["name", "distribution", "constraints", "grounding"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["attributes"],
        "additionalProperties": False,
    }


def _build_modifiers_schema() -> dict:
    """Build JSON schema for conditional modifiers hydration."""
    return {
        "type": "object",
        "properties": {
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "modifiers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "when": {"type": "string"},
                                    "multiply": {"type": ["number", "null"]},
                                    "add": {"type": ["number", "null"]},
                                    "weight_overrides": {
                                        "type": ["object", "null"],
                                        "additionalProperties": {"type": "number"},
                                    },
                                    "probability_override": {"type": ["number", "null"]},
                                },
                                "required": ["when"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["name", "modifiers"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["attributes"],
        "additionalProperties": False,
    }


# =============================================================================
# Distribution Parsers
# =============================================================================


def _parse_distribution(dist_data: dict, attr_type: str):
    """Parse distribution from response data."""
    if not dist_data:
        return _default_distribution(attr_type)

    dist_type = dist_data.get("type")

    if dist_type == "normal":
        return NormalDistribution(
            mean=dist_data.get("mean"),
            std=dist_data.get("std"),
            min=dist_data.get("min"),
            max=dist_data.get("max"),
            mean_formula=dist_data.get("mean_formula"),
            std_formula=dist_data.get("std_formula"),
        )
    elif dist_type == "lognormal":
        return LognormalDistribution(
            mean=dist_data.get("mean"),
            std=dist_data.get("std"),
            min=dist_data.get("min"),
            max=dist_data.get("max"),
            mean_formula=dist_data.get("mean_formula"),
            std_formula=dist_data.get("std_formula"),
        )
    elif dist_type == "uniform":
        return UniformDistribution(
            min=dist_data.get("min", 0),
            max=dist_data.get("max", 1),
        )
    elif dist_type == "beta":
        return BetaDistribution(
            alpha=dist_data.get("alpha", 2),
            beta=dist_data.get("beta", 2),
            min=dist_data.get("min"),
            max=dist_data.get("max"),
        )
    elif dist_type == "categorical":
        options = dist_data.get("options", [])
        weights = dist_data.get("weights")
        if not weights or len(weights) != len(options):
            weights = [1.0 / len(options)] * len(options) if options else []
        return CategoricalDistribution(
            options=options,
            weights=weights,
        )
    elif dist_type == "boolean":
        return BooleanDistribution(
            probability_true=dist_data.get("probability_true", 0.5),
        )

    return _default_distribution(attr_type)


def _default_distribution(attr_type: str):
    """Get default distribution for attribute type."""
    if attr_type == "int":
        return NormalDistribution(mean=50, std=15, min=0, max=100)
    elif attr_type == "float":
        return UniformDistribution(min=0, max=1)
    elif attr_type == "categorical":
        return CategoricalDistribution(options=["unknown"], weights=[1.0])
    else:
        return BooleanDistribution(probability_true=0.5)


def _parse_constraints(constraints_data: list[dict]) -> list[Constraint]:
    """Parse constraints from response data."""
    constraints = []
    for c in constraints_data:
        constraints.append(
            Constraint(
                type=c.get("type", "expression"),
                value=c.get("value"),
                expression=c.get("expression"),
                reason=c.get("reason"),
            )
        )
    return constraints


def _parse_modifiers(modifiers_data: list[dict] | None) -> list[Modifier]:
    """Parse modifiers from response data."""
    if not modifiers_data:
        return []

    modifiers = []
    for mod in modifiers_data:
        modifiers.append(
            Modifier(
                when=mod["when"],
                multiply=mod.get("multiply"),
                add=mod.get("add"),
                weight_overrides=mod.get("weight_overrides"),
                probability_override=mod.get("probability_override"),
            )
        )
    return modifiers


# =============================================================================
# Step 2a: Independent Attribute Hydration
# =============================================================================


def hydrate_independent(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[list[HydratedAttribute], list[str]]:
    """
    Research distributions for independent attributes (Step 2a).

    Uses GPT-5 with web search to find real-world distribution data.

    Args:
        attributes: List of DiscoveredAttribute with strategy=independent
        population: Population description (e.g., "German surgeons")
        geography: Geographic scope (e.g., "Germany")
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (list of HydratedAttribute, list of source URLs)
    """
    if not attributes:
        return [], []

    # Filter to only independent attributes
    independent_attrs = [a for a in attributes if a.strategy == "independent"]
    if not independent_attrs:
        return [], []

    geo_context = f" in {geography}" if geography else ""

    # Build attribute summary for prompt
    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}, {attr.category}): {attr.description}"
        for attr in independent_attrs
    )

    prompt = f"""Research realistic distributions for these INDEPENDENT attributes of {population}{geo_context}:

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

Example for surgeon age:
- Research shows surgeons are typically 28-65
- Set hard_min: 26 (minimum possible post-training)
- Set hard_max: 78 (rare but practicing surgeons exist)

### 3. Grounding Quality

For EACH attribute, honestly assess:
- **level**: "strong" (direct data found), "medium" (extrapolated), "low" (estimated)
- **method**: "researched", "extrapolated", or "estimated"
- **source**: URL or citation if available
- **note**: Any caveats

## Research Guidelines

- Use web search to find real statistics from official sources
- Prefer: government data, professional associations, academic studies, industry reports
- Be honest about data quality — mark as "low" if estimating
- Use {geography or "appropriate"} units and categories

Return JSON with distribution, constraints, and grounding for each attribute."""

    schema = _build_independent_schema()

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=schema,
        schema_name="independent_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Build lookup for original attributes
    attr_lookup = {a.name: a for a in independent_attrs}

    # Parse response
    hydrated = []
    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)

        if not original:
            continue

        distribution = _parse_distribution(attr_data.get("distribution", {}), original.type)
        constraints = _parse_constraints(attr_data.get("constraints", []))

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

    # Validate
    errors = validate_independent_hydration(hydrated)
    if errors:
        print(f"  [Validation warnings for independent attributes: {len(errors)}]")
        for err in errors[:3]:
            print(f"    - {err}")

    return hydrated, sources


# =============================================================================
# Step 2b: Derived Attribute Hydration
# =============================================================================


def hydrate_derived(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> list[HydratedAttribute]:
    """
    Specify formulas for derived attributes (Step 2b).

    Uses GPT-5 WITHOUT web search (formulas are deterministic).

    Args:
        attributes: List of DiscoveredAttribute with strategy=derived
        population: Population description
        geography: Geographic scope
        independent_attrs: Already hydrated independent attributes for reference
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        List of HydratedAttribute with formulas
    """
    if not attributes:
        return []

    # Filter to only derived attributes
    derived_attrs = [a for a in attributes if a.strategy == "derived"]
    if not derived_attrs:
        return []

    # Build context from independent attributes
    independent_summary = ""
    if independent_attrs:
        independent_summary = "## Available Upstream Attributes (already hydrated)\n\n"
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

    # Build attribute list for prompt
    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}): {attr.description} [depends on: {', '.join(attr.depends_on)}]"
        for attr in derived_attrs
    )

    prompt = f"""{independent_summary}Specify deterministic formulas for these DERIVED attributes of {population}:

{attr_list}

## Your Task

For EACH derived attribute, provide a Python expression that computes its value from upstream attributes.

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

**Computed value:**
```json
{{
  "name": "bmi",
  "formula": "weight / (height ** 2)"
}}
```

Return JSON array with formula for each attribute."""

    schema = _build_derived_schema()

    # Use reasoning_call (no web search needed for formulas)
    data = reasoning_call(
        prompt=prompt,
        response_schema=schema,
        schema_name="derived_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Build lookup for original attributes
    attr_lookup = {a.name: a for a in derived_attrs}

    # Parse response
    hydrated = []
    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)

        if not original:
            continue

        formula = attr_data.get("formula", "")

        # Derived attributes have computed grounding
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
                constraints=[],  # Derived don't need constraints
            )
        )

    # Validate
    all_names = {a.name for a in (independent_attrs or [])} | {a.name for a in derived_attrs}
    errors = validate_derived_hydration(hydrated, all_names)
    if errors:
        print(f"  [Validation warnings for derived attributes: {len(errors)}]")
        for err in errors[:3]:
            print(f"    - {err}")

    return hydrated


# =============================================================================
# Step 2c: Conditional Base Distributions
# =============================================================================


def hydrate_conditional_base(
    attributes: list[DiscoveredAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    derived_attrs: list[HydratedAttribute] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[list[HydratedAttribute], list[str]]:
    """
    Research BASE distributions for conditional attributes (Step 2c).

    Uses GPT-5 with web search. Does NOT include modifiers - those come in Step 2d.

    Args:
        attributes: List of DiscoveredAttribute with strategy=conditional
        population: Population description
        geography: Geographic scope
        independent_attrs: Already hydrated independent attributes
        derived_attrs: Already hydrated derived attributes
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (list of HydratedAttribute with base distributions, list of source URLs)
    """
    if not attributes:
        return [], []

    # Filter to only conditional attributes
    conditional_attrs = [a for a in attributes if a.strategy == "conditional"]
    if not conditional_attrs:
        return [], []

    geo_context = f" in {geography}" if geography else ""

    # Build context summary
    context_summary = ""
    all_hydrated = (independent_attrs or []) + (derived_attrs or [])
    if all_hydrated:
        context_summary = "## Context: Already Hydrated Attributes\n\n"
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

    # Build attribute list
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

This means: "Experience is centered around (age - 28) with std=3."
Two 50-year-olds sample from normal(22, 3) — different results due to std.

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

Same rules as independent — set hard constraints WIDER than observed data.

Expression constraints can reference dependencies:
```json
{{
  "type": "expression",
  "expression": "value <= age - 24",
  "reason": "Cannot exceed time since minimum training age"
}}
```

### Grounding

Same rules — be honest about data quality.

## Important

- Do NOT specify modifiers yet — that's the next step
- Focus on the BASE distribution
- For `mean_formula`, ensure the formula references only attributes in depends_on

Return JSON with distribution, constraints, and grounding for each attribute."""

    schema = _build_conditional_base_schema()

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=schema,
        schema_name="conditional_base_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Build lookup for original attributes
    attr_lookup = {a.name: a for a in conditional_attrs}

    # Parse response
    hydrated = []
    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)

        if not original:
            continue

        distribution = _parse_distribution(attr_data.get("distribution", {}), original.type)
        constraints = _parse_constraints(attr_data.get("constraints", []))

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
            modifiers=[],  # Will be populated in Step 2d
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

    # Validate
    errors = validate_conditional_base(hydrated)
    if errors:
        print(f"  [Validation warnings for conditional base: {len(errors)}]")
        for err in errors[:3]:
            print(f"    - {err}")

    return hydrated, sources


# =============================================================================
# Step 2d: Conditional Modifiers
# =============================================================================


def hydrate_conditional_modifiers(
    conditional_attrs: list[HydratedAttribute],
    population: str,
    geography: str | None = None,
    independent_attrs: list[HydratedAttribute] | None = None,
    derived_attrs: list[HydratedAttribute] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[list[HydratedAttribute], list[str]]:
    """
    Specify MODIFIERS for conditional attributes (Step 2d).

    Uses GPT-5 with web search to find how distributions vary by dependency values.

    Args:
        conditional_attrs: List of HydratedAttribute from Step 2c (with base distributions)
        population: Population description
        geography: Geographic scope
        independent_attrs: Already hydrated independent attributes
        derived_attrs: Already hydrated derived attributes
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (updated HydratedAttribute list with modifiers, list of source URLs)
    """
    if not conditional_attrs:
        return [], []

    geo_context = f" in {geography}" if geography else ""

    # Build full context
    context_summary = "## Full Context\n\n"

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

For EACH conditional attribute, specify how its base distribution should be MODIFIED based on the values of its dependencies.

### Type-Specific Modifiers

**For NUMERIC attributes (int/float) — use multiply/add:**

```json
{{
  "name": "income",
  "modifiers": [
    {{"when": "role_seniority == 'chief'", "multiply": 1.8, "add": 0}},
    {{"when": "role_seniority == 'senior'", "multiply": 1.3, "add": 0}},
    {{"when": "employer_type == 'private_clinic'", "multiply": 1.15, "add": 0}}
  ]
}}
```

Multiple matching modifiers STACK: base × 1.8 × 1.15 for a chief at private clinic.

**For CATEGORICAL attributes — use weight_overrides:**

```json
{{
  "name": "research_activity",
  "modifiers": [
    {{
      "when": "employer_type == 'university_hospital'",
      "weight_overrides": {{"none": 0.15, "occasional": 0.35, "active": 0.50}}
    }},
    {{
      "when": "employer_type == 'private_clinic'",
      "weight_overrides": {{"none": 0.60, "occasional": 0.30, "active": 0.10}}
    }}
  ]
}}
```

For categorical, the LAST matching modifier wins (they don't stack).

**For BOOLEAN attributes — use probability_override:**

```json
{{
  "name": "has_research_grants",
  "modifiers": [
    {{"when": "research_activity == 'active'", "probability_override": 0.70}},
    {{"when": "research_activity == 'none'", "probability_override": 0.05}}
  ]
}}
```

### Modifier Conditions

The `when` clause supports:
- Equality: `role == 'chief'`
- Inequality: `age > 50`, `experience >= 15`
- Membership: `specialty in ['cardiac', 'neuro', 'thoracic']`
- Compound: `role == 'chief' and employer_type == 'university'`
- Negation: `specialty != 'general'`

### Rules

1. Numeric attributes use multiply/add. Categorical use weight_overrides. Boolean use probability_override.
2. You don't need modifiers for every possible value — only where distribution meaningfully differs from base.
3. Don't include no-ops like {{"multiply": 1.0, "add": 0}}.
4. `when` can only reference attributes in depends_on.

### Attributes That Already Use mean_formula

If a conditional attribute already captures its dependency via `mean_formula` (e.g., experience depends on age), you may not need additional modifiers. Only add modifiers if there are ADDITIONAL categorical shifts.

Return JSON array with modifiers for each conditional attribute."""

    schema = _build_modifiers_schema()

    data, sources = agentic_research(
        prompt=prompt,
        response_schema=schema,
        schema_name="conditional_modifiers_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Build lookup for conditional attributes
    attr_lookup = {a.name: a for a in conditional_attrs}

    # Merge modifiers into existing attributes
    updated = []
    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)

        if not original:
            continue

        modifiers = _parse_modifiers(attr_data.get("modifiers", []))

        # Create updated sampling config with modifiers
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

        # Remove from lookup to track what's been processed
        del attr_lookup[name]

    # Add any unprocessed attributes (no modifiers returned by LLM)
    for name, original in attr_lookup.items():
        updated.append(original)

    # Validate
    all_attrs = {a.name: a for a in (independent_attrs or []) + (derived_attrs or []) + updated}
    errors = validate_modifiers(updated, all_attrs)
    if errors:
        print(f"  [Validation warnings for modifiers: {len(errors)}]")
        for err in errors[:3]:
            print(f"    - {err}")

    return updated, sources


# =============================================================================
# Main Orchestrator
# =============================================================================


def hydrate_attributes(
    attributes: list[DiscoveredAttribute],
    description: str,
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[list[HydratedAttribute], list[str]]:
    """
    Research distributions for discovered attributes using split hydration.

    This function orchestrates the 4-step split hydration process:
    - Step 2a: hydrate_independent() - Research distributions for independent attributes
    - Step 2b: hydrate_derived() - Specify formulas for derived attributes
    - Step 2c: hydrate_conditional_base() - Research base distributions for conditional
    - Step 2d: hydrate_conditional_modifiers() - Specify modifiers for conditional

    When context is provided (overlay mode), the model can reference
    context attributes in formulas and modifiers but should NOT define
    distributions for them.

    Args:
        attributes: List of DiscoveredAttribute from selector
        description: Original population description
        geography: Geographic scope for research
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (list of HydratedAttribute, list of source URLs)

    Example:
        # Base mode
        >>> hydrated, sources = hydrate_attributes(attrs, "German surgeons", "Germany")

        # Overlay mode - can reference base attributes in formulas
        >>> overlay_hydrated, sources = hydrate_attributes(
        ...     scenario_attrs, "AI adoption scenario", "Germany",
        ...     context=base_spec.attributes
        ... )
    """
    if not attributes:
        return [], []

    all_sources = []

    # Extract population from description
    population = description

    # Step 2a: Independent attributes
    print("  Step 2a: Hydrating independent attributes...")
    independent_attrs, independent_sources = hydrate_independent(
        attributes=attributes,
        population=population,
        geography=geography,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    all_sources.extend(independent_sources)
    print(f"    → Hydrated {len(independent_attrs)} independent attributes ({len(independent_sources)} sources)")

    # Step 2b: Derived attributes
    print("  Step 2b: Hydrating derived attributes...")
    derived_attrs = hydrate_derived(
        attributes=attributes,
        population=population,
        geography=geography,
        independent_attrs=independent_attrs,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    print(f"    → Hydrated {len(derived_attrs)} derived attributes")

    # Step 2c: Conditional base distributions
    print("  Step 2c: Hydrating conditional base distributions...")
    conditional_base_attrs, conditional_sources = hydrate_conditional_base(
        attributes=attributes,
        population=population,
        geography=geography,
        independent_attrs=independent_attrs,
        derived_attrs=derived_attrs,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    all_sources.extend(conditional_sources)
    print(f"    → Hydrated {len(conditional_base_attrs)} conditional base distributions ({len(conditional_sources)} sources)")

    # Step 2d: Conditional modifiers
    print("  Step 2d: Hydrating conditional modifiers...")
    conditional_attrs, modifier_sources = hydrate_conditional_modifiers(
        conditional_attrs=conditional_base_attrs,
        population=population,
        geography=geography,
        independent_attrs=independent_attrs,
        derived_attrs=derived_attrs,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    all_sources.extend(modifier_sources)
    print(f"    → Added modifiers to {len(conditional_attrs)} conditional attributes ({len(modifier_sources)} sources)")

    # Combine all hydrated attributes
    all_hydrated = independent_attrs + derived_attrs + conditional_attrs

    # Deduplicate sources
    unique_sources = list(set(all_sources))

    return all_hydrated, unique_sources
