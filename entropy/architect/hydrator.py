"""Step 2: Attribute Hydration (Distribution Research).

Takes discovered attributes and researches real-world distributions
using GPT-5 with agentic web search.
"""

from ..llm import agentic_research
from ..spec import (
    DiscoveredAttribute,
    HydratedAttribute,
    SamplingConfig,
    GroundingInfo,
    Constraint,
    NormalDistribution,
    UniformDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    Modifier,
)


def _build_hydration_schema(attributes: list[DiscoveredAttribute]) -> dict:
    """Build JSON schema for hydration response based on discovered attributes."""
    
    # Build per-attribute schema
    attribute_schemas = []
    for attr in attributes:
        attr_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": attr.name},
                "sampling_strategy": {
                    "type": "string",
                    "enum": ["independent", "derived", "conditional"],
                    "description": "How to sample: independent (direct), derived (formula), conditional (base + modifiers)",
                },
                # Distribution parameters (for independent/conditional)
                "distribution_type": {
                    "type": ["string", "null"],
                    "enum": ["normal", "uniform", "categorical", "boolean", None],
                },
                "mean": {"type": ["number", "null"]},
                "std": {"type": ["number", "null"]},
                "min": {"type": ["number", "null"]},
                "max": {"type": ["number", "null"]},
                "options": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "weights": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                },
                "probability_true": {"type": ["number", "null"]},
                # For derived attributes
                "formula": {
                    "type": ["string", "null"],
                    "description": "Python expression for derived attributes",
                },
                # For conditional attributes
                "modifiers": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "object",
                        "properties": {
                            "when": {"type": "string"},
                            "multiply": {"type": ["number", "null"]},
                            "add": {"type": ["number", "null"]},
                        },
                        "required": ["when"],
                        "additionalProperties": False,
                    },
                },
                # Constraints
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["min", "max", "expression"]},
                            "value": {"type": ["number", "null"]},
                            "expression": {"type": ["string", "null"]},
                        },
                        "required": ["type"],
                        "additionalProperties": False,
                    },
                },
                # Grounding
                "grounding_level": {
                    "type": "string",
                    "enum": ["strong", "medium", "low"],
                },
                "grounding_method": {
                    "type": "string",
                    "enum": ["researched", "extrapolated", "estimated", "computed"],
                },
                "grounding_source": {"type": ["string", "null"]},
                "grounding_note": {"type": ["string", "null"]},
            },
            "required": [
                "name", "sampling_strategy", "distribution_type",
                "mean", "std", "min", "max", "options", "weights", "probability_true",
                "formula", "modifiers", "constraints",
                "grounding_level", "grounding_method", "grounding_source", "grounding_note",
            ],
            "additionalProperties": False,
        }
        attribute_schemas.append(attr_schema)
    
    return {
        "type": "object",
        "properties": {
            "attributes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "sampling_strategy": {
                            "type": "string",
                            "enum": ["independent", "derived", "conditional"],
                        },
                        "distribution_type": {
                            "type": ["string", "null"],
                            "enum": ["normal", "uniform", "categorical", "boolean", None],
                        },
                        "mean": {"type": ["number", "null"]},
                        "std": {"type": ["number", "null"]},
                        "min": {"type": ["number", "null"]},
                        "max": {"type": ["number", "null"]},
                        "options": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                        "weights": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                        },
                        "probability_true": {"type": ["number", "null"]},
                        "formula": {"type": ["string", "null"]},
                        "modifiers": {
                            "type": ["array", "null"],
                            "items": {
                                "type": "object",
                                "properties": {
                                    "when": {"type": "string"},
                                    "multiply": {"type": ["number", "null"]},
                                    "add": {"type": ["number", "null"]},
                                },
                                "required": ["when"],
                                "additionalProperties": False,
                            },
                        },
                        "constraints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["min", "max", "expression"]},
                                    "value": {"type": ["number", "null"]},
                                    "expression": {"type": ["string", "null"]},
                                },
                                "required": ["type"],
                                "additionalProperties": False,
                            },
                        },
                        "grounding_level": {"type": "string", "enum": ["strong", "medium", "low"]},
                        "grounding_method": {"type": "string", "enum": ["researched", "extrapolated", "estimated", "computed"]},
                        "grounding_source": {"type": ["string", "null"]},
                        "grounding_note": {"type": ["string", "null"]},
                    },
                    "required": [
                        "name", "sampling_strategy", "distribution_type",
                        "mean", "std", "min", "max", "options", "weights", "probability_true",
                        "formula", "modifiers", "constraints",
                        "grounding_level", "grounding_method", "grounding_source", "grounding_note",
                    ],
                    "additionalProperties": False,
                },
            },
            "methodology_notes": {
                "type": "string",
                "description": "Notes on research methodology and data quality",
            },
        },
        "required": ["attributes", "methodology_notes"],
        "additionalProperties": False,
    }


def _parse_distribution(attr_data: dict, attr_type: str):
    """Parse distribution from hydration response."""
    dist_type = attr_data.get("distribution_type")
    
    if dist_type == "normal":
        return NormalDistribution(
            mean=attr_data.get("mean", 0),
            std=attr_data.get("std", 1),
            min=attr_data.get("min"),
            max=attr_data.get("max"),
        )
    elif dist_type == "uniform":
        return UniformDistribution(
            min=attr_data.get("min", 0),
            max=attr_data.get("max", 1),
        )
    elif dist_type == "categorical":
        options = attr_data.get("options", [])
        weights = attr_data.get("weights")
        if not weights or len(weights) != len(options):
            weights = [1.0 / len(options)] * len(options) if options else []
        return CategoricalDistribution(
            options=options,
            weights=weights,
        )
    elif dist_type == "boolean":
        return BooleanDistribution(
            probability_true=attr_data.get("probability_true", 0.5),
        )
    
    # Default based on attribute type
    if attr_type == "int":
        return NormalDistribution(mean=50, std=15, min=0, max=100)
    elif attr_type == "float":
        return UniformDistribution(min=0, max=1)
    elif attr_type == "categorical":
        return CategoricalDistribution(options=["unknown"], weights=[1.0])
    else:
        return BooleanDistribution(probability_true=0.5)


def _parse_modifiers(modifiers_data: list[dict] | None) -> list[Modifier]:
    """Parse modifiers from hydration response."""
    if not modifiers_data:
        return []
    
    modifiers = []
    for mod in modifiers_data:
        modifiers.append(Modifier(
            when=mod["when"],
            multiply=mod.get("multiply"),
            add=mod.get("add"),
        ))
    return modifiers


def _parse_constraints(constraints_data: list[dict]) -> list[Constraint]:
    """Parse constraints from hydration response."""
    constraints = []
    for c in constraints_data:
        constraints.append(Constraint(
            type=c["type"],
            value=c.get("value"),
            expression=c.get("expression"),
        ))
    return constraints


def hydrate_attributes(
    attributes: list[DiscoveredAttribute],
    description: str,
    geography: str | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[list[HydratedAttribute], list[str]]:
    """
    Research distributions for discovered attributes using agentic web search.
    
    The model will:
    1. Search for real demographic/statistical data
    2. Determine appropriate distributions
    3. Assess grounding quality per attribute
    
    Args:
        attributes: List of DiscoveredAttribute from selector
        description: Original population description
        geography: Geographic scope for research
        model: Model to use
        reasoning_effort: "low", "medium", or "high"
    
    Returns:
        Tuple of (list of HydratedAttribute, list of source URLs)
    """
    if not attributes:
        return [], []
    
    geo_context = f" in {geography}" if geography else ""
    
    # Build attribute list for prompt
    attr_list = "\n".join(
        f"- {attr.name} ({attr.type}, {attr.category}): {attr.description}"
        + (f" [depends on: {', '.join(attr.depends_on)}]" if attr.depends_on else "")
        for attr in attributes
    )
    
    prompt = f"""Research realistic distributions for these attributes of {description}{geo_context}:

{attr_list}

## Your Task

For EACH attribute, research and provide:

### 1. Sampling Strategy
- **independent**: Sample directly from distribution (most attributes)
- **derived**: Compute from formula using other attributes (e.g., years_practice = age - training_age)
- **conditional**: Sample from distribution, then apply modifiers based on other attributes

### 2. Distribution Parameters (for independent/conditional)
Based on attribute type, provide:
- **int/float with normal**: mean, std, min, max
- **int/float with uniform**: min, max
- **categorical**: options list and weights (probabilities summing to ~1.0)
- **boolean**: probability_true (0-1)

### 3. Formula (for derived only)
Python expression using other attribute names, e.g.:
- "max(0, age - 28)"
- "income * 0.3"

### 4. Modifiers (for conditional only)
List of conditions that modify the base distribution:
- when: Python condition (e.g., "specialty == 'cardiology'")
- multiply: factor to multiply (e.g., 1.25 for 25% increase)
- add: value to add

### 5. Constraints
Hard limits that must be satisfied:
- type: "min", "max", or "expression"
- value: for min/max
- expression: Python expression for complex constraints

### 6. Grounding Quality
For EACH attribute, honestly assess:
- **level**: "strong" (direct data found), "medium" (extrapolated), "low" (estimated)
- **method**: "researched" (from data), "extrapolated" (inferred), "estimated" (guessed), "computed" (derived)
- **source**: Citation or URL if available
- **note**: Any caveats

## Research Guidelines

- Use web search to find real statistics from official sources
- Prefer: government data, professional associations, academic studies, industry reports
- Be honest about data quality - mark as "low" if estimating
- Use {geography or "appropriate"} units and categories

Search for data on each attribute category before filling in distributions."""

    schema = _build_hydration_schema(attributes)
    
    data, sources = agentic_research(
        prompt=prompt,
        response_schema=schema,
        schema_name="attribute_hydration",
        model=model,
        reasoning_effort=reasoning_effort,
    )
    
    # Build lookup for original attributes
    attr_lookup = {a.name: a for a in attributes}
    
    # Parse response into HydratedAttributes
    hydrated = []
    for attr_data in data.get("attributes", []):
        name = attr_data.get("name")
        original = attr_lookup.get(name)
        
        if not original:
            continue
        
        # Build sampling config
        strategy = attr_data.get("sampling_strategy", "independent")
        
        sampling = SamplingConfig(
            strategy=strategy,
            distribution=_parse_distribution(attr_data, original.type) if strategy != "derived" else None,
            formula=attr_data.get("formula") if strategy == "derived" else None,
            depends_on=original.depends_on,
            modifiers=_parse_modifiers(attr_data.get("modifiers")) if strategy == "conditional" else [],
        )
        
        # Build grounding info
        grounding = GroundingInfo(
            level=attr_data.get("grounding_level", "low"),
            method=attr_data.get("grounding_method", "estimated"),
            source=attr_data.get("grounding_source"),
            note=attr_data.get("grounding_note"),
        )
        
        hydrated.append(HydratedAttribute(
            name=original.name,
            type=original.type,
            category=original.category,
            description=original.description,
            depends_on=original.depends_on,
            sampling=sampling,
            grounding=grounding,
            constraints=_parse_constraints(attr_data.get("constraints", [])),
        ))
    
    return hydrated, sources

