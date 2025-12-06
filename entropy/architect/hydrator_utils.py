"""Utilities for attribute hydration: schemas, parsers, and validation.

This module provides supporting functions for the hydration pipeline:
- JSON schema builders for LLM structured output
- Distribution and constraint parsers
- Validation functions for each hydration step
"""

import ast
import re

from ..models import (
    Constraint,
    GroundingInfo,
    HydratedAttribute,
    Modifier,
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
)


# =============================================================================
# Constants
# =============================================================================

BUILTIN_NAMES = {'True', 'False', 'None', 'abs', 'min', 'max', 'round', 'int', 'float', 'str', 'len'}


# =============================================================================
# Name Extraction Helpers
# =============================================================================


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
        return extract_names_from_condition(formula)


def extract_names_from_condition(condition: str) -> set[str]:
    """Extract attribute names referenced in a when condition.

    Uses AST parsing to correctly identify variable references
    while ignoring string literals and other constants.
    """
    try:
        tree = ast.parse(condition, mode='eval')
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
                if name not in BUILTIN_NAMES:
                    names.add(name)
        return names
    except SyntaxError:
        # Fallback to regex for malformed expressions
        # First, remove quoted strings to avoid matching their contents
        cleaned = re.sub(r"'[^']*'", '', condition)
        cleaned = re.sub(r'"[^"]*"', '', cleaned)
        tokens = re.findall(r'\b([a-z_][a-z0-9_]*)\b', cleaned, re.IGNORECASE)
        keywords = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None', 'if', 'else'}
        return {t for t in tokens if t not in keywords and t not in BUILTIN_NAMES}


# =============================================================================
# Validation Functions
# =============================================================================


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
        if not attr.sampling.formula:
            errors.append(f"{attr.name}: derived attribute missing formula")
            continue

        try:
            ast.parse(attr.sampling.formula, mode='eval')
        except SyntaxError as e:
            errors.append(f"{attr.name}: invalid formula syntax: {e}")
            continue

        used_names = extract_names_from_formula(attr.sampling.formula)
        for name in used_names:
            if name in BUILTIN_NAMES:
                continue
            if name not in attr.depends_on:
                errors.append(f"{attr.name}: formula references '{name}' not in depends_on")
            elif name not in all_attribute_names:
                errors.append(f"{attr.name}: formula references unknown attribute '{name}'")

    return errors


def validate_conditional_base(attributes: list[HydratedAttribute]) -> list[str]:
    """Validate conditional attributes with base distributions."""
    errors = []

    for attr in attributes:
        dist = attr.sampling.distribution

        if dist is None:
            errors.append(f"{attr.name}: conditional attribute missing distribution")
            continue

        if isinstance(dist, (NormalDistribution, LognormalDistribution)) and dist.mean_formula:
            used_names = extract_names_from_formula(dist.mean_formula)
            for name in used_names:
                if name not in attr.depends_on and name not in BUILTIN_NAMES:
                    errors.append(f"{attr.name}: mean_formula references '{name}' not in depends_on")

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


def build_independent_schema() -> dict:
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
                            "required": ["type", "mean", "std", "min", "max", "alpha", "beta", "options", "weights", "probability_true"],
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
                                "required": ["type", "value", "expression", "reason"],
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
                            "required": ["level", "method", "source", "note"],
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


def build_derived_schema() -> dict:
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


def build_conditional_base_schema() -> dict:
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
                            "required": ["type", "mean", "std", "mean_formula", "std_formula", "min", "max", "alpha", "beta", "options", "weights", "probability_true"],
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
                                "required": ["type", "value", "expression", "reason"],
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
                            "required": ["level", "method", "source", "note"],
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


def build_modifiers_schema() -> dict:
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
                                "required": ["when", "multiply", "add", "weight_overrides", "probability_override"],
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
# Distribution and Constraint Parsers
# =============================================================================


def parse_distribution(dist_data: dict, attr_type: str):
    """Parse distribution from LLM response data."""
    if not dist_data:
        return default_distribution(attr_type)

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
        # Handle explicit null from LLM response (get returns None, not default)
        options = dist_data.get("options") or []
        weights = dist_data.get("weights") or []
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

    return default_distribution(attr_type)


def default_distribution(attr_type: str):
    """Get default distribution for attribute type."""
    if attr_type == "int":
        return NormalDistribution(mean=50, std=15, min=0, max=100)
    elif attr_type == "float":
        return UniformDistribution(min=0, max=1)
    elif attr_type == "categorical":
        return CategoricalDistribution(options=["unknown"], weights=[1.0])
    else:
        return BooleanDistribution(probability_true=0.5)


def parse_constraints(constraints_data: list[dict]) -> list[Constraint]:
    """Parse constraints from LLM response data."""
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


def parse_modifiers(modifiers_data: list[dict] | None) -> list[Modifier]:
    """Parse modifiers from LLM response data."""
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
