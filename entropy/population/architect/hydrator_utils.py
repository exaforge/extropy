"""Utilities for attribute hydration: schemas and parsers.

This module provides supporting functions for the hydration pipeline:
- JSON schema builders for LLM structured output
- Distribution and constraint parsers
- Formula sanitization

Validation functions have been moved to population/validator/hydration.py
"""

import re

from ...core.models import (
    Constraint,
    Modifier,
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
)


# =============================================================================
# Formula Sanitization
# =============================================================================


def sanitize_formula(formula: str | None) -> str | None:
    """Sanitize a formula/expression by fixing common LLM mistakes.

    Common issues fixed:
    - Stray braces from JSON examples (e.g., "age - 28}" -> "age - 28")
    - Leading/trailing whitespace
    - Lowercase boolean literals (true/false -> True/False)

    Args:
        formula: The formula string to sanitize

    Returns:
        Sanitized formula, or None if input is None/empty
    """
    if not formula:
        return None

    result = formula.strip()

    # Remove stray braces at start/end (common LLM mistake from JSON examples)
    while result.endswith("}") and result.count("{") < result.count("}"):
        result = result[:-1].rstrip()
    while result.startswith("{") and result.count("{") > result.count("}"):
        result = result[1:].lstrip()

    # Fix lowercase boolean literals to Python style
    # Only replace standalone words, not parts of identifiers
    result = re.sub(r"\btrue\b", "True", result)
    result = re.sub(r"\bfalse\b", "False", result)

    return result if result else None


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
                                    "enum": [
                                        "normal",
                                        "lognormal",
                                        "uniform",
                                        "beta",
                                        "categorical",
                                        "boolean",
                                    ],
                                },
                                "mean": {"type": ["number", "null"]},
                                "std": {"type": ["number", "null"]},
                                "min": {"type": ["number", "null"]},
                                "max": {"type": ["number", "null"]},
                                "alpha": {"type": ["number", "null"]},
                                "beta": {"type": ["number", "null"]},
                                "options": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"},
                                },
                                "weights": {
                                    "type": ["array", "null"],
                                    "items": {"type": "number"},
                                },
                                "probability_true": {"type": ["number", "null"]},
                            },
                            "required": [
                                "type",
                                "mean",
                                "std",
                                "min",
                                "max",
                                "alpha",
                                "beta",
                                "options",
                                "weights",
                                "probability_true",
                            ],
                            "additionalProperties": False,
                        },
                        "constraints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "hard_min",
                                            "hard_max",
                                            "expression",
                                            "spec_expression",
                                        ],
                                    },
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
                                "level": {
                                    "type": "string",
                                    "enum": ["strong", "medium", "low"],
                                },
                                "method": {
                                    "type": "string",
                                    "enum": ["researched", "extrapolated", "estimated"],
                                },
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
                                    "enum": [
                                        "normal",
                                        "lognormal",
                                        "uniform",
                                        "beta",
                                        "categorical",
                                        "boolean",
                                    ],
                                },
                                "mean": {"type": ["number", "null"]},
                                "std": {"type": ["number", "null"]},
                                "mean_formula": {"type": ["string", "null"]},
                                "std_formula": {"type": ["string", "null"]},
                                "min": {"type": ["number", "null"]},
                                "max": {"type": ["number", "null"]},
                                "min_formula": {"type": ["string", "null"]},
                                "max_formula": {"type": ["string", "null"]},
                                "alpha": {"type": ["number", "null"]},
                                "beta": {"type": ["number", "null"]},
                                "options": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"},
                                },
                                "weights": {
                                    "type": ["array", "null"],
                                    "items": {"type": "number"},
                                },
                                "probability_true": {"type": ["number", "null"]},
                            },
                            "required": [
                                "type",
                                "mean",
                                "std",
                                "mean_formula",
                                "std_formula",
                                "min",
                                "max",
                                "min_formula",
                                "max_formula",
                                "alpha",
                                "beta",
                                "options",
                                "weights",
                                "probability_true",
                            ],
                            "additionalProperties": False,
                        },
                        "constraints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "hard_min",
                                            "hard_max",
                                            "expression",
                                            "spec_expression",
                                        ],
                                    },
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
                                "level": {
                                    "type": "string",
                                    "enum": ["strong", "medium", "low"],
                                },
                                "method": {
                                    "type": "string",
                                    "enum": ["researched", "extrapolated", "estimated"],
                                },
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
                                    "probability_override": {
                                        "type": ["number", "null"]
                                    },
                                },
                                "required": [
                                    "when",
                                    "multiply",
                                    "add",
                                    "weight_overrides",
                                    "probability_override",
                                ],
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
    """Parse distribution from LLM response data.

    Includes defensive checks and fallbacks for incomplete or invalid data.
    """
    if not dist_data:
        return default_distribution(attr_type)

    dist_type = dist_data.get("type")

    # If distribution type is None or unrecognized, return a default distribution
    if dist_type is None:
        return default_distribution(attr_type)

    if dist_type == "normal":
        return NormalDistribution(
            mean=dist_data.get("mean"),
            std=dist_data.get("std"),
            min=dist_data.get("min"),
            max=dist_data.get("max"),
            mean_formula=sanitize_formula(dist_data.get("mean_formula")),
            std_formula=sanitize_formula(dist_data.get("std_formula")),
            min_formula=sanitize_formula(dist_data.get("min_formula")),
            max_formula=sanitize_formula(dist_data.get("max_formula")),
        )
    elif dist_type == "lognormal":
        return LognormalDistribution(
            mean=dist_data.get("mean"),
            std=dist_data.get("std"),
            min=dist_data.get("min"),
            max=dist_data.get("max"),
            mean_formula=sanitize_formula(dist_data.get("mean_formula")),
            std_formula=sanitize_formula(dist_data.get("std_formula")),
            min_formula=sanitize_formula(dist_data.get("min_formula")),
            max_formula=sanitize_formula(dist_data.get("max_formula")),
        )
    elif dist_type == "uniform":
        return UniformDistribution(
            min=dist_data.get("min", 0),
            max=dist_data.get("max", 1),
        )
    elif dist_type == "beta":
        # Default alpha/beta to 2.0 if missing or non-positive
        alpha = dist_data.get("alpha")
        beta_val = dist_data.get("beta")
        if alpha is None or alpha <= 0:
            alpha = 2.0
        if beta_val is None or beta_val <= 0:
            beta_val = 2.0
        return BetaDistribution(
            alpha=alpha,
            beta=beta_val,
            min=dist_data.get("min"),
            max=dist_data.get("max"),
            min_formula=sanitize_formula(dist_data.get("min_formula")),
            max_formula=sanitize_formula(dist_data.get("max_formula")),
        )
    elif dist_type == "categorical":
        # Handle explicit null from LLM response (get returns None, not default)
        options = dist_data.get("options") or []
        weights = dist_data.get("weights") or []
        # If no options, return a default categorical
        if not options:
            return CategoricalDistribution(options=["unknown"], weights=[1.0])
        if not weights or len(weights) != len(options):
            weights = [1.0 / len(options)] * len(options)
        return CategoricalDistribution(
            options=options,
            weights=weights,
        )
    elif dist_type == "boolean":
        # Default probability_true to 0.5 if missing
        prob = dist_data.get("probability_true")
        if prob is None:
            prob = 0.5
        return BooleanDistribution(
            probability_true=prob,
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
                when=sanitize_formula(mod["when"]) or mod["when"],
                multiply=mod.get("multiply"),
                add=mod.get("add"),
                weight_overrides=mod.get("weight_overrides"),
                probability_override=mod.get("probability_override"),
            )
        )
    return modifiers
