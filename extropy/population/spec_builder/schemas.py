"""JSON Schema builders for LLM structured output.

These schemas define the expected format for LLM responses during hydration.
"""


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
