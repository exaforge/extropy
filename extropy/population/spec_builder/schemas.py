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


def build_household_config_schema() -> dict:
    """Build JSON schema for household config hydration.

    Uses array-of-objects patterns instead of dict/tuple schemas for LLM compatibility.
    Both Anthropic and OpenAI structured outputs require additionalProperties: false
    (not a schema) and don't support tuple-style array items.
    """
    return {
        "type": "object",
        "properties": {
            # Array of {upper_bound, label} instead of tuple [int, str]
            "age_brackets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "upper_bound": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["upper_bound", "label"],
                    "additionalProperties": False,
                },
            },
            # Array of {bracket, types: [{type, weight}]} instead of nested dict
            "household_type_weights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bracket": {"type": "string"},
                        "types": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "weight": {"type": "number"},
                                },
                                "required": ["type", "weight"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["bracket", "types"],
                    "additionalProperties": False,
                },
            },
            # Array of {group, rate} instead of dict
            "same_group_rates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group": {"type": "string"},
                        "rate": {"type": "number"},
                    },
                    "required": ["group", "rate"],
                    "additionalProperties": False,
                },
            },
            "default_same_group_rate": {"type": "number"},
            # Array of {attribute, correlation} instead of dict
            "assortative_mating": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "attribute": {"type": "string"},
                        "correlation": {"type": "number"},
                    },
                    "required": ["attribute", "correlation"],
                    "additionalProperties": False,
                },
            },
            "partner_age_gap_mean": {"type": "number"},
            "partner_age_gap_std": {"type": "number"},
            "min_adult_age": {"type": "integer"},
            "min_agent_age": {"type": "integer"},
            "child_min_parent_offset": {"type": "integer"},
            "child_max_parent_offset": {"type": "integer"},
            "max_dependent_child_age": {"type": "integer"},
            "elderly_min_offset": {"type": "integer"},
            "elderly_max_offset": {"type": "integer"},
            "life_stages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "max_age": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["max_age", "label"],
                    "additionalProperties": False,
                },
            },
            "adult_stage_label": {"type": "string"},
            "avg_household_size": {"type": "number"},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "age_brackets",
            "household_type_weights",
            "same_group_rates",
            "default_same_group_rate",
            "assortative_mating",
            "partner_age_gap_mean",
            "partner_age_gap_std",
            "min_adult_age",
            "child_min_parent_offset",
            "child_max_parent_offset",
            "max_dependent_child_age",
            "elderly_min_offset",
            "elderly_max_offset",
            "life_stages",
            "adult_stage_label",
            "avg_household_size",
            "sources",
        ],
        "additionalProperties": False,
    }


def build_name_config_schema() -> dict:
    """Build JSON schema for name config hydration."""
    name_entry = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "weight": {"type": "number"},
        },
        "required": ["name", "weight"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "male_first_names": {"type": "array", "items": name_entry},
            "female_first_names": {"type": "array", "items": name_entry},
            "last_names": {"type": "array", "items": name_entry},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "male_first_names",
            "female_first_names",
            "last_names",
            "sources",
        ],
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
