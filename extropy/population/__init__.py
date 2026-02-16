"""Population Creation module for Extropy (Phase 1).

This package handles all aspects of population creation:
- spec_builder/: Spec generation pipeline (sufficiency → selection → hydration → binding)
- sampler/: Agent sampling from specs
- validator/: Spec validation before sampling
"""

# Spec builder exports
from .spec_builder import (
    check_sufficiency,
    select_attributes,
    hydrate_attributes,
    hydrate_independent,
    hydrate_derived,
    hydrate_conditional_base,
    hydrate_conditional_modifiers,
    hydrate_household_config,
    hydrate_name_config,
    bind_constraints,
    build_spec,
)

# Sampler exports
from .sampler import (
    sample_population,
    save_json,
    save_sqlite,
    SamplingError,
    SamplingResult,
    SamplingStats,
    eval_safe,
    eval_formula,
    eval_condition,
    FormulaError,
    ConditionError,
    sample_distribution,
    coerce_to_type,
    apply_modifiers_and_sample,
)

# Validator exports
from .validator import (
    Severity,
    ValidationIssue,
    ValidationResult,
    validate_spec,
)

__all__ = [
    # Spec builder
    "check_sufficiency",
    "select_attributes",
    "hydrate_attributes",
    "hydrate_independent",
    "hydrate_derived",
    "hydrate_conditional_base",
    "hydrate_conditional_modifiers",
    "hydrate_household_config",
    "hydrate_name_config",
    "bind_constraints",
    "build_spec",
    # Sampler
    "sample_population",
    "save_json",
    "save_sqlite",
    "SamplingError",
    "SamplingResult",
    "SamplingStats",
    "eval_safe",
    "eval_formula",
    "eval_condition",
    "FormulaError",
    "ConditionError",
    "sample_distribution",
    "coerce_to_type",
    "apply_modifiers_and_sample",
    # Validator
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    "validate_spec",
]
