"""Population Creation module for Entropy (Phase 1).

This package handles all aspects of population creation:
- architect/: Spec generation pipeline (sufficiency → selection → hydration → binding)
- sampler/: Agent sampling from specs
- validator/: Spec validation before sampling
"""

# Architect layer exports
from .architect import (
    check_sufficiency,
    select_attributes,
    hydrate_attributes,
    hydrate_independent,
    hydrate_derived,
    hydrate_conditional_base,
    hydrate_conditional_modifiers,
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
    fix_modifier_conditions,
    fix_spec_file,
    ConditionFix,
    FixResult,
)

__all__ = [
    # Architect
    "check_sufficiency",
    "select_attributes",
    "hydrate_attributes",
    "hydrate_independent",
    "hydrate_derived",
    "hydrate_conditional_base",
    "hydrate_conditional_modifiers",
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
    "fix_modifier_conditions",
    "fix_spec_file",
    "ConditionFix",
    "FixResult",
]
