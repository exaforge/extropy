"""Validator module for Entropy population specs.

This module validates specs before sampling to catch systematic LLM errors.
All checks are structural or mathematical - no sampling required.

Module structure:
- spec.py: Main validate_spec() entry point
- structural.py: Categories 1-9 (ERROR checks - blocks sampling)
- semantic.py: Categories 10-12 (WARNING checks - sampling proceeds)
- llm_response.py: Fail-fast validation for LLM outputs
"""

from ...core.models.validation import (
    Severity,
    ValidationIssue,
    ValidationResult,
)

# Spec validation
from .spec import validate_spec

# LLM response validation
from .llm_response import (
    is_spec_level_constraint,
    extract_bound_from_constraint,
    validate_formula_syntax,
    validate_condition_syntax,
    validate_distribution_data,
    validate_modifier_data,
    validate_independent_response,
    validate_derived_response,
    validate_conditional_base_response,
    validate_modifiers_response,
)

__all__ = [
    # Core validation types
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    # Spec validation
    "validate_spec",
    # LLM response validation
    "validate_formula_syntax",
    "validate_condition_syntax",
    "validate_distribution_data",
    "validate_modifier_data",
    "validate_independent_response",
    "validate_derived_response",
    "validate_conditional_base_response",
    "validate_modifiers_response",
    # Utility functions
    "is_spec_level_constraint",
    "extract_bound_from_constraint",
]
