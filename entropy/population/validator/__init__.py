"""Validator module for Entropy population specs.

This module validates specs before sampling to catch systematic LLM errors.
All checks are structural or mathematical - no sampling required.

Module structure:
- syntactic.py: Categories 1-9 (ERROR checks - blocks sampling)
- semantic.py: Categories 10-12 (WARNING checks - sampling proceeds)
- probabilistic.py: Stub for post-sample analysis (Phase 2)
"""

from ...core.models import PopulationSpec
from ...core.models.validation import (
    Severity,
    ValidationIssue,
    ValidationResult,
)

from .fixer import fix_modifier_conditions, fix_spec_file, ConditionFix, FixResult


def validate_spec(spec: PopulationSpec) -> ValidationResult:
    """
    Validate a PopulationSpec for structural and semantic correctness.

    Runs all validation checks and returns a result indicating whether
    the spec is valid for sampling.

    Args:
        spec: The PopulationSpec to validate

    Returns:
        ValidationResult with errors, warnings, and info

    Example:
        >>> result = validate_spec(population_spec)
        >>> if not result.valid:
        ...     for err in result.errors:
        ...         print(f"ERROR: {err}")
    """
    from .syntactic import run_syntactic_checks
    from .semantic import run_semantic_checks

    result = ValidationResult()

    # Run syntactic checks (ERROR level)
    syntactic_issues = run_syntactic_checks(spec)
    result.issues.extend(syntactic_issues)

    # Run semantic checks (WARNING level)
    semantic_issues = run_semantic_checks(spec)
    result.issues.extend(semantic_issues)

    return result


__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    "validate_spec",
    "fix_modifier_conditions",
    "fix_spec_file",
    "ConditionFix",
    "FixResult",
]
