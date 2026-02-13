"""Spec-level validation orchestrator.

This module combines structural and semantic checks to validate
a complete PopulationSpec before sampling.
"""

from ...core.models import PopulationSpec
from ...core.models.validation import ValidationResult
from .structural import run_structural_checks
from .semantic import run_semantic_checks


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
    result = ValidationResult()

    # Run structural checks (ERROR level)
    structural_issues = run_structural_checks(spec)
    result.issues.extend(structural_issues)

    # Run semantic checks (WARNING level)
    semantic_issues = run_semantic_checks(spec)
    result.issues.extend(semantic_issues)

    return result
