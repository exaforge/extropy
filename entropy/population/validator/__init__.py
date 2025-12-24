"""Validator module for Entropy population specs.

This module validates specs before sampling to catch systematic LLM errors.
All checks are structural or mathematical - no sampling required.

Module structure:
- syntactic.py: Categories 1-9 (ERROR checks - blocks sampling)
- semantic.py: Categories 10-12 (WARNING checks - sampling proceeds)
- probabilistic.py: Stub for post-sample analysis (Phase 2)
"""

from enum import Enum
from dataclasses import dataclass, field

from ...core.models import PopulationSpec


class Severity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Blocks sampling
    WARNING = "warning"  # Sampling proceeds
    INFO = "info"        # Informational notes


@dataclass
class ValidationIssue:
    """A single validation issue found in a spec."""
    severity: Severity
    category: str
    attribute: str
    message: str
    modifier_index: int | None = None
    suggestion: str | None = None

    def __str__(self) -> str:
        loc = self.attribute
        if self.modifier_index is not None:
            loc = f"{self.attribute}[{self.modifier_index}]"
        return f"{loc}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a spec."""
    valid: bool  # True if no errors (warnings OK)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)

    @property
    def all_issues(self) -> list[ValidationIssue]:
        """All issues sorted by severity."""
        return self.errors + self.warnings + self.info

    def __str__(self) -> str:
        if self.valid and not self.warnings:
            return "Spec is valid"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)


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

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    info: list[ValidationIssue] = []

    # Run syntactic checks (ERROR level)
    syntactic_issues = run_syntactic_checks(spec)
    for issue in syntactic_issues:
        if issue.severity == Severity.ERROR:
            errors.append(issue)
        elif issue.severity == Severity.WARNING:
            warnings.append(issue)
        else:
            info.append(issue)

    # Run semantic checks (WARNING level)
    semantic_issues = run_semantic_checks(spec)
    for issue in semantic_issues:
        if issue.severity == Severity.ERROR:
            errors.append(issue)
        elif issue.severity == Severity.WARNING:
            warnings.append(issue)
        else:
            info.append(issue)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
    )


from .fixer import fix_modifier_conditions, fix_spec_file, ConditionFix, FixResult


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
