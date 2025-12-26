"""Unified validation models for Entropy.

This module provides validation types used across population specs,
scenario specs, and other validation contexts. All validation in
Entropy should use these models for consistency.

Classes:
- Severity: ERROR, WARNING, INFO levels
- ValidationIssue: A single validation issue with context
- ValidationResult: Collection of issues with helper methods
"""

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Blocks processing
    WARNING = "warning"  # Processing continues
    INFO = "info"  # Informational only


class ValidationIssue(BaseModel):
    """A single validation issue found during validation.

    This unified model supports:
    - Population spec validation (syntactic/semantic checks)
    - Scenario spec validation
    - LLM response validation (fail-fast with retry)

    Attributes:
        severity: ERROR blocks processing, WARNING/INFO do not
        category: Issue category (e.g., 'FORMULA_SYNTAX', 'WEIGHT_INVALID')
        location: Where the issue occurred (attribute name, path, etc.)
        message: Human-readable description of the issue
        suggestion: Optional fix suggestion
        modifier_index: For modifier-specific issues
        value: The problematic value (useful for LLM retry prompts)
    """

    severity: Severity = Field(
        default=Severity.ERROR,
        description="Issue severity level",
    )
    category: str = Field(
        description="Category of issue (e.g., 'FORMULA_SYNTAX', 'TYPE_MISMATCH')",
    )
    location: str = Field(
        description="Where the issue occurred (attribute name, field path, etc.)",
    )
    message: str = Field(
        description="Human-readable description of the issue",
    )
    suggestion: str | None = Field(
        default=None,
        description="How to fix the issue",
    )
    modifier_index: int | None = Field(
        default=None,
        description="Index of the modifier if issue is modifier-specific",
    )
    value: str | None = Field(
        default=None,
        description="The problematic value (for LLM retry context)",
    )

    def __str__(self) -> str:
        """Format as human-readable string."""
        loc = self.location
        if self.modifier_index is not None:
            loc = f"{self.location}[{self.modifier_index}]"
        return f"{loc}: {self.message}"

    def for_llm_retry(self) -> str:
        """Format for LLM retry prompt."""
        lines = [f"ERROR in {self.location}:"]
        if self.value:
            lines.append(f"  Value: {repr(self.value)}")
        lines.append(f"  Problem: {self.message}")
        if self.suggestion:
            lines.append(f"  Fix: {self.suggestion}")
        return "\n".join(lines)


class ValidationResult(BaseModel):
    """Result of validating a spec or LLM response.

    Provides a unified interface for all validation results with
    helper properties and methods for common operations.

    Example:
        >>> result = ValidationResult()
        >>> result.add_error("SYNTAX", "formula", "Invalid Python syntax")
        >>> result.valid
        False
        >>> print(result.format_for_retry())
        ## VALIDATION ERRORS - PLEASE FIX:
        ...
    """

    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="All validation issues found",
    )

    @property
    def valid(self) -> bool:
        """True if no ERROR-level issues."""
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """All ERROR-level issues."""
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """All WARNING-level issues."""
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def info(self) -> list[ValidationIssue]:
        """All INFO-level issues."""
        return [i for i in self.issues if i.severity == Severity.INFO]

    def add_error(
        self,
        category: str,
        location: str,
        message: str,
        suggestion: str | None = None,
        modifier_index: int | None = None,
        value: str | None = None,
    ) -> None:
        """Add an ERROR-level issue."""
        self.issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                category=category,
                location=location,
                message=message,
                suggestion=suggestion,
                modifier_index=modifier_index,
                value=value,
            )
        )

    def add_warning(
        self,
        category: str,
        location: str,
        message: str,
        suggestion: str | None = None,
        modifier_index: int | None = None,
    ) -> None:
        """Add a WARNING-level issue."""
        self.issues.append(
            ValidationIssue(
                severity=Severity.WARNING,
                category=category,
                location=location,
                message=message,
                suggestion=suggestion,
                modifier_index=modifier_index,
            )
        )

    def add_info(
        self,
        category: str,
        location: str,
        message: str,
    ) -> None:
        """Add an INFO-level issue."""
        self.issues.append(
            ValidationIssue(
                severity=Severity.INFO,
                category=category,
                location=location,
                message=message,
            )
        )

    def format_for_retry(self) -> str:
        """Format all errors as a retry prompt section for LLM."""
        if self.valid:
            return ""

        lines = [
            "## PREVIOUS ATTEMPT FAILED - PLEASE FIX THESE ERRORS:",
            "",
        ]
        for err in self.errors:
            lines.append(err.for_llm_retry())
            lines.append("")

        lines.append("Please regenerate the output with these issues fixed.")
        lines.append("")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Human-readable summary."""
        if self.valid and not self.warnings:
            return "Validation passed"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)

    def __add__(self, other: "ValidationResult") -> "ValidationResult":
        """Combine two validation results."""
        return ValidationResult(issues=self.issues + other.issues)

    def __iadd__(self, other: "ValidationResult") -> "ValidationResult":
        """In-place combine."""
        self.issues.extend(other.issues)
        return self


# Backwards compatibility aliases for scenario.py
# These map the old two-class pattern to the unified model
ValidationError = ValidationIssue
ValidationWarning = ValidationIssue
