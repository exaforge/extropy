"""Prompt content and response handling for hydration steps.

This module contains:
- FORMULA_SYNTAX_GUIDELINES: Instructions injected into prompts
- make_validator: Factory for LLM response validators
- Context formatting helpers for consistent prompt structure
"""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ....core.models import AttributeSpec, HydratedAttribute


# =============================================================================
# Prompt Content
# =============================================================================

FORMULA_SYNTAX_GUIDELINES = """
## CRITICAL: Formula Syntax Rules

All formulas and expressions must be valid Python. Common errors to AVOID:

CORRECT:
- "max(0, 0.10 * age - 1.8)"
- "'18-24' if age < 25 else '25-34' if age < 35 else '35+'"
- "age > 50 and role == 'senior'"

WRONG (will cause pipeline failure):
- "max(0, 0.10 * age - 1.8   (missing closing quote)
- "age - 28 years"            (invalid Python - 'years' is not a variable)
- "'senior' if age > 50       (missing else clause)
- "specialty == cardiology"   (missing quotes around string)

Before outputting, mentally verify:
1. All quotes are paired (matching " or ')
2. All parentheses are balanced
3. The expression is valid Python syntax
"""


# =============================================================================
# Context Formatting Helpers
# =============================================================================


def format_context_section(
    context: "list[AttributeSpec] | None",
    instruction: str = "Do NOT redefine them, but you may reference them.",
) -> str:
    """Format read-only context attributes for prompts.

    Args:
        context: List of existing attributes from base population
        instruction: What the LLM should do with these attributes

    Returns:
        Formatted section string, empty if no context
    """
    if not context:
        return ""

    lines = [
        "## READ-ONLY CONTEXT ATTRIBUTES (from base population)",
        "",
        f"These attributes already exist. {instruction}",
        "",
    ]
    for attr in context:
        lines.append(f"- {attr.name} ({attr.type}): {attr.description}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def format_hydrated_section(
    attrs: "list[HydratedAttribute] | None",
    title: str = "Available Upstream Attributes",
) -> str:
    """Format already-hydrated attributes with distribution info.

    Args:
        attrs: List of hydrated attributes to display
        title: Section title

    Returns:
        Formatted section string, empty if no attrs
    """
    if not attrs:
        return ""

    lines = [f"## {title}", ""]

    for attr in attrs:
        dist_info = _get_distribution_info(attr)
        lines.append(f"- {attr.name} ({attr.type}): {attr.description}{dist_info}")

    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def format_conditional_context(
    independent_attrs: "list[HydratedAttribute] | None",
    derived_attrs: "list[HydratedAttribute] | None",
    conditional_attrs: "list[HydratedAttribute] | None" = None,
    context: "list[AttributeSpec] | None" = None,
    show_options: bool = False,
) -> str:
    """Format complete context for conditional hydration steps.

    Args:
        independent_attrs: Hydrated independent attributes
        derived_attrs: Hydrated derived attributes
        conditional_attrs: Hydrated conditional attributes (for step 2d)
        context: Read-only context from base population
        show_options: Whether to show VALID OPTIONS for categoricals

    Returns:
        Formatted context string
    """
    sections = []

    # Context attributes (read-only from base)
    if context:
        instruction = (
            "You can reference them in 'when' conditions."
            if show_options
            else "You can reference them in formulas."
        )
        ctx_lines = ["## READ-ONLY CONTEXT ATTRIBUTES (from base population)", ""]
        ctx_lines.append(f"These attributes already exist. {instruction}")
        ctx_lines.append("")
        for attr in context:
            opt_info = ""
            if show_options and attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "options") and dist.options:
                    opt_info = f"\n    VALID OPTIONS: {dist.options}"
            ctx_lines.append(
                f"- {attr.name} ({attr.type}): {attr.description}{opt_info}"
            )
        ctx_lines.extend(["", "---", ""])
        sections.append("\n".join(ctx_lines))

    # Full context section
    sections.append("## Full Context\n")

    # Independent attributes
    if independent_attrs:
        sections.append("**Independent Attributes:**")
        for attr in independent_attrs:
            dist_info = _get_distribution_info(attr, show_options=show_options)
            sections.append(
                f"- {attr.name} ({attr.type}): {attr.description}{dist_info}"
            )
        sections.append("")

    # Derived attributes
    if derived_attrs:
        sections.append("**Derived Attributes:**")
        for attr in derived_attrs:
            formula_info = (
                f" — formula: {attr.sampling.formula}" if attr.sampling.formula else ""
            )
            sections.append(
                f"- {attr.name} ({attr.type}): {attr.description}{formula_info}"
            )
        sections.append("")

    # Conditional attributes (for step 2d)
    if conditional_attrs:
        sections.append("**Conditional Attributes (with base distributions):**")
        for attr in conditional_attrs:
            dist_info = _get_conditional_dist_info(attr)
            deps_info = f" [depends on: {', '.join(attr.depends_on)}]"
            sections.append(
                f"- {attr.name} ({attr.type}): {attr.description}{dist_info}{deps_info}"
            )
        sections.append("")

    sections.append("---\n")
    return "\n".join(sections)


def _get_distribution_info(
    attr: "HydratedAttribute",
    show_options: bool = False,
) -> str:
    """Get distribution info string for display."""
    if not attr.sampling.distribution:
        if attr.sampling.formula:
            return f" (formula: {attr.sampling.formula[:30]}...)"
        return ""

    dist = attr.sampling.distribution

    if show_options and hasattr(dist, "options") and dist.options:
        return f"\n    VALID OPTIONS: {dist.options}"
    elif hasattr(dist, "mean") and dist.mean is not None:
        std = getattr(dist, "std", "?")
        return f" — mean={dist.mean}, std={std}"
    elif hasattr(dist, "options") and dist.options:
        return f" (options: {', '.join(dist.options[:3])}...)"

    return ""


def _get_conditional_dist_info(attr: "HydratedAttribute") -> str:
    """Get distribution info for conditional attributes."""
    if not attr.sampling.distribution:
        return ""

    dist = attr.sampling.distribution

    if hasattr(dist, "mean_formula") and dist.mean_formula:
        return f" — mean_formula: {dist.mean_formula}"
    elif hasattr(dist, "mean") and dist.mean is not None:
        return f" — base mean={dist.mean}"
    elif hasattr(dist, "options") and dist.options:
        return f" — options: {', '.join(dist.options)}"

    return ""


# =============================================================================
# Response Validation
# =============================================================================


def make_validator(validator_fn: Callable, *args) -> Callable[[dict], tuple[bool, str]]:
    """Create a validator closure for LLM response validation.

    Args:
        validator_fn: The validation function (e.g., validate_independent_response)
        *args: Additional arguments to pass to validator_fn after data

    Returns:
        A closure that returns (is_valid, error_message_for_retry)
    """

    def validate_response(data: dict) -> tuple[bool, str]:
        result = validator_fn(data, *args)
        if result.valid:
            return True, ""
        return False, result.format_for_retry()

    return validate_response
