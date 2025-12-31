"""Prompt content and response handling for hydration steps.

This module contains:
- FORMULA_SYNTAX_GUIDELINES: Instructions injected into prompts
- make_validator: Factory for LLM response validators
"""

from typing import Callable


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
# Response Validation
# =============================================================================


def make_validator(
    validator_fn: Callable, *args
) -> Callable[[dict], tuple[bool, str]]:
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
