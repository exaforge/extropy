"""Safe expression evaluation for formulas and conditions.

Provides a restricted eval environment that only allows safe builtins,
preventing file access, imports, and other dangerous operations.
"""

from typing import Any

# Safe builtins allowed in formula/condition evaluation
SAFE_BUILTINS = {
    "True": True,
    "False": False,
    "None": None,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "sum": sum,
    "all": all,
    "any": any,
    "bool": bool,
}


class FormulaError(Exception):
    """Raised when formula evaluation fails."""

    pass


class ConditionError(Exception):
    """Raised when condition evaluation fails."""

    pass


def eval_safe(expression: str, context: dict[str, Any]) -> Any:
    """
    Safely evaluate a Python expression with restricted builtins.

    Args:
        expression: Python expression string (e.g., "age - 28", "role == 'chief'")
        context: Dictionary of variable names to values

    Returns:
        Result of evaluating the expression

    Raises:
        FormulaError: If evaluation fails

    Example:
        >>> eval_safe("max(0, age - 26)", {"age": 45})
        19
        >>> eval_safe("role == 'chief'", {"role": "resident"})
        False
    """
    # Create restricted globals with only safe builtins
    restricted_globals = {"__builtins__": SAFE_BUILTINS}

    # Merge context into local namespace
    local_vars = dict(context)

    try:
        return eval(expression, restricted_globals, local_vars)
    except Exception as e:
        raise FormulaError(f"Failed to evaluate '{expression}': {e}") from e


def eval_formula(formula: str, agent: dict[str, Any]) -> Any:
    """
    Evaluate a formula expression using agent attributes.

    This is used for derived attributes where the value is computed
    from other attributes (e.g., years_experience = age - 26).

    Args:
        formula: Python expression string
        agent: Dictionary of already-sampled attribute values

    Returns:
        Computed value

    Raises:
        FormulaError: If formula evaluation fails
    """
    try:
        return eval_safe(formula, agent)
    except FormulaError:
        raise
    except Exception as e:
        raise FormulaError(f"Formula '{formula}' failed: {e}") from e


def eval_condition(condition: str, agent: dict[str, Any]) -> bool:
    """
    Evaluate a condition expression to determine if a modifier applies.

    Unlike formulas, condition failures are non-fatal - they just mean
    the modifier doesn't apply.

    Args:
        condition: Python boolean expression (e.g., "age < 32")
        agent: Dictionary of already-sampled attribute values

    Returns:
        True if condition is met, False otherwise

    Note:
        Returns False (not raises) on evaluation errors, since a failed
        condition just means the modifier doesn't apply.
    """
    try:
        result = eval_safe(condition, agent)
        return bool(result)
    except Exception:
        # Condition failures are warnings, not errors
        # The modifier just doesn't apply
        return False
