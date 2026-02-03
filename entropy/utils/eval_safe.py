"""Safe expression evaluation for formulas and conditions.

Provides a restricted AST evaluator that only allows safe operations,
preventing attribute access, imports, and other dangerous operations.
"""

import ast
import operator
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


_SAFE_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}
_SAFE_BOOL_OPS = (ast.And, ast.Or)
_SAFE_CMP_OPS: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _eval_ast(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, context)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id.startswith("__"):
            raise FormulaError("dunder names are not allowed in expressions")
        if node.id in context:
            return context[node.id]
        if node.id in SAFE_BUILTINS:
            return SAFE_BUILTINS[node.id]
        raise FormulaError(f"Unknown name '{node.id}' in expression")

    if isinstance(node, ast.List):
        return [_eval_ast(elt, context) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(elt, context) for elt in node.elts)

    if isinstance(node, ast.Set):
        return {_eval_ast(elt, context) for elt in node.elts}

    if isinstance(node, ast.Dict):
        for key in node.keys:
            if key is None:
                raise FormulaError("Dict unpacking is not allowed")
        return {
            _eval_ast(key, context): _eval_ast(val, context)
            for key, val in zip(node.keys, node.values)
        }

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_UNARY_OPS:
            raise FormulaError(f"Unary operator not allowed: {op_type.__name__}")
        return _SAFE_UNARY_OPS[op_type](_eval_ast(node.operand, context))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_BIN_OPS:
            raise FormulaError(f"Binary operator not allowed: {op_type.__name__}")
        left = _eval_ast(node.left, context)
        right = _eval_ast(node.right, context)
        return _SAFE_BIN_OPS[op_type](left, right)

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _SAFE_BOOL_OPS):
            raise FormulaError(
                f"Boolean operator not allowed: {type(node.op).__name__}"
            )
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not _eval_ast(value, context):
                    return False
            return True
        # ast.Or
        for value in node.values:
            if _eval_ast(value, context):
                return True
        return False

    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _SAFE_CMP_OPS:
                raise FormulaError(
                    f"Comparison operator not allowed: {op_type.__name__}"
                )
            right = _eval_ast(comparator, context)
            if not _SAFE_CMP_OPS[op_type](left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        return (
            _eval_ast(node.body, context)
            if _eval_ast(node.test, context)
            else _eval_ast(node.orelse, context)
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FormulaError("Only direct function calls are allowed")
        func_name = node.func.id
        if func_name.startswith("__"):
            raise FormulaError("Dunder functions are not allowed")
        func = SAFE_BUILTINS.get(func_name)
        if not callable(func):
            raise FormulaError(f"Function '{func_name}' is not allowed")

        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise FormulaError("Star-args are not allowed")
            args.append(_eval_ast(arg, context))
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise FormulaError("Keyword splats are not allowed")
            kwargs[kw.arg] = _eval_ast(kw.value, context)

        return func(*args, **kwargs)

    raise FormulaError(f"Unsupported expression element: {type(node).__name__}")


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
    # Merge context into local namespace
    local_vars = dict(context)

    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_ast(tree, local_vars)
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


def eval_condition(
    condition: str,
    agent: dict[str, Any],
    *,
    raise_on_error: bool = False,
) -> bool:
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
        Returns False (not raises) on evaluation errors by default, since
        a failed condition just means the modifier doesn't apply. Set
        raise_on_error=True to raise ConditionError instead.
    """
    try:
        result = eval_safe(condition, agent)
        return bool(result)
    except Exception as e:
        if raise_on_error:
            raise ConditionError(f"Condition '{condition}' failed: {e}") from e
        # Condition failures are warnings, not errors
        # The modifier just doesn't apply
        return False
