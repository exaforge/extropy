"""Expression and formula validation primitives.

This module provides shared utilities for validating Python expressions
used in formulas, conditions, and constraints across the codebase.

All functions return simple types (str | None for errors, set[str] for names)
to avoid coupling with domain-specific validation models.
"""

import ast
import re


# =============================================================================
# Constants
# =============================================================================

BUILTIN_NAMES = frozenset(
    {
        # Boolean literals
        "True",
        "False",
        "true",
        "false",
        "None",
        # Common functions
        "abs",
        "min",
        "max",
        "round",
        "int",
        "float",
        "str",
        "len",
        "sum",
        "all",
        "any",
        "bool",
        # Logical operators (for AST walking)
        "and",
        "or",
        "not",
        "in",
        "is",
        "if",
        "else",
    }
)

PYTHON_KEYWORDS = frozenset(
    {
        "and",
        "or",
        "not",
        "in",
        "is",
        "True",
        "False",
        "true",
        "false",
        "None",
        "if",
        "else",
    }
)

SAFE_CALL_NAMES = frozenset(
    {
        "abs",
        "min",
        "max",
        "round",
        "int",
        "float",
        "str",
        "len",
        "sum",
        "all",
        "any",
        "bool",
    }
)


# =============================================================================
# Name Extraction
# =============================================================================


def extract_names_from_expression(expr: str) -> set[str]:
    """Extract variable names from a Python expression.

    Uses AST parsing to correctly identify variable references
    while ignoring string literals, builtins, and keywords.

    Args:
        expr: A Python expression string (e.g., "age + income * 0.5")

    Returns:
        Set of variable names found in the expression

    Example:
        >>> extract_names_from_expression("age > 30 and income < 50000")
        {'age', 'income'}
    """
    try:
        tree = ast.parse(expr, mode="eval")
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
                if name not in BUILTIN_NAMES:
                    names.add(name)
        return names
    except SyntaxError:
        # Fallback to regex for malformed expressions
        cleaned = re.sub(r"'[^']*'", "", expr)
        cleaned = re.sub(r'"[^"]*"', "", cleaned)
        tokens = re.findall(r"\b([a-z_][a-z0-9_]*)\b", cleaned, re.IGNORECASE)
        return {
            t for t in tokens if t not in PYTHON_KEYWORDS and t not in BUILTIN_NAMES
        }


# =============================================================================
# Syntax Validation
# =============================================================================


def validate_expression_syntax(expr: str | None) -> str | None:
    """Validate Python expression syntax.

    Performs quick structural checks before attempting AST parse.
    Returns an error message if invalid, None if valid.

    Args:
        expr: A Python expression string

    Returns:
        Error message string if invalid, None if valid

    Example:
        >>> validate_expression_syntax("age > 30")
        None
        >>> validate_expression_syntax("age > ")
        "invalid Python syntax: ..."
    """
    if not expr:
        return None

    # Check for unterminated strings
    if expr.count('"') % 2 != 0:
        return "unterminated string literal (unmatched double quote)"

    if expr.count("'") % 2 != 0:
        return "unterminated string literal (unmatched single quote)"

    # Check for unbalanced parentheses
    if expr.count("(") != expr.count(")"):
        return (
            f"unbalanced parentheses ({expr.count('(')} open, {expr.count(')')} close)"
        )

    # Check for unbalanced brackets
    if expr.count("[") != expr.count("]"):
        return f"unbalanced brackets ({expr.count('[')} open, {expr.count(']')} close)"

    # Try to parse as Python expression
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError as e:
        error_msg = str(e.msg) if hasattr(e, "msg") else str(e)
        return f"invalid Python syntax: {error_msg}"

    # Validate allowed AST nodes to keep expressions safe and consistent
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    safety_error = _validate_safe_expression_tree(tree)
    if safety_error:
        return safety_error

    return None


def _validate_safe_expression_tree(tree: ast.AST) -> str | None:
    """Validate expression AST for safe, supported nodes only."""
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.Name,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.Call,
        ast.IfExp,
        ast.Load,
        ast.keyword,
    )
    allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    allowed_unary_ops = (ast.UAdd, ast.USub, ast.Not)
    allowed_bool_ops = (ast.And, ast.Or)
    allowed_cmp_ops = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return f"unsupported expression element: {type(node).__name__}"

        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return "expression may not reference dunder names"

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_bin_ops):
                return f"operator not allowed: {type(node.op).__name__}"

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unary_ops):
                return f"unary operator not allowed: {type(node.op).__name__}"

        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, allowed_bool_ops):
                return f"boolean operator not allowed: {type(node.op).__name__}"

        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, allowed_cmp_ops):
                    return f"comparison operator not allowed: {type(op).__name__}"

        if isinstance(node, ast.Dict):
            for key in node.keys:
                if key is None:
                    return "dict unpacking is not allowed"

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return "function calls must be direct names"
            if node.func.id not in SAFE_CALL_NAMES:
                return f"function '{node.func.id}' is not allowed"
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    return "star-args are not allowed"
            for kw in node.keywords:
                if kw.arg is None:
                    return "keyword splats are not allowed"

    return None


# =============================================================================
# Comparison Extraction
# =============================================================================


def extract_comparisons_from_expression(expr: str) -> list[tuple[str, list[str]]]:
    """Extract (variable_name, [compared_values]) pairs from a condition expression.

    Uses AST parsing to correctly handle compound conditions like:
    - employer_type == 'x' and job_title in ['y', 'z']
    - (age > 50 or job_title == 'chief') and employer_type == 'university'

    Only extracts comparisons where the left-hand side is a variable name
    and the right-hand side contains string literals.

    Args:
        expr: A Python condition expression string

    Returns:
        List of (variable_name, [string_values]) tuples

    Example:
        >>> extract_comparisons_from_expression("job_title == 'chief' and age > 50")
        [('job_title', ['chief'])]
        >>> extract_comparisons_from_expression("status in ['active', 'pending']")
        [('status', ['active', 'pending'])]
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return []

    comparisons: list[tuple[str, list[str]]] = []

    def _extract_string_values(node) -> list[str]:
        """Extract string values from a node (handles lists and single values)."""
        values = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
        elif isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    values.append(elt.value)
        return values

    def _visit(node):
        """Recursively visit AST nodes to find comparisons."""
        if isinstance(node, ast.Compare):
            # Handle: attr == 'value' or attr in ['val1', 'val2']
            left = node.left
            if isinstance(left, ast.Name):
                attr_name = left.id
                values = []
                for comparator in node.comparators:
                    values.extend(_extract_string_values(comparator))
                if values:
                    comparisons.append((attr_name, values))

        # Recurse into child nodes
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return comparisons
