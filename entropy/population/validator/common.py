"""Common utilities shared across validation modules.

This module contains constants and helper functions used by multiple
validation modules to avoid code duplication.
"""

import ast
import re


# =============================================================================
# Constants
# =============================================================================

BUILTIN_NAMES = {
    "True",
    "False",
    "true",
    "false",
    "None",
    "abs",
    "min",
    "max",
    "round",
    "int",
    "float",
    "str",
    "len",
}

PYTHON_KEYWORDS = {
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


# =============================================================================
# Name Extraction Helpers
# =============================================================================


def extract_names_from_expression(expr: str) -> set[str]:
    """Extract variable names from a Python expression.

    Uses AST parsing to correctly identify variable references
    while ignoring string literals and other constants.

    Args:
        expr: A Python expression string (e.g., "age + income * 0.5")

    Returns:
        Set of variable names found in the expression
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


# Aliases for backwards compatibility
extract_names_from_formula = extract_names_from_expression
extract_names_from_condition = extract_names_from_expression
