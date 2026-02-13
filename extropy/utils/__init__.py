"""Pure utility functions for Extropy.

This module contains pure functions with ZERO dependencies on extropy models
or other extropy modules. These are foundational utilities that can be
imported from anywhere without circular import risk.

Modules:
- graphs: Topological sort and cycle detection
- expressions: AST parsing and syntax validation
- distributions: Distribution parameter validation
- eval_safe: Safe expression evaluation
"""

from .graphs import topological_sort, CircularDependencyError
from .expressions import (
    extract_names_from_expression,
    validate_expression_syntax,
    extract_comparisons_from_expression,
    BUILTIN_NAMES,
    PYTHON_KEYWORDS,
)
from .distributions import (
    validate_weight_sum,
    validate_weights_options_match,
    validate_probability_range,
    validate_min_max,
    validate_std_positive,
    validate_beta_params,
    validate_options_not_empty,
)
from .eval_safe import (
    eval_safe,
    eval_formula,
    eval_condition,
    FormulaError,
    ConditionError,
    SAFE_BUILTINS,
)
from .paths import (
    resolve_relative_to,
    make_relative_to,
)

__all__ = [
    # Graphs
    "topological_sort",
    "CircularDependencyError",
    # Expressions
    "extract_names_from_expression",
    "validate_expression_syntax",
    "extract_comparisons_from_expression",
    "BUILTIN_NAMES",
    "PYTHON_KEYWORDS",
    # Distributions
    "validate_weight_sum",
    "validate_weights_options_match",
    "validate_probability_range",
    "validate_min_max",
    "validate_std_positive",
    "validate_beta_params",
    "validate_options_not_empty",
    # Eval
    "eval_safe",
    "eval_formula",
    "eval_condition",
    "FormulaError",
    "ConditionError",
    "SAFE_BUILTINS",
    # Paths
    "resolve_relative_to",
    "make_relative_to",
]
