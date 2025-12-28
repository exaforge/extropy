"""Shared validation primitives for Entropy.

This module re-exports utilities from entropy.utils for backwards compatibility.
New code should import directly from entropy.utils instead.

Modules (now in utils/):
    expressions: Expression/formula syntax validation and name extraction
    distributions: Distribution parameter validation (weights, ranges, etc.)
    graphs: Dependency graph analysis (topological sort, cycle detection)
"""

# Re-export from utils for backwards compatibility
from ..utils.expressions import (
    BUILTIN_NAMES,
    PYTHON_KEYWORDS,
    extract_names_from_expression,
    extract_comparisons_from_expression,
    validate_expression_syntax,
)

from ..utils.distributions import (
    validate_weight_sum,
    validate_weights_options_match,
    validate_probability_range,
    validate_min_max,
    validate_std_positive,
    validate_beta_params,
    validate_options_not_empty,
)

from ..utils.graphs import (
    topological_sort,
    CircularDependencyError,
)

__all__ = [
    # Expression utilities
    "BUILTIN_NAMES",
    "PYTHON_KEYWORDS",
    "extract_names_from_expression",
    "extract_comparisons_from_expression",
    "validate_expression_syntax",
    # Distribution utilities
    "validate_weight_sum",
    "validate_weights_options_match",
    "validate_probability_range",
    "validate_min_max",
    "validate_std_positive",
    "validate_beta_params",
    "validate_options_not_empty",
    # Graph utilities
    "topological_sort",
    "CircularDependencyError",
]

