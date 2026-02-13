"""Distribution parameter validation primitives.

This module provides shared utilities for validating distribution parameters
(weights, probabilities, ranges, etc.) across the codebase.

All functions return simple error messages (str | None) to avoid coupling
with domain-specific validation models.
"""

from typing import Sequence


# =============================================================================
# Weight Validation
# =============================================================================


def validate_weight_sum(
    weights: Sequence[float] | None,
    tolerance: float = 0.02,
) -> str | None:
    """Validate that weights sum to approximately 1.0.

    Args:
        weights: List of probability weights
        tolerance: Allowed deviation from 1.0 (default 0.02)

    Returns:
        Error message if invalid, None if valid

    Example:
        >>> validate_weight_sum([0.3, 0.3, 0.4])
        None
        >>> validate_weight_sum([0.5, 0.5, 0.5])
        "weights sum to 1.500, should be 1.0"
    """
    if weights is None or len(weights) == 0:
        return None

    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > tolerance:
        return f"weights sum to {weight_sum:.3f}, should be 1.0"

    return None


def validate_weights_options_match(
    weights: Sequence[float] | None,
    options: Sequence[str] | None,
) -> str | None:
    """Validate that weights and options arrays have matching lengths.

    Args:
        weights: List of probability weights
        options: List of categorical options

    Returns:
        Error message if mismatched, None if valid

    Example:
        >>> validate_weights_options_match([0.5, 0.5], ["a", "b"])
        None
        >>> validate_weights_options_match([0.5], ["a", "b"])
        "weights (1) and options (2) length mismatch"
    """
    if weights is None or options is None:
        return None

    if len(weights) != len(options):
        return f"weights ({len(weights)}) and options ({len(options)}) length mismatch"

    return None


# =============================================================================
# Probability Validation
# =============================================================================


def validate_probability_range(prob: float | None) -> str | None:
    """Validate that a probability is between 0 and 1.

    Args:
        prob: Probability value

    Returns:
        Error message if out of range, None if valid

    Example:
        >>> validate_probability_range(0.5)
        None
        >>> validate_probability_range(1.5)
        "probability (1.5) must be between 0 and 1"
    """
    if prob is None:
        return None

    if prob < 0 or prob > 1:
        return f"probability ({prob}) must be between 0 and 1"

    return None


# =============================================================================
# Range Validation
# =============================================================================


def validate_min_max(
    min_val: float | None,
    max_val: float | None,
) -> str | None:
    """Validate that min is less than max.

    Args:
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Error message if min >= max, None if valid

    Example:
        >>> validate_min_max(0, 100)
        None
        >>> validate_min_max(100, 50)
        "min (100) must be less than max (50)"
    """
    if min_val is None or max_val is None:
        return None

    if min_val >= max_val:
        return f"min ({min_val}) must be less than max ({max_val})"

    return None


# =============================================================================
# Standard Deviation Validation
# =============================================================================


def validate_std_positive(std: float | None) -> str | None:
    """Validate that standard deviation is positive.

    Args:
        std: Standard deviation value

    Returns:
        Error message if non-positive, None if valid

    Example:
        >>> validate_std_positive(10.0)
        None
        >>> validate_std_positive(-5.0)
        "std (-5.0) must be positive"
    """
    if std is None:
        return None

    if std < 0:
        return f"std ({std}) cannot be negative"

    if std == 0:
        return "std is 0 (no variance) — consider using derived strategy"

    return None


# =============================================================================
# Beta Distribution Validation
# =============================================================================


def validate_beta_params(
    alpha: float | None,
    beta: float | None,
) -> str | None:
    """Validate beta distribution parameters.

    Both alpha and beta must be positive.

    Args:
        alpha: Alpha parameter
        beta: Beta parameter

    Returns:
        Error message if invalid, None if valid

    Example:
        >>> validate_beta_params(2.0, 5.0)
        None
        >>> validate_beta_params(-1.0, 5.0)
        "alpha (-1.0) must be positive"
    """
    if alpha is not None and alpha <= 0:
        return f"alpha ({alpha}) must be positive"

    if beta is not None and beta <= 0:
        return f"beta ({beta}) must be positive"

    return None


# =============================================================================
# Options Validation
# =============================================================================


def validate_options_not_empty(options: Sequence[str] | None) -> str | None:
    """Validate that categorical options list is not empty.

    Args:
        options: List of categorical options

    Returns:
        Error message if empty, None if valid

    Example:
        >>> validate_options_not_empty(["a", "b", "c"])
        None
        >>> validate_options_not_empty([])
        "categorical distribution requires at least one option"
    """
    if options is not None and len(options) == 0:
        return "categorical distribution requires at least one option"

    return None
