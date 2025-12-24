"""Type-specific distribution sampling.

Handles sampling from all 6 distribution types:
- normal: Gaussian with optional min/max clamping
- lognormal: Log-normal distribution
- uniform: Uniform between min and max
- beta: Beta distribution (0-1 output, useful for proportions)
- categorical: Weighted random selection from options
- boolean: Bernoulli trial with probability_true
"""

import random
import re
from typing import Any

from ...core.models import (
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    Distribution,
)
from .eval_safe import eval_safe, FormulaError


def sample_distribution(
    dist: Distribution,
    rng: random.Random,
    agent: dict[str, Any] | None = None,
) -> Any:
    """
    Sample a value from a distribution.

    Args:
        dist: Distribution configuration
        rng: Random number generator (seeded for reproducibility)
        agent: Current agent's already-sampled values (for mean_formula/std_formula)

    Returns:
        Sampled value (float, str, or bool depending on distribution type)

    Raises:
        FormulaError: If mean_formula/std_formula evaluation fails
        ValueError: If distribution type is unknown
    """
    if isinstance(dist, NormalDistribution):
        return _sample_normal(dist, rng, agent)
    elif isinstance(dist, LognormalDistribution):
        return _sample_lognormal(dist, rng, agent)
    elif isinstance(dist, UniformDistribution):
        return _sample_uniform(dist, rng)
    elif isinstance(dist, BetaDistribution):
        return _sample_beta(dist, rng)
    elif isinstance(dist, CategoricalDistribution):
        return _sample_categorical(dist, rng)
    elif isinstance(dist, BooleanDistribution):
        return _sample_boolean(dist, rng)
    else:
        raise ValueError(f"Unknown distribution type: {type(dist)}")


def _resolve_param(
    static_value: float | None,
    formula: str | None,
    agent: dict[str, Any] | None,
    param_name: str,
) -> float:
    """Resolve a parameter that may be static or formula-based."""
    if static_value is not None:
        return static_value
    if formula is not None:
        if agent is None:
            raise FormulaError(
                f"{param_name}_formula '{formula}' requires agent context"
            )
        result = eval_safe(formula, agent)
        return float(result)
    raise FormulaError(f"Neither {param_name} nor {param_name}_formula provided")


def _sample_normal(
    dist: NormalDistribution,
    rng: random.Random,
    agent: dict[str, Any] | None,
) -> float:
    """Sample from normal distribution with optional formula-based parameters."""
    mean = _resolve_param(dist.mean, dist.mean_formula, agent, "mean")
    std = _resolve_param(dist.std, dist.std_formula, agent, "std") if (dist.std is not None or dist.std_formula is not None) else 1.0

    value = rng.gauss(mean, std)

    # Apply min/max clamping
    if dist.min is not None:
        value = max(value, dist.min)
    if dist.max is not None:
        value = min(value, dist.max)

    return value


def _sample_lognormal(
    dist: LognormalDistribution,
    rng: random.Random,
    agent: dict[str, Any] | None,
) -> float:
    """Sample from lognormal distribution.
    
    Note: The spec provides mean and std as the actual lognormal distribution
    parameters, but Python's lognormvariate expects log-space (mu, sigma).
    We convert using the standard formulas.
    """
    import math
    
    mean = _resolve_param(dist.mean, dist.mean_formula, agent, "mean")
    std = _resolve_param(dist.std, dist.std_formula, agent, "std") if (dist.std is not None or dist.std_formula is not None) else mean * 0.5
    
    # Convert from actual mean/std to log-space mu/sigma
    # mu = log(mean^2 / sqrt(mean^2 + std^2))
    # sigma = sqrt(log(1 + std^2/mean^2))
    variance = std ** 2
    mean_sq = mean ** 2
    mu = math.log(mean_sq / math.sqrt(mean_sq + variance))
    sigma = math.sqrt(math.log(1 + variance / mean_sq))
    
    value = rng.lognormvariate(mu, sigma)

    # Apply min/max clamping
    if dist.min is not None:
        value = max(value, dist.min)
    if dist.max is not None:
        value = min(value, dist.max)

    return value


def _sample_uniform(dist: UniformDistribution, rng: random.Random) -> float:
    """Sample from uniform distribution."""
    return rng.uniform(dist.min, dist.max)


def _sample_beta(dist: BetaDistribution, rng: random.Random) -> float:
    """Sample from beta distribution (outputs 0-1, optionally scaled)."""
    value = rng.betavariate(dist.alpha, dist.beta)

    # Scale if min/max provided
    if dist.min is not None and dist.max is not None:
        value = dist.min + value * (dist.max - dist.min)

    return value


def _sample_categorical(
    dist: CategoricalDistribution,
    rng: random.Random,
    weights: list[float] | None = None,
) -> str:
    """
    Sample from categorical distribution.

    Args:
        dist: Categorical distribution with options and weights
        rng: Random number generator
        weights: Override weights (used when modifiers apply)

    Returns:
        Selected option string
    """
    use_weights = weights if weights is not None else dist.weights
    return rng.choices(dist.options, weights=use_weights, k=1)[0]


def _sample_boolean(
    dist: BooleanDistribution,
    rng: random.Random,
    probability: float | None = None,
) -> bool:
    """
    Sample from boolean distribution.

    Args:
        dist: Boolean distribution with probability_true
        rng: Random number generator
        probability: Override probability (used when modifiers apply)

    Returns:
        True or False
    """
    p = probability if probability is not None else dist.probability_true
    return rng.random() < p


def coerce_to_type(value: Any, attr_type: str) -> Any:
    """
    Coerce sampled value to the declared attribute type.

    Handles mismatches between distribution output and attr.type:
    - Categorical options like "6+" → int 6
    - Float → int via rounding
    - String booleans → bool

    Args:
        value: Sampled value
        attr_type: Declared type ("int", "float", "categorical", "boolean")

    Returns:
        Value coerced to the appropriate type
    """
    if attr_type == "int":
        if isinstance(value, str):
            # Handle categorical options like "1", "2", "5+", "6+"
            # Strip non-digit characters and use numeric floor
            digits = re.sub(r"[^\d]", "", value)
            if digits:
                return int(digits)
            # If no digits found, try direct conversion
            try:
                return int(float(value))
            except ValueError:
                return 0
        elif isinstance(value, (int, float)):
            return round(value)
        return int(value)

    elif attr_type == "float":
        if isinstance(value, str):
            digits = re.sub(r"[^\d.]", "", value)
            if digits:
                return float(digits)
            try:
                return float(value)
            except ValueError:
                return 0.0
        return float(value)

    elif attr_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    elif attr_type == "categorical":
        # Keep as string
        return str(value)

    return value
