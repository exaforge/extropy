"""Modifier application for conditional sampling.

Modifiers adjust distributions based on agent attributes:
- Numeric (multiply/add): All matching modifiers stack
- Categorical (weight_overrides): Last matching modifier wins
- Boolean (probability_override): Last matching modifier wins
"""

import logging
import random
from typing import Any

from ...core.models import (
    Modifier,
    Distribution,
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
)
from .eval_safe import eval_condition
from .distributions import (
    _sample_normal,
    _sample_lognormal,
    _sample_uniform,
    _sample_beta,
    _sample_categorical,
    _sample_boolean,
    _resolve_optional_param,
)

logger = logging.getLogger(__name__)


def apply_modifiers_and_sample(
    dist: Distribution,
    modifiers: list[Modifier],
    rng: random.Random,
    agent: dict[str, Any],
) -> tuple[Any, list[int]]:
    """
    Apply matching modifiers to a distribution and sample.

    Args:
        dist: Base distribution configuration
        modifiers: List of conditional modifiers
        rng: Random number generator
        agent: Current agent's already-sampled attribute values

    Returns:
        Tuple of (sampled_value, list of indices of triggered modifiers)
    """
    triggered_indices: list[int] = []

    # Check which modifiers apply
    matching_modifiers: list[tuple[int, Modifier]] = []
    for i, mod in enumerate(modifiers):
        try:
            if eval_condition(mod.when, agent):
                matching_modifiers.append((i, mod))
                triggered_indices.append(i)
        except Exception as e:
            # Log warning but continue - condition failure means modifier doesn't apply
            logger.warning(f"Modifier condition '{mod.when}' failed: {e}")

    # Route to type-specific handler
    if isinstance(dist, (NormalDistribution, LognormalDistribution)):
        value = _apply_numeric_modifiers(dist, matching_modifiers, rng, agent)
    elif isinstance(dist, UniformDistribution):
        value = _apply_uniform_modifiers(dist, matching_modifiers, rng)
    elif isinstance(dist, BetaDistribution):
        value = _apply_beta_modifiers(dist, matching_modifiers, rng)
    elif isinstance(dist, CategoricalDistribution):
        value = _apply_categorical_modifiers(dist, matching_modifiers, rng)
    elif isinstance(dist, BooleanDistribution):
        value = _apply_boolean_modifiers(dist, matching_modifiers, rng)
    else:
        raise ValueError(f"Unknown distribution type: {type(dist)}")

    return value, triggered_indices


def _apply_numeric_modifiers(
    dist: NormalDistribution | LognormalDistribution,
    matching: list[tuple[int, Modifier]],
    rng: random.Random,
    agent: dict[str, Any],
) -> float:
    """
    Apply numeric modifiers (multiply/add stack).

    All matching modifiers are applied in sequence:
    - multiply values are multiplied together
    - add values are summed
    - Final: (base_sample * total_multiply) + total_add
    """
    # Sample from base distribution first
    if isinstance(dist, NormalDistribution):
        base_value = _sample_normal(dist, rng, agent)
    else:
        base_value = _sample_lognormal(dist, rng, agent)

    if not matching:
        return base_value

    # Stack modifiers
    total_multiply = 1.0
    total_add = 0.0

    for _, mod in matching:
        if mod.multiply is not None:
            total_multiply *= mod.multiply
        if mod.add is not None:
            total_add += mod.add

    modified_value = (base_value * total_multiply) + total_add

    # Re-apply min/max clamping after modification
    # Use formula bounds if available (they take precedence over static bounds)
    min_bound = _resolve_optional_param(
        dist.min, getattr(dist, 'min_formula', None), agent
    )
    max_bound = _resolve_optional_param(
        dist.max, getattr(dist, 'max_formula', None), agent
    )
    
    if min_bound is not None:
        modified_value = max(modified_value, min_bound)
    if max_bound is not None:
        modified_value = min(modified_value, max_bound)

    return modified_value


def _apply_uniform_modifiers(
    dist: UniformDistribution,
    matching: list[tuple[int, Modifier]],
    rng: random.Random,
) -> float:
    """Apply modifiers to uniform distribution (multiply/add on sampled value)."""
    base_value = _sample_uniform(dist, rng)

    if not matching:
        return base_value

    total_multiply = 1.0
    total_add = 0.0

    for _, mod in matching:
        if mod.multiply is not None:
            total_multiply *= mod.multiply
        if mod.add is not None:
            total_add += mod.add

    return (base_value * total_multiply) + total_add


def _apply_beta_modifiers(
    dist: BetaDistribution,
    matching: list[tuple[int, Modifier]],
    rng: random.Random,
) -> float:
    """Apply modifiers to beta distribution (multiply/add on sampled value)."""
    base_value = _sample_beta(dist, rng)

    if not matching:
        return base_value

    total_multiply = 1.0
    total_add = 0.0

    for _, mod in matching:
        if mod.multiply is not None:
            total_multiply *= mod.multiply
        if mod.add is not None:
            total_add += mod.add

    modified_value = (base_value * total_multiply) + total_add

    # Clamp to [0, 1] for proportion attributes
    if dist.min is None and dist.max is None:
        modified_value = max(0.0, min(1.0, modified_value))

    return modified_value


def _apply_categorical_modifiers(
    dist: CategoricalDistribution,
    matching: list[tuple[int, Modifier]],
    rng: random.Random,
) -> str:
    """
    Apply categorical modifiers (last weight_override wins).

    Note: If modifiers only have multiply/add (legacy numeric modifiers on categorical),
    we ignore them and use base weights.
    """
    # Find the last modifier with weight_overrides
    override_weights: list[float] | None = None

    for _, mod in matching:
        if mod.weight_overrides:
            # Build new weights list from overrides
            new_weights = []
            for option in dist.options:
                if option in mod.weight_overrides:
                    new_weights.append(mod.weight_overrides[option])
                else:
                    # Keep original weight for options not in override
                    idx = dist.options.index(option)
                    new_weights.append(dist.weights[idx])
            override_weights = new_weights

    return _sample_categorical(dist, rng, override_weights)


def _apply_boolean_modifiers(
    dist: BooleanDistribution,
    matching: list[tuple[int, Modifier]],
    rng: random.Random,
) -> bool:
    """
    Apply boolean modifiers (last probability_override wins).

    If modifiers use multiply/add instead of probability_override,
    apply to probability: new_prob = (base_prob * multiply) + add, clamped to [0,1].
    """
    probability = dist.probability_true

    for _, mod in matching:
        if mod.probability_override is not None:
            # Direct override wins
            probability = mod.probability_override
        else:
            # Apply multiply/add to probability
            if mod.multiply is not None:
                probability *= mod.multiply
            if mod.add is not None:
                probability += mod.add

    # Clamp probability to [0, 1]
    probability = max(0.0, min(1.0, probability))

    return _sample_boolean(dist, rng, probability)
