"""Modifier application for conditional sampling.

Modifiers adjust distributions based on agent attributes:
- Numeric (multiply/add): All matching modifiers stack
- Categorical (weight_overrides): One deterministic winner
- Boolean (probability_override): One deterministic winner
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
from ..modifier_precedence import choose_modifier_precedence
from ...utils.eval_safe import eval_condition
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
) -> tuple[Any, list[int], list[str]]:
    """
    Apply matching modifiers to a distribution and sample.

    Args:
        dist: Base distribution configuration
        modifiers: List of conditional modifiers
        rng: Random number generator
        agent: Current agent's already-sampled attribute values

    Returns:
        Tuple of (sampled_value, list of indices of triggered modifiers, condition warnings)
    """
    triggered_indices: list[int] = []
    condition_warnings: list[str] = []

    # Check which modifiers apply
    matching_modifiers: list[tuple[int, Modifier]] = []
    for i, mod in enumerate(modifiers):
        try:
            if eval_condition(mod.when, agent, raise_on_error=True):
                matching_modifiers.append((i, mod))
        except Exception as e:
            warning = f"modifier[{i}] when='{mod.when}' eval failed: {e}"
            condition_warnings.append(warning)
            logger.warning(warning)

    # Route to type-specific handler
    if isinstance(dist, (NormalDistribution, LognormalDistribution)):
        value = _apply_numeric_modifiers(dist, matching_modifiers, rng, agent)
        triggered_indices = [idx for idx, _ in matching_modifiers]
    elif isinstance(dist, UniformDistribution):
        value = _apply_uniform_modifiers(dist, matching_modifiers, rng)
        triggered_indices = [idx for idx, _ in matching_modifiers]
    elif isinstance(dist, BetaDistribution):
        value = _apply_beta_modifiers(dist, matching_modifiers, rng)
        triggered_indices = [idx for idx, _ in matching_modifiers]
    elif isinstance(dist, CategoricalDistribution):
        value, winner_index = _apply_categorical_modifiers(
            dist, matching_modifiers, rng
        )
        triggered_indices = [winner_index] if winner_index is not None else []
    elif isinstance(dist, BooleanDistribution):
        value, winner_index = _apply_boolean_modifiers(dist, matching_modifiers, rng)
        triggered_indices = [winner_index] if winner_index is not None else []
    else:
        raise ValueError(f"Unknown distribution type: {type(dist)}")

    return value, triggered_indices, condition_warnings


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
        dist.min, getattr(dist, "min_formula", None), agent
    )
    max_bound = _resolve_optional_param(
        dist.max, getattr(dist, "max_formula", None), agent
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
) -> tuple[str, int | None]:
    """
    Apply categorical modifiers with deterministic winner selection.

    Note: If modifiers only have multiply/add (legacy numeric modifiers on categorical),
    they are ignored and base weights are used.
    """
    weighted_modifiers = [(idx, mod) for idx, mod in matching if mod.weight_overrides]
    winner_index, winner_modifier = _select_winner_modifier(weighted_modifiers)

    if winner_modifier is None or winner_modifier.weight_overrides is None:
        return _sample_categorical(dist, rng, None), None

    base_weights = dict(zip(dist.options, dist.weights, strict=False))
    override_weights = [
        winner_modifier.weight_overrides.get(option, base_weights[option])
        for option in dist.options
    ]
    return _sample_categorical(dist, rng, override_weights), winner_index


def _apply_boolean_modifiers(
    dist: BooleanDistribution,
    matching: list[tuple[int, Modifier]],
    rng: random.Random,
) -> tuple[bool, int | None]:
    """
    Apply boolean modifiers with deterministic winner selection.

    If modifiers use multiply/add instead of probability_override,
    apply to probability using the selected winning modifier:
    new_prob = (base_prob * multiply) + add, clamped to [0,1].
    """
    probability = dist.probability_true

    effective = []
    for idx, mod in matching:
        has_effect = (
            mod.probability_override is not None
            or (mod.multiply is not None and mod.multiply != 1.0)
            or (mod.add is not None and mod.add != 0)
        )
        if has_effect:
            effective.append((idx, mod))

    winner_index, winner_modifier = _select_winner_modifier(effective)
    if winner_modifier is not None:
        if winner_modifier.probability_override is not None:
            probability = winner_modifier.probability_override
        else:
            if winner_modifier.multiply is not None:
                probability *= winner_modifier.multiply
            if winner_modifier.add is not None:
                probability += winner_modifier.add

    # Clamp probability to [0, 1]
    probability = max(0.0, min(1.0, probability))

    return _sample_boolean(dist, rng, probability), winner_index


def _select_winner_modifier(
    matching: list[tuple[int, Modifier]],
) -> tuple[int | None, Modifier | None]:
    """Pick one deterministic winner from matching modifiers."""
    if not matching:
        return None, None

    decision = choose_modifier_precedence([(idx, mod.when) for idx, mod in matching])
    if decision is None:
        return None, None

    by_index = {idx: mod for idx, mod in matching}
    winner = by_index.get(decision.winner_index)
    if winner is None:
        return None, None

    return decision.winner_index, winner
