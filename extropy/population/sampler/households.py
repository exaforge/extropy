"""Household-based sampling for co-sampling correlated agent pairs.

All household composition rates and correlation tables are read from
a HouseholdConfig instance (populated by LLM research at spec time,
with US Census defaults as the safety net).

Attribute scopes are now spec-driven:
- scope="household": shared across all household members
- scope="partner_correlated": correlated between partners using assortative mating
- scope="individual": sampled independently for each person
"""

from __future__ import annotations

import math
import random
from typing import Any, TYPE_CHECKING, Literal

from ...core.models.population import Dependent, HouseholdConfig, HouseholdType
from ..names.generator import generate_name

if TYPE_CHECKING:
    from ...core.models.population import NameConfig


# Legacy name-based mapping for backward compatibility only.
# New specs should drive partner-correlation behavior via metadata/policy.
_LEGACY_POLICY_HINTS: dict[str, dict[str, str]] = {
    "age": {"semantic_type": "age"},
    "race_ethnicity": {"identity_type": "race_ethnicity"},
    "race": {"identity_type": "race_ethnicity"},
    "ethnicity": {"identity_type": "race_ethnicity"},
    "country": {"identity_type": "citizenship"},
}


def _age_bracket(age: int, config: HouseholdConfig) -> str:
    """Map age to bracket key using config age brackets."""
    for upper_bound, label in config.age_brackets:
        if age < upper_bound:
            return label
    # Fallback to last bracket
    return config.age_brackets[-1][1] if config.age_brackets else "65+"


def sample_household_type(
    primary_age: int, rng: random.Random, config: HouseholdConfig
) -> HouseholdType:
    """Sample a household type based on the primary adult's age bracket."""
    bracket = _age_bracket(primary_age, config)
    weights = config.household_type_weights.get(bracket, {})
    if not weights:
        # Fallback: pick first available bracket
        for b_weights in config.household_type_weights.values():
            if b_weights:
                weights = b_weights
                break
    if not weights:
        return HouseholdType.SINGLE

    types = [HouseholdType(k) for k in weights.keys()]
    probs = list(weights.values())
    return rng.choices(types, weights=probs, k=1)[0]


def household_needs_partner(htype: HouseholdType) -> bool:
    """Whether this household type includes a second adult partner."""
    return htype in (
        HouseholdType.COUPLE,
        HouseholdType.COUPLE_WITH_KIDS,
        HouseholdType.MULTI_GENERATIONAL,
    )


def household_needs_kids(htype: HouseholdType) -> bool:
    """Whether this household type includes children."""
    return htype in (
        HouseholdType.SINGLE_PARENT,
        HouseholdType.COUPLE_WITH_KIDS,
        HouseholdType.MULTI_GENERATIONAL,
    )


def correlate_partner_attribute(
    attr_name: str,
    attr_type: str,
    primary_value: Any,
    correlation_rate: float | None,
    rng: random.Random,
    config: HouseholdConfig,
    available_options: list[str] | None = None,
    semantic_type: str | None = None,
    identity_type: str | None = None,
    partner_correlation_policy: Literal[
        "gaussian_offset", "same_group_rate", "same_value_probability"
    ]
    | None = None,
) -> Any:
    """Produce a correlated value for a partner based on the primary's value.

    Policy resolution order:
    1. Explicit `partner_correlation_policy` from attribute metadata
    2. semantic_type / identity_type metadata
    3. Legacy name-based compatibility mapping
    4. Default same-value probability behavior

    Args:
        attr_name: Name of the attribute
        attr_type: Type of the attribute (int, float, categorical, boolean)
        primary_value: The primary partner's value
        correlation_rate: Probability (0-1) that partner has same value, or None for defaults
        rng: Random number generator
        config: HouseholdConfig with default rates
        available_options: For categorical attrs, list of valid options to sample from

        semantic_type: Optional semantic type from AttributeSpec
        identity_type: Optional identity type from AttributeSpec
        partner_correlation_policy: Optional explicit policy from AttributeSpec

    Returns:
        Correlated value for the partner.
    """
    policy = _resolve_partner_policy(
        attr_name=attr_name,
        attr_type=attr_type,
        correlation_rate=correlation_rate,
        config=config,
        semantic_type=semantic_type,
        identity_type=identity_type,
        partner_correlation_policy=partner_correlation_policy,
    )

    if policy == "gaussian_offset":
        partner_age = int(
            round(
                rng.gauss(
                    primary_value + config.partner_age_gap_mean,
                    config.partner_age_gap_std,
                )
            )
        )
        return max(config.min_adult_age, partner_age)

    if policy == "same_group_rate":
        same_rate = config.same_group_rates.get(
            str(primary_value).lower(), config.default_same_group_rate
        )
        if rng.random() < same_rate:
            return primary_value
        if available_options:
            others = [o for o in available_options if o != primary_value]
            if others:
                return rng.choice(others)
        return primary_value

    rate = _resolve_same_value_rate(
        attr_name=attr_name,
        correlation_rate=correlation_rate,
        config=config,
        identity_type=identity_type,
    )
    if rng.random() < rate:
        return primary_value

    # Sample a different value
    if available_options:
        others = [o for o in available_options if o != primary_value]
        if others:
            return rng.choice(others)

    return primary_value


def _resolve_partner_policy(
    attr_name: str,
    attr_type: str,
    correlation_rate: float | None,
    config: HouseholdConfig,
    semantic_type: str | None = None,
    identity_type: str | None = None,
    partner_correlation_policy: Literal[
        "gaussian_offset", "same_group_rate", "same_value_probability"
    ]
    | None = None,
) -> Literal["gaussian_offset", "same_group_rate", "same_value_probability"]:
    """Resolve which partner-correlation algorithm to use."""
    if partner_correlation_policy is not None:
        return partner_correlation_policy

    inferred_semantic = semantic_type
    inferred_identity = identity_type

    # Backward compatibility: infer missing metadata from legacy names.
    if inferred_semantic is None and inferred_identity is None:
        hints = _LEGACY_POLICY_HINTS.get(attr_name.lower())
        if hints:
            inferred_semantic = hints.get("semantic_type")
            inferred_identity = hints.get("identity_type")

    if inferred_semantic == "age" and attr_type in ("int", "float"):
        return "gaussian_offset"

    if inferred_identity == "race_ethnicity":
        return "same_group_rate"

    # Treat citizenship-like identity as same-country behavior.
    if inferred_identity == "citizenship":
        return "same_value_probability"

    # Default fallback remains same-value probability.
    _ = correlation_rate
    _ = config
    return "same_value_probability"


def _resolve_same_value_rate(
    attr_name: str,
    correlation_rate: float | None,
    config: HouseholdConfig,
    identity_type: str | None = None,
) -> float:
    """Resolve same-value probability for partner correlation."""
    if correlation_rate is not None:
        return correlation_rate

    inferred_identity = identity_type
    if inferred_identity is None:
        hints = _LEGACY_POLICY_HINTS.get(attr_name.lower())
        if hints:
            inferred_identity = hints.get("identity_type")

    if inferred_identity == "citizenship":
        return config.same_country_rate

    if attr_name in config.assortative_mating:
        return config.assortative_mating[attr_name]

    return config.default_same_group_rate


def generate_dependents(
    household_type: HouseholdType,
    household_size: int,
    num_adults: int,
    primary_age: int,
    rng: random.Random,
    config: HouseholdConfig,
    ethnicity: str | None = None,
    name_config: NameConfig | None = None,
) -> list[Dependent]:
    """Generate NPC dependents for a household.

    Fills the gap between num_adults and household_size with children
    or elderly dependents based on household type and primary adult age.
    """
    num_dependents = max(0, household_size - num_adults)
    if num_dependents == 0:
        return []

    dependents: list[Dependent] = []

    # Multi-generational households may include an elderly parent
    elderly_count = 0
    if household_type == HouseholdType.MULTI_GENERATIONAL and num_dependents > 0:
        elderly_count = 1
        elderly_age = primary_age + rng.randint(
            config.elderly_min_offset, config.elderly_max_offset
        )
        elderly_gender = rng.choice(["male", "female"])
        relationship = "father" if elderly_gender == "male" else "mother"
        dep_first, _ = generate_name(
            gender=elderly_gender,
            ethnicity=ethnicity,
            age=elderly_age,
            seed=rng.randint(0, 2**31),
            name_config=name_config,
        )
        dependents.append(
            Dependent(
                name=dep_first,
                age=elderly_age,
                gender=elderly_gender,
                relationship=relationship,
                school_status=None,
            )
        )

    # Remaining dependents are children
    num_children = num_dependents - elderly_count
    for _c in range(num_children):
        child_age = _sample_child_age(primary_age, rng, config)
        child_gender = rng.choice(["male", "female"])
        relationship = "son" if child_gender == "male" else "daughter"
        school_status = _life_stage(child_age, config)
        child_first, _ = generate_name(
            gender=child_gender,
            ethnicity=ethnicity,
            age=child_age,
            seed=rng.randint(0, 2**31),
            name_config=name_config,
        )
        dependents.append(
            Dependent(
                name=child_first,
                age=child_age,
                gender=child_gender,
                relationship=relationship,
                school_status=school_status,
            )
        )

    return dependents


def _sample_child_age(
    parent_age: int, rng: random.Random, config: HouseholdConfig
) -> int:
    """Sample a realistic child age given parent age."""
    max_child_age = max(0, parent_age - config.child_min_parent_offset)
    min_child_age = max(0, parent_age - config.child_max_parent_offset)
    max_dep = config.max_dependent_child_age
    lo = min(max_dep, min_child_age)
    hi = min(max_dep, max_child_age)
    if hi <= lo:
        return max(0, hi)
    age = rng.randint(lo, hi)
    return max(0, age)


def _life_stage(age: int, config: HouseholdConfig) -> str:
    """Determine life stage from age using config thresholds."""
    for stage in config.life_stages:
        if age < stage.max_age:
            return stage.label
    return config.adult_stage_label


def estimate_household_count(target_agents: int, config: HouseholdConfig) -> int:
    """Estimate number of households needed to produce target_agents individuals."""
    return max(1, math.ceil(target_agents / config.avg_household_size))
