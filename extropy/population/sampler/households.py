"""Household-based sampling for co-sampling correlated agent pairs.

All household composition rates and correlation tables are read from
a HouseholdConfig instance (scenario-owned config, with deterministic
defaults as the safety net).

Attribute scopes are now spec-driven:
- scope="household": shared across all household members
- scope="partner_correlated": correlated between partners using assortative mating
- scope="individual": sampled independently for each person
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal

from ...core.models.population import Dependent, HouseholdConfig, HouseholdType
from ..names.generator import generate_name


def _normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _pair_key(left: str, right: str) -> tuple[str, str]:
    a = _normalize_label(left)
    b = _normalize_label(right)
    return (a, b) if a <= b else (b, a)


def choose_partner_gender(
    primary_gender: str | None,
    available_genders: list[str] | None,
    rng: random.Random,
    config: HouseholdConfig,
) -> str | None:
    """Choose partner gender using scenario-owned pairing policy."""
    options = [g for g in (available_genders or []) if isinstance(g, str) and g.strip()]
    if not options:
        return None

    if config.partner_gender_mode != "weighted" or not config.partner_gender_pair_weights:
        return rng.choice(options)

    primary = primary_gender if isinstance(primary_gender, str) else None
    if not primary:
        return rng.choice(options)

    weights_by_pair: dict[tuple[str, str], float] = {}
    for row in config.partner_gender_pair_weights:
        key = _pair_key(row.left, row.right)
        weights_by_pair[key] = max(0.0, weights_by_pair.get(key, 0.0) + row.weight)

    candidate_weights: list[float] = []
    for option in options:
        key = _pair_key(primary, option)
        candidate_weights.append(max(0.0, weights_by_pair.get(key, 0.0)))

    if sum(candidate_weights) <= 0:
        return rng.choice(options)

    return rng.choices(options, weights=candidate_weights, k=1)[0]


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
    rng: random.Random,
    config: HouseholdConfig,
    available_options: list[str] | None = None,
    semantic_type: str | None = None,
    identity_type: str | None = None,
    policy_override: Literal[
        "gaussian_offset",
        "same_group_rate",
        "same_country_rate",
        "same_value_probability",
    ]
    | None = None,
) -> Any:
    """Produce a correlated value for a partner based on the primary's value.

    Runtime fallback chain:
    1) household_config.assortative_mating[attr_name] (if configured)
    2) Policy defaults (age/race/country special handling + general default)

    Args:
        attr_name: Name of the attribute
        attr_type: Type of the attribute (int, float, categorical, boolean)
        primary_value: The primary partner's value
        rng: Random number generator
        config: HouseholdConfig with default rates
        available_options: For categorical attrs, list of valid options to sample from
        semantic_type: Optional semantic metadata from AttributeSpec
        identity_type: Optional identity metadata from AttributeSpec

    Returns:
        The correlated value for the partner.
    """
    policy = _resolve_partner_correlation_policy(
        attr_name=attr_name,
        attr_type=attr_type,
        semantic_type=semantic_type,
        identity_type=identity_type,
        policy_override=policy_override,
    )

    # Age-like attributes use gaussian offset, not simple same-value probability
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

    # Race/ethnicity-like attributes use per-group rates
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

    # Country/citizenship-like attributes use same_country_rate default
    if policy == "same_country_rate":
        rate = _resolve_same_value_rate(
            attr_name=attr_name,
            config=config,
            default_rate=config.same_country_rate,
        )
        if rng.random() < rate:
            return primary_value
        if available_options:
            others = [o for o in available_options if o != primary_value]
            if others:
                return rng.choice(others)
        return primary_value

    # General same-value policy
    rate = _resolve_same_value_rate(
        attr_name=attr_name,
        config=config,
        default_rate=config.default_same_group_rate,
    )
    if rng.random() < rate:
        return primary_value

    # Sample a different value
    if available_options:
        others = [o for o in available_options if o != primary_value]
        if others:
            return rng.choice(others)

    return primary_value


def _resolve_partner_correlation_policy(
    attr_name: str,
    attr_type: str,
    semantic_type: str | None,
    identity_type: str | None,
    policy_override: Literal[
        "gaussian_offset",
        "same_group_rate",
        "same_country_rate",
        "same_value_probability",
    ]
    | None = None,
) -> str:
    """Resolve partner-correlation policy without hardcoded branching spread."""
    if policy_override in {
        "gaussian_offset",
        "same_group_rate",
        "same_country_rate",
        "same_value_probability",
    }:
        return policy_override

    normalized_name = attr_name.strip().lower()
    if semantic_type == "age":
        return "gaussian_offset"
    if attr_type in {"int", "float"} and normalized_name == "age":
        return "gaussian_offset"

    if identity_type == "race_ethnicity" or normalized_name in {
        "race_ethnicity",
        "ethnicity",
        "race",
    }:
        return "same_group_rate"

    if identity_type == "citizenship" or normalized_name in {
        "country",
        "nationality",
        "citizenship",
    }:
        return "same_country_rate"

    return "same_value_probability"


def _resolve_same_value_rate(
    attr_name: str,
    config: HouseholdConfig,
    default_rate: float,
) -> float:
    """Resolve deterministic same-value probability for partner correlation."""
    if attr_name in config.assortative_mating:
        return config.assortative_mating[attr_name]
    return default_rate


def generate_dependents(
    household_type: HouseholdType,
    household_size: int,
    num_adults: int,
    primary_age: int,
    rng: random.Random,
    config: HouseholdConfig,
    ethnicity: str | None = None,
    country: str = "US",
    household_last_name: str | None = None,
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
            country=country,
            age=elderly_age,
            seed=rng.randint(0, 2**31),
        )
        dep_name = (
            f"{dep_first} {household_last_name}".strip()
            if household_last_name
            else dep_first
        )
        dependents.append(
            Dependent(
                name=dep_name,
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
            country=country,
            age=child_age,
            seed=rng.randint(0, 2**31),
        )
        child_name = (
            f"{child_first} {household_last_name}".strip()
            if household_last_name
            else child_first
        )
        dependents.append(
            Dependent(
                name=child_name,
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
