"""Household-based sampling for co-sampling correlated agent pairs.

All household composition rates and correlation tables are read from
a HouseholdConfig instance (populated by LLM research at spec time,
with US Census defaults as the safety net).
"""

import math
import random
from typing import Any

from ...core.models.population import Dependent, HouseholdConfig, HouseholdType
from ..names.generator import generate_name


# Attributes that are always shared within a household
HOUSEHOLD_SHARED_ATTRIBUTES = [
    "state",
    "urban_rural",
    "household_income",
    "household_size",
]

# Attributes correlated between partners (not copied, but biased)
PARTNER_CORRELATED_ATTRIBUTES = [
    "age",
    "race_ethnicity",
    "education_level",
    "religious_affiliation",
    "political_orientation",
]

# Attributes sampled independently for each partner
PARTNER_INDEPENDENT_ATTRIBUTES = [
    "personality",
    "occupation_sector",
]


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
    primary_value: Any,
    rng: random.Random,
    config: HouseholdConfig,
    available_options: list[str] | None = None,
) -> Any:
    """Produce a correlated value for a partner based on the primary's value.

    For categorical attributes, uses assortative mating rates to decide
    whether to copy or re-sample.  For age, applies a Gaussian offset.

    Returns the correlated value, or None if the attribute isn't in the
    correlation tables (caller should sample independently).
    """
    if attr_name == "age" and isinstance(primary_value, (int, float)):
        partner_age = int(
            round(
                rng.gauss(
                    primary_value + config.partner_age_gap_mean,
                    config.partner_age_gap_std,
                )
            )
        )
        return max(config.min_adult_age, partner_age)

    if attr_name == "race_ethnicity":
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

    if attr_name in config.assortative_mating:
        correlation = config.assortative_mating[attr_name]
        if rng.random() < correlation:
            return primary_value
        if available_options:
            others = [o for o in available_options if o != primary_value]
            if others:
                return rng.choice(others)
        return primary_value

    return None  # Not a correlated attribute


def generate_dependents(
    household_type: HouseholdType,
    household_size: int,
    num_adults: int,
    primary_age: int,
    rng: random.Random,
    config: HouseholdConfig,
    ethnicity: str | None = None,
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
