"""Household-based sampling for co-sampling correlated agent pairs.

Contains Census-derived household composition rates and correlation tables
for assortative mating on demographics.
"""

import math
import random
from typing import Any

from ...core.models.population import Dependent, HouseholdType
from ..names.generator import generate_name


# =============================================================================
# Census-derived household composition rates by age bracket of primary adult
# =============================================================================

# Keys: age bracket of Adult 1; values: dict of HouseholdType -> probability
HOUSEHOLD_TYPE_WEIGHTS: dict[str, dict[HouseholdType, float]] = {
    "18-29": {
        HouseholdType.SINGLE: 0.45,
        HouseholdType.COUPLE: 0.25,
        HouseholdType.SINGLE_PARENT: 0.08,
        HouseholdType.COUPLE_WITH_KIDS: 0.15,
        HouseholdType.MULTI_GENERATIONAL: 0.07,
    },
    "30-44": {
        HouseholdType.SINGLE: 0.20,
        HouseholdType.COUPLE: 0.15,
        HouseholdType.SINGLE_PARENT: 0.12,
        HouseholdType.COUPLE_WITH_KIDS: 0.40,
        HouseholdType.MULTI_GENERATIONAL: 0.13,
    },
    "45-64": {
        HouseholdType.SINGLE: 0.25,
        HouseholdType.COUPLE: 0.35,
        HouseholdType.SINGLE_PARENT: 0.08,
        HouseholdType.COUPLE_WITH_KIDS: 0.20,
        HouseholdType.MULTI_GENERATIONAL: 0.12,
    },
    "65+": {
        HouseholdType.SINGLE: 0.35,
        HouseholdType.COUPLE: 0.40,
        HouseholdType.SINGLE_PARENT: 0.02,
        HouseholdType.COUPLE_WITH_KIDS: 0.05,
        HouseholdType.MULTI_GENERATIONAL: 0.18,
    },
}

# Intermarriage rates: probability partner shares same value.
# Key: race_ethnicity group; value: probability of same-race partner.
INTERMARRIAGE_RATES: dict[str, float] = {
    "white": 0.90,
    "black": 0.82,
    "hispanic": 0.78,
    "asian": 0.75,
    "other": 0.50,
}
_DEFAULT_SAME_RACE_RATE = 0.85

# Assortative mating correlation coefficients.
# Higher = more likely partner shares similar value.
ASSORTATIVE_MATING: dict[str, float] = {
    "education_level": 0.6,
    "religious_affiliation": 0.7,
    "political_orientation": 0.5,
}

# Age gap parameters by gender combination.
# (mean_offset, std) where offset is Adult2_age - Adult1_age.
AGE_GAP_PARAMS: dict[str, tuple[float, float]] = {
    "default": (-2.0, 3.0),
}

# Average household size used for estimating number of households from N.
AVG_HOUSEHOLD_SIZE = 2.5


def _age_bracket(age: int) -> str:
    """Map age to bracket key for HOUSEHOLD_TYPE_WEIGHTS."""
    if age < 30:
        return "18-29"
    elif age < 45:
        return "30-44"
    elif age < 65:
        return "45-64"
    else:
        return "65+"


def sample_household_type(primary_age: int, rng: random.Random) -> HouseholdType:
    """Sample a household type based on the primary adult's age bracket."""
    bracket = _age_bracket(primary_age)
    weights = HOUSEHOLD_TYPE_WEIGHTS[bracket]
    types = list(weights.keys())
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


def correlate_partner_attribute(
    attr_name: str,
    primary_value: Any,
    rng: random.Random,
    available_options: list[str] | None = None,
) -> Any:
    """Produce a correlated value for a partner based on the primary's value.

    For categorical attributes, uses assortative mating rates to decide
    whether to copy or re-sample.  For age, applies a Gaussian offset.

    Returns the correlated value, or None if the attribute isn't in the
    correlation tables (caller should sample independently).
    """
    if attr_name == "age" and isinstance(primary_value, (int, float)):
        mean_offset, std = AGE_GAP_PARAMS.get("default", (-2.0, 3.0))
        partner_age = int(round(rng.gauss(primary_value + mean_offset, std)))
        return max(18, partner_age)

    if attr_name == "race_ethnicity":
        same_rate = INTERMARRIAGE_RATES.get(
            str(primary_value).lower(), _DEFAULT_SAME_RACE_RATE
        )
        if rng.random() < same_rate:
            return primary_value
        # Pick a different value from available options
        if available_options:
            others = [o for o in available_options if o != primary_value]
            if others:
                return rng.choice(others)
        return primary_value

    if attr_name in ASSORTATIVE_MATING:
        correlation = ASSORTATIVE_MATING[attr_name]
        if rng.random() < correlation:
            return primary_value
        # Pick a different value from available options
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
        elderly_age = primary_age + rng.randint(22, 35)
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
    for c in range(num_children):
        child_age = _sample_child_age(primary_age, rng)
        child_gender = rng.choice(["male", "female"])
        relationship = "son" if child_gender == "male" else "daughter"
        school_status = _school_status(child_age)
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


def _sample_child_age(parent_age: int, rng: random.Random) -> int:
    """Sample a realistic child age given parent age."""
    # Parent had child between age 20-40 typically
    max_child_age = max(0, parent_age - 20)
    min_child_age = max(0, parent_age - 40)
    # Clamp to 0-17 range (children only)
    lo = min(17, min_child_age)
    hi = min(17, max_child_age)
    if hi <= lo:
        return max(0, hi)
    age = rng.randint(lo, hi)
    return max(0, age)


def _school_status(age: int) -> str:
    """Determine school status from age."""
    if age < 5:
        return "home"
    elif age < 11:
        return "elementary"
    elif age < 14:
        return "middle_school"
    elif age < 18:
        return "high_school"
    else:
        return "adult"


def estimate_household_count(target_agents: int) -> int:
    """Estimate number of households needed to produce target_agents individuals."""
    return max(1, math.ceil(target_agents / AVG_HOUSEHOLD_SIZE))
