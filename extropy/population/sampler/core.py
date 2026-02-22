"""Core sampling loop for generating agents from a PopulationSpec.

The sampler is a generic spec interpreter - it doesn't know about surgeons
or farmers, it just executes whatever spec it's given.

Supports two modes:
- Independent sampling (legacy): each agent sampled independently
- Household sampling: agents are grouped into households with correlated demographics
"""

import json
import logging
import random
import sqlite3
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Literal

from ...core.models import (
    PopulationSpec,
    AttributeSpec,
    SamplingStats,
    SamplingResult,
    HouseholdConfig,
    HouseholdType,
    SamplingSemanticRoles,
    MaritalRoles,
)
from ...utils.callbacks import ItemProgressCallback
from .distributions import sample_distribution, coerce_to_type
from .households import (
    sample_household_type,
    household_needs_partner,
    household_needs_kids,
    correlate_partner_attribute,
    generate_dependents,
    choose_partner_gender,
)
from .modifiers import apply_modifiers_and_sample
from ...utils.eval_safe import eval_formula, FormulaError
from ..names import generate_name
from ..names.generator import age_to_birth_decade


# =============================================================================
# Minor Normalization (for promoted dependents)
# =============================================================================


def _coerce_minor_education(value: Any, age: int, options: list[str]) -> Any:
    """Coerce education to age-appropriate level for minors."""
    if age >= 18:
        return value

    # Max education stage by age
    if age <= 10:
        max_stage = 1  # elementary
    elif age <= 13:
        max_stage = 2  # middle school
    elif age <= 17:
        max_stage = 3  # high school
    else:
        max_stage = 4  # some college

    # Try to find age-appropriate option
    age_appropriate = ["none", "elementary", "middle_school", "high_school"][
        : max_stage + 1
    ]
    for opt in reversed(age_appropriate):
        for available in options:
            if opt in available.lower():
                return available

    # Fallback to first option or "none"
    return options[0] if options else "none"


def _coerce_minor_employment(value: Any, age: int, options: list[str]) -> Any:
    """Coerce employment to age-appropriate status for minors."""
    if age >= 18:
        return value

    # Minors should be students or not employed
    preferred = ["student", "none", "unemployed", "not_employed", "n/a"]
    if age >= 16:
        preferred = ["student", "part_time", "intern", "none", "unemployed"]

    for pref in preferred:
        for opt in options:
            if pref in opt.lower():
                return opt

    return options[0] if options else "student"


def _coerce_minor_occupation(value: Any, options: list[str]) -> Any:
    """Coerce occupation to age-appropriate value for minors."""
    preferred = ["student", "none", "n/a", "not_applicable"]
    for pref in preferred:
        for opt in options:
            if pref in opt.lower():
                return opt
    return options[0] if options else "student"


def _coerce_minor_income(value: Any, options: list[str] | None) -> Any:
    """Coerce income to zero for minors."""
    if isinstance(value, (int, float)):
        return 0
    if options:
        for opt in options:
            if "none" in opt.lower() or "0" in opt or "zero" in opt.lower():
                return opt
        return options[0]
    return 0


def _coerce_young_adult_education(value: Any, age: int, options: list[str]) -> Any:
    """Coerce implausible early education stages for young adults."""
    if age >= 22:
        return value
    if not isinstance(value, str):
        return value

    def stage(option: str) -> int:
        token = _normalize_attr_token(option)
        if any(k in token for k in ("phd", "doctorate", "doctoral", "graduate", "masters")):
            return 5
        if any(k in token for k in ("bachelor", "ba", "bs")):
            return 4
        if "associate" in token:
            return 3
        if "some_college" in token or "college" in token:
            return 2
        if "high_school" in token or "hs" in token or "ged" in token:
            return 1
        return 0

    current_stage = stage(value)
    if age < 20 and current_stage >= 4:
        allowed_max = 2
    elif age < 22 and current_stage >= 5:
        allowed_max = 4
    else:
        return value

    candidates = [opt for opt in options if stage(opt) <= allowed_max]
    if not candidates:
        return value
    return max(candidates, key=stage)


def _coerce_early_retirement(value: Any, age: int, options: list[str]) -> Any:
    """Coerce retirement status for very young adults."""
    if age >= 30:
        return value
    if not isinstance(value, str):
        return value

    value_token = _normalize_attr_token(value)
    if "retired" not in value_token:
        return value

    preferred_tokens = (
        "student",
        "intern",
        "part_time",
        "gig",
        "self",
        "private",
        "public",
        "government",
        "unemployed",
        "labor_force",
        "employed",
    )
    for token in preferred_tokens:
        for opt in options:
            norm = _normalize_attr_token(opt)
            if "retired" in norm:
                continue
            if token in norm:
                return opt

    for opt in options:
        if "retired" not in _normalize_attr_token(opt):
            return opt
    return value


def _normalize_young_adult_attributes(
    agent: dict[str, Any],
    spec: PopulationSpec,
) -> None:
    """Normalize rare but implausible lifecycle combinations for young adults."""
    age = agent.get("age")
    if not isinstance(age, (int, float)):
        return
    age_int = int(age)
    if age_int >= 30:
        return

    for attr in spec.attributes:
        if attr.name not in agent:
            continue
        if (
            attr.sampling.distribution is None
            or not hasattr(attr.sampling.distribution, "options")
            or not attr.sampling.distribution.options
        ):
            continue
        options = list(attr.sampling.distribution.options)
        value = agent[attr.name]
        if attr.semantic_type == "education":
            agent[attr.name] = _coerce_young_adult_education(value, age_int, options)
        elif attr.semantic_type == "employment":
            agent[attr.name] = _coerce_early_retirement(value, age_int, options)


def _normalize_minor_attributes(
    agent: dict[str, Any],
    spec: PopulationSpec,
    parent: dict[str, Any],
) -> None:
    """Normalize attributes for minor dependents based on semantic_type.

    Uses the semantic_type field on AttributeSpec (set by LLM during spec creation)
    to determine which attributes need age-appropriate normalization.
    """
    age = agent.get("age")
    if age is None or age >= 18:
        return

    for attr in spec.attributes:
        if attr.name not in agent:
            continue

        value = agent[attr.name]
        # Get options from categorical distribution if available
        options: list[str] = []
        if (
            attr.sampling.distribution
            and hasattr(attr.sampling.distribution, "options")
            and attr.sampling.distribution.options
        ):
            dist_options = attr.sampling.distribution.options
            # CategoricalDistribution.options is list[str]
            if isinstance(dist_options, list):
                options = list(dist_options)
            # In case it's ever a dict (weighted options)
            elif isinstance(dist_options, dict):
                options = list(dist_options.keys())

        if attr.semantic_type == "education":
            agent[attr.name] = _coerce_minor_education(value, int(age), options)
        elif attr.semantic_type == "employment":
            agent[attr.name] = _coerce_minor_employment(value, int(age), options)
        elif attr.semantic_type == "occupation":
            agent[attr.name] = _coerce_minor_occupation(value, options)
        elif attr.semantic_type == "income":
            # Preserve household income, zero out personal income
            if attr.scope == "household" and attr.name in parent:
                agent[attr.name] = parent[attr.name]
            else:
                agent[attr.name] = _coerce_minor_income(
                    value, options if options else None
                )


logger = logging.getLogger(__name__)


class SamplingError(Exception):
    """Raised when sampling fails for an agent."""

    pass


def _has_household_attributes(spec: PopulationSpec) -> bool:
    """Check if the spec has household-scoped attributes, indicating household mode."""
    return any(attr.scope == "household" for attr in spec.attributes)


_GENERIC_CITIZENSHIP_VALUES = {
    "citizen",
    "citizenship",
    "noncitizen",
    "non_citizen",
    "non citizen",
    "resident",
    "permanent_resident",
    "permanent resident",
    "immigrant",
    "non_immigrant",
    "non immigrant",
    "unknown",
    "na",
    "n_a",
    "n/a",
}


def _extract_country_from_citizenship(value: str) -> str | None:
    """Extract country-like token from citizenship-style values.

    Examples:
    - "US citizen" -> "US"
    - "citizen of India" -> "India"
    - "non_citizen" -> None
    """
    text = value.strip()
    if not text:
        return None

    normalized = _normalize_attr_token(text)
    if normalized in _GENERIC_CITIZENSHIP_VALUES:
        return None

    cleaned = text
    for pattern in (
        r"\bcitizenship\b",
        r"\bcitizen\b",
        r"\bnationality\b",
        r"\bnational\b",
        r"\bpermanent resident\b",
        r"\bresident\b",
        r"\bpassport\b",
        r"\bholder\b",
        r"\bof\b",
    ):
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;/")
    if not cleaned:
        return None

    cleaned_norm = _normalize_attr_token(cleaned)
    if cleaned_norm in _GENERIC_CITIZENSHIP_VALUES:
        return None

    return cleaned


def _resolve_country_hint(
    agent: dict[str, Any],
    default_geography: str | None = None,
    semantic_roles: SamplingSemanticRoles | None = None,
) -> str:
    """Resolve geography hint for name generation.

    Precedence:
    1. Explicit country-like agent fields
    2. Spec-level geography (pipeline scope anchor)
    3. Broader region/location fields
    4. US fallback
    """
    geo_roles = semantic_roles.geo_roles if semantic_roles else None

    prioritized_keys: list[str] = []
    if geo_roles and geo_roles.country_attr:
        prioritized_keys.append(geo_roles.country_attr)
    prioritized_keys.extend(("country", "country_code", "nationality"))

    for key in prioritized_keys:
        value = agent.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    citizenship_value = agent.get("citizenship")
    if citizenship_value is not None:
        parsed_country = _extract_country_from_citizenship(str(citizenship_value))
        if parsed_country:
            return parsed_country

    if default_geography:
        text = str(default_geography).strip()
        if text:
            return text

    region_keys: list[str] = []
    if geo_roles and geo_roles.region_attr:
        region_keys.append(geo_roles.region_attr)
    region_keys.extend(("region", "geography", "location"))

    for key in region_keys:
        value = agent.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    return "US"


def sample_population(
    spec: PopulationSpec,
    count: int,
    seed: int | None = None,
    on_progress: ItemProgressCallback | None = None,
    household_config: HouseholdConfig | None = None,
    agent_focus_mode: Literal["primary_only", "couples", "all"] | None = None,
    semantic_roles: SamplingSemanticRoles | None = None,
) -> SamplingResult:
    """
    Generate agents from a PopulationSpec.

    If the spec contains household-scoped attributes, agents are sampled in
    household units with correlated demographics between partners.  Otherwise,
    agents are sampled independently (legacy behavior).

    Args:
        spec: The population specification to sample from
        count: Number of agents to generate (required)
        seed: Random seed for reproducibility (None = random)
        on_progress: Optional callback(current, total) for progress updates
        household_config: Household composition config (required if spec has household attributes)

    Returns:
        SamplingResult with agents list, metadata, and statistics

    Raises:
        SamplingError: If sampling fails for any agent (e.g., formula error)
    """
    # Resolve count
    n = count

    # Initialize RNG
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = random.Random(seed)

    # Build attribute lookup for quick access
    attr_map: dict[str, AttributeSpec] = {attr.name: attr for attr in spec.attributes}

    # Determine ID padding based on count
    id_width = len(str(n - 1))

    # Initialize stats
    stats = SamplingStats()
    for attr in spec.attributes:
        if attr.type in ("int", "float"):
            stats.attribute_means[attr.name] = 0.0
            stats.attribute_stds[attr.name] = 0.0
        elif attr.type == "categorical":
            stats.categorical_counts[attr.name] = {}
        elif attr.type == "boolean":
            stats.boolean_counts[attr.name] = {True: 0, False: 0}

        # Initialize modifier trigger counts
        if attr.sampling.modifiers:
            stats.modifier_triggers[attr.name] = {
                i: 0 for i in range(len(attr.sampling.modifiers))
            }

    # Collect numeric values for std calculation
    numeric_values: dict[str, list[float]] = {
        attr.name: [] for attr in spec.attributes if attr.type in ("int", "float")
    }

    use_households = _has_household_attributes(spec)

    # Use provided household_config or default
    hh_config = household_config or HouseholdConfig()

    if use_households:
        agents, households = _sample_population_households(
            spec,
            attr_map,
            rng,
            n,
            id_width,
            stats,
            numeric_values,
            on_progress,
            hh_config,
            agent_focus_mode=agent_focus_mode,
            semantic_roles=semantic_roles,
        )
        _reconcile_household_coherence(
            agents,
            attr_map,
            stats,
            semantic_roles=semantic_roles,
            household_config=hh_config,
        )
    else:
        agents = _sample_population_independent(
            spec,
            attr_map,
            rng,
            n,
            id_width,
            stats,
            numeric_values,
            on_progress,
            semantic_roles=semantic_roles,
        )
        households = []

    # Recompute descriptive stats from finalized agent payloads
    _recompute_descriptive_stats(spec, agents, stats)

    # Check expression constraints
    _check_expression_constraints(spec, agents, stats)

    # Build metadata
    meta: dict[str, Any] = {
        "spec": spec.meta.description,
        "count": len(agents),
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
    }
    if households:
        meta["household_count"] = len(households)
        meta["household_mode"] = True
        # Household type distribution
        type_counts: dict[str, int] = {}
        for hh in households:
            ht = hh["household_type"]
            type_counts[ht] = type_counts.get(ht, 0) + 1
        meta["household_type_distribution"] = type_counts

    result = SamplingResult(agents=agents, meta=meta, stats=stats)
    # Attach households for DB persistence (not part of SamplingResult model,
    # but accessible as an ad-hoc attribute for save_sample_result)
    result._households = households  # type: ignore[attr-defined]
    return result


def _sample_population_independent(
    spec: PopulationSpec,
    attr_map: dict[str, AttributeSpec],
    rng: random.Random,
    n: int,
    id_width: int,
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
    on_progress: ItemProgressCallback | None = None,
    semantic_roles: SamplingSemanticRoles | None = None,
) -> list[dict[str, Any]]:
    """Sample N agents independently (legacy path)."""
    agents: list[dict[str, Any]] = []
    for i in range(n):
        agent = _sample_single_agent(
            spec,
            attr_map,
            rng,
            i,
            id_width,
            stats,
            numeric_values,
            semantic_roles=semantic_roles,
        )
        agents.append(agent)
        if on_progress:
            on_progress(i + 1, n)
    return agents


def _generate_npc_partner(
    primary: dict[str, Any],
    attr_map: dict[str, AttributeSpec],
    categorical_options: dict[str, list[str]],
    gender_attr: str | None,
    rng: random.Random,
    config: HouseholdConfig,
    geography_hint: str | None = None,
    semantic_roles: SamplingSemanticRoles | None = None,
) -> dict[str, Any]:
    """Generate a lightweight NPC partner profile for context.

    Not a full agent — just enough for persona prompts and conversations.
    Uses attr.scope from the spec to determine which attributes to include.
    """
    partner: dict[str, Any] = {}

    if gender_attr:
        partner_gender = choose_partner_gender(
            primary_gender=(
                str(primary.get(gender_attr))
                if primary.get(gender_attr) is not None
                else None
            ),
            available_genders=categorical_options.get(gender_attr),
            rng=rng,
            config=config,
        )
        if partner_gender:
            partner[gender_attr] = partner_gender
            partner["gender"] = partner_gender
        else:
            partner["gender"] = rng.choice(["male", "female"])
    else:
        # Legacy fallback when no gender attribute exists in spec.
        partner["gender"] = rng.choice(["male", "female"])

    # Always correlate age if present (essential for NPC identity, regardless of scope)
    if "age" in primary:
        partner["age"] = correlate_partner_attribute(
            "age",
            "int",
            primary["age"],
            rng,
            config,
            semantic_type=getattr(attr_map.get("age"), "semantic_type", "age"),
            identity_type=getattr(attr_map.get("age"), "identity_type", None),
            policy_override=_resolve_partner_policy_override("age", semantic_roles),
        )

    # Process attributes based on their scope
    for attr_name, attr in attr_map.items():
        if attr_name not in primary or attr_name == "age":
            continue

        if attr.scope == "household":
            # Shared: copy from primary
            partner[attr_name] = primary[attr_name]
        elif attr.scope == "partner_correlated":
            # Correlated: use assortative mating
            partner[attr_name] = correlate_partner_attribute(
                attr_name,
                attr.type,
                primary[attr_name],
                rng,
                config,
                available_options=categorical_options.get(attr_name),
                semantic_type=attr.semantic_type,
                identity_type=attr.identity_type,
                policy_override=_resolve_partner_policy_override(
                    attr_name, semantic_roles
                ),
            )
        # Individual scope: skip for NPC (not enough data to sample fully)

    # Generate name for partner
    partner_age = partner.get("age")
    birth_decade = age_to_birth_decade(partner_age) if partner_age is not None else None
    ethnicity = (
        partner.get("race_ethnicity") or partner.get("ethnicity") or partner.get("race")
    )
    first_name, _ = generate_name(
        gender=partner["gender"],
        ethnicity=str(ethnicity) if ethnicity else None,
        birth_decade=birth_decade,
        country=_resolve_country_hint(
            primary,
            geography_hint,
            semantic_roles=semantic_roles,
        ),
        seed=rng.randint(0, 2**31),
    )
    partner["first_name"] = first_name

    if primary.get("last_name"):
        partner["last_name"] = primary["last_name"]

    partner["relationship"] = "partner"
    return partner


def _sample_dependent_as_agent(
    spec: PopulationSpec,
    attr_map: dict[str, AttributeSpec],
    rng: random.Random,
    index: int,
    id_width: int,
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
    dependent: Any,
    parent: dict[str, Any],
    household_id: str,
    semantic_roles: SamplingSemanticRoles | None = None,
) -> dict[str, Any]:
    """Promote a dependent to a full agent with all attributes sampled.

    Uses the dependent's known attributes (age, gender) as seeds,
    then samples remaining attributes normally.
    """
    agent = _sample_single_agent(
        spec,
        attr_map,
        rng,
        index,
        id_width,
        stats,
        numeric_values,
        semantic_roles=semantic_roles,
    )

    # Override with dependent's known attributes
    agent["age"] = dependent.age
    agent["gender"] = dependent.gender
    agent["household_id"] = household_id
    agent["household_role"] = f"dependent_{dependent.relationship}"
    agent["relationship_to_primary"] = dependent.relationship
    dep_name = str(getattr(dependent, "name", "")).strip()
    if dep_name:
        agent["first_name"] = dep_name.split()[0]
    if parent.get("last_name"):
        agent["last_name"] = parent["last_name"]

    # Copy household-scoped attributes from parent
    for attr in spec.attributes:
        if attr.scope == "household" and attr.name in parent:
            agent[attr.name] = parent[attr.name]

    # Normalize age-inappropriate attributes for minors
    _normalize_minor_attributes(agent, spec, parent)

    return agent


def _sample_population_households(
    spec: PopulationSpec,
    attr_map: dict[str, AttributeSpec],
    rng: random.Random,
    target_n: int,
    id_width: int,
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
    on_progress: ItemProgressCallback | None = None,
    config: HouseholdConfig | None = None,
    agent_focus_mode: Literal["primary_only", "couples", "all"] | None = None,
    semantic_roles: SamplingSemanticRoles | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample agents in household units with correlated demographics.

    Returns (agents, households) where households is a list of household
    metadata dicts for DB persistence.
    """
    if config is None:
        config = HouseholdConfig()
    focus_mode = (
        agent_focus_mode
        if agent_focus_mode in ("all", "couples", "primary_only")
        else "primary_only"
    )

    hh_id_width = len(str(target_n - 1))  # safe upper bound for household IDs

    agents: list[dict[str, Any]] = []
    households: list[dict[str, Any]] = []
    agent_index = 0

    # Identify household-scoped attributes and collect categorical options
    household_attrs = {
        attr.name for attr in spec.attributes if attr.scope == "household"
    }
    marital_attr = _resolve_marital_attr_name(attr_map, semantic_roles)
    household_size_attr = _resolve_household_size_attr_name(attr_map, semantic_roles)
    gender_attr = _resolve_gender_attr_name(attr_map)
    parental_status_attrs = _resolve_parental_status_attrs(attr_map)
    marital_spec = attr_map.get(marital_attr) if marital_attr else None
    categorical_options: dict[str, list[str]] = {}
    for attr in spec.attributes:
        if attr.type == "categorical" and attr.sampling.distribution:
            dist = attr.sampling.distribution
            if hasattr(dist, "options"):
                categorical_options[attr.name] = dist.options

    hh_idx = 0
    while agent_index < target_n:
        household_id = f"household_{hh_idx:0{hh_id_width}d}"

        # Sample Adult 1 (primary) — always an agent
        adult1 = _sample_single_agent(
            spec,
            attr_map,
            rng,
            agent_index,
            id_width,
            stats,
            numeric_values,
            semantic_roles=semantic_roles,
        )
        adult1_age = adult1.get("age", 35)
        agent_index += 1

        # Determine household type
        htype = sample_household_type(adult1_age, rng, config)

        has_partner = household_needs_partner(htype)
        has_kids = household_needs_kids(htype)
        if has_partner and focus_mode in ("couples", "all") and agent_index >= target_n:
            # In couples/all mode, avoid creating fallback NPC partners when there is
            # no capacity left for a second agent.
            has_partner = False
        num_adults = 2 if has_partner else 1

        # Determine sampled household_size from agent if present, else estimate.
        # Then enforce household-type semantics deterministically (kids/no-kids).
        sampled_household_size = adult1.get(
            "household_size", num_adults + (1 if has_kids else 0)
        )
        if isinstance(sampled_household_size, (int, float)):
            sampled_household_size = max(num_adults, int(sampled_household_size))
        else:
            sampled_household_size = num_adults + (1 if has_kids else 0)

        target_dependents = max(0, sampled_household_size - num_adults)
        if htype in (HouseholdType.SINGLE, HouseholdType.COUPLE):
            target_dependents = 0
        elif htype in (
            HouseholdType.SINGLE_PARENT,
            HouseholdType.COUPLE_WITH_KIDS,
            HouseholdType.MULTI_GENERATIONAL,
        ):
            target_dependents = max(1, target_dependents)

        household_size = num_adults + target_dependents

        # Annotate Adult 1 with household fields
        adult1["household_id"] = household_id
        adult1["household_role"] = "adult_primary"

        adult_ids = [adult1["_id"]]
        adult2_added = False

        if has_partner:
            if focus_mode in ("couples", "all") and agent_index < target_n:
                # Partner is a full agent
                adult2 = _sample_partner_agent(
                    spec,
                    attr_map,
                    rng,
                    agent_index,
                    id_width,
                    stats,
                    numeric_values,
                    adult1,
                    household_attrs,
                    categorical_options,
                    config,
                    gender_attr,
                    semantic_roles=semantic_roles,
                )
                adult2["household_id"] = household_id
                adult2["household_role"] = "adult_secondary"
                if adult1.get("last_name"):
                    adult2["last_name"] = adult1["last_name"]
                adult2["partner_id"] = adult1["_id"]
                adult1["partner_id"] = adult2["_id"]
                adult_ids.append(adult2["_id"])
                agent_index += 1
                adult2_added = True
            else:
                # Partner is NPC context on the primary agent
                npc_partner = _generate_npc_partner(
                    adult1,
                    attr_map,
                    categorical_options,
                    gender_attr,
                    rng,
                    config,
                    geography_hint=spec.meta.geography,
                    semantic_roles=semantic_roles,
                )
                adult1["partner_npc"] = npc_partner
                adult1["partner_id"] = None
        else:
            adult1["partner_id"] = None

        # Dependents
        dependents = generate_dependents(
            htype,
            household_size,
            num_adults,
            adult1_age,
            rng,
            config,
            ethnicity=adult1.get("race_ethnicity"),
            country=_resolve_country_hint(
                adult1,
                spec.meta.geography,
                semantic_roles=semantic_roles,
            ),
            household_last_name=adult1.get("last_name"),
        )

        if has_kids and focus_mode == "all":
            # Kids old enough become full agents; younger ones stay as NPCs
            dep_dicts = []
            promoted_dependents: list[dict[str, Any]] = []
            for dep in dependents:
                if agent_index >= target_n or dep.age < config.min_agent_age:
                    # Too young or at target — stays as NPC data
                    dep_dicts.append(dep.model_dump())
                    continue
                kid_agent = _sample_dependent_as_agent(
                    spec,
                    attr_map,
                    rng,
                    agent_index,
                    id_width,
                    stats,
                    numeric_values,
                    dep,
                    adult1,
                    household_id,
                    semantic_roles=semantic_roles,
                )
                promoted_dependents.append(kid_agent)
                adult_ids.append(kid_agent["_id"])
                agent_index += 1
            # Any overflow dependents attached as NPC data
            adult1["dependents"] = dep_dicts
        else:
            # Kids are NPCs
            dep_dicts = [d.model_dump() for d in dependents]
            adult1["dependents"] = dep_dicts
            promoted_dependents = []

        household_members = [adult1]
        if adult2_added:
            household_members.append(adult2)
        household_members.extend(promoted_dependents)
        adult_members = [
            member
            for member in household_members
            if str(member.get("household_role", "")).startswith("adult_")
        ]

        # In household-mode, household-coupled fields should reflect realized structure
        # before post-sampling reconciliation/gating runs.
        if household_size_attr:
            npc_partner_count = 1 if has_partner and not adult2_added else 0
            realized_size = len(household_members) + npc_partner_count + len(dep_dicts)
            for member in adult_members:
                member[household_size_attr] = realized_size

        if parental_status_attrs:
            child_dependents_present = any(
                isinstance(dep, dict)
                and _dependent_is_child(dep, child_age_max=config.max_dependent_child_age)
                for dep in dep_dicts
            )
            if not child_dependents_present:
                child_dependents_present = any(
                    str(member.get("household_role", "")).startswith("dependent_")
                    and _dependent_is_child(
                        {
                            "relationship": str(member.get("household_role", ""))[
                                len("dependent_") :
                            ],
                            "age": member.get("age"),
                        },
                        child_age_max=config.max_dependent_child_age,
                    )
                    for member in household_members
                )

            for member in adult_members:
                for attr_name in parental_status_attrs:
                    attr_spec = attr_map.get(attr_name)
                    if attr_spec is None:
                        continue
                    desired = _pick_parental_status_value(
                        attr_spec,
                        child_dependents_present,
                        current_value=member.get(attr_name),
                    )
                    if desired is not None:
                        member[attr_name] = desired

        if marital_spec and marital_attr:
            for member in adult_members:
                has_partner_member = bool(
                    member.get("partner_id") or member.get("partner_npc")
                )
                desired = _pick_marital_value(
                    marital_spec,
                    has_partner_member,
                    marital_roles=(
                        semantic_roles.marital_roles if semantic_roles else None
                    ),
                    current_value=member.get(marital_attr),
                )
                if desired is not None:
                    member[marital_attr] = desired

        agents.extend(promoted_dependents)

        agents.append(adult1)

        if adult2_added:
            adult2["dependents"] = adult1.get("dependents", [])
            agents.append(adult2)

        # Build household record
        shared_attrs = {}
        for attr_name in household_attrs:
            if attr_name in adult1:
                shared_attrs[attr_name] = adult1[attr_name]

        households.append(
            {
                "id": household_id,
                "household_type": htype.value,
                "adult_ids": adult_ids,
                "dependent_data": [d.model_dump() for d in dependents],
                "shared_attributes": shared_attrs,
            }
        )

        hh_idx += 1

        if on_progress:
            on_progress(min(agent_index, target_n), target_n)

    # Trim excess agents (last household in couples/all mode may overshoot)
    if len(agents) > target_n:
        trimmed_ids = {a["_id"] for a in agents[target_n:]}
        for a in agents[:target_n]:
            if a.get("partner_id") in trimmed_ids:
                a["partner_id"] = None
        agents = agents[:target_n]

    return agents, households


def _sample_partner_agent(
    spec: PopulationSpec,
    attr_map: dict[str, AttributeSpec],
    rng: random.Random,
    index: int,
    id_width: int,
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
    primary: dict[str, Any],
    household_attrs: set[str],
    categorical_options: dict[str, list[str]],
    config: HouseholdConfig | None = None,
    gender_attr: str | None = None,
    semantic_roles: SamplingSemanticRoles | None = None,
) -> dict[str, Any]:
    """Sample a partner agent with correlated demographics.

    Uses attr.scope from the spec to determine sampling behavior:
    - scope="household": copy from primary
    - scope="partner_correlated": use assortative mating correlation
    - scope="individual": sample independently
    """
    if config is None:
        config = HouseholdConfig()
    agent: dict[str, Any] = {"_id": f"agent_{index:0{id_width}d}"}

    for attr_name in spec.sampling_order:
        attr = attr_map.get(attr_name)
        if attr is None:
            continue

        # Household-scoped: copy from primary
        if attr.scope == "household" and attr_name in primary:
            value = primary[attr_name]
        # Partner-correlated: use assortative mating.
        # Age-like fields and scenario policy overrides are correlated even when
        # scope metadata is incomplete, to prevent unrealistic partner drift.
        elif (
            attr_name in primary
            and (
                attr.scope == "partner_correlated"
                or attr.semantic_type == "age"
                or _resolve_partner_policy_override(attr_name, semantic_roles)
                is not None
            )
        ):
            value = correlate_partner_attribute(
                attr_name,
                attr.type,
                primary[attr_name],
                rng,
                config,
                available_options=categorical_options.get(attr_name),
                semantic_type=attr.semantic_type,
                identity_type=attr.identity_type,
                policy_override=_resolve_partner_policy_override(
                    attr_name, semantic_roles
                ),
            )
        else:
            # Individual scope: sample independently
            try:
                value = _sample_attribute(attr, rng, agent, stats)
            except FormulaError as e:
                raise SamplingError(
                    f"Agent {index}: Failed to sample '{attr_name}': {e}"
                ) from e

        value = coerce_to_type(value, attr.type)
        value = _apply_hard_constraints(value, attr)
        agent[attr_name] = value
        _update_stats(attr, value, stats, numeric_values)

    _normalize_young_adult_attributes(agent, spec)

    if gender_attr and gender_attr in agent:
        partner_gender = choose_partner_gender(
            primary_gender=(
                str(primary.get(gender_attr))
                if primary.get(gender_attr) is not None
                else None
            ),
            available_genders=categorical_options.get(gender_attr),
            rng=rng,
            config=config,
        )
        if partner_gender is not None:
            agent[gender_attr] = partner_gender

    # Generate first name for partner agent
    gender = None
    if gender_attr:
        gender = agent.get(gender_attr)
    if gender is None:
        gender = agent.get("gender") or agent.get("sex")
    ethnicity = (
        agent.get("race_ethnicity") or agent.get("ethnicity") or agent.get("race")
    )
    age = agent.get("age")
    birth_decade = age_to_birth_decade(age) if age is not None else None
    first_name, _ = generate_name(
        gender=str(gender) if gender is not None else None,
        ethnicity=str(ethnicity) if ethnicity is not None else None,
        birth_decade=birth_decade,
        country=_resolve_country_hint(
            agent,
            spec.meta.geography,
            semantic_roles=semantic_roles,
        ),
        seed=index,
    )
    agent["first_name"] = first_name

    return agent


def _sample_single_agent(
    spec: PopulationSpec,
    attr_map: dict[str, AttributeSpec],
    rng: random.Random,
    index: int,
    id_width: int,
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
    semantic_roles: SamplingSemanticRoles | None = None,
) -> dict[str, Any]:
    """Sample a single agent following the sampling order."""
    agent: dict[str, Any] = {"_id": f"agent_{index:0{id_width}d}"}

    for attr_name in spec.sampling_order:
        attr = attr_map.get(attr_name)
        if attr is None:
            logger.warning(f"Attribute '{attr_name}' in sampling_order not found")
            continue

        try:
            value = _sample_attribute(attr, rng, agent, stats)
        except FormulaError as e:
            raise SamplingError(
                f"Agent {index}: Failed to sample '{attr_name}': {e}"
            ) from e

        # Coerce to declared type
        value = coerce_to_type(value, attr.type)

        # Apply hard constraints (min/max clamping)
        value = _apply_hard_constraints(value, attr)

        agent[attr_name] = value

        # Update stats
        _update_stats(attr, value, stats, numeric_values)

    _normalize_young_adult_attributes(agent, spec)

    # Generate demographically-plausible name
    gender = agent.get("gender") or agent.get("sex")
    ethnicity = (
        agent.get("race_ethnicity") or agent.get("ethnicity") or agent.get("race")
    )
    age = agent.get("age")
    birth_decade = age_to_birth_decade(age) if age is not None else None
    first_name, last_name = generate_name(
        gender=str(gender) if gender is not None else None,
        ethnicity=str(ethnicity) if ethnicity is not None else None,
        birth_decade=birth_decade,
        country=_resolve_country_hint(
            agent,
            spec.meta.geography,
            semantic_roles=semantic_roles,
        ),
        seed=index,
    )
    agent["first_name"] = first_name
    agent["last_name"] = last_name

    return agent


def _sample_attribute(
    attr: AttributeSpec,
    rng: random.Random,
    agent: dict[str, Any],
    stats: SamplingStats,
) -> Any:
    """Sample a single attribute based on its strategy."""
    strategy = attr.sampling.strategy

    if strategy == "derived":
        # Compute from formula
        if not attr.sampling.formula:
            raise FormulaError(f"Derived attribute '{attr.name}' has no formula")
        return eval_formula(attr.sampling.formula, agent)

    elif strategy == "independent":
        # Sample directly from distribution
        if not attr.sampling.distribution:
            raise FormulaError(
                f"Independent attribute '{attr.name}' has no distribution"
            )
        return sample_distribution(attr.sampling.distribution, rng, agent)

    elif strategy == "conditional":
        # Sample with modifiers
        if not attr.sampling.distribution:
            raise FormulaError(
                f"Conditional attribute '{attr.name}' has no distribution"
            )

        if not attr.sampling.modifiers:
            # No modifiers, sample directly
            return sample_distribution(attr.sampling.distribution, rng, agent)

        value, triggered, condition_warnings = apply_modifiers_and_sample(
            attr.sampling.distribution,
            attr.sampling.modifiers,
            rng,
            agent,
        )

        # Update modifier trigger stats
        if attr.name in stats.modifier_triggers:
            for idx in triggered:
                stats.modifier_triggers[attr.name][idx] += 1
        for warning in condition_warnings:
            if len(stats.condition_warnings) < 100:
                stats.condition_warnings.append(f"{attr.name}: {warning}")
            elif len(stats.condition_warnings) == 100:
                stats.condition_warnings.append(
                    "... additional condition warnings truncated"
                )

        return value

    else:
        raise FormulaError(f"Unknown sampling strategy: {strategy}")


def _apply_hard_constraints(value: Any, attr: AttributeSpec) -> Any:
    """Apply hard_min and hard_max constraints (clamping)."""
    if attr.type not in ("int", "float"):
        return value

    for constraint in attr.constraints:
        # Handle both legacy "min"/"max" and new "hard_min"/"hard_max"
        if constraint.type in ("hard_min", "min") and constraint.value is not None:
            if isinstance(value, (int, float)):
                value = max(value, constraint.value)
        elif constraint.type in ("hard_max", "max") and constraint.value is not None:
            if isinstance(value, (int, float)):
                value = min(value, constraint.value)

    return value


def _update_stats(
    attr: AttributeSpec,
    value: Any,
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
) -> None:
    """Update running statistics for an attribute."""
    if attr.type in ("int", "float") and isinstance(value, (int, float)):
        numeric_values[attr.name].append(float(value))
    elif attr.type == "categorical":
        str_value = str(value)
        if str_value not in stats.categorical_counts[attr.name]:
            stats.categorical_counts[attr.name][str_value] = 0
        stats.categorical_counts[attr.name][str_value] += 1
    elif attr.type == "boolean":
        bool_value = bool(value)
        stats.boolean_counts[attr.name][bool_value] += 1


def _finalize_stats(
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
    n: int,
) -> None:
    """Compute final mean/std for numeric attributes."""
    for name, values in numeric_values.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        stats.attribute_means[name] = mean
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            stats.attribute_stds[name] = variance**0.5
        else:
            stats.attribute_stds[name] = 0.0


def _recompute_descriptive_stats(
    spec: PopulationSpec,
    agents: list[dict[str, Any]],
    stats: SamplingStats,
) -> None:
    """Recompute descriptive stats from finalized agents."""
    stats.attribute_means = {}
    stats.attribute_stds = {}
    stats.categorical_counts = {}
    stats.boolean_counts = {}

    numeric_values: dict[str, list[float]] = {}
    for attr in spec.attributes:
        if attr.type in ("int", "float"):
            numeric_values[attr.name] = []
        elif attr.type == "categorical":
            stats.categorical_counts[attr.name] = {}
        elif attr.type == "boolean":
            stats.boolean_counts[attr.name] = {True: 0, False: 0}

    for agent in agents:
        for attr in spec.attributes:
            if attr.name not in agent:
                continue
            value = agent[attr.name]
            if attr.type in ("int", "float") and isinstance(value, (int, float)):
                numeric_values[attr.name].append(float(value))
            elif attr.type == "categorical":
                key = str(value)
                stats.categorical_counts[attr.name][key] = (
                    stats.categorical_counts[attr.name].get(key, 0) + 1
                )
            elif attr.type == "boolean":
                key = bool(value)
                stats.boolean_counts[attr.name][key] = (
                    stats.boolean_counts[attr.name].get(key, 0) + 1
                )

    _finalize_stats(stats, numeric_values, len(agents))


def _normalize_attr_token(name: str) -> str:
    """Normalize attribute tokens for tolerant matching."""
    token = name.strip().lower()
    token = re.sub(r"[\s\-]+", "_", token)
    token = re.sub(r"[^a-z0-9_]", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _find_attr_name(
    attr_map: dict[str, AttributeSpec],
    aliases: set[str],
) -> str | None:
    """Find an attribute by normalized alias set."""
    normalized_aliases = {_normalize_attr_token(a) for a in aliases}
    for name in attr_map:
        if _normalize_attr_token(name) in normalized_aliases:
            return name
    return None


def _resolve_marital_attr_name(
    attr_map: dict[str, AttributeSpec],
    semantic_roles: SamplingSemanticRoles | None = None,
) -> str | None:
    """Resolve marital attribute via scenario semantic roles, then legacy fallback."""
    if (
        semantic_roles
        and semantic_roles.marital_roles
        and semantic_roles.marital_roles.attr
    ):
        candidate = semantic_roles.marital_roles.attr
        if candidate in attr_map:
            return candidate

    return _find_attr_name(
        attr_map,
        {"marital_status", "marital", "relationship_status"},
    )


def _resolve_household_size_attr_name(
    attr_map: dict[str, AttributeSpec],
    semantic_roles: SamplingSemanticRoles | None = None,
) -> str | None:
    """Resolve household-size attribute via scenario semantic roles, then fallback."""
    if (
        semantic_roles
        and semantic_roles.household_roles
        and semantic_roles.household_roles.household_size_attr
    ):
        candidate = semantic_roles.household_roles.household_size_attr
        attr = attr_map.get(candidate)
        if attr is not None and attr.type in ("int", "float"):
            return candidate

    fallback = _find_attr_name(attr_map, {"household_size"})
    if fallback is None:
        return None
    attr = attr_map.get(fallback)
    if attr is not None and attr.type in ("int", "float"):
        return fallback
    return None


def _resolve_parental_status_attrs(
    attr_map: dict[str, AttributeSpec],
) -> list[str]:
    """Resolve parental-status attributes with semantic role first, aliases second."""
    names: list[str] = []
    for name, attr in attr_map.items():
        if attr.identity_type == "parental_status":
            names.append(name)

    if names:
        return names

    for alias in (
        "has_children",
        "is_parent",
        "parental_status",
        "has_kids",
        "children_at_home",
    ):
        candidate = _find_attr_name(attr_map, {alias})
        if candidate and candidate not in names:
            names.append(candidate)
    return names


def _dependent_is_child(dep: dict[str, Any], child_age_max: int = 17) -> bool:
    """Detect child dependents from relationship/age/school metadata."""
    rel = _normalize_attr_token(str(dep.get("relationship", "")))
    if any(
        token in rel
        for token in ("son", "daughter", "child", "stepchild", "foster_child")
    ):
        return True

    age_raw = dep.get("age")
    if isinstance(age_raw, str) and age_raw.strip().isdigit():
        age_raw = int(age_raw.strip())
    if isinstance(age_raw, (int, float)) and 0 <= int(age_raw) <= child_age_max:
        return True

    school_status = _normalize_attr_token(str(dep.get("school_status", "")))
    school_tokens = {
        "home",
        "elementary",
        "middle_school",
        "high_school",
        "primary",
        "secondary",
        "k12",
    }
    return school_status in school_tokens


def _pick_parental_status_value(
    attr: AttributeSpec,
    has_child_dependents: bool,
    current_value: Any | None = None,
) -> Any | None:
    """Pick a coherent parental-status value based on realized household children."""
    if attr.type == "boolean":
        return has_child_dependents

    if (
        attr.type != "categorical"
        or attr.sampling.distribution is None
        or not hasattr(attr.sampling.distribution, "options")
    ):
        return None

    options = list(getattr(attr.sampling.distribution, "options", []) or [])
    if not options:
        return None

    if current_value in options:
        norm_current = _normalize_attr_token(str(current_value))
        if has_child_dependents and any(
            token in norm_current for token in ("child", "children", "parent")
        ):
            return current_value
        if not has_child_dependents and any(
            token in norm_current
            for token in (
                "no_child",
                "no_children",
                "non_parent",
                "not_parent",
                "childfree",
            )
        ):
            return current_value

    yes_tokens = ("child", "children", "parent", "has_kid", "has_child", "yes", "true")
    no_tokens = (
        "no_child",
        "no_children",
        "non_parent",
        "not_parent",
        "childfree",
        "none",
        "no",
        "false",
    )

    token_set = yes_tokens if has_child_dependents else no_tokens
    for opt in options:
        norm = _normalize_attr_token(str(opt))
        if any(token in norm for token in token_set):
            return opt
    return None


def _resolve_gender_attr_name(attr_map: dict[str, AttributeSpec]) -> str | None:
    """Resolve gender attribute by metadata first, then legacy alias fallback."""
    for name, attr in attr_map.items():
        if attr.identity_type == "gender_identity":
            return name

    fallback = _find_attr_name(attr_map, {"gender", "sex", "gender_identity"})
    if fallback is None:
        return None
    attr = attr_map.get(fallback)
    if attr is None or attr.type != "categorical":
        return None
    return fallback


def _resolve_partner_policy_override(
    attr_name: str,
    semantic_roles: SamplingSemanticRoles | None,
) -> str | None:
    """Resolve scenario-owned partner-correlation policy override."""
    if semantic_roles is None:
        return None
    override = semantic_roles.partner_correlation_roles.get(attr_name)
    return str(override) if override else None


def _pick_marital_value(
    attr: AttributeSpec,
    has_partner: bool,
    marital_roles: MaritalRoles | None = None,
    current_value: Any | None = None,
) -> Any | None:
    """Pick a coherent marital value based on realized partner status."""
    if attr.type == "boolean":
        return has_partner

    if (
        attr.type != "categorical"
        or attr.sampling.distribution is None
        or not hasattr(attr.sampling.distribution, "options")
    ):
        return None

    options = list(getattr(attr.sampling.distribution, "options", []) or [])
    if not options:
        return None

    if marital_roles and marital_roles.attr:
        if _normalize_attr_token(attr.name) == _normalize_attr_token(
            marital_roles.attr
        ):
            if current_value in options:
                if has_partner and current_value in marital_roles.partnered_values:
                    return current_value
                if not has_partner and current_value in marital_roles.single_values:
                    return current_value
            preferred = (
                marital_roles.partnered_values
                if has_partner
                else marital_roles.single_values
            )
            for value in preferred:
                if value in options:
                    return value

    partnered_tokens = (
        "married",
        "partner",
        "cohabit",
        "relationship",
        "civil_union",
    )
    single_tokens = ("single", "unmarried", "not_married", "divorced", "widowed")

    if current_value in options:
        norm_current = _normalize_attr_token(str(current_value))
        if has_partner and any(token in norm_current for token in partnered_tokens):
            return current_value
        if not has_partner and any(token in norm_current for token in single_tokens):
            return current_value

    normalized = {opt: _normalize_attr_token(opt) for opt in options}
    token_set = partnered_tokens if has_partner else single_tokens
    for opt, norm in normalized.items():
        if any(token in norm for token in token_set):
            return opt
    return options[0]


def _extract_surname_from_name(name: Any) -> str | None:
    """Extract surname token from a full-name-like value."""
    if not isinstance(name, str):
        return None
    parts = [p for p in name.strip().split() if p]
    if len(parts) < 2:
        return None
    return parts[-1]


def _household_surname_seed(
    primary: dict[str, Any], members: list[dict[str, Any]]
) -> str | None:
    """Choose canonical household surname from available household context."""
    for candidate in (primary.get("last_name"),):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for member in members:
        candidate = member.get("last_name")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    partner_npc = primary.get("partner_npc")
    if isinstance(partner_npc, dict):
        candidate = partner_npc.get("last_name")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    dependents = primary.get("dependents")
    if isinstance(dependents, list):
        for dep in dependents:
            if not isinstance(dep, dict):
                continue
            candidate = _extract_surname_from_name(dep.get("name"))
            if candidate:
                return candidate

    return None


def _reconcile_household_coherence(
    agents: list[dict[str, Any]],
    attr_map: dict[str, AttributeSpec],
    stats: SamplingStats,
    semantic_roles: SamplingSemanticRoles | None = None,
    household_config: HouseholdConfig | None = None,
) -> None:
    """Deterministically reconcile household coherence after realization."""
    by_household: dict[str, list[dict[str, Any]]] = {}
    for agent in agents:
        hid = agent.get("household_id")
        if not isinstance(hid, str) or not hid:
            continue
        by_household.setdefault(hid, []).append(agent)

    if not by_household:
        return

    marital_attr = _resolve_marital_attr_name(attr_map, semantic_roles)
    household_size_attr = _resolve_household_size_attr_name(attr_map, semantic_roles)
    parental_status_attrs = _resolve_parental_status_attrs(attr_map)
    child_age_max = (
        household_config.max_dependent_child_age if household_config else 17
    )

    marital_updates = 0
    household_size_updates = 0
    surname_updates = 0
    parental_status_updates = 0

    for members in by_household.values():
        primary = next(
            (m for m in members if m.get("household_role") == "adult_primary"),
            members[0],
        )
        adult_members = [
            member
            for member in members
            if str(member.get("household_role", "")).startswith("adult_")
        ]
        if not adult_members:
            adult_members = members
        dependents = primary.get("dependents", [])
        npc_partner_count = 1 if primary.get("partner_npc") else 0
        npc_dependents_count = len(dependents) if isinstance(dependents, list) else 0
        realized_size = len(members) + npc_partner_count + npc_dependents_count
        household_surname = _household_surname_seed(primary, members)

        if household_size_attr:
            for member in members:
                current = member.get(household_size_attr)
                if current != realized_size:
                    member[household_size_attr] = realized_size
                    household_size_updates += 1

        if household_surname:
            for member in members:
                if member.get("last_name") != household_surname:
                    member["last_name"] = household_surname
                    surname_updates += 1

            partner_npc = primary.get("partner_npc")
            if isinstance(partner_npc, dict):
                if partner_npc.get("last_name") != household_surname:
                    partner_npc["last_name"] = household_surname
                    surname_updates += 1

            # Dependents are represented as NPC dicts with `name`.
            # Normalize them to the household surname for consistency.
            if isinstance(dependents, list):
                for dep in dependents:
                    if not isinstance(dep, dict):
                        continue
                    dep_name = dep.get("name")
                    if isinstance(dep_name, str) and dep_name.strip():
                        first = dep_name.strip().split()[0]
                        corrected = f"{first} {household_surname}"
                        if dep_name != corrected:
                            dep["name"] = corrected
                            surname_updates += 1

        child_dependents_present = False
        if isinstance(dependents, list):
            child_dependents_present = any(
                isinstance(dep, dict) and _dependent_is_child(dep, child_age_max)
                for dep in dependents
            )
        if not child_dependents_present:
            child_dependents_present = any(
                str(member.get("household_role", "")).startswith("dependent_")
                and _dependent_is_child(
                    {
                        "relationship": str(member.get("household_role", ""))[
                            len("dependent_") :
                        ],
                        "age": member.get("age"),
                    },
                    child_age_max,
                )
                for member in members
            )

        if parental_status_attrs:
            for member in adult_members:
                for attr_name in parental_status_attrs:
                    attr_spec = attr_map.get(attr_name)
                    if attr_spec is None:
                        continue
                    desired = _pick_parental_status_value(
                        attr_spec,
                        child_dependents_present,
                        current_value=member.get(attr_name),
                    )
                    if desired is None:
                        continue
                    if member.get(attr_name) != desired:
                        member[attr_name] = desired
                        parental_status_updates += 1

        if marital_attr:
            marital_spec = attr_map[marital_attr]
            for member in adult_members:
                has_partner = bool(
                    member.get("partner_id") or member.get("partner_npc")
                )
                desired = _pick_marital_value(
                    marital_spec,
                    has_partner,
                    marital_roles=(
                        semantic_roles.marital_roles if semantic_roles else None
                    ),
                    current_value=member.get(marital_attr),
                )
                if desired is None:
                    continue
                if member.get(marital_attr) != desired:
                    member[marital_attr] = desired
                    marital_updates += 1

    reconciliation: dict[str, int] = {}
    if marital_updates:
        reconciliation["marital_status_updates"] = marital_updates
    if household_size_updates:
        reconciliation["household_size_updates"] = household_size_updates
    if surname_updates:
        reconciliation["household_surname_updates"] = surname_updates
    if parental_status_updates:
        reconciliation["parental_status_updates"] = parental_status_updates
    if reconciliation:
        stats.reconciliation_counts.update(reconciliation)


def _check_expression_constraints(
    spec: PopulationSpec,
    agents: list[dict[str, Any]],
    stats: SamplingStats,
) -> None:
    """Check expression constraints and count violations.

    Only checks constraints with type='expression' (agent-level constraints).
    Constraints with type='spec_expression' are spec-level validations
    (e.g., sum(weights)==1) and are NOT evaluated against individual agents.
    """
    from ...utils.eval_safe import eval_condition

    for attr in spec.attributes:
        for constraint in attr.constraints:
            # Only check agent-level expression constraints
            # spec_expression constraints validate the YAML spec itself, not agents
            if constraint.type == "expression" and constraint.expression:
                violation_count = 0
                for agent in agents:
                    # Add 'value' to context for constraints that reference it
                    context = dict(agent)
                    if attr.name in agent:
                        context["value"] = agent[attr.name]

                    try:
                        if not eval_condition(constraint.expression, context):
                            violation_count += 1
                    except Exception:
                        # Skip malformed constraints
                        pass

                if violation_count > 0:
                    key = f"{attr.name}: {constraint.expression}"
                    stats.constraint_violations[key] = violation_count


def save_json(result: SamplingResult, path: Path | str) -> None:
    """Save sampling result to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": result.meta,
        "agents": result.agents,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def save_sqlite(result: SamplingResult, path: Path | str) -> None:
    """Save sampling result to SQLite database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to start fresh
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE agents (
            id TEXT PRIMARY KEY,
            attributes JSON
        )
    """
    )

    # Insert metadata
    for key, value in result.meta.items():
        cursor.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value, default=str)),
        )

    # Insert agents
    for agent in result.agents:
        agent_id = agent.get("_id", "")
        # Store full agent as JSON (including _id for consistency)
        cursor.execute(
            "INSERT INTO agents (id, attributes) VALUES (?, ?)",
            (agent_id, json.dumps(agent, default=str)),
        )

    conn.commit()
    conn.close()
