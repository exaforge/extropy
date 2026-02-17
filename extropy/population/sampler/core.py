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
from typing import Any, Literal

from ...core.models import (
    PopulationSpec,
    AttributeSpec,
    SamplingStats,
    SamplingResult,
    HouseholdConfig,
    NameConfig,
)
from ...utils.callbacks import ItemProgressCallback
from .distributions import sample_distribution, coerce_to_type
from .households import (
    sample_household_type,
    household_needs_partner,
    household_needs_kids,
    correlate_partner_attribute,
    generate_dependents,
    estimate_household_count,
)
from .modifiers import apply_modifiers_and_sample
from ...utils.eval_safe import eval_formula, FormulaError
from ..names import generate_name
from ..names.generator import age_to_birth_decade

logger = logging.getLogger(__name__)


class SamplingError(Exception):
    """Raised when sampling fails for an agent."""

    pass


def _classify_agent_focus(
    agent_focus: str | None,
) -> Literal["all", "couples", "primary_only"]:
    """Determine household agent scope from agent_focus metadata.

    Returns:
        "all" — everyone in household is an agent (families, communities)
        "couples" — both partners are agents, kids are NPCs (retired couples, married couples)
        "primary_only" — only the primary adult is an agent, partner + kids are NPCs (surgeons, students, subscribers)
    """
    if not agent_focus:
        return "primary_only"

    focus_lower = agent_focus.lower()

    if any(kw in focus_lower for kw in ("famil", "household", "everyone")):
        return "all"

    if any(kw in focus_lower for kw in ("couple", "pair", "partners", "spouses")):
        return "couples"

    return "primary_only"


def _has_household_attributes(spec: PopulationSpec) -> bool:
    """Check if the spec has household-scoped attributes, indicating household mode."""
    return any(attr.scope == "household" for attr in spec.attributes)


def sample_population(
    spec: PopulationSpec,
    count: int,
    seed: int | None = None,
    on_progress: ItemProgressCallback | None = None,
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
    household_config = spec.meta.household_config

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
            household_config,
        )
    else:
        agents = _sample_population_independent(
            spec, attr_map, rng, n, id_width, stats, numeric_values, on_progress
        )
        households = []

    # Compute final statistics
    _finalize_stats(stats, numeric_values, len(agents))

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
) -> list[dict[str, Any]]:
    """Sample N agents independently (legacy path)."""
    agents: list[dict[str, Any]] = []
    for i in range(n):
        agent = _sample_single_agent(
            spec, attr_map, rng, i, id_width, stats, numeric_values
        )
        agents.append(agent)
        if on_progress:
            on_progress(i + 1, n)
    return agents


def _generate_npc_partner(
    primary: dict[str, Any],
    attr_map: dict[str, AttributeSpec],
    categorical_options: dict[str, list[str]],
    rng: random.Random,
    config: HouseholdConfig,
    name_config: NameConfig | None = None,
) -> dict[str, Any]:
    """Generate a lightweight NPC partner profile for context.

    Not a full agent — just enough for persona prompts and conversations.
    Uses attr.scope from the spec to determine which attributes to include.
    """
    partner: dict[str, Any] = {}

    # Always include gender
    partner["gender"] = rng.choice(["male", "female"])

    # Always correlate age if present (essential for NPC identity, regardless of scope)
    if "age" in primary:
        partner["age"] = correlate_partner_attribute(
            "age",
            "int",
            primary["age"],
            None,  # Uses gaussian offset
            rng,
            config,
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
                attr.correlation_rate,
                rng,
                config,
                available_options=categorical_options.get(attr_name),
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
        seed=rng.randint(0, 2**31),
        name_config=name_config,
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
) -> dict[str, Any]:
    """Promote a dependent to a full agent with all attributes sampled.

    Uses the dependent's known attributes (age, gender) as seeds,
    then samples remaining attributes normally.
    """
    agent = _sample_single_agent(
        spec, attr_map, rng, index, id_width, stats, numeric_values
    )

    # Override with dependent's known attributes
    agent["age"] = dependent.age
    agent["gender"] = dependent.gender
    agent["household_id"] = household_id
    agent["household_role"] = f"dependent_{dependent.relationship}"
    agent["relationship_to_primary"] = dependent.relationship
    dep_name = str(getattr(dependent, "name", "")).strip()
    if dep_name:
        agent["first_name"] = dep_name
    if parent.get("last_name"):
        agent["last_name"] = parent["last_name"]

    # Copy household-scoped attributes from parent
    for attr in spec.attributes:
        if attr.scope == "household" and attr.name in parent:
            agent[attr.name] = parent[attr.name]

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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample agents in household units with correlated demographics.

    Returns (agents, households) where households is a list of household
    metadata dicts for DB persistence.
    """
    if config is None:
        config = HouseholdConfig()
    focus_mode = _classify_agent_focus(spec.meta.agent_focus)

    num_households = estimate_household_count(target_n, config)
    hh_id_width = len(str(num_households - 1))

    agents: list[dict[str, Any]] = []
    households: list[dict[str, Any]] = []
    agent_index = 0

    # Identify household-scoped attributes and collect categorical options
    household_attrs = {
        attr.name for attr in spec.attributes if attr.scope == "household"
    }
    categorical_options: dict[str, list[str]] = {}
    for attr in spec.attributes:
        if attr.type == "categorical" and attr.sampling.distribution:
            dist = attr.sampling.distribution
            if hasattr(dist, "options"):
                categorical_options[attr.name] = dist.options

    for hh_idx in range(num_households):
        if agent_index >= target_n:
            break

        household_id = f"household_{hh_idx:0{hh_id_width}d}"

        # Sample Adult 1 (primary) — always an agent
        adult1 = _sample_single_agent(
            spec, attr_map, rng, agent_index, id_width, stats, numeric_values
        )
        adult1_age = adult1.get("age", 35)
        agent_index += 1

        # Determine household type
        htype = sample_household_type(adult1_age, rng, config)

        has_partner = household_needs_partner(htype)
        has_kids = household_needs_kids(htype)
        num_adults = 2 if has_partner else 1

        # Determine household_size from agent if present, else estimate
        household_size = adult1.get(
            "household_size", num_adults + (1 if has_kids else 0)
        )
        if isinstance(household_size, (int, float)):
            household_size = max(num_adults, int(household_size))
        else:
            household_size = num_adults + (1 if has_kids else 0)

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
                    rng,
                    config,
                    name_config=spec.meta.name_config,
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
            name_config=spec.meta.name_config,
        )

        if has_kids and focus_mode == "all":
            # Kids old enough become full agents; younger ones stay as NPCs
            dep_dicts = []
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
                )
                agents.append(kid_agent)
                adult_ids.append(kid_agent["_id"])
                agent_index += 1
            # Any overflow dependents attached as NPC data
            adult1["dependents"] = dep_dicts
        else:
            # Kids are NPCs
            dep_dicts = [d.model_dump() for d in dependents]
            adult1["dependents"] = dep_dicts

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

        if on_progress:
            on_progress(min(agent_index, target_n), target_n)

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
        # Partner-correlated: use assortative mating
        elif attr.scope == "partner_correlated" and attr_name in primary:
            value = correlate_partner_attribute(
                attr_name,
                attr.type,
                primary[attr_name],
                attr.correlation_rate,
                rng,
                config,
                available_options=categorical_options.get(attr_name),
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

    return agent


def _sample_single_agent(
    spec: PopulationSpec,
    attr_map: dict[str, AttributeSpec],
    rng: random.Random,
    index: int,
    id_width: int,
    stats: SamplingStats,
    numeric_values: dict[str, list[float]],
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

    # Generate demographically-plausible name
    name_config = spec.meta.name_config
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
        seed=index,
        name_config=name_config,
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

        value, triggered = apply_modifiers_and_sample(
            attr.sampling.distribution,
            attr.sampling.modifiers,
            rng,
            agent,
        )

        # Update modifier trigger stats
        if attr.name in stats.modifier_triggers:
            for idx in triggered:
                stats.modifier_triggers[attr.name][idx] += 1

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
