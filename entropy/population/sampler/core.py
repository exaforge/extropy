"""Core sampling loop for generating agents from a PopulationSpec.

The sampler is a generic spec interpreter - it doesn't know about surgeons
or farmers, it just executes whatever spec it's given.
"""

import json
import logging
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ...core.models import PopulationSpec, AttributeSpec
from .distributions import sample_distribution, coerce_to_type
from .modifiers import apply_modifiers_and_sample
from .eval_safe import eval_formula, FormulaError

logger = logging.getLogger(__name__)


class SamplingError(Exception):
    """Raised when sampling fails for an agent."""

    pass


@dataclass
class SamplingStats:
    """Statistics collected during sampling."""

    # Attribute-level stats
    attribute_means: dict[str, float] = field(default_factory=dict)
    attribute_stds: dict[str, float] = field(default_factory=dict)
    categorical_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    boolean_counts: dict[str, dict[bool, int]] = field(default_factory=dict)

    # Modifier trigger counts: attr_name -> modifier_index -> count
    modifier_triggers: dict[str, dict[int, int]] = field(default_factory=dict)

    # Constraint violations (expression constraints)
    constraint_violations: dict[str, int] = field(default_factory=dict)

    # Condition evaluation warnings
    condition_warnings: list[str] = field(default_factory=list)


@dataclass
class SamplingResult:
    """Result of sampling a population."""

    agents: list[dict[str, Any]]
    meta: dict[str, Any]
    stats: SamplingStats


def sample_population(
    spec: PopulationSpec,
    count: int | None = None,
    seed: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> SamplingResult:
    """
    Generate agents from a PopulationSpec.

    Args:
        spec: The population specification to sample from
        count: Number of agents to generate (defaults to spec.meta.size)
        seed: Random seed for reproducibility (None = random)
        on_progress: Optional callback(current, total) for progress updates

    Returns:
        SamplingResult with agents list, metadata, and statistics

    Raises:
        SamplingError: If sampling fails for any agent (e.g., formula error)
    """
    # Resolve count
    n = count if count is not None else spec.meta.size

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

    agents: list[dict[str, Any]] = []

    for i in range(n):
        agent = _sample_single_agent(
            spec, attr_map, rng, i, id_width, stats, numeric_values
        )
        agents.append(agent)

        if on_progress:
            on_progress(i + 1, n)

    # Compute final statistics
    _finalize_stats(stats, numeric_values, n)

    # Check expression constraints
    _check_expression_constraints(spec, agents, stats)

    # Build metadata
    meta = {
        "spec": spec.meta.description,
        "count": n,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
    }

    return SamplingResult(agents=agents, meta=meta, stats=stats)


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
    """Check expression constraints and count violations."""
    from .eval_safe import eval_condition

    for attr in spec.attributes:
        for constraint in attr.constraints:
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
    cursor.execute("""
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE agents (
            id TEXT PRIMARY KEY,
            attributes JSON
        )
    """)

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
