"""Aggregate computation for simulation results.

Computes population-level statistics including per-timestep summaries,
segment breakdowns, and outcome distributions.
"""

from collections import defaultdict
from typing import Any

from ..core.models import (
    OutcomeDefinition,
    OutcomeType,
    PopulationSpec,
    TimestepSummary,
)
from .state import StateManager


def compute_timestep_summary(
    timestep: int,
    state_manager: StateManager,
    prev_summary: TimestepSummary | None = None,
) -> TimestepSummary:
    """Compute summary statistics for a single timestep.

    Args:
        timestep: Current timestep
        state_manager: State manager with current state
        prev_summary: Previous timestep's summary (optional)

    Returns:
        TimestepSummary with computed statistics
    """
    exposure_rate = state_manager.get_exposure_rate()
    position_distribution = state_manager.get_position_distribution()
    average_sentiment = state_manager.get_average_sentiment()
    average_conviction = state_manager.get_average_conviction()
    sentiment_variance = state_manager.get_sentiment_variance()

    return TimestepSummary(
        timestep=timestep,
        new_exposures=0,  # Will be updated by engine
        agents_reasoned=0,  # Will be updated by engine
        shares_occurred=0,  # Will be updated by engine
        state_changes=0,  # Will be updated by engine
        exposure_rate=exposure_rate,
        position_distribution=position_distribution,
        average_sentiment=average_sentiment,
        average_conviction=average_conviction,
        sentiment_variance=sentiment_variance,
    )


def compute_final_aggregates(
    state_manager: StateManager,
    agents: list[dict[str, Any]],
    population_spec: PopulationSpec,
) -> dict[str, Any]:
    """Compute final aggregates at end of simulation.

    Args:
        state_manager: State manager with final state
        agents: List of agent dictionaries
        population_spec: Population specification

    Returns:
        Dict containing final aggregate statistics
    """
    final_states = state_manager.export_final_states()

    # Basic stats
    total = len(final_states)
    aware = sum(1 for s in final_states if s["aware"])
    reasoned = sum(1 for s in final_states if s["last_reasoning_timestep"] >= 0)
    sharing = sum(1 for s in final_states if s["will_share"])

    # Position distribution
    positions: dict[str, int] = defaultdict(int)
    for state in final_states:
        if state["position"]:
            positions[state["position"]] += 1

    # Sentiment distribution
    sentiments = [s["sentiment"] for s in final_states if s["sentiment"] is not None]
    sentiment_stats = {}
    if sentiments:
        sentiment_stats = {
            "mean": sum(sentiments) / len(sentiments),
            "std": (
                sum((s - sum(sentiments) / len(sentiments)) ** 2 for s in sentiments)
                / len(sentiments)
            )
            ** 0.5,
            "min": min(sentiments),
            "max": max(sentiments),
        }

    # Outcome distributions
    outcome_distributions: dict[str, dict[str, float]] = {}
    if total > 0:
        for position, count in positions.items():
            outcome_distributions[position] = count / total

    return {
        "population_size": total,
        "aware_count": aware,
        "aware_rate": aware / total if total > 0 else 0,
        "reasoned_count": reasoned,
        "sharing_count": sharing,
        "sharing_rate": sharing / total if total > 0 else 0,
        "position_distribution": dict(positions),
        "position_percentages": outcome_distributions,
        "sentiment_stats": sentiment_stats,
    }


def compute_segment_aggregates(
    state_manager: StateManager,
    agents: list[dict[str, Any]],
    segment_attribute: str,
) -> list[dict[str, Any]]:
    """Break down results by any attribute.

    Args:
        state_manager: State manager with final state
        agents: List of agent dictionaries
        segment_attribute: Attribute to segment by

    Returns:
        List of segment aggregate dictionaries
    """
    final_states = state_manager.export_final_states()

    # Group agents by segment value
    segments: dict[str, list[str]] = defaultdict(list)
    for i, agent in enumerate(agents):
        agent_id = agent.get("_id", str(i))
        value = agent.get(segment_attribute, "unknown")
        segments[str(value)].append(agent_id)

    # Map agent_id to state
    state_map = {s["agent_id"]: s for s in final_states}

    results = []
    for segment_value, agent_ids in segments.items():
        states = [state_map.get(aid) for aid in agent_ids if aid in state_map]
        states = [s for s in states if s]

        if not states:
            continue

        # Compute distributions for this segment
        position_counts: dict[str, int] = defaultdict(int)
        sentiments: list[float] = []

        for state in states:
            if state["position"]:
                position_counts[state["position"]] += 1
            if state["sentiment"] is not None:
                sentiments.append(state["sentiment"])

        # Normalize position counts
        total = len(states)
        position_dist = {pos: count / total for pos, count in position_counts.items()}

        results.append(
            {
                "segment_attribute": segment_attribute,
                "segment_value": segment_value,
                "agent_count": len(agent_ids),
                "aware_count": sum(1 for s in states if s["aware"]),
                "position_distribution": position_dist,
                "position_counts": dict(position_counts),
                "average_sentiment": (
                    sum(sentiments) / len(sentiments) if sentiments else None
                ),
            }
        )

    # Sort by agent count (largest segments first)
    results.sort(key=lambda x: x["agent_count"], reverse=True)

    return results


def compute_outcome_distributions(
    state_manager: StateManager,
    outcomes: list[OutcomeDefinition],
) -> dict[str, dict[str, float]]:
    """Compute distributions for all outcomes.

    Args:
        state_manager: State manager with final state
        outcomes: List of outcome definitions from scenario

    Returns:
        Dict mapping outcome name to distribution
    """
    final_states = state_manager.export_final_states()
    total = len(final_states)

    if total == 0:
        return {}

    distributions: dict[str, dict[str, float]] = {}

    for outcome in outcomes:
        if outcome.type == OutcomeType.CATEGORICAL:
            counts: dict[str, int] = defaultdict(int)
            for state in final_states:
                value = state["outcomes"].get(outcome.name)
                if value:
                    counts[str(value)] += 1

            if counts:
                distributions[outcome.name] = {k: v / total for k, v in counts.items()}

        elif outcome.type == OutcomeType.BOOLEAN:
            true_count = 0
            false_count = 0
            for state in final_states:
                value = state["outcomes"].get(outcome.name)
                if value is True:
                    true_count += 1
                elif value is False:
                    false_count += 1

            total_bool = true_count + false_count
            if total_bool > 0:
                distributions[outcome.name] = {
                    "true": true_count / total_bool,
                    "false": false_count / total_bool,
                }

        elif outcome.type == OutcomeType.FLOAT:
            values = []
            for state in final_states:
                value = state["outcomes"].get(outcome.name)
                if value is not None:
                    try:
                        values.append(float(value))
                    except (ValueError, TypeError):
                        pass

            if values:
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                distributions[outcome.name] = {
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

    return distributions


def compute_timeline_aggregates(
    summaries: list[TimestepSummary],
) -> list[dict[str, Any]]:
    """Compute timeline aggregates from timestep summaries.

    Args:
        summaries: List of timestep summaries

    Returns:
        List of timeline point dictionaries
    """
    timeline = []

    cumulative_shares = 0
    for summary in summaries:
        cumulative_shares += summary.shares_occurred

        # Normalize position distribution
        total_positions = sum(summary.position_distribution.values())
        position_pct = {}
        if total_positions > 0:
            position_pct = {
                k: v / total_positions for k, v in summary.position_distribution.items()
            }

        timeline.append(
            {
                "timestep": summary.timestep,
                "exposure_rate": summary.exposure_rate,
                "position_distribution": position_pct,
                "average_sentiment": summary.average_sentiment,
                "average_conviction": summary.average_conviction,
                "sentiment_variance": summary.sentiment_variance,
                "cumulative_shares": cumulative_shares,
                "new_exposures": summary.new_exposures,
                "agents_reasoned": summary.agents_reasoned,
            }
        )

    return timeline
