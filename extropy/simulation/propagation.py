"""Exposure logic for seed exposure and network propagation.

Handles how agents become exposed to the event through:
1. Seed exposure (from channels based on Phase 2 rules)
2. Network propagation (from other agents who share)
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

from ..core.models import (
    ScenarioSpec,
    Event,
    ExposureRule,
    SpreadConfig,
    ExposureRecord,
    SimulationEvent,
    SimulationEventType,
    TimelineEvent,
)
from ..population.sampler import eval_condition, ConditionError
from .state import StateManager

logger = logging.getLogger(__name__)

_SOFT_SATURATION_MAX = 0.97


@dataclass(frozen=True)
class TimelineExposureResult:
    """Outcome of applying timeline exposures for a timestep."""

    new_exposure_count: int
    active_event: Event | None
    direct_exposed_agent_ids: set[str]
    info_epoch: int | None
    re_reasoning_intensity: str | None


def _soft_saturate_probability(raw_probability: float) -> float:
    """Apply smooth saturation for high probabilities.

    Keeps low-mid probabilities unchanged while compressing extreme values
    so stacked modifiers don't collapse to near-deterministic sharing.
    """
    if raw_probability <= 1.0:
        return raw_probability

    # For overflow above 1.0, smoothly compress toward a high but non-deterministic ceiling.
    overflow = raw_probability - 1.0
    return 1.0 - (1.0 - _SOFT_SATURATION_MAX) * (1.0 - math.exp(-overflow))


def evaluate_exposure_rule(
    rule: ExposureRule,
    agent: dict[str, Any],
    timestep: int,
) -> bool:
    """Evaluate if an exposure rule applies to an agent.

    Args:
        rule: Exposure rule from scenario
        agent: Agent attributes dictionary
        timestep: Current timestep

    Returns:
        True if rule applies (before probability check)
    """
    # Check timestep
    if rule.timestep != timestep:
        return False

    # Evaluate condition
    if rule.when.lower() == "true" or rule.when == "1":
        return True

    try:
        return eval_condition(rule.when, agent, raise_on_error=True)
    except ConditionError as e:
        logger.warning(f"Failed to evaluate exposure rule '{rule.when}': {e}")
        return False


def get_channel_credibility(
    scenario: ScenarioSpec,
    channel_name: str,
) -> float:
    """Get credibility modifier for a channel.

    Args:
        scenario: Scenario specification
        channel_name: Name of the channel

    Returns:
        Credibility modifier (default 1.0)
    """
    for channel in scenario.seed_exposure.channels:
        if channel.name == channel_name:
            return channel.credibility_modifier
    return 1.0


def apply_seed_exposures(
    timestep: int,
    scenario: ScenarioSpec,
    agents: list[dict[str, Any]],
    state_manager: StateManager,
    rng: random.Random,
) -> int:
    """Apply Phase 2 exposure rules for this timestep.

    Args:
        timestep: Current timestep
        scenario: Scenario specification
        agents: List of all agents
        state_manager: State manager for recording exposures
        rng: Random number generator

    Returns:
        Count of new exposures
    """
    new_exposures = 0

    for rule in scenario.seed_exposure.rules:
        if rule.timestep != timestep:
            continue

        channel_credibility = get_channel_credibility(scenario, rule.channel)
        event_credibility = scenario.event.credibility

        for i, agent in enumerate(agents):
            agent_id = agent.get("_id", str(i))

            if not evaluate_exposure_rule(rule, agent, timestep):
                continue

            # Probabilistic exposure
            if rng.random() > rule.probability:
                continue

            exposure = ExposureRecord(
                timestep=timestep,
                channel=rule.channel,
                source_agent_id=None,
                content=scenario.event.content,
                credibility=min(1.0, event_credibility * channel_credibility),
            )

            state_manager.record_exposure(agent_id, exposure)
            state_manager.log_event(
                SimulationEvent(
                    timestep=timestep,
                    event_type=SimulationEventType.SEED_EXPOSURE,
                    agent_id=agent_id,
                    details={"channel": rule.channel},
                )
            )
            new_exposures += 1

    return new_exposures


def apply_timeline_exposures(
    timestep: int,
    scenario: ScenarioSpec,
    agents: list[dict[str, Any]],
    state_manager: StateManager,
    rng: random.Random,
) -> TimelineExposureResult:
    """Apply timeline event exposures for this timestep.

    Timeline events represent scenario developments (new information, escalations,
    resolutions) that occur at specific timesteps. Each timeline event can have
    custom exposure rules or reuse the seed exposure rules with updated content.

    Args:
        timestep: Current timestep
        scenario: Scenario specification
        agents: List of all agents
        state_manager: State manager for recording exposures
        rng: Random number generator

    Returns:
        TimelineExposureResult with counts, active event, direct exposure IDs,
        and provenance metadata for this timeline epoch.
    """
    if not scenario.timeline:
        return TimelineExposureResult(
            new_exposure_count=0,
            active_event=None,
            direct_exposed_agent_ids=set(),
            info_epoch=None,
            re_reasoning_intensity=None,
        )

    # Find timeline event for this timestep
    active_event: TimelineEvent | None = None
    for te in scenario.timeline:
        if te.timestep == timestep:
            active_event = te
            break

    if active_event is None:
        return TimelineExposureResult(
            new_exposure_count=0,
            active_event=None,
            direct_exposed_agent_ids=set(),
            info_epoch=None,
            re_reasoning_intensity=None,
        )

    logger.info(
        f"[TIMELINE] Timestep {timestep}: Applying timeline event - "
        f"{active_event.description or active_event.event.content[:50]}"
    )

    # Determine which exposure rules to use
    if active_event.exposure_rules is not None:
        rules = active_event.exposure_rules
    else:
        # Reuse seed exposure rules but substitute with timeline event content
        rules = scenario.seed_exposure.rules

    new_exposures = 0
    event_content = active_event.event.content
    event_credibility = active_event.event.credibility
    info_epoch = active_event.timestep
    intensity = active_event.re_reasoning_intensity or "normal"
    direct_exposed_agent_ids: set[str] = set()

    for rule in rules:
        # For timeline events, ignore the rule's timestep field — we're applying now
        # (Rules are designed for t=0 seed exposure but we reuse them for timeline)
        channel_credibility = get_channel_credibility(scenario, rule.channel)

        for i, agent in enumerate(agents):
            agent_id = agent.get("_id", str(i))

            # Evaluate the "when" condition (skip timestep check since we're applying now)
            when_cond = rule.when.lower()
            if when_cond != "true" and when_cond != "1":
                try:
                    if not eval_condition(rule.when, agent, raise_on_error=True):
                        continue
                except ConditionError as e:
                    logger.warning(
                        f"Failed to evaluate timeline exposure rule '{rule.when}': {e}"
                    )
                    continue

            # Probabilistic exposure
            if rng.random() > rule.probability:
                continue

            exposure = ExposureRecord(
                timestep=timestep,
                channel=rule.channel,
                source_agent_id=None,
                content=event_content,
                credibility=min(1.0, event_credibility * channel_credibility),
                info_epoch=info_epoch,
                force_rereason=True,
            )

            state_manager.record_exposure(agent_id, exposure)
            state_manager.log_event(
                SimulationEvent(
                    timestep=timestep,
                    event_type=SimulationEventType.SEED_EXPOSURE,
                    agent_id=agent_id,
                    details={
                        "channel": rule.channel,
                        "timeline_event": True,
                        "description": active_event.description,
                    },
                )
            )
            new_exposures += 1
            direct_exposed_agent_ids.add(agent_id)

    logger.info(f"[TIMELINE] Timestep {timestep}: {new_exposures} new exposures")

    return TimelineExposureResult(
        new_exposure_count=new_exposures,
        active_event=active_event.event,
        direct_exposed_agent_ids=direct_exposed_agent_ids,
        info_epoch=info_epoch,
        re_reasoning_intensity=intensity,
    )


def get_neighbors(
    network: dict[str, Any],
    agent_id: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Get neighbors of an agent from the network.

    Args:
        network: Network data (with edges list)
        agent_id: Agent ID

    Returns:
        List of (neighbor_id, edge_data) tuples
    """
    neighbors = []
    edges = network.get("edges", [])

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source == agent_id:
            neighbors.append((target, edge))
        elif target == agent_id:
            neighbors.append((source, edge))

    return neighbors


def calculate_share_probability(
    agent: dict[str, Any],
    edge_data: dict[str, Any],
    spread_config: SpreadConfig,
    rng: random.Random,
    hop_depth: int = 0,
) -> float:
    """Calculate probability that an agent shares to a specific neighbor.

    Args:
        agent: Agent attributes dictionary
        edge_data: Edge attributes
        spread_config: Spread configuration from scenario
        rng: Random number generator (unused, but available for stochastic modifiers)
        hop_depth: Current propagation depth from seed exposure

    Returns:
        Share probability (0-1)
    """
    base_prob = spread_config.share_probability

    # Apply modifiers
    for modifier in spread_config.share_modifiers:
        try:
            # Create context with both agent and edge attributes
            context = dict(agent)
            context["edge_type"] = edge_data.get("type", "unknown")
            context["edge_weight"] = edge_data.get("weight", 0.5)

            if eval_condition(modifier.when, context, raise_on_error=True):
                base_prob = base_prob * modifier.multiply + modifier.add
        except ConditionError:
            # Skip modifier if condition fails
            pass

    # Apply per-hop decay after modifiers.
    if hop_depth > 0 and spread_config.decay_per_hop > 0:
        decay_factor = (1.0 - spread_config.decay_per_hop) ** hop_depth
        base_prob *= decay_factor

    # Clamp negatives, then softly saturate high values.
    base_prob = max(0.0, base_prob)
    softened = _soft_saturate_probability(base_prob)

    return max(0.0, min(1.0, softened))


def propagate_through_network(
    timestep: int,
    scenario: ScenarioSpec,
    agents: list[dict[str, Any]],
    network: dict[str, Any],
    state_manager: StateManager,
    rng: random.Random,
    adjacency: dict[str, list[tuple[str, dict]]] | None = None,
    agent_map: dict[str, dict[str, Any]] | None = None,
    timeline_intensity_by_epoch: dict[int, str] | None = None,
) -> int:
    """Propagate information through network from sharing agents.

    Agents who have will_share=True spread to their neighbors.

    Args:
        timestep: Current timestep
        scenario: Scenario specification
        agents: List of all agents
        network: Network data
        state_manager: State manager
        rng: Random number generator
        adjacency: Pre-built adjacency list (optional, avoids O(E) scan per agent)
        agent_map: Pre-built agent lookup dict (optional, avoids rebuild)
        timeline_intensity_by_epoch: Optional map of info epoch -> intensity policy

    Returns:
        Count of new exposures via network
    """
    new_exposures = 0
    if agent_map is None:
        agent_map = {a.get("_id", str(i)): a for i, a in enumerate(agents)}

    # Get agents who will share
    sharers = state_manager.get_sharers()

    for sharer_id in sharers:
        sharer_agent = agent_map.get(sharer_id)
        if not sharer_agent:
            continue
        sharer_hop = state_manager.get_network_hop_depth(sharer_id)
        next_hop = (sharer_hop + 1) if sharer_hop is not None else 1

        if scenario.spread.max_hops is not None and next_hop > scenario.spread.max_hops:
            continue

        # Get neighbors from network (use adjacency list if available)
        if adjacency is not None:
            neighbors = adjacency.get(sharer_id, [])
        else:
            neighbors = get_neighbors(network, sharer_id)

        # One-shot sharing: filter to neighbors not yet shared to
        # (or shared to with a different position — allows re-share on position change)
        neighbor_ids = [nid for nid, _ in neighbors]
        sharer_state = state_manager.get_agent_state(sharer_id)
        sharer_public_position = sharer_state.public_position or sharer_state.position
        eligible_ids = set(
            state_manager.get_unshared_neighbors(
                sharer_id, neighbor_ids, sharer_public_position
            )
        )

        # Build lookup for edge data
        neighbor_edge_map = {nid: edge for nid, edge in neighbors}

        for neighbor_id in eligible_ids:
            neighbor_agent = agent_map.get(neighbor_id)
            if not neighbor_agent:
                continue

            edge_data = neighbor_edge_map[neighbor_id]

            # Calculate share probability for this edge
            prob = calculate_share_probability(
                sharer_agent,
                edge_data,
                scenario.spread,
                rng,
                hop_depth=next_hop,
            )

            # Record the share attempt regardless of probability outcome
            # (prevents retrying the same neighbor every timestep)
            state_manager.record_share(
                sharer_id, neighbor_id, timestep, sharer_public_position
            )

            if rng.random() > prob:
                continue

            # Record exposure (even if already aware - for multi-touch)
            decay_factor = (
                (1.0 - scenario.spread.decay_per_hop) ** next_hop
                if scenario.spread.decay_per_hop > 0
                else 1.0
            )
            exposure = ExposureRecord(
                timestep=timestep,
                channel="network",
                source_agent_id=sharer_id,
                content=scenario.event.content,
                credibility=max(0.05, min(1.0, 0.85 * decay_factor)),
                info_epoch=(
                    sharer_state.latest_info_epoch
                    if sharer_state.latest_info_epoch >= 0
                    else None
                ),
                force_rereason=(
                    (
                        timeline_intensity_by_epoch.get(sharer_state.latest_info_epoch)
                        in {"high", "extreme"}
                    )
                    if (
                        timeline_intensity_by_epoch
                        and sharer_state.latest_info_epoch is not None
                        and sharer_state.latest_info_epoch >= 0
                    )
                    else False
                ),
            )

            state_manager.record_exposure(neighbor_id, exposure)
            state_manager.log_event(
                SimulationEvent(
                    timestep=timestep,
                    event_type=SimulationEventType.NETWORK_EXPOSURE,
                    agent_id=neighbor_id,
                    details={
                        "source": sharer_id,
                        "edge_type": edge_data.get("type", "unknown"),
                    },
                )
            )
            new_exposures += 1

    return new_exposures
