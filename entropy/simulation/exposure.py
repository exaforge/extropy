"""Exposure logic for seed exposure and network propagation.

Handles how agents become exposed to the event through:
1. Seed exposure (from channels based on Phase 2 rules)
2. Network propagation (from other agents who share)
"""

import logging
import random
from typing import Any

from ..models.scenario import ScenarioSpec, ExposureRule, SpreadConfig
from ..sampler import eval_condition, ConditionError

from .models import ExposureRecord, SimulationEvent, SimulationEventType
from .state import StateManager

logger = logging.getLogger(__name__)


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
        return eval_condition(rule.when, agent)
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
    agent_map = {a.get("_id", str(i)): a for i, a in enumerate(agents)}

    for rule in scenario.seed_exposure.rules:
        if rule.timestep != timestep:
            continue

        channel_credibility = get_channel_credibility(scenario, rule.channel)
        event_credibility = scenario.event.credibility

        for agent in agents:
            agent_id = agent.get("_id", str(agents.index(agent)))

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
                credibility=event_credibility * channel_credibility,
            )

            state_manager.record_exposure(agent_id, exposure)
            state_manager.log_event(SimulationEvent(
                timestep=timestep,
                event_type=SimulationEventType.SEED_EXPOSURE,
                agent_id=agent_id,
                details={"channel": rule.channel},
            ))
            new_exposures += 1

    return new_exposures


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
) -> float:
    """Calculate probability that an agent shares to a specific neighbor.

    Args:
        agent: Agent attributes dictionary
        edge_data: Edge attributes
        spread_config: Spread configuration from scenario
        rng: Random number generator (unused, but available for stochastic modifiers)

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

            if eval_condition(modifier.when, context):
                base_prob = base_prob * modifier.multiply + modifier.add
        except ConditionError:
            # Skip modifier if condition fails
            pass

    # Clamp to valid range
    return max(0.0, min(1.0, base_prob))


def propagate_through_network(
    timestep: int,
    scenario: ScenarioSpec,
    agents: list[dict[str, Any]],
    network: dict[str, Any],
    state_manager: StateManager,
    rng: random.Random,
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

    Returns:
        Count of new exposures via network
    """
    new_exposures = 0
    agent_map = {a.get("_id", str(i)): a for i, a in enumerate(agents)}

    # Get agents who will share
    sharers = state_manager.get_sharers()

    for sharer_id in sharers:
        sharer_agent = agent_map.get(sharer_id)
        if not sharer_agent:
            continue

        # Get neighbors from network
        neighbors = get_neighbors(network, sharer_id)

        for neighbor_id, edge_data in neighbors:
            neighbor_agent = agent_map.get(neighbor_id)
            if not neighbor_agent:
                continue

            # Calculate share probability for this edge
            prob = calculate_share_probability(
                sharer_agent,
                edge_data,
                scenario.spread,
                rng,
            )

            if rng.random() > prob:
                continue

            # Record exposure (even if already aware - for multi-touch)
            exposure = ExposureRecord(
                timestep=timestep,
                channel="network",
                source_agent_id=sharer_id,
                content=scenario.event.content,
                credibility=0.85,  # Peer credibility
            )

            state_manager.record_exposure(neighbor_id, exposure)
            state_manager.log_event(SimulationEvent(
                timestep=timestep,
                event_type=SimulationEventType.NETWORK_EXPOSURE,
                agent_id=neighbor_id,
                details={
                    "source": sharer_id,
                    "edge_type": edge_data.get("type", "unknown"),
                },
            ))
            new_exposures += 1

    return new_exposures


def count_seed_exposures_for_timestep(
    timestep: int,
    scenario: ScenarioSpec,
) -> int:
    """Estimate number of seed exposure rules for a timestep.

    Useful for progress reporting.

    Args:
        timestep: Timestep to check
        scenario: Scenario specification

    Returns:
        Count of rules for this timestep
    """
    count = 0
    for rule in scenario.seed_exposure.rules:
        if rule.timestep == timestep:
            count += 1
    return count


def get_exposure_timeline(
    scenario: ScenarioSpec,
) -> dict[int, list[str]]:
    """Get mapping of timesteps to exposure channels.

    Args:
        scenario: Scenario specification

    Returns:
        Dict mapping timestep to list of channel names
    """
    timeline: dict[int, list[str]] = {}

    for rule in scenario.seed_exposure.rules:
        if rule.timestep not in timeline:
            timeline[rule.timestep] = []
        if rule.channel not in timeline[rule.timestep]:
            timeline[rule.timestep].append(rule.channel)

    return timeline
