"""Tests for exposure propagation and network spreading.

Tests seed exposure application, network propagation, share probability
calculation, and credibility assignments.
Functions under test in extropy/simulation/propagation.py.
"""

import random
from datetime import datetime

import pytest

from extropy.core.models import AgentState, ExposureRecord
from extropy.core.models.scenario import (
    Event,
    EventType,
    ExposureChannel,
    ExposureRule,
    InteractionConfig,
    InteractionType,
    OutcomeConfig,
    OutcomeDefinition,
    OutcomeType,
    ScenarioMeta,
    ScenarioSpec,
    SeedExposure,
    SimulationConfig,
    SpreadConfig,
    SpreadModifier,
    TimestepUnit,
)
from extropy.simulation.propagation import (
    apply_seed_exposures,
    calculate_share_probability,
    evaluate_exposure_rule,
    get_channel_credibility,
    get_neighbors,
    propagate_through_network,
)
from extropy.simulation.state import StateManager


def _make_scenario(
    rules=None,
    channels=None,
    event_credibility=0.9,
    share_probability=0.3,
    share_modifiers=None,
):
    """Create a scenario with configurable exposure rules and spread config."""
    if channels is None:
        channels = [
            ExposureChannel(
                name="broadcast",
                description="Mass broadcast",
                reach="broadcast",
                credibility_modifier=1.0,
            ),
        ]
    if rules is None:
        rules = [
            ExposureRule(
                channel="broadcast",
                timestep=0,
                when="true",
                probability=1.0,
            ),
        ]

    return ScenarioSpec(
        meta=ScenarioMeta(
            name="test",
            description="Test scenario",
            population_spec="test.yaml",
            study_db="study.db",
            population_id="default",
            network_id="default",
            created_at=datetime(2024, 1, 1),
        ),
        event=Event(
            type=EventType.PRODUCT_LAUNCH,
            content="A new product is launching.",
            source="Test Corp",
            credibility=event_credibility,
            ambiguity=0.2,
            emotional_valence=0.3,
        ),
        seed_exposure=SeedExposure(channels=channels, rules=rules),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Observe",
        ),
        spread=SpreadConfig(
            share_probability=share_probability,
            share_modifiers=share_modifiers or [],
        ),
        outcomes=OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="adoption",
                    description="Adoption decision",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["adopt", "reject", "undecided"],
                ),
            ],
        ),
        simulation=SimulationConfig(max_timesteps=5, timestep_unit=TimestepUnit.HOUR),
    )


# ============================================================================
# Exposure Rule Evaluation
# ============================================================================


class TestEvaluateExposureRule:
    """Test evaluate_exposure_rule(rule, agent, timestep)."""

    def test_matching_timestep_and_true_condition(self):
        rule = ExposureRule(
            channel="broadcast", timestep=0, when="true", probability=1.0
        )
        agent = {"_id": "a0", "age": 30}
        assert evaluate_exposure_rule(rule, agent, 0) is True

    def test_wrong_timestep(self):
        rule = ExposureRule(
            channel="broadcast", timestep=0, when="true", probability=1.0
        )
        agent = {"_id": "a0", "age": 30}
        assert evaluate_exposure_rule(rule, agent, 1) is False

    def test_condition_filters_agent(self):
        rule = ExposureRule(
            channel="broadcast", timestep=0, when="age > 40", probability=1.0
        )
        agent = {"_id": "a0", "age": 30}
        assert evaluate_exposure_rule(rule, agent, 0) is False

    def test_condition_matches_agent(self):
        rule = ExposureRule(
            channel="broadcast", timestep=0, when="age > 40", probability=1.0
        )
        agent = {"_id": "a0", "age": 50}
        assert evaluate_exposure_rule(rule, agent, 0) is True

    def test_numeric_true_condition(self):
        """when='1' should also be treated as always true."""
        rule = ExposureRule(channel="broadcast", timestep=0, when="1", probability=1.0)
        agent = {"_id": "a0"}
        assert evaluate_exposure_rule(rule, agent, 0) is True


# ============================================================================
# Channel Credibility
# ============================================================================


class TestGetChannelCredibility:
    """Test get_channel_credibility(scenario, channel_name)."""

    def test_existing_channel(self):
        scenario = _make_scenario(
            channels=[
                ExposureChannel(
                    name="email",
                    description="Email",
                    reach="targeted",
                    credibility_modifier=0.8,
                ),
            ]
        )
        assert get_channel_credibility(scenario, "email") == 0.8

    def test_missing_channel_returns_default(self):
        scenario = _make_scenario()
        assert get_channel_credibility(scenario, "nonexistent") == 1.0


# ============================================================================
# Seed Exposure Application
# ============================================================================


class TestApplySeedExposures:
    """Test apply_seed_exposures(timestep, scenario, agents, state_manager, rng)."""

    def test_all_agents_exposed_broadcast(self, tmp_path, ten_agents, rng):
        """probability=1.0, when='true' → all 10 agents exposed."""
        scenario = _make_scenario()
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)

        count = apply_seed_exposures(0, scenario, ten_agents, sm, rng)

        assert count == 10
        for agent in ten_agents:
            state = sm.get_agent_state(agent["_id"])
            assert state.aware is True
            assert state.exposure_count == 1

    def test_no_exposure_wrong_timestep(self, tmp_path, ten_agents, rng):
        """Rule at timestep=5, called at timestep=0 → no exposures."""
        rules = [
            ExposureRule(channel="broadcast", timestep=5, when="true", probability=1.0),
        ]
        scenario = _make_scenario(rules=rules)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)

        count = apply_seed_exposures(0, scenario, ten_agents, sm, rng)
        assert count == 0

    def test_conditional_exposure(self, tmp_path, ten_agents, rng):
        """Only agents matching condition get exposed."""
        # ten_agents: roles cycle junior, mid, senior. Seniors at indices 2,5,8
        rules = [
            ExposureRule(
                channel="broadcast",
                timestep=0,
                when="role == 'senior'",
                probability=1.0,
            ),
        ]
        scenario = _make_scenario(rules=rules)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)

        count = apply_seed_exposures(0, scenario, ten_agents, sm, rng)

        # Seniors are at indices 2, 5, 8 (i % 3 == 2 in ten_agents fixture)
        assert count == 3
        for i in range(10):
            state = sm.get_agent_state(f"a{i}")
            if ten_agents[i]["role"] == "senior":
                assert state.aware is True
            else:
                assert state.aware is False

    def test_probabilistic_exposure(self, tmp_path, ten_agents):
        """prob=0.5, seeded rng → deterministic subset."""
        rules = [
            ExposureRule(channel="broadcast", timestep=0, when="true", probability=0.5),
        ]
        scenario = _make_scenario(rules=rules)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        count = apply_seed_exposures(0, scenario, ten_agents, sm, rng)

        # With seed 42 and prob=0.5, some but not all should be exposed
        assert 0 < count < 10

        # Deterministic: same seed, same result
        sm2 = StateManager(tmp_path / "test2.db", agents=ten_agents)
        rng2 = random.Random(42)
        count2 = apply_seed_exposures(0, scenario, ten_agents, sm2, rng2)
        assert count == count2

    def test_credibility_calculation(self, tmp_path, rng):
        """Exposure credibility = event_credibility * channel_credibility_modifier."""
        agents = [{"_id": "a0", "age": 30}]
        channels = [
            ExposureChannel(
                name="email",
                description="Email",
                reach="targeted",
                credibility_modifier=0.8,
            ),
        ]
        rules = [
            ExposureRule(channel="email", timestep=0, when="true", probability=1.0),
        ]
        scenario = _make_scenario(rules=rules, channels=channels, event_credibility=0.9)
        sm = StateManager(tmp_path / "test.db", agents=agents)

        apply_seed_exposures(0, scenario, agents, sm, rng)

        state = sm.get_agent_state("a0")
        assert len(state.exposures) == 1
        assert state.exposures[0].credibility == pytest.approx(0.72)  # 0.9 * 0.8

    def test_already_aware_agent_gets_new_exposure(self, tmp_path, rng):
        """An already-aware agent still receives additional exposures."""
        agents = [{"_id": "a0", "age": 30}]
        scenario = _make_scenario()
        sm = StateManager(tmp_path / "test.db", agents=agents)

        # First exposure
        apply_seed_exposures(0, scenario, agents, sm, rng)
        state = sm.get_agent_state("a0")
        assert state.exposure_count == 1

        # Add another rule at timestep=1
        rules2 = [
            ExposureRule(channel="broadcast", timestep=1, when="true", probability=1.0),
        ]
        scenario2 = _make_scenario(rules=rules2)
        apply_seed_exposures(1, scenario2, agents, sm, rng)

        state2 = sm.get_agent_state("a0")
        assert state2.exposure_count == 2
        assert state2.aware is True


# ============================================================================
# Network Neighbors
# ============================================================================


class TestGetNeighbors:
    """Test get_neighbors(network, agent_id)."""

    def test_hub_of_star(self, star_network):
        neighbors = get_neighbors(star_network, "a0")
        assert len(neighbors) == 9

    def test_spoke_of_star(self, star_network):
        neighbors = get_neighbors(star_network, "a1")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "a0"

    def test_chain_middle(self, linear_network):
        neighbors = get_neighbors(linear_network, "a5")
        assert len(neighbors) == 2
        neighbor_ids = {n[0] for n in neighbors}
        assert neighbor_ids == {"a4", "a6"}

    def test_chain_endpoint(self, linear_network):
        neighbors = get_neighbors(linear_network, "a0")
        assert len(neighbors) == 1
        assert neighbors[0][0] == "a1"

    def test_isolated_agent(self):
        """Agent with no edges has no neighbors."""
        network = {
            "meta": {"node_count": 3},
            "nodes": [{"id": "a0"}, {"id": "a1"}, {"id": "a2"}],
            "edges": [{"source": "a0", "target": "a1", "type": "colleague"}],
        }
        neighbors = get_neighbors(network, "a2")
        assert len(neighbors) == 0

    def test_bidirectional(self):
        """Edge source→target means both can see each other."""
        network = {
            "meta": {"node_count": 2},
            "nodes": [{"id": "a0"}, {"id": "a1"}],
            "edges": [{"source": "a0", "target": "a1", "type": "colleague"}],
        }
        assert len(get_neighbors(network, "a0")) == 1
        assert len(get_neighbors(network, "a1")) == 1


# ============================================================================
# Share Probability
# ============================================================================


class TestCalculateShareProbability:
    """Test calculate_share_probability(agent, edge_data, spread_config, rng)."""

    def test_base_probability_no_modifiers(self):
        config = SpreadConfig(share_probability=0.3)
        agent = {"_id": "a0", "age": 30}
        edge = {"type": "colleague", "weight": 0.5}
        rng = random.Random(42)

        prob = calculate_share_probability(agent, edge, config, rng)
        assert prob == 0.3

    def test_modifier_multiply(self):
        config = SpreadConfig(
            share_probability=0.3,
            share_modifiers=[SpreadModifier(when="True", multiply=2.0, add=0.0)],
        )
        agent = {"_id": "a0"}
        edge = {"type": "colleague"}
        rng = random.Random(42)

        prob = calculate_share_probability(agent, edge, config, rng)
        assert prob == pytest.approx(0.6)

    def test_modifier_add(self):
        config = SpreadConfig(
            share_probability=0.3,
            share_modifiers=[SpreadModifier(when="True", multiply=1.0, add=0.1)],
        )
        agent = {"_id": "a0"}
        edge = {"type": "colleague"}
        rng = random.Random(42)

        prob = calculate_share_probability(agent, edge, config, rng)
        assert prob == pytest.approx(0.4)

    def test_clamp_upper(self):
        """Modifier overflow above 1.0 is softly saturated."""
        config = SpreadConfig(
            share_probability=0.8,
            share_modifiers=[SpreadModifier(when="True", multiply=2.0, add=0.0)],
        )
        agent = {"_id": "a0"}
        edge = {"type": "colleague"}
        rng = random.Random(42)

        prob = calculate_share_probability(agent, edge, config, rng)
        assert 0.95 < prob < 1.0

    def test_clamp_lower(self):
        """Modifier pushing probability below 0 gets clamped."""
        config = SpreadConfig(
            share_probability=0.1,
            share_modifiers=[SpreadModifier(when="True", multiply=1.0, add=-0.5)],
        )
        agent = {"_id": "a0"}
        edge = {"type": "colleague"}
        rng = random.Random(42)

        prob = calculate_share_probability(agent, edge, config, rng)
        assert prob == 0.0

    def test_condition_not_met_skips_modifier(self):
        config = SpreadConfig(
            share_probability=0.3,
            share_modifiers=[SpreadModifier(when="age > 100", multiply=5.0, add=0.0)],
        )
        agent = {"_id": "a0", "age": 30}
        edge = {"type": "colleague"}
        rng = random.Random(42)

        prob = calculate_share_probability(agent, edge, config, rng)
        assert prob == 0.3  # modifier skipped

    def test_edge_type_in_context(self):
        """Modifier can reference edge_type from edge_data."""
        config = SpreadConfig(
            share_probability=0.3,
            share_modifiers=[
                SpreadModifier(when="edge_type == 'mentor'", multiply=2.0, add=0.0)
            ],
        )
        agent = {"_id": "a0", "age": 30}
        edge_mentor = {"type": "mentor"}
        edge_colleague = {"type": "colleague"}
        rng = random.Random(42)

        prob_mentor = calculate_share_probability(agent, edge_mentor, config, rng)
        prob_colleague = calculate_share_probability(agent, edge_colleague, config, rng)

        assert prob_mentor == pytest.approx(0.6)
        assert prob_colleague == 0.3  # condition not met

    def test_hop_decay_lowers_probability(self):
        """Higher hop depth should reduce probability when decay_per_hop > 0."""
        config = SpreadConfig(share_probability=0.6, decay_per_hop=0.1)
        agent = {"_id": "a0"}
        edge = {"type": "colleague"}
        rng = random.Random(42)

        p_hop0 = calculate_share_probability(agent, edge, config, rng, hop_depth=0)
        p_hop2 = calculate_share_probability(agent, edge, config, rng, hop_depth=2)

        assert p_hop0 == pytest.approx(0.6)
        assert p_hop2 == pytest.approx(0.6 * 0.9 * 0.9)


# ============================================================================
# Network Propagation
# ============================================================================


class TestPropagateNetwork:
    """Test propagate_through_network(...)."""

    def _setup_sharer(self, sm, agent_id, timestep=0):
        """Make an agent aware and willing to share."""
        sm.record_exposure(
            agent_id,
            ExposureRecord(
                timestep=timestep,
                channel="broadcast",
                content="test",
                credibility=0.9,
            ),
        )
        sm.update_agent_state(
            agent_id,
            AgentState(
                agent_id=agent_id,
                aware=True,
                will_share=True,
                position="adopt",
                sentiment=0.5,
                conviction=0.5,
            ),
            timestep=timestep,
        )

    def test_single_sharer_star(self, tmp_path, ten_agents, star_network):
        """Hub shares to all spokes."""
        scenario = _make_scenario(share_probability=1.0)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        self._setup_sharer(sm, "a0")

        count = propagate_through_network(
            1, scenario, ten_agents, star_network, sm, rng
        )

        # Hub a0 has 9 neighbors, share_prob=1.0 → all 9 exposed
        assert count == 9
        for i in range(1, 10):
            state = sm.get_agent_state(f"a{i}")
            assert state.aware is True

    def test_no_sharers_no_propagation(self, tmp_path, ten_agents, star_network):
        """No agents with will_share=True → zero propagation."""
        scenario = _make_scenario(share_probability=1.0)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        count = propagate_through_network(
            0, scenario, ten_agents, star_network, sm, rng
        )
        assert count == 0

    def test_single_hop_in_chain(self, tmp_path, ten_agents, linear_network):
        """In a chain, a0 sharing reaches a1 only (one hop per timestep call)."""
        scenario = _make_scenario(share_probability=1.0)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        self._setup_sharer(sm, "a0")

        count = propagate_through_network(
            1, scenario, ten_agents, linear_network, sm, rng
        )

        # a0 has one neighbor (a1) in the chain
        assert count == 1
        assert sm.get_agent_state("a1").aware is True
        assert sm.get_agent_state("a2").aware is False

    def test_peer_credibility_is_085(self, tmp_path, ten_agents, linear_network):
        """Network exposures have credibility = 0.85."""
        scenario = _make_scenario(share_probability=1.0)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        self._setup_sharer(sm, "a0")
        propagate_through_network(1, scenario, ten_agents, linear_network, sm, rng)

        state = sm.get_agent_state("a1")
        assert len(state.exposures) == 1
        assert state.exposures[0].credibility == 0.85
        assert state.exposures[0].channel == "network"
        assert state.exposures[0].source_agent_id == "a0"

    def test_already_aware_still_gets_exposure(
        self, tmp_path, ten_agents, linear_network
    ):
        """Already-aware agents still receive new exposures (multi-touch)."""
        scenario = _make_scenario(share_probability=1.0)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        # Make a1 already aware via seed
        sm.record_exposure(
            "a1",
            ExposureRecord(
                timestep=0, channel="broadcast", content="test", credibility=0.9
            ),
        )

        self._setup_sharer(sm, "a0")
        propagate_through_network(1, scenario, ten_agents, linear_network, sm, rng)

        state = sm.get_agent_state("a1")
        assert state.exposure_count == 2  # seed + network

    def test_probabilistic_sharing(self, tmp_path, ten_agents, star_network):
        """share_prob=0.5 with seeded rng → deterministic subset."""
        scenario = _make_scenario(share_probability=0.5)
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        self._setup_sharer(sm, "a0")
        count = propagate_through_network(
            1, scenario, ten_agents, star_network, sm, rng
        )

        # With prob=0.5 and 9 neighbors, expect some but not all
        assert 0 < count < 9

    def test_max_hops_blocks_deeper_reshares(
        self, tmp_path, ten_agents, linear_network
    ):
        """Agents beyond max_hops should not receive propagated exposures."""
        scenario = _make_scenario(share_probability=1.0)
        scenario.spread.max_hops = 1
        sm = StateManager(tmp_path / "test.db", agents=ten_agents)
        rng = random.Random(42)

        # Seed a0, then allow only a0 to share.
        self._setup_sharer(sm, "a0", timestep=0)
        first = propagate_through_network(
            1, scenario, ten_agents, linear_network, sm, rng
        )
        assert first == 1  # a0 -> a1

        # Make a1 a sharer, stop a0 from sharing again.
        sm.update_agent_state(
            "a0",
            AgentState(agent_id="a0", aware=True, will_share=False, position="adopt"),
            timestep=1,
        )
        sm.update_agent_state(
            "a1",
            AgentState(agent_id="a1", aware=True, will_share=True, position="adopt"),
            timestep=1,
        )

        second = propagate_through_network(
            2, scenario, ten_agents, linear_network, sm, rng
        )
        assert second == 0
        assert sm.get_agent_state("a2").aware is False
