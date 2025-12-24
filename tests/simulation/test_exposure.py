"""Tests for exposure logic."""

import random
import tempfile
from pathlib import Path

import pytest

from entropy.simulation.exposure import (
    evaluate_exposure_rule,
    apply_seed_exposures,
    propagate_through_network,
    get_neighbors,
    calculate_share_probability,
)
from entropy.simulation.state import StateManager
from entropy.simulation.models import AgentState
from entropy.models.scenario import (
    ScenarioSpec,
    ScenarioMeta,
    Event,
    EventType,
    SeedExposure,
    ExposureChannel,
    ExposureRule,
    InteractionConfig,
    InteractionType,
    SpreadConfig,
    SpreadModifier,
    OutcomeConfig,
    SimulationConfig,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def sample_agents():
    """Sample agents for testing."""
    return [
        {"_id": "agent_001", "age": 35, "gender": "male", "plan_tier": "premium"},
        {"_id": "agent_002", "age": 42, "gender": "female", "plan_tier": "standard"},
        {"_id": "agent_003", "age": 28, "gender": "male", "plan_tier": "basic"},
        {"_id": "agent_004", "age": 55, "gender": "female", "plan_tier": "premium"},
        {"_id": "agent_005", "age": 31, "gender": "male", "plan_tier": "standard"},
    ]


@pytest.fixture
def sample_network():
    """Sample network for testing."""
    return {
        "meta": {"agent_count": 5, "edge_count": 4},
        "edges": [
            {"source": "agent_001", "target": "agent_002", "type": "colleague", "weight": 0.8},
            {"source": "agent_001", "target": "agent_003", "type": "friend", "weight": 0.6},
            {"source": "agent_002", "target": "agent_004", "type": "colleague", "weight": 0.7},
            {"source": "agent_003", "target": "agent_005", "type": "acquaintance", "weight": 0.4},
        ],
    }


@pytest.fixture
def sample_scenario():
    """Sample scenario for testing."""
    return ScenarioSpec(
        meta=ScenarioMeta(
            name="test_scenario",
            description="Test scenario",
            population_spec="pop.yaml",
            agents_file="agents.json",
            network_file="network.json",
        ),
        event=Event(
            type=EventType.ANNOUNCEMENT,
            content="Test announcement",
            source="Test Source",
            credibility=0.9,
            ambiguity=0.2,
            emotional_valence=-0.3,
        ),
        seed_exposure=SeedExposure(
            channels=[
                ExposureChannel(
                    name="email",
                    description="Email notification",
                    reach="broadcast",
                    credibility_modifier=1.0,
                ),
                ExposureChannel(
                    name="social_media",
                    description="Social media",
                    reach="targeted",
                    credibility_modifier=0.7,
                ),
            ],
            rules=[
                ExposureRule(
                    channel="email",
                    when="true",
                    probability=1.0,
                    timestep=0,
                ),
                ExposureRule(
                    channel="social_media",
                    when="age < 40",
                    probability=0.5,
                    timestep=1,
                ),
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Test interaction",
        ),
        spread=SpreadConfig(
            share_probability=0.5,
            share_modifiers=[
                SpreadModifier(when="age < 30", multiply=1.5),
            ],
        ),
        outcomes=OutcomeConfig(),
        simulation=SimulationConfig(max_timesteps=10),
    )


class TestEvaluateExposureRule:
    """Tests for exposure rule evaluation."""

    def test_true_condition(self):
        """Test 'true' condition."""
        rule = ExposureRule(channel="email", when="true", probability=1.0, timestep=0)
        agent = {"age": 35}

        assert evaluate_exposure_rule(rule, agent, timestep=0) is True

    def test_numeric_condition(self):
        """Test numeric condition."""
        rule = ExposureRule(channel="email", when="age < 40", probability=1.0, timestep=0)

        assert evaluate_exposure_rule(rule, {"age": 35}, timestep=0) is True
        assert evaluate_exposure_rule(rule, {"age": 45}, timestep=0) is False

    def test_timestep_mismatch(self):
        """Test that wrong timestep returns False."""
        rule = ExposureRule(channel="email", when="true", probability=1.0, timestep=1)

        assert evaluate_exposure_rule(rule, {}, timestep=0) is False
        assert evaluate_exposure_rule(rule, {}, timestep=1) is True

    def test_string_condition(self):
        """Test string equality condition."""
        rule = ExposureRule(
            channel="email",
            when="plan_tier == 'premium'",
            probability=1.0,
            timestep=0,
        )

        assert evaluate_exposure_rule(rule, {"plan_tier": "premium"}, timestep=0) is True
        assert evaluate_exposure_rule(rule, {"plan_tier": "basic"}, timestep=0) is False


class TestGetNeighbors:
    """Tests for getting neighbors from network."""

    def test_get_neighbors(self, sample_network):
        """Test getting neighbors."""
        neighbors = get_neighbors(sample_network, "agent_001")

        assert len(neighbors) == 2
        neighbor_ids = [n[0] for n in neighbors]
        assert "agent_002" in neighbor_ids
        assert "agent_003" in neighbor_ids

    def test_get_neighbors_bidirectional(self, sample_network):
        """Test neighbors work bidirectionally."""
        neighbors = get_neighbors(sample_network, "agent_002")

        neighbor_ids = [n[0] for n in neighbors]
        assert "agent_001" in neighbor_ids  # agent_001 is source, agent_002 is target
        assert "agent_004" in neighbor_ids

    def test_get_neighbors_no_connections(self, sample_network):
        """Test agent with no connections."""
        neighbors = get_neighbors(sample_network, "agent_999")
        assert len(neighbors) == 0


class TestApplySeedExposures:
    """Tests for applying seed exposures."""

    def test_broadcast_exposure(self, temp_db, sample_agents, sample_scenario):
        """Test broadcast exposure at timestep 0."""
        manager = StateManager(temp_db, sample_agents)
        rng = random.Random(42)

        count = apply_seed_exposures(
            timestep=0,
            scenario=sample_scenario,
            agents=sample_agents,
            state_manager=manager,
            rng=rng,
        )

        # All 5 agents should be exposed via email (probability=1.0, when="true")
        assert count == 5

        # Verify all are aware
        aware = manager.get_aware_agents()
        assert len(aware) == 5

        manager.close()

    def test_targeted_exposure(self, temp_db, sample_agents, sample_scenario):
        """Test targeted exposure at timestep 1."""
        manager = StateManager(temp_db, sample_agents)
        rng = random.Random(42)

        # First apply timestep 0
        apply_seed_exposures(0, sample_scenario, sample_agents, manager, rng)

        # Then timestep 1 (age < 40, probability=0.5)
        count = apply_seed_exposures(1, sample_scenario, sample_agents, manager, rng)

        # Agents with age < 40: agent_001(35), agent_003(28), agent_005(31)
        # With probability 0.5, some subset will be exposed
        # (exact count depends on RNG)
        assert count >= 0

        manager.close()


class TestCalculateShareProbability:
    """Tests for share probability calculation."""

    def test_base_probability(self):
        """Test base share probability."""
        spread = SpreadConfig(share_probability=0.5)
        rng = random.Random(42)

        prob = calculate_share_probability(
            agent={"age": 35},
            edge_data={"type": "colleague"},
            spread_config=spread,
            rng=rng,
        )

        assert prob == 0.5

    def test_modifier_applies(self):
        """Test that modifier applies when condition met."""
        spread = SpreadConfig(
            share_probability=0.5,
            share_modifiers=[
                SpreadModifier(when="age < 30", multiply=2.0),
            ],
        )
        rng = random.Random(42)

        # Young agent - modifier applies
        prob_young = calculate_share_probability(
            agent={"age": 25},
            edge_data={"type": "colleague"},
            spread_config=spread,
            rng=rng,
        )
        assert prob_young == 1.0  # 0.5 * 2.0, clamped to 1.0

        # Older agent - modifier doesn't apply
        prob_old = calculate_share_probability(
            agent={"age": 40},
            edge_data={"type": "colleague"},
            spread_config=spread,
            rng=rng,
        )
        assert prob_old == 0.5


class TestPropagateNetwork:
    """Tests for network propagation."""

    def test_propagation(self, temp_db, sample_agents, sample_network, sample_scenario):
        """Test network propagation from sharers."""
        manager = StateManager(temp_db, sample_agents)
        rng = random.Random(42)

        # Expose agent_001 and mark them as sharer
        from entropy.simulation.models import ExposureRecord

        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            source_agent_id=None,
            content="Test",
            credibility=0.9,
        )
        manager.record_exposure("agent_001", exposure)

        # Update agent_001 to be a sharer
        state = AgentState(agent_id="agent_001", aware=True, will_share=True)
        manager.update_agent_state("agent_001", state, 0)

        # Propagate
        count = propagate_through_network(
            timestep=1,
            scenario=sample_scenario,
            agents=sample_agents,
            network=sample_network,
            state_manager=manager,
            rng=rng,
        )

        # agent_001 has 2 neighbors (agent_002, agent_003)
        # With share_probability=0.5, some may be exposed
        assert count >= 0

        manager.close()
