"""Mock-based integration tests for the SimulationEngine.

Tests seed exposure application, state persistence, flip resistance,
conviction-gated sharing, and the sub-function decomposition.
"""

from datetime import datetime

import pytest

from entropy.core.models import (
    AgentState,
    ConvictionLevel,
    CONVICTION_MAP,
    ExposureRecord,
    ReasoningResponse,
    SimulationRunConfig,
)
from entropy.core.models.scenario import (
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
    TimestepUnit,
)
from entropy.simulation.engine import SimulationEngine


@pytest.fixture
def minimal_scenario():
    """Create a minimal scenario spec for engine tests."""
    return ScenarioSpec(
        meta=ScenarioMeta(
            name="test_scenario",
            description="Test scenario",
            population_spec="test.yaml",
            agents_file="test.json",
            network_file="test_network.json",
            created_at=datetime(2024, 1, 1),
        ),
        event=Event(
            type=EventType.PRODUCT_LAUNCH,
            content="A new product is launching.",
            source="Test Corp",
            credibility=0.9,
            ambiguity=0.2,
            emotional_valence=0.3,
        ),
        seed_exposure=SeedExposure(
            channels=[
                ExposureChannel(
                    name="broadcast",
                    description="Mass broadcast channel",
                    reach="broadcast",
                    credibility_modifier=1.0,
                ),
            ],
            rules=[
                ExposureRule(
                    channel="broadcast",
                    timestep=0,
                    when="true",
                    probability=1.0,
                ),
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Agents observe each other's public statements",
        ),
        spread=SpreadConfig(
            share_probability=0.3,
        ),
        outcomes=OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="adoption",
                    description="Whether the agent adopts the product",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["adopt", "reject", "undecided"],
                ),
            ],
        ),
        simulation=SimulationConfig(
            max_timesteps=5,
            timestep_unit=TimestepUnit.HOUR,
        ),
    )


@pytest.fixture
def simple_agents():
    """Three simple agents for testing."""
    return [
        {"_id": "a0", "age": 30, "role": "junior"},
        {"_id": "a1", "age": 40, "role": "senior"},
        {"_id": "a2", "age": 50, "role": "mid"},
    ]


@pytest.fixture
def simple_network():
    """Simple network connecting all three agents."""
    return {
        "meta": {"node_count": 3},
        "nodes": [{"id": "a0"}, {"id": "a1"}, {"id": "a2"}],
        "edges": [
            {"source": "a0", "target": "a1", "type": "colleague"},
            {"source": "a1", "target": "a2", "type": "colleague"},
        ],
    }


@pytest.fixture
def minimal_pop_spec(minimal_population_spec):
    """Reuse the conftest minimal population spec."""
    return minimal_population_spec


def _make_reasoning_response(**kwargs) -> ReasoningResponse:
    """Helper to create a ReasoningResponse with sensible defaults."""
    defaults = {
        "position": "adopt",
        "sentiment": 0.5,
        "conviction": CONVICTION_MAP[ConvictionLevel.MODERATE],
        "public_statement": "I think this is good.",
        "reasoning_summary": "Positive initial reaction.",
        "action_intent": "Will try it",
        "will_share": True,
        "reasoning": "I considered the product and find it appealing.",
        "outcomes": {"adoption": "adopt"},
    }
    defaults.update(kwargs)
    return ReasoningResponse(**defaults)


class TestEngineInit:
    """Test engine initialization."""

    def test_adjacency_list_built(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )
        # a0-a1 edge => a0 has [a1], a1 has [a0, a2], a2 has [a1]
        assert len(engine.adjacency.get("a0", [])) == 1
        assert len(engine.adjacency.get("a1", [])) == 2
        assert len(engine.adjacency.get("a2", [])) == 1

    def test_agent_map_built(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )
        assert "a0" in engine.agent_map
        assert "a1" in engine.agent_map
        assert "a2" in engine.agent_map


class TestFlipResistance:
    """Test conviction-based flip resistance logic."""

    def test_firm_agent_rejects_weak_flip(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """A firm-conviction agent should reject a position flip when new conviction < moderate."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        old_state = AgentState(
            agent_id="a0",
            position="adopt",
            conviction=CONVICTION_MAP[ConvictionLevel.FIRM],
        )

        # Response tries to flip to "reject" with very_uncertain conviction
        response = _make_reasoning_response(
            position="reject",
            conviction=CONVICTION_MAP[ConvictionLevel.VERY_UNCERTAIN],
        )

        # Run the state update logic
        results = [("a0", response)]
        old_states = {"a0": old_state}

        agents_reasoned, state_changes, shares_occurred = engine._apply_state_updates(
            timestep=1, results=results, old_states=old_states
        )

        # The flip should have been rejected — check the state
        final_state = engine.state_manager.get_agent_state("a0")
        assert final_state.position == "adopt"  # kept old position

    def test_firm_agent_accepts_moderate_flip(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """A firm-conviction agent should accept a flip when new conviction >= moderate."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        old_state = AgentState(
            agent_id="a0",
            position="adopt",
            conviction=CONVICTION_MAP[ConvictionLevel.FIRM],
        )

        response = _make_reasoning_response(
            position="reject",
            conviction=CONVICTION_MAP[ConvictionLevel.MODERATE],
        )

        results = [("a0", response)]
        old_states = {"a0": old_state}

        engine._apply_state_updates(timestep=1, results=results, old_states=old_states)

        final_state = engine.state_manager.get_agent_state("a0")
        assert final_state.position == "reject"  # flip accepted


class TestConvictionGatedSharing:
    """Test that very_uncertain agents don't share."""

    def test_very_uncertain_agent_wont_share(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        old_state = AgentState(agent_id="a0")
        response = _make_reasoning_response(
            conviction=CONVICTION_MAP[ConvictionLevel.VERY_UNCERTAIN],
            will_share=True,
        )

        results = [("a0", response)]
        old_states = {"a0": old_state}

        engine._apply_state_updates(timestep=0, results=results, old_states=old_states)

        final_state = engine.state_manager.get_agent_state("a0")
        assert final_state.will_share is False  # sharing gated


class TestSeedExposure:
    """Test seed exposure application."""

    def test_seed_exposure_creates_awareness(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        with engine.state_manager.transaction():
            new_exposures = engine._apply_exposures(0)

        # All 3 agents should be exposed (probability=1.0, when="true")
        assert new_exposures == 3

        for aid in ["a0", "a1", "a2"]:
            state = engine.state_manager.get_agent_state(aid)
            assert state.aware is True


class TestStateManagerTransaction:
    """Test the StateManager transaction context manager."""

    def test_transaction_commits_on_success(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Test that transaction commits changes when no exception occurs."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        # Apply changes within a transaction
        with engine.state_manager.transaction():
            exposure = ExposureRecord(
                timestep=0,
                channel="broadcast",
                content="Test content",
                credibility=0.9,
            )
            engine.state_manager.record_exposure("a0", exposure)

        # Changes should be committed
        state = engine.state_manager.get_agent_state("a0")
        assert state.aware is True
        assert state.exposure_count == 1
        assert len(state.exposures) == 1

    def test_transaction_rolls_back_on_exception(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Test that transaction rolls back changes when exception is raised."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        # Verify initial state
        initial_state = engine.state_manager.get_agent_state("a0")
        assert initial_state.aware is False
        assert initial_state.exposure_count == 0

        # Try to apply changes but raise an exception
        try:
            with engine.state_manager.transaction():
                exposure = ExposureRecord(
                    timestep=0,
                    channel="broadcast",
                    content="Test content",
                    credibility=0.9,
                )
                engine.state_manager.record_exposure("a0", exposure)

                # Raise an exception to trigger rollback
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Changes should be rolled back
        state = engine.state_manager.get_agent_state("a0")
        assert state.aware is False
        assert state.exposure_count == 0
        assert len(state.exposures) == 0

    def test_transaction_nested_operations(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Test transaction with multiple state changes."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        # Apply multiple changes in one transaction
        with engine.state_manager.transaction():
            # Expose multiple agents
            for agent_id in ["a0", "a1", "a2"]:
                exposure = ExposureRecord(
                    timestep=0,
                    channel="broadcast",
                    content="Test content",
                    credibility=0.9,
                )
                engine.state_manager.record_exposure(agent_id, exposure)

            # Update one agent's state
            state = AgentState(
                agent_id="a0",
                position="adopt",
                sentiment=0.8,
                conviction=0.7,
            )
            engine.state_manager.update_agent_state("a0", state, timestep=0)

        # All changes should be committed
        for agent_id in ["a0", "a1", "a2"]:
            state = engine.state_manager.get_agent_state(agent_id)
            assert state.aware is True

        # Check specific state update
        a0_state = engine.state_manager.get_agent_state("a0")
        assert a0_state.position == "adopt"
        assert a0_state.sentiment == 0.8

    def test_transaction_partial_rollback(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Test that partial changes are rolled back on exception."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        # Try to apply multiple changes but fail partway through
        try:
            with engine.state_manager.transaction():
                # Expose first agent (should be rolled back)
                exposure = ExposureRecord(
                    timestep=0,
                    channel="broadcast",
                    content="Test content",
                    credibility=0.9,
                )
                engine.state_manager.record_exposure("a0", exposure)

                # Expose second agent (should be rolled back)
                engine.state_manager.record_exposure("a1", exposure)

                # Raise exception before committing
                raise RuntimeError("Failed midway")
        except RuntimeError:
            pass

        # Both agents should remain unaware
        assert engine.state_manager.get_agent_state("a0").aware is False
        assert engine.state_manager.get_agent_state("a1").aware is False


class TestCommittedFlag:
    """Test that engine sets committed flag based on conviction."""

    def test_firm_agent_becomes_committed(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Agent with conviction >= FIRM should get committed=True."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        old_state = AgentState(agent_id="a0")
        response = _make_reasoning_response(
            conviction=CONVICTION_MAP[ConvictionLevel.FIRM],
        )

        results = [("a0", response)]
        old_states = {"a0": old_state}

        engine._apply_state_updates(timestep=0, results=results, old_states=old_states)

        final_state = engine.state_manager.get_agent_state("a0")
        assert final_state.committed is True

    def test_moderate_agent_not_committed(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Agent with conviction < FIRM should NOT get committed=True."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        old_state = AgentState(agent_id="a0")
        response = _make_reasoning_response(
            conviction=CONVICTION_MAP[ConvictionLevel.MODERATE],
        )

        results = [("a0", response)]
        old_states = {"a0": old_state}

        engine._apply_state_updates(timestep=0, results=results, old_states=old_states)

        final_state = engine.state_manager.get_agent_state("a0")
        assert final_state.committed is False


class TestOneShotSharing:
    """Test that one-shot sharing prevents duplicate network propagation."""

    def test_agent_shares_once_per_neighbor(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """After sharing once, a second propagate call should produce 0 new exposures to same neighbor."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        # Set up a0 as a sharer with will_share=True
        state = AgentState(
            agent_id="a0",
            aware=True,
            will_share=True,
            position="adopt",
            conviction=0.7,
            committed=True,
        )
        engine.state_manager.update_agent_state("a0", state, timestep=0)
        # Also make a0 aware
        exposure = ExposureRecord(
            timestep=0, channel="broadcast", content="Test", credibility=0.9
        )
        engine.state_manager.record_exposure("a0", exposure)

        from entropy.simulation.propagation import propagate_through_network
        import random

        # Use rng with seed for deterministic share_probability pass
        rng = random.Random(42)

        # Set share probability to 1.0 so sharing always succeeds
        minimal_scenario.spread.share_probability = 1.0

        # First propagation — should share to a1 (a0's only neighbor)
        new1 = propagate_through_network(
            timestep=1,
            scenario=minimal_scenario,
            agents=simple_agents,
            network=simple_network,
            state_manager=engine.state_manager,
            rng=rng,
            adjacency=engine.adjacency,
            agent_map=engine.agent_map,
        )
        assert new1 >= 1  # At least one new exposure

        # Second propagation — same position, should produce 0
        new2 = propagate_through_network(
            timestep=2,
            scenario=minimal_scenario,
            agents=simple_agents,
            network=simple_network,
            state_manager=engine.state_manager,
            rng=rng,
            adjacency=engine.adjacency,
            agent_map=engine.agent_map,
        )
        assert new2 == 0  # No new exposures — already shared
