"""Mock-based integration tests for the SimulationEngine.

Tests seed exposure application, state persistence, flip resistance,
conviction-gated sharing, and the sub-function decomposition.
"""

from datetime import datetime

import pytest

from unittest.mock import patch

from entropy.core.models import (
    AgentState,
    ConvictionLevel,
    CONVICTION_MAP,
    ExposureRecord,
    ReasoningResponse,
    SimulationRunConfig,
)
from entropy.simulation.progress import SimulationProgress
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
from entropy.simulation.reasoning import BatchTokenUsage


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

        agents_reasoned, state_changes, shares_occurred = (
            engine._process_reasoning_chunk(
                timestep=1, results=results, old_states=old_states
            )
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
        """Public stance can flip while private behavior remains anchored."""
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

        engine._process_reasoning_chunk(
            timestep=1, results=results, old_states=old_states
        )

        final_state = engine.state_manager.get_agent_state("a0")
        assert final_state.public_position == "reject"
        assert final_state.position == "adopt"


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

        engine._process_reasoning_chunk(
            timestep=0, results=results, old_states=old_states
        )

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

        engine._process_reasoning_chunk(
            timestep=0, results=results, old_states=old_states
        )

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

        engine._process_reasoning_chunk(
            timestep=0, results=results, old_states=old_states
        )

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


class TestSimulationMetadata:
    """Test simulation_metadata table and checkpoint methods."""

    def test_metadata_round_trip(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """save_metadata / get_metadata round trip."""
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

        engine.state_manager.save_metadata("foo", "bar")
        assert engine.state_manager.get_metadata("foo") == "bar"
        assert engine.state_manager.get_metadata("missing") is None

    def test_last_completed_timestep_empty(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Returns -1 when no timestep summaries exist."""
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
        assert engine.state_manager.get_last_completed_timestep() == -1

    def test_last_completed_timestep_with_data(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Returns correct max timestep from summaries."""
        from entropy.core.models import TimestepSummary

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

        for ts in [0, 1, 2]:
            summary = TimestepSummary(timestep=ts, exposure_rate=0.5)
            engine.state_manager.save_timestep_summary(summary)

        assert engine.state_manager.get_last_completed_timestep() == 2

    def test_checkpoint_lifecycle(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """mark_timestep_started sets checkpoint, mark_timestep_completed clears it."""
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

        assert engine.state_manager.get_checkpoint_timestep() is None

        engine.state_manager.mark_timestep_started(5)
        assert engine.state_manager.get_checkpoint_timestep() == 5

        engine.state_manager.mark_timestep_completed(5)
        assert engine.state_manager.get_checkpoint_timestep() is None

    def test_agents_already_reasoned(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """get_agents_already_reasoned_this_timestep filters by last_reasoning_timestep."""
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

        # Update a0 as having reasoned at timestep 3
        state = AgentState(
            agent_id="a0",
            aware=True,
            position="adopt",
            conviction=0.5,
        )
        engine.state_manager.update_agent_state("a0", state, timestep=3)

        already = engine.state_manager.get_agents_already_reasoned_this_timestep(3)
        assert "a0" in already
        assert "a1" not in already

        # Different timestep — should be empty for timestep 4
        already_4 = engine.state_manager.get_agents_already_reasoned_this_timestep(4)
        assert "a0" not in already_4


class TestResumeLogic:
    """Test engine resume/checkpoint logic."""

    def test_resume_skips_completed_timesteps(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Engine should start from the next timestep after last completed."""
        from entropy.core.models import TimestepSummary

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

        # Simulate 2 completed timesteps
        for ts in [0, 1]:
            summary = TimestepSummary(timestep=ts, exposure_rate=0.5)
            engine.state_manager.save_timestep_summary(summary)

        assert engine._get_resume_timestep() == 2

    def test_resume_from_checkpoint(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """If checkpoint_timestep is set, resume that timestep."""
        from entropy.core.models import TimestepSummary

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

        # Simulate: timestep 0 completed, timestep 1 started but crashed
        summary = TimestepSummary(timestep=0, exposure_rate=0.5)
        engine.state_manager.save_timestep_summary(summary)
        engine.state_manager.mark_timestep_started(1)

        assert engine._get_resume_timestep() == 1

    def test_resume_fresh_start(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Fresh database should start from timestep 0."""
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

        assert engine._get_resume_timestep() == 0

    def test_resume_skips_processed_agents(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """On resume, agents already reasoned this timestep should be skipped."""
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

        # Expose all agents so they need reasoning
        for aid in ["a0", "a1", "a2"]:
            exposure = ExposureRecord(
                timestep=0, channel="broadcast", content="Test", credibility=0.9
            )
            engine.state_manager.record_exposure(aid, exposure)

        # Mark a0 as already reasoned at timestep 0
        state = AgentState(
            agent_id="a0",
            aware=True,
            position="adopt",
            conviction=0.5,
        )
        engine.state_manager.update_agent_state("a0", state, timestep=0)

        # Get agents to reason — a0 should be skipped because it already reasoned
        agents_to_reason = engine.state_manager.get_agents_to_reason(
            timestep=0, threshold=3
        )
        already_done = engine.state_manager.get_agents_already_reasoned_this_timestep(0)
        filtered = [a for a in agents_to_reason if a not in already_done]

        assert "a0" not in filtered
        # a1 and a2 should still need reasoning (never reasoned)
        assert "a1" in filtered
        assert "a2" in filtered


class TestTimelineAppend:
    """Test that timeline uses append mode for resume support."""

    def test_timeline_preserves_events_on_reopen(self, tmp_path):
        """Events written before close should persist when file is reopened."""
        from entropy.simulation.timeline import TimelineManager, TimelineReader
        from entropy.core.models import SimulationEvent, SimulationEventType

        timeline_path = tmp_path / "timeline.jsonl"

        # Write first batch
        tm1 = TimelineManager(timeline_path)
        tm1.log_event(
            SimulationEvent(
                timestep=0,
                event_type=SimulationEventType.AGENT_REASONED,
                agent_id="a0",
                details={"batch": 1},
            )
        )
        tm1.flush()
        tm1.close()

        # Reopen and write second batch (append mode)
        tm2 = TimelineManager(timeline_path)
        tm2.log_event(
            SimulationEvent(
                timestep=1,
                event_type=SimulationEventType.AGENT_REASONED,
                agent_id="a1",
                details={"batch": 2},
            )
        )
        tm2.flush()
        tm2.close()

        # Read all events — both batches should be present
        reader = TimelineReader(timeline_path)
        events = reader.get_all_events()
        assert len(events) == 2
        assert events[0]["agent_id"] == "a0"
        assert events[1]["agent_id"] == "a1"


class TestChunkedCommit:
    """Test that chunked reasoning commits partial results."""

    def test_chunk_size_respected(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Engine should use configured chunk_size."""
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
            chunk_size=2,
        )
        assert engine.chunk_size == 2

    def test_process_reasoning_chunk_commits_results(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """After processing a chunk, agent states should be updated in DB."""
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
            chunk_size=1,
        )

        old_state_a0 = AgentState(agent_id="a0")
        old_state_a1 = AgentState(agent_id="a1")
        old_states = {"a0": old_state_a0, "a1": old_state_a1}

        # Process first chunk (just a0)
        response_a0 = _make_reasoning_response(position="adopt", conviction=0.5)
        chunk1_results = [("a0", response_a0)]

        with engine.state_manager.transaction():
            r1, c1, s1 = engine._process_reasoning_chunk(
                timestep=0, results=chunk1_results, old_states=old_states
            )

        # a0 should be updated
        state_a0 = engine.state_manager.get_agent_state("a0")
        assert state_a0.position == "adopt"
        assert state_a0.last_reasoning_timestep == 0

        # a1 should still be unprocessed
        state_a1 = engine.state_manager.get_agent_state("a1")
        assert state_a1.position is None
        assert state_a1.last_reasoning_timestep == -1


class TestProgressState:
    """Test that engine wires SimulationProgress correctly."""

    def test_progress_state_updated(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Progress state should reflect agents processed after reasoning."""
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

        progress = SimulationProgress()
        engine.set_progress_state(progress)

        # Mock batch_reason_agents to return fixed results
        response_a0 = _make_reasoning_response(position="adopt", conviction=0.5)
        response_a1 = _make_reasoning_response(position="reject", conviction=0.7)

        def fake_batch(contexts, scenario, cfg, rate_limiter=None, on_agent_done=None):
            results = []
            for ctx in contexts:
                if ctx.agent_id == "a0":
                    resp = response_a0
                elif ctx.agent_id == "a1":
                    resp = response_a1
                else:
                    resp = _make_reasoning_response()
                results.append((ctx.agent_id, resp))
                if on_agent_done:
                    on_agent_done(ctx.agent_id, resp)
            return results, BatchTokenUsage()

        # Expose all agents so they need reasoning
        for aid in ["a0", "a1", "a2"]:
            exposure = ExposureRecord(
                timestep=0, channel="broadcast", content="Test", credibility=0.9
            )
            engine.state_manager.record_exposure(aid, exposure)

        with patch(
            "entropy.simulation.engine.batch_reason_agents", side_effect=fake_batch
        ):
            engine._reason_agents(0)

        snap = progress.snapshot()
        assert snap["agents_done"] == 3
        assert "adopt" in snap["position_counts"]
        assert "reject" in snap["position_counts"]

    def test_on_agent_done_callback_passed(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """batch_reason_agents should receive on_agent_done kwarg when progress is set."""
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

        progress = SimulationProgress()
        engine.set_progress_state(progress)

        # Expose one agent
        exposure = ExposureRecord(
            timestep=0, channel="broadcast", content="Test", credibility=0.9
        )
        engine.state_manager.record_exposure("a0", exposure)

        received_kwargs = {}

        def fake_batch(contexts, scenario, cfg, rate_limiter=None, on_agent_done=None):
            received_kwargs["on_agent_done"] = on_agent_done
            resp = _make_reasoning_response()
            return [(ctx.agent_id, resp) for ctx in contexts], BatchTokenUsage()

        with patch(
            "entropy.simulation.engine.batch_reason_agents", side_effect=fake_batch
        ):
            engine._reason_agents(0)

        assert received_kwargs.get("on_agent_done") is not None

    def test_progress_updated_when_zero_agents(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Progress should still update timestep/exposure even when 0 agents need reasoning."""
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

        progress = SimulationProgress()
        engine.set_progress_state(progress)

        # Don't expose any agents — 0 agents will need reasoning
        engine._reason_agents(0)

        snap = progress.snapshot()
        # begin_timestep should have been called even with 0 agents
        assert snap["max_timesteps"] == minimal_scenario.simulation.max_timesteps
        assert snap["agents_total"] == 0
        assert snap["agents_done"] == 0


class TestDynamicsUpdates:
    """Test bounded-confidence and conviction-decay dynamics."""

    def test_bounded_confidence_dampens_state_shift(
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

        old_state = AgentState(
            agent_id="a0",
            aware=True,
            position="adopt",
            sentiment=0.0,
            conviction=0.9,
            will_share=True,
        )
        response = _make_reasoning_response(
            position="reject",
            sentiment=-1.0,
            conviction=0.1,
            will_share=True,
        )

        engine._process_reasoning_chunk(
            timestep=1,
            results=[("a0", response)],
            old_states={"a0": old_state},
        )

        final_state = engine.state_manager.get_agent_state("a0")
        # Public uses rho=0.35, private uses rho=0.12 toward public.
        assert final_state.public_sentiment == pytest.approx(-0.35)
        assert final_state.public_conviction == pytest.approx(0.62)
        assert final_state.sentiment == pytest.approx(-0.042)
        assert final_state.conviction == pytest.approx(0.8664)

    def test_conviction_decay_updates_non_reasoning_agents(
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

        # Agent has prior state at t=0 and does not reason at t=1.
        engine.state_manager.record_exposure(
            "a0",
            ExposureRecord(
                timestep=0,
                channel="broadcast",
                content="seed",
                credibility=0.9,
            ),
        )
        engine.state_manager.update_agent_state(
            "a0",
            AgentState(
                agent_id="a0",
                aware=True,
                position="adopt",
                conviction=0.7,
                sentiment=0.4,
                will_share=True,
                committed=True,
            ),
            timestep=0,
        )

        with engine.state_manager.transaction():
            changed = engine.state_manager.apply_conviction_decay(
                timestep=1,
                decay_rate=0.05,
                sharing_threshold=CONVICTION_MAP[ConvictionLevel.VERY_UNCERTAIN],
                firm_threshold=CONVICTION_MAP[ConvictionLevel.FIRM],
            )

        assert changed == 1
        final_state = engine.state_manager.get_agent_state("a0")
        assert final_state.conviction == pytest.approx(0.665)
        assert final_state.committed is False


class TestTokenAccumulation:
    """Test that engine accumulates token usage from batch reasoning."""

    def test_tokens_accumulate_across_chunks(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """Token usage should sum across reasoning chunks."""
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            chunk_size=2,
        )

        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        # Expose all agents so they need reasoning
        for aid in ["a0", "a1", "a2"]:
            exposure = ExposureRecord(
                timestep=0, channel="broadcast", content="Test", credibility=0.9
            )
            engine.state_manager.record_exposure(aid, exposure)

        call_count = [0]

        def fake_batch(contexts, scenario, cfg, rate_limiter=None, on_agent_done=None):
            call_count[0] += 1
            resp = _make_reasoning_response()
            results = [(ctx.agent_id, resp) for ctx in contexts]
            usage = BatchTokenUsage(
                pivotal_input_tokens=100 * len(contexts),
                pivotal_output_tokens=50 * len(contexts),
                routine_input_tokens=30 * len(contexts),
                routine_output_tokens=10 * len(contexts),
            )
            return results, usage

        with patch(
            "entropy.simulation.engine.batch_reason_agents", side_effect=fake_batch
        ):
            engine._reason_agents(0)

        # 3 agents, chunk_size=2 -> 2 chunks (2 + 1)
        assert call_count[0] == 2
        assert engine.pivotal_input_tokens == 300  # 100 * 3
        assert engine.pivotal_output_tokens == 150  # 50 * 3
        assert engine.routine_input_tokens == 90  # 30 * 3
        assert engine.routine_output_tokens == 30  # 10 * 3

    def test_cost_in_meta_json(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """meta.json should contain cost block with token counts and estimated_usd."""
        import json

        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            model="gpt-5",
            routine_model="gpt-5-mini",
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        # Set some token counts directly
        engine.pivotal_input_tokens = 1_000_000
        engine.pivotal_output_tokens = 500_000
        engine.routine_input_tokens = 200_000
        engine.routine_output_tokens = 100_000

        # Run export (needs timeline to exist)
        engine.timeline.flush()
        engine.timeline.close()
        engine._export_results()

        with open(tmp_path / "output" / "meta.json") as f:
            meta = json.load(f)

        assert "cost" in meta
        cost = meta["cost"]
        assert cost["pivotal_input_tokens"] == 1_000_000
        assert cost["pivotal_output_tokens"] == 500_000
        assert cost["routine_input_tokens"] == 200_000
        assert cost["routine_output_tokens"] == 100_000
        assert cost["total_input_tokens"] == 1_200_000
        assert cost["total_output_tokens"] == 600_000
        assert cost["estimated_usd"] is not None
        assert cost["estimated_usd"] > 0

    def test_cost_unknown_model_returns_null_usd(
        self,
        minimal_scenario,
        simple_agents,
        simple_network,
        minimal_pop_spec,
        tmp_path,
    ):
        """When both models are unknown, estimated_usd should be null."""
        import json

        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            model="unknown-model-xyz",
            routine_model="unknown-model-abc",
        )
        engine = SimulationEngine(
            scenario=minimal_scenario,
            population_spec=minimal_pop_spec,
            agents=simple_agents,
            network=simple_network,
            config=config,
        )

        engine.pivotal_input_tokens = 1000
        engine.pivotal_output_tokens = 500

        engine.timeline.flush()
        engine.timeline.close()
        engine._export_results()

        with open(tmp_path / "output" / "meta.json") as f:
            meta = json.load(f)

        assert meta["cost"]["estimated_usd"] is None
