"""Integration tests for the simulation engine timestep loop.

Tests the full timestep cycle (exposure → reasoning → state update)
and multi-timestep dynamics with mocked LLM calls. No real API calls.
Functions under test in entropy/simulation/engine.py.
"""

from datetime import datetime
from unittest.mock import patch

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


def _make_scenario(
    max_timesteps=10,
    rules=None,
    share_probability=1.0,
    stop_conditions=None,
):
    """Create a configurable scenario for engine tests."""
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
                    description="Broadcast",
                    reach="broadcast",
                    credibility_modifier=1.0,
                ),
            ],
            rules=rules,
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Observe",
        ),
        spread=SpreadConfig(share_probability=share_probability),
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
        simulation=SimulationConfig(
            max_timesteps=max_timesteps,
            timestep_unit=TimestepUnit.HOUR,
            stop_conditions=stop_conditions,
        ),
    )


def _make_engine(scenario, agents, network, tmp_path):
    """Create a SimulationEngine with the given parameters."""
    config = SimulationRunConfig(
        scenario_path="test.yaml",
        output_dir=str(tmp_path / "output"),
        random_seed=42,
    )
    return SimulationEngine(
        scenario=scenario,
        population_spec=pytest.importorskip("tests.conftest").minimal_population_spec
        if False
        else _minimal_pop_spec(),
        agents=agents,
        network=network,
        config=config,
    )


def _minimal_pop_spec():
    """Inline minimal population spec to avoid fixture dependency."""
    from entropy.core.models.population import (
        AttributeSpec,
        GroundingInfo,
        GroundingSummary,
        NormalDistribution,
        PopulationSpec,
        SamplingConfig,
        SpecMeta,
    )

    return PopulationSpec(
        meta=SpecMeta(
            description="Test population",
            size=100,
            geography="Test Region",
            created_at=datetime(2024, 1, 1),
            version="1.0",
        ),
        grounding=GroundingSummary(
            overall="medium",
            sources_count=0,
            strong_count=0,
            medium_count=2,
            low_count=0,
            sources=[],
        ),
        attributes=[
            AttributeSpec(
                name="age",
                type="int",
                category="universal",
                description="Age",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal", mean=45.0, std=10.0, min=25.0, max=70.0
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            ),
        ],
        sampling_order=["age"],
    )


def _mock_batch_reason(responses_map=None, default_response=None):
    """Create a mock for batch_reason_agents that returns canned responses.

    Args:
        responses_map: Optional dict of agent_id → ReasoningResponse
        default_response: Fallback response for agents not in responses_map
    """
    if default_response is None:
        default_response = _make_reasoning_response()

    def mock_fn(contexts, scenario, config, **kwargs):
        results = []
        for ctx in contexts:
            if responses_map and ctx.agent_id in responses_map:
                results.append((ctx.agent_id, responses_map[ctx.agent_id]))
            else:
                results.append((ctx.agent_id, default_response))
        return results

    return mock_fn


# ============================================================================
# Engine Initialization
# ============================================================================


class TestEngineInit:
    """Test engine initialization."""

    def test_adjacency_not_used_in_engine_propagation(
        self, ten_agents, linear_network, tmp_path
    ):
        """Engine builds agent_map from agents list."""
        scenario = _make_scenario()
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        assert "a0" in engine.agent_map
        assert "a9" in engine.agent_map
        assert len(engine.agent_map) == 10

    def test_personas_pre_generated(self, ten_agents, linear_network, tmp_path):
        """All agents get personas at init time."""
        scenario = _make_scenario()
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        assert len(engine._personas) == 10
        for i in range(10):
            assert f"a{i}" in engine._personas
            assert len(engine._personas[f"a{i}"]) > 0


# ============================================================================
# Single Timestep
# ============================================================================


class TestSingleTimestep:
    """Test a single timestep execution with mocked LLM."""

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_seed_exposure_then_reasoning(
        self, mock_batch, ten_agents, linear_network, tmp_path
    ):
        """Timestep 0: seed exposure → agents reason → state updated."""
        mock_batch.side_effect = _mock_batch_reason()
        scenario = _make_scenario(max_timesteps=5)
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        summary = engine._run_timestep(0)

        # All 10 agents exposed at timestep 0 (broadcast, prob=1.0)
        assert summary.new_exposures == 10
        assert summary.agents_reasoned == 10

        # All agents should now have position and sentiment
        for i in range(10):
            state = engine.state_manager.get_agent_state(f"a{i}")
            assert state.aware is True
            assert state.position == "adopt"
            assert state.sentiment == 0.5

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_no_reasoning_without_exposure(
        self, mock_batch, ten_agents, linear_network, tmp_path
    ):
        """Timestep with no exposure rules → no reasoning."""
        mock_batch.return_value = []
        # Rules only at timestep=5, running timestep=0
        rules = [
            ExposureRule(channel="broadcast", timestep=5, when="true", probability=1.0),
        ]
        scenario = _make_scenario(max_timesteps=10, rules=rules)
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        summary = engine._run_timestep(0)

        assert summary.new_exposures == 0
        assert summary.agents_reasoned == 0
        # No agents to reason → batch_reason_agents is never called
        mock_batch.assert_not_called()

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_memory_entry_saved(self, mock_batch, ten_agents, linear_network, tmp_path):
        """Reasoning produces a memory entry for each agent."""
        mock_batch.side_effect = _mock_batch_reason()
        scenario = _make_scenario(max_timesteps=5)
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        engine._run_timestep(0)

        traces = engine.state_manager.get_memory_traces("a0")
        assert len(traces) == 1
        assert traces[0].summary == "Positive initial reaction."

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_flip_resistance_applied(
        self, mock_batch, ten_agents, linear_network, tmp_path
    ):
        """Firm agent with weak new conviction → position flip rejected."""
        scenario = _make_scenario(max_timesteps=5)
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        # Manually set a0 to firm conviction with "adopt" position
        engine.state_manager.record_exposure(
            "a0",
            ExposureRecord(
                timestep=0, channel="broadcast", content="test", credibility=0.9
            ),
        )
        engine.state_manager.update_agent_state(
            "a0",
            AgentState(
                agent_id="a0",
                aware=True,
                position="adopt",
                conviction=CONVICTION_MAP[ConvictionLevel.FIRM],
                sentiment=0.5,
            ),
            timestep=0,
        )

        # Mock: a0 tries to flip to "reject" with very_uncertain conviction
        mock_batch.side_effect = _mock_batch_reason(
            responses_map={
                "a0": _make_reasoning_response(
                    position="reject",
                    conviction=CONVICTION_MAP[ConvictionLevel.VERY_UNCERTAIN],
                    will_share=True,
                ),
            }
        )

        # Need to give a0 enough new exposures to trigger re-reasoning
        for t in range(1, 5):
            engine.state_manager.record_exposure(
                "a0",
                ExposureRecord(
                    timestep=t,
                    channel="network",
                    content="test",
                    credibility=0.85,
                    source_agent_id=f"a{t}",
                ),
            )

        engine._run_timestep(1)

        state = engine.state_manager.get_agent_state("a0")
        # Flip rejected — kept old position
        assert state.position == "adopt"
        # But sharing also gated because conviction is very_uncertain
        assert state.will_share is False

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_conviction_gated_sharing(
        self, mock_batch, ten_agents, linear_network, tmp_path
    ):
        """Very uncertain agent gets will_share forced to False."""
        mock_batch.side_effect = _mock_batch_reason(
            default_response=_make_reasoning_response(
                conviction=CONVICTION_MAP[ConvictionLevel.VERY_UNCERTAIN],
                will_share=True,  # Agent wants to share but conviction too low
            )
        )
        scenario = _make_scenario(max_timesteps=5)
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        engine._run_timestep(0)

        for i in range(10):
            state = engine.state_manager.get_agent_state(f"a{i}")
            assert state.will_share is False  # gated by low conviction


# ============================================================================
# Multi-Timestep Dynamics
# ============================================================================


class TestMultiTimestepDynamics:
    """Test multi-timestep simulation behavior."""

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_information_cascade_through_chain(
        self, mock_batch, ten_agents, linear_network, tmp_path
    ):
        """Information propagates through chain network over timesteps.

        a0 exposed at t=0, shares to a1 at t=1, a1 shares to a2 at t=2, etc.
        """
        mock_batch.side_effect = _mock_batch_reason(
            default_response=_make_reasoning_response(
                will_share=True,
                conviction=CONVICTION_MAP[ConvictionLevel.MODERATE],
            )
        )

        # Only seed a0 at timestep 0
        rules = [
            ExposureRule(
                channel="broadcast",
                timestep=0,
                when="_id == 'a0'",
                probability=1.0,
            ),
        ]
        scenario = _make_scenario(
            max_timesteps=12,
            rules=rules,
            share_probability=1.0,
        )
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=linear_network,
            config=config,
        )

        # Run several timesteps
        for t in range(6):
            engine._run_timestep(t)

        # After 6 timesteps in a chain, information should have reached
        # several hops from a0
        a0_state = engine.state_manager.get_agent_state("a0")
        assert a0_state.aware is True

        a1_state = engine.state_manager.get_agent_state("a1")
        assert a1_state.aware is True

        # Verify exposure rate is increasing
        rate = engine.state_manager.get_exposure_rate()
        assert rate > 0.1  # At minimum a0 and a1 are aware

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_stopping_condition_triggers(
        self, mock_batch, ten_agents, star_network, tmp_path
    ):
        """Simulation stops early when stop condition is met."""
        mock_batch.side_effect = _mock_batch_reason(
            default_response=_make_reasoning_response(
                will_share=True,
                conviction=CONVICTION_MAP[ConvictionLevel.MODERATE],
            )
        )

        scenario = _make_scenario(
            max_timesteps=50,
            share_probability=1.0,
            stop_conditions=["exposure_rate > 0.5"],
        )
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=star_network,
            config=config,
        )

        result = engine.run()

        # Should have stopped before max_timesteps
        assert result.total_timesteps < 50
        assert result.stopped_reason is not None

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_isolated_agent_never_exposed(self, mock_batch, tmp_path):
        """Agent with no network edges never gets network exposure."""
        mock_batch.side_effect = _mock_batch_reason(
            default_response=_make_reasoning_response(will_share=True)
        )

        agents = [
            {"_id": "a0", "age": 30, "role": "junior"},
            {"_id": "a1", "age": 40, "role": "senior"},
            {"_id": "isolated", "age": 50, "role": "mid"},
        ]
        network = {
            "meta": {"node_count": 3},
            "nodes": [{"id": "a0"}, {"id": "a1"}, {"id": "isolated"}],
            "edges": [{"source": "a0", "target": "a1", "type": "colleague"}],
        }

        # Only expose a0
        rules = [
            ExposureRule(
                channel="broadcast",
                timestep=0,
                when="_id == 'a0'",
                probability=1.0,
            ),
        ]
        scenario = _make_scenario(
            max_timesteps=5,
            rules=rules,
            share_probability=1.0,
        )
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=agents,
            network=network,
            config=config,
        )

        for t in range(5):
            engine._run_timestep(t)

        iso_state = engine.state_manager.get_agent_state("isolated")
        assert iso_state.aware is False

    @patch("entropy.simulation.engine.batch_reason_agents")
    def test_exposure_rate_never_decreases(
        self, mock_batch, ten_agents, star_network, tmp_path
    ):
        """Exposure rate should be monotonically non-decreasing."""
        mock_batch.side_effect = _mock_batch_reason(
            default_response=_make_reasoning_response(
                will_share=True,
                conviction=CONVICTION_MAP[ConvictionLevel.MODERATE],
            )
        )

        scenario = _make_scenario(max_timesteps=8, share_probability=1.0)
        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(tmp_path / "output"),
            random_seed=42,
        )
        engine = SimulationEngine(
            scenario=scenario,
            population_spec=_minimal_pop_spec(),
            agents=ten_agents,
            network=star_network,
            config=config,
        )

        prev_rate = 0.0
        for t in range(5):
            engine._run_timestep(t)
            rate = engine.state_manager.get_exposure_rate()
            assert rate >= prev_rate
            prev_rate = rate
