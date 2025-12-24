"""Tests for simulation engine."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from entropy.simulation.engine import SimulationEngine
from entropy.simulation.models import SimulationRunConfig, ReasoningResponse
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
    OutcomeConfig,
    SimulationConfig,
)
from entropy.models.spec import PopulationSpec, SpecMeta, GroundingSummary


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_agents():
    """Sample agents for testing."""
    return [
        {"_id": "agent_001", "age": 35, "gender": "male", "plan_tier": "premium"},
        {"_id": "agent_002", "age": 42, "gender": "female", "plan_tier": "standard"},
        {"_id": "agent_003", "age": 28, "gender": "male", "plan_tier": "basic"},
    ]


@pytest.fixture
def sample_network():
    """Sample network for testing."""
    return {
        "meta": {"agent_count": 3, "edge_count": 2},
        "edges": [
            {"source": "agent_001", "target": "agent_002", "type": "colleague", "weight": 0.8},
            {"source": "agent_002", "target": "agent_003", "type": "friend", "weight": 0.6},
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
            ],
            rules=[
                ExposureRule(
                    channel="email",
                    when="true",
                    probability=1.0,
                    timestep=0,
                ),
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Test interaction",
        ),
        spread=SpreadConfig(share_probability=0.5),
        outcomes=OutcomeConfig(),
        simulation=SimulationConfig(max_timesteps=3),
    )


@pytest.fixture
def sample_population_spec():
    """Sample population spec for testing."""
    return PopulationSpec(
        meta=SpecMeta(
            description="Test population",
            size=3,
            geography="Test",
        ),
        grounding=GroundingSummary(
            overall="low",
            sources_count=0,
            strong_count=0,
            medium_count=0,
            low_count=0,
        ),
        attributes=[],
        sampling_order=[],
    )


@pytest.fixture
def sample_run_config(temp_output_dir):
    """Sample run config for testing."""
    return SimulationRunConfig(
        scenario_path="test_scenario.yaml",
        output_dir=str(temp_output_dir),
        model="gpt-5-mini",
        reasoning_effort="low",
        multi_touch_threshold=3,
        random_seed=42,
    )


class TestSimulationEngine:
    """Tests for the simulation engine."""

    def test_engine_initialization(
        self,
        sample_agents,
        sample_network,
        sample_scenario,
        sample_population_spec,
        sample_run_config,
    ):
        """Test engine can be initialized."""
        engine = SimulationEngine(
            scenario=sample_scenario,
            population_spec=sample_population_spec,
            agents=sample_agents,
            network=sample_network,
            config=sample_run_config,
        )

        assert engine.scenario == sample_scenario
        assert engine.agents == sample_agents
        assert engine.network == sample_network
        assert engine.config == sample_run_config

    @patch("entropy.simulation.engine.reason_agent")
    def test_engine_run_with_mocked_llm(
        self,
        mock_reasoning,
        temp_output_dir,
        sample_agents,
        sample_network,
        sample_scenario,
        sample_population_spec,
    ):
        """Test engine can run with mocked LLM."""
        # Mock the reasoning response
        mock_reasoning.return_value = ReasoningResponse(
            reasoning="I considered the announcement...",
            position="neutral",
            sentiment=0.0,
            action_intent="wait_and_see",
            will_share=False,
            outcomes={},
        )

        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(temp_output_dir),
            random_seed=42,
        )

        engine = SimulationEngine(
            scenario=sample_scenario,
            population_spec=sample_population_spec,
            agents=sample_agents,
            network=sample_network,
            config=config,
        )

        # Run simulation
        result = engine.run()

        assert result is not None
        assert result.total_timesteps >= 0
        assert result.final_exposure_rate >= 0

    @patch("entropy.simulation.engine.reason_agent")
    def test_engine_creates_output_files(
        self,
        mock_reasoning,
        temp_output_dir,
        sample_agents,
        sample_network,
        sample_scenario,
        sample_population_spec,
    ):
        """Test engine creates expected output files."""
        mock_reasoning.return_value = ReasoningResponse(
            reasoning="Test reasoning",
            position="neutral",
            sentiment=0.0,
            action_intent="none",
            will_share=False,
            outcomes={},
        )

        # Use max_timesteps=1 for quick test
        sample_scenario.simulation = SimulationConfig(max_timesteps=1)

        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(temp_output_dir),
            random_seed=42,
        )

        engine = SimulationEngine(
            scenario=sample_scenario,
            population_spec=sample_population_spec,
            agents=sample_agents,
            network=sample_network,
            config=config,
        )

        engine.run()

        # Check output files exist
        assert (temp_output_dir / "meta.json").exists()
        assert (temp_output_dir / "timeline.jsonl").exists()


class TestEngineExposure:
    """Tests for exposure handling in engine."""

    @patch("entropy.simulation.engine.reason_agent")
    def test_seed_exposure_applied(
        self,
        mock_reasoning,
        temp_output_dir,
        sample_agents,
        sample_network,
        sample_scenario,
        sample_population_spec,
    ):
        """Test that seed exposure is applied at timestep 0."""
        mock_reasoning.return_value = ReasoningResponse(
            reasoning="Test",
            position="neutral",
            sentiment=0.0,
            action_intent="none",
            will_share=False,
            outcomes={},
        )

        # Set max_timesteps to 1 for quick test
        sample_scenario.simulation = SimulationConfig(max_timesteps=1)

        config = SimulationRunConfig(
            scenario_path="test.yaml",
            output_dir=str(temp_output_dir),
            random_seed=42,
        )

        engine = SimulationEngine(
            scenario=sample_scenario,
            population_spec=sample_population_spec,
            agents=sample_agents,
            network=sample_network,
            config=config,
        )

        result = engine.run()

        # With probability=1.0 and when="true", all agents should be exposed
        assert result is not None
        assert result.final_exposure_rate > 0  # At least some agents exposed
