"""Tests for the simulation module components."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from entropy.core.models.simulation import (
    SimulationEventType,
    ExposureRecord,
    AgentState,
    SimulationEvent,
    PeerOpinion,
    ReasoningContext,
    ReasoningResponse,
    SimulationRunConfig,
    TimestepSummary,
)
from entropy.simulation.state import StateManager
from entropy.simulation.persona import (
    generate_persona,
    generate_persona_for_reasoning,
    _format_age,
    _format_gender,
    _format_role,
    _format_experience,
    _get_descriptor,
    OPENNESS_DESCRIPTORS,
)


class TestSimulationModels:
    """Tests for simulation model classes."""

    def test_exposure_record_creation(self):
        """Test creating an exposure record."""
        record = ExposureRecord(
            timestep=0,
            channel="email",
            content="Important announcement",
            credibility=0.9,
        )
        assert record.timestep == 0
        assert record.channel == "email"
        assert record.source_agent_id is None

    def test_exposure_record_from_network(self):
        """Test exposure record from network propagation."""
        record = ExposureRecord(
            timestep=2,
            channel="network",
            source_agent_id="agent_042",
            content="Did you hear about...",
            credibility=0.7,
        )
        assert record.source_agent_id == "agent_042"

    def test_agent_state_defaults(self):
        """Test agent state default values."""
        state = AgentState(agent_id="agent_000")

        assert state.aware is False
        assert state.exposure_count == 0
        assert state.exposures == []
        assert state.last_reasoning_timestep == -1
        assert state.position is None
        assert state.sentiment is None
        assert state.will_share is False
        assert state.outcomes == {}

    def test_agent_state_with_values(self):
        """Test agent state with all values set."""
        state = AgentState(
            agent_id="agent_001",
            aware=True,
            exposure_count=3,
            position="supportive",
            sentiment=0.7,
            will_share=True,
            outcomes={"adoption_intent": "will_adopt"},
        )
        assert state.aware is True
        assert state.sentiment == 0.7

    def test_simulation_event_types(self):
        """Test simulation event type enum."""
        assert SimulationEventType.SEED_EXPOSURE.value == "seed_exposure"
        assert SimulationEventType.NETWORK_EXPOSURE.value == "network_exposure"
        assert SimulationEventType.AGENT_REASONED.value == "agent_reasoned"
        assert SimulationEventType.AGENT_SHARED.value == "agent_shared"
        assert SimulationEventType.STATE_CHANGED.value == "state_changed"

    def test_simulation_event_creation(self):
        """Test creating a simulation event."""
        event = SimulationEvent(
            timestep=5,
            event_type=SimulationEventType.AGENT_REASONED,
            agent_id="agent_010",
            details={"position": "resistant", "sentiment": -0.3},
        )
        assert event.timestep == 5
        assert event.agent_id == "agent_010"
        assert event.timestamp is not None

    def test_peer_opinion(self):
        """Test peer opinion model."""
        opinion = PeerOpinion(
            agent_id="agent_005",
            relationship="colleague",
            position="supportive",
            sentiment=0.8,
        )
        assert opinion.relationship == "colleague"
        assert opinion.position == "supportive"

    def test_reasoning_context(self):
        """Test reasoning context creation."""
        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            content="Announcement",
            credibility=0.9,
        )
        context = ReasoningContext(
            agent_id="agent_001",
            persona="You are a 45-year-old surgeon...",
            event_content="Hospital announces AI tool",
            exposure_history=[exposure],
            peer_opinions=[],
        )
        assert context.agent_id == "agent_001"
        assert len(context.exposure_history) == 1

    def test_reasoning_response(self):
        """Test reasoning response model."""
        response = ReasoningResponse(
            position="considering",
            sentiment=0.2,
            action_intent="wait_and_see",
            will_share=False,
            reasoning="I want to see more evidence before deciding.",
            outcomes={"adoption_intent": "considering"},
        )
        assert response.position == "considering"
        assert response.will_share is False

    def test_simulation_run_config_defaults(self):
        """Test simulation run config defaults."""
        config = SimulationRunConfig(
            scenario_path="scenario.yaml",
            output_dir="results/",
        )
        assert config.model == "gpt-5-mini"
        assert config.reasoning_effort == "low"
        assert config.multi_touch_threshold == 3

    def test_timestep_summary(self):
        """Test timestep summary model."""
        summary = TimestepSummary(
            timestep=5,
            new_exposures=20,
            agents_reasoned=15,
            shares_occurred=5,
            state_changes=10,
            exposure_rate=0.45,
            position_distribution={"supportive": 10, "resistant": 5},
            average_sentiment=0.3,
        )
        assert summary.timestep == 5
        assert summary.exposure_rate == 0.45


class TestStateManager:
    """Tests for the StateManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            yield db_path

    @pytest.fixture
    def agents(self):
        """Sample agents for testing."""
        return [
            {"_id": "agent_000", "age": 45},
            {"_id": "agent_001", "age": 35},
            {"_id": "agent_002", "age": 55},
        ]

    def test_state_manager_creation(self, temp_db):
        """Test creating a state manager."""
        with StateManager(temp_db) as manager:
            assert manager.db_path == temp_db
            assert temp_db.exists()

    def test_initialize_agents(self, temp_db, agents):
        """Test initializing agents in state manager."""
        with StateManager(temp_db, agents=agents) as manager:
            all_ids = manager.get_all_agent_ids()
            assert len(all_ids) == 3
            assert "agent_000" in all_ids

    def test_get_agent_state_default(self, temp_db, agents):
        """Test getting default agent state."""
        with StateManager(temp_db, agents=agents) as manager:
            state = manager.get_agent_state("agent_000")

            assert state.agent_id == "agent_000"
            assert state.aware is False
            assert state.exposure_count == 0

    def test_record_exposure(self, temp_db, agents):
        """Test recording an exposure."""
        with StateManager(temp_db, agents=agents) as manager:
            exposure = ExposureRecord(
                timestep=0,
                channel="email",
                content="Announcement",
                credibility=0.9,
            )
            manager.record_exposure("agent_000", exposure)

            state = manager.get_agent_state("agent_000")
            assert state.aware is True
            assert state.exposure_count == 1
            assert len(state.exposures) == 1

    def test_multiple_exposures(self, temp_db, agents):
        """Test recording multiple exposures."""
        with StateManager(temp_db, agents=agents) as manager:
            exposure1 = ExposureRecord(
                timestep=0, channel="email", content="First", credibility=0.9
            )
            exposure2 = ExposureRecord(
                timestep=1, channel="network", source_agent_id="agent_001",
                content="Second", credibility=0.7
            )

            manager.record_exposure("agent_000", exposure1)
            manager.record_exposure("agent_000", exposure2)

            state = manager.get_agent_state("agent_000")
            assert state.exposure_count == 2
            assert len(state.exposures) == 2

    def test_update_agent_state(self, temp_db, agents):
        """Test updating agent state after reasoning."""
        with StateManager(temp_db, agents=agents) as manager:
            # First expose the agent
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            manager.record_exposure("agent_000", exposure)

            # Update state after reasoning
            new_state = AgentState(
                agent_id="agent_000",
                aware=True,
                position="supportive",
                sentiment=0.7,
                will_share=True,
                raw_reasoning="This seems beneficial.",
            )
            manager.update_agent_state("agent_000", new_state, timestep=0)

            # Verify update
            state = manager.get_agent_state("agent_000")
            assert state.position == "supportive"
            assert state.sentiment == 0.7
            assert state.will_share is True
            assert state.last_reasoning_timestep == 0

    def test_get_unaware_agents(self, temp_db, agents):
        """Test getting unaware agents."""
        with StateManager(temp_db, agents=agents) as manager:
            # Initially all unaware
            unaware = manager.get_unaware_agents()
            assert len(unaware) == 3

            # Expose one agent
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            manager.record_exposure("agent_000", exposure)

            unaware = manager.get_unaware_agents()
            assert len(unaware) == 2
            assert "agent_000" not in unaware

    def test_get_aware_agents(self, temp_db, agents):
        """Test getting aware agents."""
        with StateManager(temp_db, agents=agents) as manager:
            # Expose some agents
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            manager.record_exposure("agent_000", exposure)
            manager.record_exposure("agent_001", exposure)

            aware = manager.get_aware_agents()
            assert len(aware) == 2
            assert "agent_000" in aware
            assert "agent_001" in aware

    def test_get_sharers(self, temp_db, agents):
        """Test getting agents who will share."""
        with StateManager(temp_db, agents=agents) as manager:
            # Set up agents
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            manager.record_exposure("agent_000", exposure)
            manager.record_exposure("agent_001", exposure)

            # Only agent_000 will share
            state = AgentState(
                agent_id="agent_000", aware=True, will_share=True, position="supportive"
            )
            manager.update_agent_state("agent_000", state, 0)

            sharers = manager.get_sharers()
            assert len(sharers) == 1
            assert "agent_000" in sharers

    def test_get_exposure_rate(self, temp_db, agents):
        """Test calculating exposure rate."""
        with StateManager(temp_db, agents=agents) as manager:
            # Initially 0%
            assert manager.get_exposure_rate() == 0.0

            # Expose one of three
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            manager.record_exposure("agent_000", exposure)

            rate = manager.get_exposure_rate()
            assert rate == pytest.approx(1/3, abs=0.01)

    def test_get_agents_to_reason(self, temp_db, agents):
        """Test getting agents who should reason."""
        with StateManager(temp_db, agents=agents) as manager:
            # Expose agent_000
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            manager.record_exposure("agent_000", exposure)

            # Should reason (never reasoned)
            to_reason = manager.get_agents_to_reason(timestep=0, threshold=3)
            assert "agent_000" in to_reason

    def test_log_event(self, temp_db, agents):
        """Test logging a simulation event."""
        with StateManager(temp_db, agents=agents) as manager:
            event = SimulationEvent(
                timestep=0,
                event_type=SimulationEventType.SEED_EXPOSURE,
                agent_id="agent_000",
                details={"channel": "email"},
            )
            manager.log_event(event)

            timeline = manager.export_timeline()
            assert len(timeline) == 1
            assert timeline[0]["agent_id"] == "agent_000"

    def test_save_and_get_timestep_summary(self, temp_db, agents):
        """Test saving and retrieving timestep summaries."""
        with StateManager(temp_db, agents=agents) as manager:
            summary = TimestepSummary(
                timestep=0,
                new_exposures=2,
                agents_reasoned=2,
                exposure_rate=0.67,
                average_sentiment=0.5,
            )
            manager.save_timestep_summary(summary)

            summaries = manager.get_timestep_summaries()
            assert len(summaries) == 1
            assert summaries[0].timestep == 0
            assert summaries[0].new_exposures == 2

    def test_export_final_states(self, temp_db, agents):
        """Test exporting final states."""
        with StateManager(temp_db, agents=agents) as manager:
            # Set up some state
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            manager.record_exposure("agent_000", exposure)

            states = manager.export_final_states()
            assert len(states) == 3

            agent_000 = next(s for s in states if s["agent_id"] == "agent_000")
            assert agent_000["aware"] is True

    def test_get_population_count(self, temp_db, agents):
        """Test getting population count."""
        with StateManager(temp_db, agents=agents) as manager:
            count = manager.get_population_count()
            assert count == 3


class TestPersonaGeneration:
    """Tests for persona generation."""

    def test_format_age(self):
        """Test age formatting."""
        assert _format_age(45) == "45-year-old"
        assert _format_age(30.5) == "30-year-old"

    def test_format_gender(self):
        """Test gender formatting."""
        assert _format_gender("male") == "man"
        assert _format_gender("female") == "woman"
        assert _format_gender("other") == "person"
        assert _format_gender(None) == "person"
        assert _format_gender("M") == "man"

    def test_format_role(self):
        """Test role formatting."""
        agent = {
            "role_seniority": "chief_physician_Chefarzt",
            "surgical_specialty": "cardiology",
        }
        role = _format_role(agent)
        assert "chief_physician_Chefarzt" in role
        assert "cardiology" in role

    def test_format_experience(self):
        """Test experience formatting."""
        agent = {"years_experience": 15}
        result = _format_experience(agent)
        assert "15 years" in result

    def test_get_descriptor(self):
        """Test personality descriptor selection."""
        # Low openness
        desc = _get_descriptor(0.2, OPENNESS_DESCRIPTORS)
        assert "routine" in desc

        # High openness
        desc = _get_descriptor(0.8, OPENNESS_DESCRIPTORS)
        assert "curious" in desc or "creative" in desc

    def test_generate_persona_basic(self, sample_agents):
        """Test basic persona generation."""
        agent = sample_agents[0]
        persona = generate_persona(agent)

        assert len(persona) > 0
        assert "45-year-old" in persona
        assert "man" in persona

    def test_generate_persona_with_role(self, sample_agents):
        """Test persona with professional role."""
        agent = sample_agents[0]
        persona = generate_persona(agent)

        assert "chief_physician_Chefarzt" in persona or "cardiology" in persona

    def test_generate_persona_with_location(self, sample_agents):
        """Test persona with location."""
        agent = sample_agents[0]
        persona = generate_persona(agent)

        assert "Bayern" in persona

    def test_generate_persona_minimal_agent(self):
        """Test persona generation with minimal agent data."""
        agent = {"_id": "agent_test"}
        persona = generate_persona(agent)

        # Should generate something
        assert len(persona) > 0

    def test_generate_persona_for_reasoning(self, sample_agents, complex_population_spec):
        """Test enhanced persona for reasoning."""
        agent = sample_agents[0]
        persona = generate_persona_for_reasoning(agent, complex_population_spec)

        # Should include framing text
        assert "simulating" in persona.lower()
        assert len(persona) > len(generate_persona(agent))


class TestStateManagerEdgeCases:
    """Tests for edge cases in StateManager."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            yield db_path

    def test_empty_database(self, temp_db):
        """Test operations on empty database."""
        with StateManager(temp_db) as manager:
            assert manager.get_population_count() == 0
            assert manager.get_exposure_rate() == 0.0
            assert manager.get_unaware_agents() == []

    def test_get_nonexistent_agent(self, temp_db):
        """Test getting state for nonexistent agent."""
        with StateManager(temp_db) as manager:
            state = manager.get_agent_state("nonexistent")
            # Should return default state
            assert state.agent_id == "nonexistent"
            assert state.aware is False

    def test_batch_update_states(self, temp_db):
        """Test batch updating multiple agent states."""
        agents = [{"_id": f"agent_{i}"} for i in range(5)]

        with StateManager(temp_db, agents=agents) as manager:
            # Expose all agents
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )
            for agent in agents:
                manager.record_exposure(agent["_id"], exposure)

            # Batch update
            updates = [
                (f"agent_{i}", AgentState(
                    agent_id=f"agent_{i}",
                    aware=True,
                    position="supportive" if i % 2 == 0 else "resistant",
                    sentiment=0.5 if i % 2 == 0 else -0.5,
                ))
                for i in range(5)
            ]
            manager.batch_update_states(updates, timestep=0)

            # Verify
            dist = manager.get_position_distribution()
            assert dist.get("supportive", 0) == 3  # agents 0, 2, 4
            assert dist.get("resistant", 0) == 2  # agents 1, 3

    def test_position_distribution(self, temp_db):
        """Test getting position distribution."""
        agents = [{"_id": f"agent_{i}"} for i in range(10)]

        with StateManager(temp_db, agents=agents) as manager:
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )

            # Set up different positions
            positions = ["supportive", "supportive", "resistant", "considering", "supportive"]
            for i, pos in enumerate(positions):
                manager.record_exposure(f"agent_{i}", exposure)
                state = AgentState(agent_id=f"agent_{i}", aware=True, position=pos)
                manager.update_agent_state(f"agent_{i}", state, 0)

            dist = manager.get_position_distribution()
            assert dist["supportive"] == 3
            assert dist["resistant"] == 1
            assert dist["considering"] == 1

    def test_average_sentiment(self, temp_db):
        """Test calculating average sentiment."""
        agents = [{"_id": f"agent_{i}"} for i in range(4)]

        with StateManager(temp_db, agents=agents) as manager:
            exposure = ExposureRecord(
                timestep=0, channel="email", content="Test", credibility=0.9
            )

            sentiments = [0.8, 0.4, -0.2, 0.6]  # Average = 0.4
            for i, sent in enumerate(sentiments):
                manager.record_exposure(f"agent_{i}", exposure)
                state = AgentState(agent_id=f"agent_{i}", aware=True, sentiment=sent)
                manager.update_agent_state(f"agent_{i}", state, 0)

            avg = manager.get_average_sentiment()
            assert avg == pytest.approx(0.4, abs=0.01)
