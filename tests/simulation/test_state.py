"""Tests for the SQLite state manager."""

import tempfile
from pathlib import Path

import pytest

from entropy.simulation.state import StateManager
from entropy.simulation.models import ExposureRecord, SimulationEvent, SimulationEventType


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def sample_agents():
    """Sample agents for testing."""
    return [
        {"_id": "agent_001", "age": 35, "gender": "male"},
        {"_id": "agent_002", "age": 42, "gender": "female"},
        {"_id": "agent_003", "age": 28, "gender": "male"},
        {"_id": "agent_004", "age": 55, "gender": "female"},
        {"_id": "agent_005", "age": 31, "gender": "male"},
    ]


class TestStateManager:
    """Tests for StateManager class."""

    def test_initialize_agents(self, temp_db, sample_agents):
        """Test agent initialization."""
        manager = StateManager(temp_db, sample_agents)

        # All agents should be initialized
        all_ids = manager.get_all_agent_ids()
        assert len(all_ids) == 5
        assert "agent_001" in all_ids
        assert "agent_005" in all_ids

        manager.close()

    def test_get_agent_state_initial(self, temp_db, sample_agents):
        """Test getting initial agent state."""
        manager = StateManager(temp_db, sample_agents)

        state = manager.get_agent_state("agent_001")

        assert state.agent_id == "agent_001"
        assert state.aware is False
        assert state.exposure_count == 0
        assert state.position is None
        assert state.will_share is False

        manager.close()

    def test_record_exposure(self, temp_db, sample_agents):
        """Test recording an exposure."""
        manager = StateManager(temp_db, sample_agents)

        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            source_agent_id=None,
            content="Test event",
            credibility=0.9,
        )

        manager.record_exposure("agent_001", exposure)

        state = manager.get_agent_state("agent_001")
        assert state.aware is True
        assert state.exposure_count == 1
        assert len(state.exposures) == 1
        assert state.exposures[0].channel == "email"

        manager.close()

    def test_multiple_exposures(self, temp_db, sample_agents):
        """Test multiple exposures for same agent."""
        manager = StateManager(temp_db, sample_agents)

        for i in range(3):
            exposure = ExposureRecord(
                timestep=i,
                channel="network" if i > 0 else "email",
                source_agent_id=f"agent_00{i}" if i > 0 else None,
                content="Test event",
                credibility=0.8,
            )
            manager.record_exposure("agent_001", exposure)

        state = manager.get_agent_state("agent_001")
        assert state.exposure_count == 3
        assert len(state.exposures) == 3

        manager.close()

    def test_get_unaware_agents(self, temp_db, sample_agents):
        """Test getting unaware agents."""
        manager = StateManager(temp_db, sample_agents)

        # Initially all unaware
        unaware = manager.get_unaware_agents()
        assert len(unaware) == 5

        # Expose one agent
        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            source_agent_id=None,
            content="Test",
            credibility=0.9,
        )
        manager.record_exposure("agent_001", exposure)

        unaware = manager.get_unaware_agents()
        assert len(unaware) == 4
        assert "agent_001" not in unaware

        manager.close()

    def test_get_aware_agents(self, temp_db, sample_agents):
        """Test getting aware agents."""
        manager = StateManager(temp_db, sample_agents)

        # Initially none aware
        aware = manager.get_aware_agents()
        assert len(aware) == 0

        # Expose two agents
        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            source_agent_id=None,
            content="Test",
            credibility=0.9,
        )
        manager.record_exposure("agent_001", exposure)
        manager.record_exposure("agent_002", exposure)

        aware = manager.get_aware_agents()
        assert len(aware) == 2
        assert "agent_001" in aware
        assert "agent_002" in aware

        manager.close()

    def test_update_agent_state(self, temp_db, sample_agents):
        """Test updating agent state after reasoning."""
        from entropy.simulation.models import AgentState

        manager = StateManager(temp_db, sample_agents)

        # First expose the agent
        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            source_agent_id=None,
            content="Test",
            credibility=0.9,
        )
        manager.record_exposure("agent_001", exposure)

        # Update with reasoning results
        new_state = AgentState(
            agent_id="agent_001",
            aware=True,
            exposure_count=1,
            position="oppose",
            sentiment=-0.5,
            will_share=True,
            outcomes={"cancel_intent": "considering"},
            raw_reasoning="I don't like this change.",
        )

        manager.update_agent_state("agent_001", new_state, timestep=1)

        state = manager.get_agent_state("agent_001")
        assert state.position == "oppose"
        assert state.sentiment == -0.5
        assert state.will_share is True
        assert state.outcomes == {"cancel_intent": "considering"}

        manager.close()

    def test_get_sharers(self, temp_db, sample_agents):
        """Test getting agents who will share."""
        from entropy.simulation.models import AgentState

        manager = StateManager(temp_db, sample_agents)

        # Expose and update two agents
        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            source_agent_id=None,
            content="Test",
            credibility=0.9,
        )

        for agent_id in ["agent_001", "agent_002"]:
            manager.record_exposure(agent_id, exposure)

        # Only agent_001 will share
        state1 = AgentState(
            agent_id="agent_001",
            aware=True,
            will_share=True,
        )
        state2 = AgentState(
            agent_id="agent_002",
            aware=True,
            will_share=False,
        )

        manager.update_agent_state("agent_001", state1, 1)
        manager.update_agent_state("agent_002", state2, 1)

        sharers = manager.get_sharers()
        assert len(sharers) == 1
        assert "agent_001" in sharers

        manager.close()

    def test_exposure_rate(self, temp_db, sample_agents):
        """Test exposure rate calculation."""
        manager = StateManager(temp_db, sample_agents)

        # Initially 0%
        assert manager.get_exposure_rate() == 0.0

        # Expose 2 of 5 agents
        exposure = ExposureRecord(
            timestep=0,
            channel="email",
            source_agent_id=None,
            content="Test",
            credibility=0.9,
        )
        manager.record_exposure("agent_001", exposure)
        manager.record_exposure("agent_002", exposure)

        assert manager.get_exposure_rate() == 0.4  # 2/5

        manager.close()

    def test_log_event(self, temp_db, sample_agents):
        """Test logging timeline events."""
        manager = StateManager(temp_db, sample_agents)

        event = SimulationEvent(
            timestep=0,
            event_type=SimulationEventType.SEED_EXPOSURE,
            agent_id="agent_001",
            details={"channel": "email"},
        )

        manager.log_event(event)

        # Export timeline and verify
        timeline = manager.export_timeline()
        assert len(timeline) == 1
        assert timeline[0]["agent_id"] == "agent_001"
        assert timeline[0]["event_type"] == "seed_exposure"

        manager.close()

    def test_context_manager(self, temp_db, sample_agents):
        """Test using state manager as context manager."""
        with StateManager(temp_db, sample_agents) as manager:
            assert manager.get_population_count() == 5

        # Should be closed after context
        # (can't easily test this, but it shouldn't raise)
