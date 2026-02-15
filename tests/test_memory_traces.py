"""Tests for memory trace management and multi-touch reasoning triggers.

Tests the sliding window memory trace (max 3 entries per agent),
the get_agents_to_reason multi-touch logic, and aggregation queries.
Functions under test in extropy/simulation/state.py.
"""

import pytest

from extropy.core.models import (
    AgentState,
    ExposureRecord,
    MemoryEntry,
)
from extropy.simulation.state import StateManager


def _make_exposure(timestep=0, channel="broadcast", source_agent_id=None):
    """Factory for ExposureRecord."""
    return ExposureRecord(
        timestep=timestep,
        channel=channel,
        source_agent_id=source_agent_id,
        content="test content",
        credibility=0.9,
    )


def _make_memory(timestep=0, sentiment=0.5, conviction=0.5, summary="Test thought"):
    """Factory for MemoryEntry."""
    return MemoryEntry(
        timestep=timestep,
        sentiment=sentiment,
        conviction=conviction,
        summary=summary,
    )


@pytest.fixture
def five_agents():
    return [{"_id": f"a{i}"} for i in range(5)]


@pytest.fixture
def state_mgr(tmp_path, five_agents):
    """StateManager with 5 initialized agents."""
    return StateManager(tmp_path / "test.db", agents=five_agents)


# ============================================================================
# Memory Trace Sliding Window
# ============================================================================


class TestMemoryTraceWindow:
    """Test save_memory_entry and get_memory_traces sliding window."""

    def test_single_entry(self, state_mgr):
        state_mgr.save_memory_entry("a0", _make_memory(timestep=0))

        traces = state_mgr.get_memory_traces("a0")
        assert len(traces) == 1
        assert traces[0].timestep == 0

    def test_three_entries_retained(self, state_mgr):
        for t in range(3):
            state_mgr.save_memory_entry(
                "a0", _make_memory(timestep=t, summary=f"thought_{t}")
            )

        traces = state_mgr.get_memory_traces("a0")
        assert len(traces) == 3
        # Ordered oldest first
        assert traces[0].timestep == 0
        assert traces[2].timestep == 2

    def test_fourth_entry_retained(self, state_mgr):
        """After 4 inserts, all 4 entries are retained (no eviction)."""
        for t in range(4):
            state_mgr.save_memory_entry(
                "a0", _make_memory(timestep=t, summary=f"thought_{t}")
            )

        traces = state_mgr.get_memory_traces("a0")
        assert len(traces) == 4
        assert traces[0].timestep == 0
        assert traces[3].timestep == 3

    def test_fifth_entry_all_retained(self, state_mgr):
        """After 5 inserts, all 5 traces are retained."""
        for t in range(5):
            state_mgr.save_memory_entry(
                "a0", _make_memory(timestep=t, summary=f"thought_{t}")
            )

        traces = state_mgr.get_memory_traces("a0")
        assert len(traces) == 5
        assert traces[0].timestep == 0
        assert traces[4].timestep == 4

    def test_separate_agents_independent(self, state_mgr):
        """Each agent has its own independent memory trace."""
        for t in range(4):
            state_mgr.save_memory_entry("a0", _make_memory(timestep=t))
        state_mgr.save_memory_entry("a1", _make_memory(timestep=10))

        assert len(state_mgr.get_memory_traces("a0")) == 4
        assert len(state_mgr.get_memory_traces("a1")) == 1

    def test_no_traces_for_new_agent(self, state_mgr):
        traces = state_mgr.get_memory_traces("a0")
        assert len(traces) == 0

    def test_memory_preserves_content(self, state_mgr):
        state_mgr.save_memory_entry(
            "a0",
            _make_memory(
                timestep=5, sentiment=-0.3, conviction=0.7, summary="I'm skeptical"
            ),
        )

        traces = state_mgr.get_memory_traces("a0")
        assert traces[0].sentiment == pytest.approx(-0.3)
        assert traces[0].conviction == pytest.approx(0.7)
        assert traces[0].summary == "I'm skeptical"


# ============================================================================
# Multi-Touch Reasoning Triggers
# ============================================================================


class TestGetAgentsToReason:
    """Test get_agents_to_reason(timestep, threshold) multi-touch logic."""

    def test_never_reasoned_aware_agent(self, state_mgr):
        """Aware agent who never reasoned should be in the list."""
        state_mgr.record_exposure("a0", _make_exposure(timestep=0))

        agents = state_mgr.get_agents_to_reason(timestep=0, threshold=3)
        assert "a0" in agents

    def test_unaware_agent_excluded(self, state_mgr):
        """Agent without exposure should not be in the list."""
        agents = state_mgr.get_agents_to_reason(timestep=0, threshold=3)
        assert "a0" not in agents

    def test_multi_touch_below_threshold(self, state_mgr):
        """Agent who reasoned and got < threshold unique source exposures → excluded."""
        # Make aware and reason
        state_mgr.record_exposure("a0", _make_exposure(timestep=0))
        state_mgr.update_agent_state(
            "a0",
            AgentState(
                agent_id="a0",
                aware=True,
                position="adopt",
                sentiment=0.5,
                conviction=0.5,
            ),
            timestep=0,
        )

        # Add 2 network exposures from unique sources (threshold is 3)
        state_mgr.record_exposure(
            "a0", _make_exposure(timestep=1, channel="network", source_agent_id="s1")
        )
        state_mgr.record_exposure(
            "a0", _make_exposure(timestep=2, channel="network", source_agent_id="s2")
        )

        agents = state_mgr.get_agents_to_reason(timestep=2, threshold=3)
        assert "a0" not in agents

    def test_multi_touch_at_threshold(self, state_mgr):
        """Agent who reasoned and got exactly threshold unique source exposures → included."""
        state_mgr.record_exposure("a0", _make_exposure(timestep=0))
        state_mgr.update_agent_state(
            "a0",
            AgentState(
                agent_id="a0",
                aware=True,
                position="adopt",
                sentiment=0.5,
                conviction=0.5,
            ),
            timestep=0,
        )

        # Add 3 network exposures from 3 unique sources (threshold is 3)
        for i in range(3):
            state_mgr.record_exposure(
                "a0",
                _make_exposure(
                    timestep=i + 1, channel="network", source_agent_id=f"src_{i}"
                ),
            )

        agents = state_mgr.get_agents_to_reason(timestep=3, threshold=3)
        assert "a0" in agents

    def test_multi_touch_above_threshold(self, state_mgr):
        """Agent who reasoned and got > threshold unique source exposures → included."""
        state_mgr.record_exposure("a0", _make_exposure(timestep=0))
        state_mgr.update_agent_state(
            "a0",
            AgentState(
                agent_id="a0",
                aware=True,
                position="adopt",
                sentiment=0.5,
                conviction=0.5,
            ),
            timestep=0,
        )

        # Add 5 network exposures from unique sources
        for t in range(1, 6):
            state_mgr.record_exposure(
                "a0",
                _make_exposure(
                    timestep=t, channel="network", source_agent_id=f"src_{t}"
                ),
            )

        agents = state_mgr.get_agents_to_reason(timestep=5, threshold=3)
        assert "a0" in agents

    def test_recently_reasoned_no_new_exposures(self, state_mgr):
        """Agent who reasoned recently with no new exposures → excluded."""
        state_mgr.record_exposure("a0", _make_exposure(timestep=0))
        state_mgr.update_agent_state(
            "a0",
            AgentState(
                agent_id="a0",
                aware=True,
                position="adopt",
                sentiment=0.5,
                conviction=0.5,
            ),
            timestep=5,
        )

        agents = state_mgr.get_agents_to_reason(timestep=6, threshold=3)
        assert "a0" not in agents

    def test_multiple_agents_mixed(self, state_mgr):
        """Test with multiple agents in different states."""
        # a0: aware, never reasoned → should reason
        state_mgr.record_exposure("a0", _make_exposure(timestep=0))

        # a1: reasoned, 3 unique-source network exposures → should reason (multi-touch)
        state_mgr.record_exposure("a1", _make_exposure(timestep=0))
        state_mgr.update_agent_state(
            "a1",
            AgentState(
                agent_id="a1",
                aware=True,
                position="adopt",
                sentiment=0.5,
                conviction=0.5,
            ),
            timestep=0,
        )
        for t in range(1, 4):
            state_mgr.record_exposure(
                "a1",
                _make_exposure(
                    timestep=t, channel="network", source_agent_id=f"src_{t}"
                ),
            )

        # a2: not aware → should not reason
        # a3: reasoned, only 1 unique-source network exposure → should not reason
        state_mgr.record_exposure("a3", _make_exposure(timestep=0))
        state_mgr.update_agent_state(
            "a3",
            AgentState(
                agent_id="a3",
                aware=True,
                position="reject",
                sentiment=-0.3,
                conviction=0.3,
            ),
            timestep=0,
        )
        state_mgr.record_exposure(
            "a3",
            _make_exposure(timestep=1, channel="network", source_agent_id="src_x"),
        )

        agents = state_mgr.get_agents_to_reason(timestep=3, threshold=3)
        assert "a0" in agents
        assert "a1" in agents
        assert "a2" not in agents
        assert "a3" not in agents


# ============================================================================
# Aggregation Queries
# ============================================================================


class TestAggregationQueries:
    """Test analytics queries on StateManager."""

    def test_get_sentiment_variance(self, state_mgr):
        """Variance of [0.1, 0.3, 0.5, 0.7, 0.9] = 0.08."""
        sentiments = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, s in enumerate(sentiments):
            state_mgr.record_exposure(f"a{i}", _make_exposure(timestep=0))
            state_mgr.update_agent_state(
                f"a{i}",
                AgentState(agent_id=f"a{i}", sentiment=s, conviction=0.5),
                timestep=0,
            )

        variance = state_mgr.get_sentiment_variance()
        # mean = 0.5, variance = ((0.4^2 + 0.2^2 + 0 + 0.2^2 + 0.4^2) / 5) = 0.08
        assert variance == pytest.approx(0.08, abs=0.001)

    def test_get_sentiment_variance_none_when_no_data(self, state_mgr):
        """No agents with sentiment → None."""
        variance = state_mgr.get_sentiment_variance()
        assert variance is None

    def test_get_average_conviction(self, state_mgr):
        convictions = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, c in enumerate(convictions):
            state_mgr.record_exposure(f"a{i}", _make_exposure(timestep=0))
            state_mgr.update_agent_state(
                f"a{i}",
                AgentState(agent_id=f"a{i}", sentiment=0.0, conviction=c),
                timestep=0,
            )

        avg = state_mgr.get_average_conviction()
        assert avg == pytest.approx(0.5)

    def test_get_population_count(self, state_mgr):
        assert state_mgr.get_population_count() == 5

    def test_get_exposure_rate(self, state_mgr):
        """3 out of 5 agents aware → 0.6."""
        for i in range(3):
            state_mgr.record_exposure(f"a{i}", _make_exposure(timestep=0))

        rate = state_mgr.get_exposure_rate()
        assert rate == pytest.approx(0.6)

    def test_get_exposure_rate_zero(self, state_mgr):
        """No aware agents → 0.0."""
        assert state_mgr.get_exposure_rate() == 0.0
