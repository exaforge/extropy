"""Tests for stopping conditions."""

import tempfile
from pathlib import Path

import pytest

from entropy.simulation.stopping import (
    evaluate_stopping_conditions,
    parse_comparison,
    evaluate_convergence,
)
from entropy.simulation.models import TimestepSummary
from entropy.simulation.state import StateManager
from entropy.models.scenario import SimulationConfig


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def sample_agents():
    """Sample agents for testing."""
    return [
        {"_id": "agent_001", "age": 35},
        {"_id": "agent_002", "age": 42},
        {"_id": "agent_003", "age": 28},
    ]


class TestParseComparison:
    """Tests for parse_comparison function."""

    def test_greater_than(self):
        """Test parsing > operator."""
        result = parse_comparison("exposure_rate > 0.95")
        assert result == ("exposure_rate", ">", 0.95)

    def test_greater_equal(self):
        """Test parsing >= operator."""
        result = parse_comparison("exposure_rate >= 0.9")
        assert result == ("exposure_rate", ">=", 0.9)

    def test_less_than(self):
        """Test parsing < operator."""
        result = parse_comparison("sentiment < -0.5")
        assert result is None  # Negative numbers not supported in simple parse

    def test_equal(self):
        """Test parsing == operator."""
        result = parse_comparison("count == 10")
        assert result == ("count", "==", 10.0)

    def test_invalid_format(self):
        """Test invalid format returns None."""
        result = parse_comparison("not a valid condition")
        assert result is None


class TestStoppingConditions:
    """Tests for stopping condition evaluation."""

    def test_max_timesteps_reached(self, temp_db, sample_agents):
        """Test stopping when max timesteps reached."""
        config = SimulationConfig(max_timesteps=10)
        manager = StateManager(temp_db, sample_agents)
        history: list[TimestepSummary] = []

        try:
            # At timestep 9 (0-indexed), we should stop for max_timesteps=10
            should_stop, reason = evaluate_stopping_conditions(
                timestep=9,
                config=config,
                state_manager=manager,
                recent_summaries=history,
            )

            assert should_stop is True
            assert reason == "max_timesteps_reached"
        finally:
            manager.close()

    def test_not_max_timesteps(self, temp_db, sample_agents):
        """Test no stop when under max timesteps."""
        config = SimulationConfig(max_timesteps=10)
        manager = StateManager(temp_db, sample_agents)
        history: list[TimestepSummary] = []

        try:
            should_stop, reason = evaluate_stopping_conditions(
                timestep=5,
                config=config,
                state_manager=manager,
                recent_summaries=history,
            )

            assert should_stop is False
            assert reason is None
        finally:
            manager.close()

    def test_high_max_timesteps(self, temp_db, sample_agents):
        """Test with high max timesteps."""
        config = SimulationConfig(max_timesteps=100)
        manager = StateManager(temp_db, sample_agents)

        try:
            # At timestep 99, should stop for max_timesteps=100
            should_stop, reason = evaluate_stopping_conditions(
                timestep=99,
                config=config,
                state_manager=manager,
                recent_summaries=[],
            )

            assert should_stop is True
            assert reason == "max_timesteps_reached"
        finally:
            manager.close()


class TestConvergence:
    """Tests for convergence detection."""

    def test_convergence_stable_distribution(self):
        """Test convergence with stable distribution."""
        # Create history with stable positions
        history = [
            TimestepSummary(
                timestep=i,
                aware_count=100,
                new_exposures=0,
                agents_reasoned=0,
                position_distribution={"positive": 50, "neutral": 30, "negative": 20},
            )
            for i in range(6)
        ]

        result = evaluate_convergence(history, window=5, tolerance=0.01)
        assert result is True

    def test_no_convergence_changing_distribution(self):
        """Test no convergence when distribution is changing."""
        # Create history with changing positions
        history = [
            TimestepSummary(
                timestep=i,
                aware_count=100,
                new_exposures=0,
                agents_reasoned=0,
                position_distribution={"positive": 50 + i * 10, "neutral": 50 - i * 10},
            )
            for i in range(6)
        ]

        result = evaluate_convergence(history, window=5, tolerance=0.01)
        assert result is False

    def test_no_convergence_insufficient_history(self):
        """Test no convergence when history is too short."""
        history = [
            TimestepSummary(
                timestep=0,
                aware_count=100,
                new_exposures=0,
                agents_reasoned=0,
                position_distribution={"positive": 50},
            )
        ]

        result = evaluate_convergence(history, window=5, tolerance=0.01)
        assert result is False
