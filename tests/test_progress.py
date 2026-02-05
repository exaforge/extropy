"""Tests for SimulationProgress thread-safe progress tracking."""

import pytest

from entropy.simulation.progress import SimulationProgress


class TestSimulationProgress:
    """Test SimulationProgress dataclass."""

    def test_initial_state(self):
        """Fresh instance has zeros and empty dicts."""
        p = SimulationProgress()
        assert p.timestep == 0
        assert p.max_timesteps == 0
        assert p.agents_total == 0
        assert p.agents_done == 0
        assert p.exposure_rate == 0.0
        assert p.position_counts == {}
        assert p.avg_sentiment is None
        assert p.avg_conviction is None

    def test_begin_timestep(self):
        """begin_timestep sets fields and resets agents_done."""
        p = SimulationProgress()
        p.agents_done = 10  # simulate leftover from previous timestep
        p.begin_timestep(
            timestep=3,
            max_timesteps=100,
            agents_total=50,
            exposure_rate=0.65,
        )
        assert p.timestep == 3
        assert p.max_timesteps == 100
        assert p.agents_total == 50
        assert p.agents_done == 0
        assert p.exposure_rate == 0.65

    def test_record_agent_done_increments(self):
        """Each call increments agents_done."""
        p = SimulationProgress()
        p.record_agent_done(position="adopt", sentiment=0.5, conviction=0.7)
        assert p.agents_done == 1
        p.record_agent_done(position="reject", sentiment=-0.3, conviction=0.3)
        assert p.agents_done == 2

    def test_record_agent_done_position_counts(self):
        """Position counts accumulate correctly."""
        p = SimulationProgress()
        p.record_agent_done(position="adopt", sentiment=0.5, conviction=0.7)
        p.record_agent_done(position="adopt", sentiment=0.3, conviction=0.5)
        p.record_agent_done(position="reject", sentiment=-0.5, conviction=0.3)
        assert p.position_counts == {"adopt": 2, "reject": 1}

    def test_record_agent_done_none_position(self):
        """None position increments agents_done but not position_counts."""
        p = SimulationProgress()
        p.record_agent_done(position=None, sentiment=0.5, conviction=0.7)
        assert p.agents_done == 1
        assert p.position_counts == {}

    def test_running_averages(self):
        """avg_sentiment and avg_conviction computed from running sums."""
        p = SimulationProgress()
        p.record_agent_done(position="adopt", sentiment=0.8, conviction=0.7)
        p.record_agent_done(position="reject", sentiment=0.2, conviction=0.3)

        assert p.avg_sentiment == pytest.approx(0.5)
        assert p.avg_conviction == pytest.approx(0.5)

    def test_running_averages_none_values(self):
        """None sentiment/conviction values are excluded from averages."""
        p = SimulationProgress()
        p.record_agent_done(position="adopt", sentiment=0.6, conviction=None)
        p.record_agent_done(position="reject", sentiment=None, conviction=0.4)

        assert p.avg_sentiment == pytest.approx(0.6)
        assert p.avg_conviction == pytest.approx(0.4)

    def test_snapshot_returns_copy(self):
        """Modifying snapshot dict doesn't affect original state."""
        p = SimulationProgress()
        p.record_agent_done(position="adopt", sentiment=0.5, conviction=0.7)

        snap = p.snapshot()
        snap["position_counts"]["adopt"] = 999
        snap["agents_done"] = 999

        assert p.agents_done == 1
        assert p.position_counts["adopt"] == 1

    def test_snapshot_fields(self):
        """Snapshot contains all expected fields."""
        p = SimulationProgress()
        p.begin_timestep(
            timestep=2, max_timesteps=50, agents_total=100, exposure_rate=0.4
        )
        p.record_agent_done(position="adopt", sentiment=0.5, conviction=0.7)

        snap = p.snapshot()
        assert snap["timestep"] == 2
        assert snap["max_timesteps"] == 50
        assert snap["agents_total"] == 100
        assert snap["agents_done"] == 1
        assert snap["exposure_rate"] == 0.4
        assert snap["position_counts"] == {"adopt": 1}
        assert snap["avg_sentiment"] == pytest.approx(0.5)
        assert snap["avg_conviction"] == pytest.approx(0.7)

    def test_position_counts_cumulative_across_timesteps(self):
        """Position counts accumulate across begin_timestep calls."""
        p = SimulationProgress()
        p.begin_timestep(
            timestep=0, max_timesteps=10, agents_total=5, exposure_rate=0.5
        )
        p.record_agent_done(position="adopt", sentiment=0.5, conviction=0.5)

        p.begin_timestep(
            timestep=1, max_timesteps=10, agents_total=3, exposure_rate=0.7
        )
        p.record_agent_done(position="adopt", sentiment=0.3, conviction=0.3)
        p.record_agent_done(position="reject", sentiment=-0.2, conviction=0.5)

        assert p.position_counts == {"adopt": 2, "reject": 1}
