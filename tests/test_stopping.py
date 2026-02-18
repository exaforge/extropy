"""Tests for stopping condition evaluation.

Tests condition parsing, comparison evaluation, convergence detection,
and the top-level stopping condition evaluator.
Functions under test in extropy/simulation/stopping.py.
"""

from extropy.core.models import TimestepSummary
from extropy.core.models.scenario import ScenarioSimConfig
from extropy.simulation.state import StateManager
from extropy.simulation.stopping import (
    evaluate_convergence,
    evaluate_no_state_changes,
    evaluate_stopping_conditions,
    parse_comparison,
)


def _make_summary(timestep=0, **kwargs):
    """Factory for TimestepSummary with sensible defaults."""
    defaults = {
        "timestep": timestep,
        "new_exposures": 0,
        "agents_reasoned": 0,
        "shares_occurred": 0,
        "state_changes": 0,
        "exposure_rate": 0.0,
        "position_distribution": {},
        "average_sentiment": None,
        "average_conviction": None,
        "sentiment_variance": None,
    }
    defaults.update(kwargs)
    return TimestepSummary(**defaults)


class TestParseComparison:
    """Test condition string parsing."""

    def test_greater_than(self):
        assert parse_comparison("exposure_rate > 0.95") == ("exposure_rate", ">", 0.95)

    def test_less_than(self):
        assert parse_comparison("average_sentiment < 0.0") == (
            "average_sentiment",
            "<",
            0.0,
        )

    def test_greater_equal(self):
        assert parse_comparison("exposure_rate >= 1.0") == (
            "exposure_rate",
            ">=",
            1.0,
        )

    def test_less_equal(self):
        assert parse_comparison("exposure_rate <= 0.5") == (
            "exposure_rate",
            "<=",
            0.5,
        )

    def test_equal(self):
        assert parse_comparison("state_changes == 0") == ("state_changes", "==", 0.0)

    def test_not_equal(self):
        assert parse_comparison("exposure_rate != 0.0") == (
            "exposure_rate",
            "!=",
            0.0,
        )

    def test_invalid_format_returns_none(self):
        assert parse_comparison("not a condition") is None

    def test_whitespace_handling(self):
        assert parse_comparison("  exposure_rate  >  0.95  ") == (
            "exposure_rate",
            ">",
            0.95,
        )

    def test_integer_value(self):
        assert parse_comparison("state_changes > 5") == ("state_changes", ">", 5.0)

    def test_trailing_tokens_rejected(self):
        assert parse_comparison("exposure_rate > 0.95 and convergence") is None


class TestEvaluateNoStateChanges:
    """Test no_state_changes_for condition evaluation."""

    def test_stable_exceeds_threshold(self):
        """11 stable timesteps satisfies > 10."""
        summaries = [_make_summary(i, state_changes=0) for i in range(11)]
        assert evaluate_no_state_changes("no_state_changes_for > 10", summaries) is True

    def test_exactly_at_threshold(self):
        """10 stable timesteps satisfies > 10 because we check the last N."""
        summaries = [_make_summary(i, state_changes=0) for i in range(10)]
        assert evaluate_no_state_changes("no_state_changes_for > 10", summaries) is True

    def test_not_enough_summaries(self):
        """5 summaries can't satisfy > 10."""
        summaries = [_make_summary(i, state_changes=0) for i in range(5)]
        assert (
            evaluate_no_state_changes("no_state_changes_for > 10", summaries) is False
        )

    def test_recent_change_breaks_stability(self):
        """A state change in the last N timesteps fails the check."""
        summaries = [_make_summary(i, state_changes=0) for i in range(15)]
        summaries[-1] = _make_summary(14, state_changes=1)
        assert (
            evaluate_no_state_changes("no_state_changes_for > 10", summaries) is False
        )

    def test_empty_summaries(self):
        assert evaluate_no_state_changes("no_state_changes_for > 5", []) is False

    def test_invalid_format(self):
        summaries = [_make_summary(i) for i in range(10)]
        assert evaluate_no_state_changes("bad_format", summaries) is False


class TestEvaluateConvergence:
    """Test position distribution convergence detection."""

    def test_stable_distribution_converged(self):
        """Identical distributions over window → converged."""
        dist = {"adopt": 70, "reject": 30}
        summaries = [_make_summary(i, position_distribution=dist) for i in range(5)]
        assert evaluate_convergence(summaries, window=5, tolerance=0.01) is True

    def test_shifting_distribution_not_converged(self):
        """Varying distributions → not converged."""
        summaries = [
            _make_summary(0, position_distribution={"adopt": 40, "reject": 60}),
            _make_summary(1, position_distribution={"adopt": 50, "reject": 50}),
            _make_summary(2, position_distribution={"adopt": 60, "reject": 40}),
            _make_summary(3, position_distribution={"adopt": 70, "reject": 30}),
            _make_summary(4, position_distribution={"adopt": 80, "reject": 20}),
        ]
        assert evaluate_convergence(summaries, window=5, tolerance=0.01) is False

    def test_insufficient_window(self):
        """Fewer summaries than window → not converged."""
        dist = {"adopt": 70, "reject": 30}
        summaries = [_make_summary(i, position_distribution=dist) for i in range(3)]
        assert evaluate_convergence(summaries, window=5, tolerance=0.01) is False

    def test_high_tolerance_accepts_drift(self):
        """Moderate drift within high tolerance → converged."""
        summaries = [
            _make_summary(0, position_distribution={"adopt": 68, "reject": 32}),
            _make_summary(1, position_distribution={"adopt": 70, "reject": 30}),
            _make_summary(2, position_distribution={"adopt": 72, "reject": 28}),
            _make_summary(3, position_distribution={"adopt": 70, "reject": 30}),
            _make_summary(4, position_distribution={"adopt": 71, "reject": 29}),
        ]
        assert evaluate_convergence(summaries, window=5, tolerance=0.1) is True

    def test_empty_distributions(self):
        """All empty position distributions → not converged (no positions to check)."""
        summaries = [_make_summary(i, position_distribution={}) for i in range(5)]
        assert evaluate_convergence(summaries, window=5) is False


class TestEvaluateStoppingConditions:
    """Test the top-level stopping condition evaluator."""

    def _make_state_manager(self, tmp_path, agents, aware_ids=None):
        """Create a StateManager with some agents optionally made aware."""
        from extropy.core.models import ExposureRecord

        sm = StateManager(tmp_path / "test.db", agents=agents)
        for aid in aware_ids or []:
            sm.record_exposure(
                aid,
                ExposureRecord(
                    timestep=0,
                    channel="broadcast",
                    content="test",
                    credibility=0.9,
                ),
            )
        return sm

    def test_max_timesteps_reached(self, tmp_path):
        """Stops when timestep >= max_timesteps - 1."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100)

        should_stop, reason = evaluate_stopping_conditions(99, config, sm, [])
        assert should_stop is True
        assert reason == "max_timesteps_reached"

    def test_before_max_timesteps(self, tmp_path):
        """Doesn't stop before max_timesteps with no conditions."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100)

        should_stop, reason = evaluate_stopping_conditions(50, config, sm, [])
        assert should_stop is False
        assert reason is None

    def test_exposure_rate_condition_met(self, tmp_path):
        """Stops when exposure rate exceeds threshold."""
        agents = [{"_id": f"a{i}"} for i in range(10)]
        # Make 10/10 agents aware → 100% exposure rate
        sm = self._make_state_manager(
            tmp_path, agents, aware_ids=[f"a{i}" for i in range(10)]
        )
        config = ScenarioSimConfig(
            max_timesteps=100,
            stop_conditions=["exposure_rate > 0.95"],
        )

        should_stop, reason = evaluate_stopping_conditions(10, config, sm, [])
        assert should_stop is True
        assert "exposure_rate" in reason

    def test_boolean_and_condition(self, tmp_path):
        """Boolean AND condition should evaluate both clauses."""
        agents = [{"_id": f"a{i}"} for i in range(10)]
        sm = self._make_state_manager(
            tmp_path, agents, aware_ids=[f"a{i}" for i in range(10)]
        )
        config = ScenarioSimConfig(
            max_timesteps=100,
            stop_conditions=["exposure_rate > 0.95 and no_state_changes_for > 3"],
        )
        summaries = [_make_summary(i, state_changes=0) for i in range(5)]

        should_stop, reason = evaluate_stopping_conditions(10, config, sm, summaries)
        assert should_stop is True
        assert reason == "exposure_rate > 0.95 and no_state_changes_for > 3"

    def test_boolean_or_parentheses_condition(self, tmp_path):
        """Parser should support OR + parentheses in stop conditions."""
        agents = [{"_id": f"a{i}"} for i in range(10)]
        sm = self._make_state_manager(
            tmp_path, agents, aware_ids=[f"a{i}" for i in range(5)]
        )
        config = ScenarioSimConfig(
            max_timesteps=100,
            stop_conditions=[
                "(exposure_rate > 0.95 and no_state_changes_for > 10) or convergence"
            ],
        )
        dist = {"upskill": 60, "wait": 40}
        summaries = [_make_summary(i, position_distribution=dist) for i in range(5)]

        should_stop, reason = evaluate_stopping_conditions(10, config, sm, summaries)
        assert should_stop is True
        assert (
            reason
            == "(exposure_rate > 0.95 and no_state_changes_for > 10) or convergence"
        )

    def test_exposure_rate_condition_not_met(self, tmp_path):
        """Doesn't stop when exposure rate is below threshold."""
        agents = [{"_id": f"a{i}"} for i in range(10)]
        # Make 3/10 agents aware → 30% exposure rate
        sm = self._make_state_manager(tmp_path, agents, aware_ids=["a0", "a1", "a2"])
        config = ScenarioSimConfig(
            max_timesteps=100,
            stop_conditions=["exposure_rate > 0.95"],
        )

        should_stop, reason = evaluate_stopping_conditions(10, config, sm, [])
        assert should_stop is False
        assert reason is None

    def test_no_stop_conditions(self, tmp_path):
        """No custom conditions and not at max → doesn't stop."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100, stop_conditions=None)

        should_stop, reason = evaluate_stopping_conditions(50, config, sm, [])
        assert should_stop is False

    def test_convergence_condition(self, tmp_path):
        """Convergence condition triggers when distribution is stable."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(
            max_timesteps=100,
            stop_conditions=["convergence"],
        )
        dist = {"adopt": 70, "reject": 30}
        summaries = [_make_summary(i, position_distribution=dist) for i in range(10)]

        should_stop, reason = evaluate_stopping_conditions(10, config, sm, summaries)
        assert should_stop is True

    def test_no_state_changes_condition(self, tmp_path):
        """No state changes condition triggers after enough stable timesteps."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(
            max_timesteps=100,
            stop_conditions=["no_state_changes_for > 5"],
        )
        summaries = [_make_summary(i, state_changes=0) for i in range(10)]

        should_stop, reason = evaluate_stopping_conditions(10, config, sm, summaries)
        assert should_stop is True

    def test_quiescence_auto_stop(self, tmp_path):
        """Simulation stops when no agents reason for 3 consecutive timesteps."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100)

        summaries = [
            _make_summary(0, agents_reasoned=5),
            _make_summary(1, agents_reasoned=0),
            _make_summary(2, agents_reasoned=0),
            _make_summary(3, agents_reasoned=0),
        ]

        should_stop, reason = evaluate_stopping_conditions(3, config, sm, summaries)
        assert should_stop is True
        assert reason == "simulation_quiescent"

    def test_quiescence_not_triggered_with_reasoning(self, tmp_path):
        """Quiescence should NOT trigger if agents reasoned in one of the last 3 timesteps."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100)

        summaries = [
            _make_summary(0, agents_reasoned=5),
            _make_summary(1, agents_reasoned=0),
            _make_summary(2, agents_reasoned=1),  # one agent reasoned
            _make_summary(3, agents_reasoned=0),
        ]

        should_stop, reason = evaluate_stopping_conditions(3, config, sm, summaries)
        assert should_stop is False

    def test_quiescence_needs_3_timesteps(self, tmp_path):
        """Quiescence should NOT trigger with fewer than 3 summaries."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100)

        summaries = [
            _make_summary(0, agents_reasoned=0),
            _make_summary(1, agents_reasoned=0),
        ]

        should_stop, reason = evaluate_stopping_conditions(1, config, sm, summaries)
        assert should_stop is False

    def test_auto_stops_suppressed_with_future_timeline(self, tmp_path):
        """Convergence/quiescence should be suppressed when future timeline exists in auto mode."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100, allow_early_convergence=None)
        dist = {"adopt": 70, "reject": 30}
        summaries = [_make_summary(i, position_distribution=dist) for i in range(4)]

        should_stop, reason = evaluate_stopping_conditions(
            3,
            config,
            sm,
            summaries,
            has_future_timeline_events=True,
        )
        assert should_stop is False
        assert reason is None

    def test_auto_stops_enabled_with_override_true(self, tmp_path):
        """allow_early_convergence=true should force convergence/quiescence auto-stops."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100, allow_early_convergence=True)
        dist = {"adopt": 70, "reject": 30}
        summaries = [_make_summary(i, position_distribution=dist) for i in range(4)]

        should_stop, reason = evaluate_stopping_conditions(
            3,
            config,
            sm,
            summaries,
            has_future_timeline_events=True,
        )
        assert should_stop is True
        assert reason == "converged"

    def test_auto_stops_disabled_with_override_false(self, tmp_path):
        """allow_early_convergence=false should suppress convergence/quiescence."""
        agents = [{"_id": "a0"}]
        sm = self._make_state_manager(tmp_path, agents)
        config = ScenarioSimConfig(max_timesteps=100, allow_early_convergence=False)
        dist = {"adopt": 70, "reject": 30}
        summaries = [_make_summary(i, position_distribution=dist) for i in range(4)]

        should_stop, reason = evaluate_stopping_conditions(
            3,
            config,
            sm,
            summaries,
            has_future_timeline_events=False,
        )
        assert should_stop is False
        assert reason is None
