"""Tests for deterministic post-processing in scenario sufficiency."""

from extropy.scenario import sufficiency


def test_static_scenario_requires_explicit_timestep_unit(monkeypatch):
    """Static scenarios without explicit unit should request clarification."""

    def _fake_simple_call(*args, **kwargs):
        return {
            "sufficient": True,
            "inferred_duration": "shortly after announcement",
            "inferred_timestep_unit": None,
            "inferred_scenario_type": "static",
            "has_explicit_outcomes": False,
            "inferred_agent_focus_mode": "primary_only",
            "questions": [],
        }

    monkeypatch.setattr(sufficiency, "simple_call", _fake_simple_call)

    result = sufficiency.check_scenario_sufficiency(
        "City announces a congestion fee increase."
    )

    assert result.sufficient is False
    assert any(q.id == "timestep_unit" for q in result.questions)


def test_timeline_markers_force_evolving_and_unit(monkeypatch):
    """Timeline markers like 'month 0' should force evolving + month unit."""

    def _fake_simple_call(*args, **kwargs):
        return {
            "sufficient": True,
            "inferred_duration": None,
            "inferred_timestep_unit": None,
            "inferred_scenario_type": "static",
            "has_explicit_outcomes": False,
            "inferred_agent_focus_mode": "primary_only",
            "questions": [],
        }

    monkeypatch.setattr(sufficiency, "simple_call", _fake_simple_call)

    result = sufficiency.check_scenario_sufficiency(
        "Month 0: tax announced. Month 1: protests begin."
    )

    assert result.inferred_scenario_type == "evolving"
    assert result.inferred_timestep_unit == "month"


def test_infers_unit_from_duration_text_when_missing(monkeypatch):
    """If LLM omits unit but duration names one, deterministic inference fills it."""

    def _fake_simple_call(*args, **kwargs):
        return {
            "sufficient": True,
            "inferred_duration": "over 6 weeks",
            "inferred_timestep_unit": None,
            "inferred_scenario_type": "evolving",
            "has_explicit_outcomes": False,
            "inferred_agent_focus_mode": "primary_only",
            "questions": [],
        }

    monkeypatch.setattr(sufficiency, "simple_call", _fake_simple_call)

    result = sufficiency.check_scenario_sufficiency(
        "Policy adoption unfolds over 6 weeks."
    )

    assert result.sufficient is True
    assert result.inferred_timestep_unit == "week"
