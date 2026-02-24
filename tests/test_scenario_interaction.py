"""Tests for scenario interaction prompt grounding."""

from extropy.core.models import Event, EventType
from extropy.scenario import interaction as interaction_module


def test_interaction_prompt_uses_full_event_content_and_option_vocab(
    minimal_population_spec, monkeypatch
):
    captured: dict[str, str] = {}

    def fake_reasoning_call(*, prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "interaction_description": "Test dynamics",
            "share_probability": 0.35,
            "share_modifiers": [
                {"when": "gender == 'female'", "multiply": 1.1, "add": 0.0},
            ],
            "decay_per_hop": 0.1,
            "max_hops": 4,
            "reasoning": "test",
        }

    monkeypatch.setattr(interaction_module, "reasoning_call", fake_reasoning_call)

    long_content = "x" * 240 + "TAIL_MARKER_FOR_FULL_CONTENT"
    event = Event(
        type=EventType.ANNOUNCEMENT,
        content=long_content,
        source="Test Source",
        credibility=0.9,
        ambiguity=0.2,
        emotional_valence=0.1,
    )

    interaction_module.determine_interaction_model(
        event=event,
        population_spec=minimal_population_spec,
    )

    prompt = captured.get("prompt", "")
    assert "TAIL_MARKER_FOR_FULL_CONTENT" in prompt
    assert "gender (categorical) options=" in prompt
    assert "Strict literal contract for conditions" in prompt
