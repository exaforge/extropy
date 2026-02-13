"""Tests for persona categorical rendering behavior."""

from entropy.population.persona.config import CategoricalPhrasing
from entropy.population.persona.renderer import _format_categorical_value


def test_categorical_null_option_prefers_null_phrase():
    phrasing = CategoricalPhrasing(
        attribute="preferred_third_party_app",
        phrases={"none": "I access Reddit none"},
        null_options=["none"],
        null_phrase="I don't use a third-party app for Reddit",
        fallback=None,
    )

    rendered = _format_categorical_value("none", phrasing)
    assert rendered == "I don't use a third-party app for Reddit"


def test_categorical_null_option_keeps_natural_phrase_when_null_phrase_missing():
    phrasing = CategoricalPhrasing(
        attribute="preferred_third_party_app",
        phrases={"none": "I don't use any third-party app to access Reddit"},
        null_options=["none"],
        null_phrase=None,
        fallback=None,
    )

    rendered = _format_categorical_value("none", phrasing)
    assert rendered == "I don't use any third-party app to access Reddit"


def test_categorical_null_option_uses_fallback_when_phrase_is_raw_token():
    phrasing = CategoricalPhrasing(
        attribute="moderation_tool_dependency",
        phrases={"not_applicable": "My dependence is not_applicable"},
        null_options=["not_applicable"],
        null_phrase=None,
        fallback="This doesn't apply to me",
    )

    rendered = _format_categorical_value("not_applicable", phrasing)
    assert rendered == "This doesn't apply to me"


def test_categorical_non_null_options_render_normally():
    phrasing = CategoricalPhrasing(
        attribute="user_role_type",
        phrases={
            "moderator": "I moderate one or more communities",
            "reader": "I mostly read rather than post",
        },
        null_options=[],
        null_phrase=None,
        fallback=None,
    )

    rendered = _format_categorical_value("moderator", phrasing)
    assert rendered == "I moderate one or more communities"
