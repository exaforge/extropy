"""Tests for persona categorical rendering behavior."""

from extropy.population.persona.config import (
    AttributeGroup,
    AttributePhrasing,
    AttributeTreatment,
    CategoricalPhrasing,
    ConcretePhrasing,
    PersonaConfig,
)
from extropy.population.persona.renderer import (
    _format_categorical_value,
    render_persona,
)


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


def _make_contextual_test_config() -> PersonaConfig:
    return PersonaConfig(
        population_description="Test population",
        intro_template="I'm {age} years old.",
        treatments=[
            AttributeTreatment(attribute="age", treatment="concrete", group="identity"),
            AttributeTreatment(
                attribute="employment_status", treatment="concrete", group="work"
            ),
            AttributeTreatment(
                attribute="occupation", treatment="concrete", group="identity"
            ),
        ],
        groups=[
            AttributeGroup(
                name="identity",
                label="Who I Am",
                attributes=["age", "occupation"],
            ),
            AttributeGroup(
                name="work",
                label="Work",
                attributes=["employment_status"],
            ),
        ],
        phrasings=AttributePhrasing(
            boolean=[],
            categorical=[
                CategoricalPhrasing(
                    attribute="employment_status",
                    phrases={
                        "Unemployed": "I'm currently unemployed and looking for work."
                    },
                    null_options=[],
                    null_phrase=None,
                    fallback=None,
                ),
                CategoricalPhrasing(
                    attribute="occupation",
                    phrases={"Tech": "I work in software engineering."},
                    null_options=[],
                    null_phrase=None,
                    fallback=None,
                ),
            ],
            relative=[],
            concrete=[
                ConcretePhrasing(
                    attribute="age",
                    template="I'm {value} years old",
                )
            ],
        ),
    )


def test_render_persona_adjusts_occupation_when_unemployed():
    config = _make_contextual_test_config()
    agent = {
        "age": 32,
        "employment_status": "Unemployed",
        "occupation": "Tech",
    }

    rendered = render_persona(
        agent,
        config,
        semantic_type_map={
            "employment_status": "employment",
            "occupation": "occupation",
        },
    )

    assert "My background is in software engineering." in rendered
    assert "I work in software engineering." not in rendered


def test_render_persona_uses_more_about_me_instead_of_duplicate_header():
    config = _make_contextual_test_config()
    agent = {
        "age": 32,
        "employment_status": "Unemployed",
        "occupation": "Tech",
    }

    rendered = render_persona(agent, config)
    assert rendered.count("## Who I Am") == 1
    assert "## More About Me" in rendered
