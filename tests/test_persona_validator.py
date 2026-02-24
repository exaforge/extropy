"""Tests for persona config validator."""

from datetime import datetime

from extropy.population.persona import (
    PersonaConfig,
    AttributeTreatment,
    TreatmentType,
    AttributeGroup,
    AttributePhrasing,
    CategoricalPhrasing,
    ConcretePhrasing,
    validate_persona_config,
)


def _valid_persona_config() -> PersonaConfig:
    return PersonaConfig(
        population_description="Test population",
        created_at=datetime(2024, 1, 1),
        intro_template="I'm {age} years old.",
        treatments=[
            AttributeTreatment(
                attribute="age", treatment=TreatmentType.CONCRETE, group="identity"
            ),
            AttributeTreatment(
                attribute="gender",
                treatment=TreatmentType.CONCRETE,
                group="identity",
            ),
        ],
        groups=[
            AttributeGroup(
                name="identity", label="Who I Am", attributes=["age", "gender"]
            )
        ],
        phrasings=AttributePhrasing(
            categorical=[
                CategoricalPhrasing(
                    attribute="gender",
                    phrases={
                        "male": "I'm male",
                        "female": "I'm female",
                        "other": "I identify outside the binary",
                    },
                    null_options=[],
                    null_phrase=None,
                )
            ],
            concrete=[
                ConcretePhrasing(
                    attribute="age",
                    template="I'm {value} years old",
                    format_spec=".0f",
                    prefix="",
                    suffix="",
                )
            ],
        ),
    )


def test_validate_persona_config_valid(minimal_population_spec):
    config = _valid_persona_config()
    result = validate_persona_config(minimal_population_spec, config)
    assert result.valid, result.errors


def test_validate_persona_config_detects_missing_group_membership(
    minimal_population_spec,
):
    config = _valid_persona_config()
    config.groups[0].attributes = ["age"]
    result = validate_persona_config(minimal_population_spec, config)
    assert not result.valid
    assert any(issue.category == "PERSONA_GROUP" for issue in result.errors)


def test_validate_persona_config_detects_unknown_intro_reference(
    minimal_population_spec,
):
    config = _valid_persona_config()
    config.intro_template = "I live in {unknown_attr}."
    result = validate_persona_config(minimal_population_spec, config)
    assert not result.valid
    assert any(issue.category == "PERSONA_INTRO" for issue in result.errors)
