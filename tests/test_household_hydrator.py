from extropy.population.spec_builder.schemas import build_household_config_schema
from extropy.population.spec_builder.hydrators.household import _parse_household_config


def test_household_schema_constrains_assortative_attribute_enum():
    schema = build_household_config_schema(
        allowed_assortative_attributes=["education_level", "political_identity"]
    )

    attribute_schema = schema["properties"]["assortative_mating"]["items"][
        "properties"
    ]["attribute"]
    assert attribute_schema["type"] == "string"
    assert attribute_schema["enum"] == ["education_level", "political_identity"]


def test_parse_household_config_filters_unknown_assortative_attributes():
    data = {
        "assortative_mating": [
            {"attribute": "education_level", "correlation": 0.65},
            {"attribute": "technology_adoption_posture", "correlation": 0.45},
        ]
    }

    config = _parse_household_config(data, allowed_attributes=["education_level"])

    assert config.assortative_mating == {"education_level": 0.65}
