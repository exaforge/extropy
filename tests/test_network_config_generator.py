"""Tests for network config generation conversions."""

from extropy.core.models.population import (
    AttributeSpec,
    BooleanDistribution,
    GroundingInfo,
    NormalDistribution,
    SamplingConfig,
)
from extropy.population.network.config_generator import _convert_to_network_config


def test_convert_network_config_coerces_degree_multiplier_condition_types(
    minimal_population_spec,
):
    pop_spec = minimal_population_spec.model_copy(deep=True)
    pop_spec.attributes.extend(
        [
            AttributeSpec(
                name="is_student",
                type="boolean",
                category="population_specific",
                description="Whether agent is a student",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=BooleanDistribution(probability_true=0.3),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
            AttributeSpec(
                name="income",
                type="float",
                category="population_specific",
                description="Income",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(mean=50000.0, std=10000.0),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
        ]
    )
    pop_spec.sampling_order.extend(["is_student", "income"])

    data = {
        "reasoning": "test",
        "avg_degree": 12.0,
        "attribute_weights": [],
        "degree_multipliers": [
            {
                "attribute": "is_student",
                "condition_value": "true",
                "multiplier": 1.4,
                "rationale": "student hubs",
            },
            {
                "attribute": "age",
                "condition_value": "42",
                "multiplier": 1.2,
                "rationale": "age hubs",
            },
            {
                "attribute": "income",
                "condition_value": "55000.5",
                "multiplier": 1.1,
                "rationale": "income hubs",
            },
            {
                "attribute": "gender",
                "condition_value": "male",
                "multiplier": 1.05,
                "rationale": "categorical control",
            },
        ],
        "edge_type_rules": [],
        "influence_factors": [],
        "default_edge_type": "acquaintance",
    }

    config = _convert_to_network_config(data, pop_spec)
    by_attr = {dm.attribute: dm.condition for dm in config.degree_multipliers}

    assert by_attr["is_student"] is True
    assert by_attr["age"] == 42
    assert isinstance(by_attr["age"], int)
    assert by_attr["income"] == 55000.5
    assert isinstance(by_attr["income"], float)
    assert by_attr["gender"] == "male"
