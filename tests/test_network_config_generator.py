"""Tests for network config generation conversions."""

from extropy.core.models.population import (
    AttributeSpec,
    BooleanDistribution,
    CategoricalDistribution,
    GroundingInfo,
    NormalDistribution,
    SamplingConfig,
)
from extropy.population.network.config_generator import (
    _build_structural_role_prompt,
    _convert_to_network_config,
    _resolve_structural_roles_deterministic,
)


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


def test_resolve_structural_roles_deterministic_avoids_boolean_dependents(
    minimal_population_spec,
):
    pop_spec = minimal_population_spec.model_copy(deep=True)
    pop_spec.attributes.extend(
        [
            AttributeSpec(
                name="household_id",
                type="categorical",
                category="population_specific",
                description="Household identifier",
                scope="household",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        options=["h1", "h2"],
                        weights=[0.5, 0.5],
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
            AttributeSpec(
                name="partner_id",
                type="categorical",
                category="population_specific",
                description="Partner agent id",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        options=["a1", "a2"],
                        weights=[0.5, 0.5],
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
            AttributeSpec(
                name="religious_affiliation",
                type="categorical",
                category="population_specific",
                description="Faith affiliation",
                identity_type="religious_affiliation",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        options=["none", "faith_a"],
                        weights=[0.6, 0.4],
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
            AttributeSpec(
                name="has_children",
                type="boolean",
                category="population_specific",
                description="Whether agent has children",
                identity_type="parental_status",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=BooleanDistribution(probability_true=0.4),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
            AttributeSpec(
                name="dependents_count",
                type="int",
                category="population_specific",
                description="Number of dependents in household",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        mean=1.2, std=0.8, min=0.0, max=5.0
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
        ]
    )
    pop_spec.sampling_order.extend(
        [
            "household_id",
            "partner_id",
            "religious_affiliation",
            "has_children",
            "dependents_count",
        ]
    )

    agents_sample = [
        {
            "household_id": "h1",
            "partner_id": "agent_1",
            "religious_affiliation": "faith_a",
            "has_children": True,
            "dependents_count": 2,
        }
    ]
    resolved, ambiguous = _resolve_structural_roles_deterministic(
        pop_spec, agents_sample=agents_sample
    )

    assert resolved["household_id"] == "household_id"
    assert resolved["partner_id"] == "partner_id"
    assert resolved["religion"] == "religious_affiliation"
    assert resolved["dependents"] != "has_children"
    assert "dependents" in ambiguous or resolved["dependents"] == "dependents_count"


def test_structural_role_prompt_includes_full_attribute_set(minimal_population_spec):
    pop_spec = minimal_population_spec.model_copy(deep=True)
    for idx in range(60):
        name = f"extra_attr_{idx:02d}"
        pop_spec.attributes.append(
            AttributeSpec(
                name=name,
                type="categorical",
                category="context_specific",
                description=f"Extra attribute {idx}",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        options=["a", "b"],
                        weights=[0.5, 0.5],
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            )
        )
        pop_spec.sampling_order.append(name)

    prompt = _build_structural_role_prompt(pop_spec)
    assert "extra_attr_59" in prompt
