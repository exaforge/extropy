"""Shared fixtures and configuration for Entropy tests."""

import random
from datetime import datetime

import pytest

from entropy.core.models.population import (
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    AttributeSpec,
    SamplingConfig,
    GroundingInfo,
    NormalDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    UniformDistribution,
    BetaDistribution,
    Modifier,
)


@pytest.fixture
def rng() -> random.Random:
    """Provide a seeded random number generator for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def simple_normal_distribution() -> NormalDistribution:
    """A simple normal distribution for testing."""
    return NormalDistribution(
        type="normal",
        mean=50.0,
        std=10.0,
        min=20.0,
        max=80.0,
    )


@pytest.fixture
def simple_categorical_distribution() -> CategoricalDistribution:
    """A simple categorical distribution for testing."""
    return CategoricalDistribution(
        type="categorical",
        options=["A", "B", "C"],
        weights=[0.5, 0.3, 0.2],
    )


@pytest.fixture
def simple_boolean_distribution() -> BooleanDistribution:
    """A simple boolean distribution for testing."""
    return BooleanDistribution(
        type="boolean",
        probability_true=0.7,
    )


@pytest.fixture
def simple_uniform_distribution() -> UniformDistribution:
    """A simple uniform distribution for testing."""
    return UniformDistribution(
        type="uniform",
        min=0.0,
        max=100.0,
    )


@pytest.fixture
def simple_beta_distribution() -> BetaDistribution:
    """A simple beta distribution for testing."""
    return BetaDistribution(
        type="beta",
        alpha=2.0,
        beta=5.0,
    )


@pytest.fixture
def simple_attribute_spec() -> AttributeSpec:
    """A simple independent attribute spec for testing."""
    return AttributeSpec(
        name="age",
        type="int",
        category="universal",
        description="Age of the agent",
        sampling=SamplingConfig(
            strategy="independent",
            distribution=NormalDistribution(
                type="normal",
                mean=45.0,
                std=10.0,
                min=25.0,
                max=70.0,
            ),
        ),
        grounding=GroundingInfo(
            level="medium",
            method="estimated",
        ),
    )


@pytest.fixture
def derived_attribute_spec() -> AttributeSpec:
    """A derived attribute spec for testing."""
    return AttributeSpec(
        name="years_experience",
        type="int",
        category="population_specific",
        description="Years of experience",
        sampling=SamplingConfig(
            strategy="derived",
            formula="max(0, age - 26)",
            depends_on=["age"],
        ),
        grounding=GroundingInfo(
            level="medium",
            method="computed",
        ),
    )


@pytest.fixture
def conditional_attribute_spec() -> AttributeSpec:
    """A conditional attribute spec with modifiers for testing."""
    return AttributeSpec(
        name="salary",
        type="float",
        category="universal",
        description="Annual salary",
        sampling=SamplingConfig(
            strategy="conditional",
            distribution=NormalDistribution(
                type="normal",
                mean=80000.0,
                std=15000.0,
                min=40000.0,
                max=200000.0,
            ),
            depends_on=["role"],
            modifiers=[
                Modifier(
                    when="role == 'senior'",
                    multiply=1.5,
                    add=10000.0,
                ),
                Modifier(
                    when="role == 'junior'",
                    multiply=0.7,
                    add=0.0,
                ),
            ],
        ),
        grounding=GroundingInfo(
            level="low",
            method="estimated",
        ),
    )


@pytest.fixture
def minimal_population_spec() -> PopulationSpec:
    """A minimal valid population spec for testing."""
    return PopulationSpec(
        meta=SpecMeta(
            description="Test population",
            size=100,
            geography="Test Region",
            created_at=datetime(2024, 1, 1),
            version="1.0",
        ),
        grounding=GroundingSummary(
            overall="medium",
            sources_count=0,
            strong_count=0,
            medium_count=2,
            low_count=0,
            sources=[],
        ),
        attributes=[
            AttributeSpec(
                name="age",
                type="int",
                category="universal",
                description="Age of the agent",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal",
                        mean=45.0,
                        std=10.0,
                        min=25.0,
                        max=70.0,
                    ),
                ),
                grounding=GroundingInfo(
                    level="medium",
                    method="estimated",
                ),
            ),
            AttributeSpec(
                name="gender",
                type="categorical",
                category="universal",
                description="Gender of the agent",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["male", "female", "other"],
                        weights=[0.5, 0.48, 0.02],
                    ),
                ),
                grounding=GroundingInfo(
                    level="medium",
                    method="estimated",
                ),
            ),
        ],
        sampling_order=["age", "gender"],
    )


@pytest.fixture
def complex_population_spec() -> PopulationSpec:
    """A more complex population spec with derived and conditional attributes."""
    return PopulationSpec(
        meta=SpecMeta(
            description="Complex test population",
            size=500,
            geography="Germany",
            created_at=datetime(2024, 1, 1),
            version="1.0",
        ),
        grounding=GroundingSummary(
            overall="medium",
            sources_count=2,
            strong_count=1,
            medium_count=2,
            low_count=1,
            sources=["Source 1", "Source 2"],
        ),
        attributes=[
            # Independent: age
            AttributeSpec(
                name="age",
                type="int",
                category="universal",
                description="Age of the agent",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal",
                        mean=45.0,
                        std=10.0,
                        min=25.0,
                        max=70.0,
                    ),
                ),
                grounding=GroundingInfo(
                    level="strong",
                    method="researched",
                    source="Census data",
                ),
            ),
            # Independent: role
            AttributeSpec(
                name="role",
                type="categorical",
                category="population_specific",
                description="Job role",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["junior", "mid", "senior"],
                        weights=[0.3, 0.5, 0.2],
                    ),
                ),
                grounding=GroundingInfo(
                    level="medium",
                    method="estimated",
                ),
            ),
            # Derived: years_experience (depends on age)
            AttributeSpec(
                name="years_experience",
                type="int",
                category="population_specific",
                description="Years of professional experience",
                sampling=SamplingConfig(
                    strategy="derived",
                    formula="max(0, age - 26)",
                    depends_on=["age"],
                ),
                grounding=GroundingInfo(
                    level="medium",
                    method="computed",
                ),
            ),
            # Conditional: salary (depends on role)
            AttributeSpec(
                name="salary",
                type="float",
                category="universal",
                description="Annual salary",
                sampling=SamplingConfig(
                    strategy="conditional",
                    distribution=NormalDistribution(
                        type="normal",
                        mean=70000.0,
                        std=10000.0,
                        min=40000.0,
                        max=150000.0,
                    ),
                    depends_on=["role"],
                    modifiers=[
                        Modifier(
                            when="role == 'senior'",
                            multiply=1.5,
                            add=10000.0,
                        ),
                        Modifier(
                            when="role == 'junior'",
                            multiply=0.7,
                            add=-5000.0,
                        ),
                    ],
                ),
                grounding=GroundingInfo(
                    level="low",
                    method="estimated",
                ),
            ),
        ],
        sampling_order=["age", "role", "years_experience", "salary"],
    )


@pytest.fixture
def ten_agents() -> list[dict]:
    """Ten agents with varied attributes for propagation/integration tests."""
    roles = ["junior", "mid", "senior"]
    return [
        {"_id": f"a{i}", "age": 25 + i * 5, "role": roles[i % 3]} for i in range(10)
    ]


@pytest.fixture
def linear_network() -> dict:
    """Chain network: a0-a1-a2-...-a9."""
    return {
        "meta": {"node_count": 10},
        "nodes": [{"id": f"a{i}"} for i in range(10)],
        "edges": [
            {"source": f"a{i}", "target": f"a{i + 1}", "type": "colleague"}
            for i in range(9)
        ],
    }


@pytest.fixture
def star_network() -> dict:
    """Star network: a0 is hub, a1-a9 are spokes."""
    return {
        "meta": {"node_count": 10},
        "nodes": [{"id": f"a{i}"} for i in range(10)],
        "edges": [
            {"source": "a0", "target": f"a{i}", "type": "colleague"}
            for i in range(1, 10)
        ],
    }


@pytest.fixture
def sample_agents() -> list[dict]:
    """Sample agent data for network tests."""
    return [
        {
            "_id": "agent_000",
            "age": 45,
            "gender": "male",
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "federal_state": "Bayern",
            "role_seniority": "chief_physician_Chefarzt",
            "years_experience": 20,
            "participation_in_research": True,
            "professional_society_membership": True,
            "teaching_responsibility": True,
        },
        {
            "_id": "agent_001",
            "age": 35,
            "gender": "female",
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "federal_state": "Bayern",
            "role_seniority": "senior_physician_Oberarzt",
            "years_experience": 10,
            "participation_in_research": True,
            "professional_society_membership": True,
            "teaching_responsibility": False,
        },
        {
            "_id": "agent_002",
            "age": 30,
            "gender": "male",
            "employer_type": "university_hospital",
            "surgical_specialty": "neurology",
            "federal_state": "Bayern",
            "role_seniority": "resident",
            "years_experience": 4,
            "participation_in_research": False,
            "professional_society_membership": False,
            "teaching_responsibility": False,
        },
        {
            "_id": "agent_003",
            "age": 50,
            "gender": "female",
            "employer_type": "private_clinic",
            "surgical_specialty": "cardiology",
            "federal_state": "Berlin",
            "role_seniority": "senior_physician_Oberarzt",
            "years_experience": 25,
            "participation_in_research": False,
            "professional_society_membership": True,
            "teaching_responsibility": True,
        },
        {
            "_id": "agent_004",
            "age": 40,
            "gender": "male",
            "employer_type": "private_clinic",
            "surgical_specialty": "orthopedics",
            "federal_state": "Berlin",
            "role_seniority": "specialist_attending",
            "years_experience": 15,
            "participation_in_research": False,
            "professional_society_membership": False,
            "teaching_responsibility": False,
        },
    ]
