"""Tests for deterministic modifier precedence and overlap warnings."""

import random
from datetime import datetime

from extropy.core.models.population import (
    AttributeSpec,
    CategoricalDistribution,
    GroundingInfo,
    GroundingSummary,
    Modifier,
    PopulationSpec,
    SamplingConfig,
    SpecMeta,
    UniformDistribution,
    BooleanDistribution,
)
from extropy.population.modifier_precedence import choose_modifier_precedence
from extropy.population.sampler.modifiers import apply_modifiers_and_sample
from extropy.population.validator.semantic import run_semantic_checks


def _categorical_attr(name: str, options: list[str]) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        type="categorical",
        category="population_specific",
        description=name,
        sampling=SamplingConfig(
            strategy="independent",
            distribution=CategoricalDistribution(
                type="categorical",
                options=options,
                weights=[1.0 / len(options)] * len(options),
            ),
        ),
        grounding=GroundingInfo(level="low", method="estimated"),
    )


def _int_attr(name: str, low: float = 0.0, high: float = 100.0) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        type="int",
        category="universal",
        description=name,
        sampling=SamplingConfig(
            strategy="independent",
            distribution=UniformDistribution(type="uniform", min=low, max=high),
        ),
        grounding=GroundingInfo(level="low", method="estimated"),
    )


def _spec(attrs: list[AttributeSpec], order: list[str]) -> PopulationSpec:
    return PopulationSpec(
        meta=SpecMeta(
            description="modifier precedence test",
            geography="test",
            created_at=datetime(2024, 1, 1),
            version="1.0",
        ),
        grounding=GroundingSummary(
            overall="low",
            sources_count=0,
            strong_count=0,
            medium_count=0,
            low_count=1,
            sources=[],
        ),
        attributes=attrs,
        sampling_order=order,
    )


def test_choose_modifier_precedence_subset_wins():
    decision = choose_modifier_precedence(
        [
            (3, "age >= 50"),
            (7, "age >= 65"),
        ]
    )
    assert decision is not None
    assert decision.winner_index == 7
    assert decision.reason == "subset"


def test_choose_modifier_precedence_specificity_wins():
    decision = choose_modifier_precedence(
        [
            (2, "education_level == 'Graduate degree'"),
            (
                4,
                "education_level == 'Graduate degree' and race_ethnicity == 'Asian'",
            ),
        ]
    )
    assert decision is not None
    assert decision.winner_index == 4
    assert decision.reason in {"subset", "specificity"}


def test_choose_modifier_precedence_order_wins_on_tie():
    decision = choose_modifier_precedence(
        [
            (1, "us_region == 'West' and race_ethnicity == 'Asian'"),
            (6, "us_region == 'West' and education_level == 'Graduate degree'"),
        ]
    )
    assert decision is not None
    assert decision.winner_index == 6
    assert decision.reason == "order"


def test_boolean_modifiers_use_single_deterministic_winner():
    distribution = BooleanDistribution(type="boolean", probability_true=0.2)
    modifiers = [
        Modifier(when="age >= 18", add=0.2),
        Modifier(when="age >= 18 and household_income >= 100000", add=0.5),
    ]
    _, triggered, warnings = apply_modifiers_and_sample(
        dist=distribution,
        modifiers=modifiers,
        rng=random.Random(42),
        agent={"age": 42, "household_income": 120000},
    )
    assert warnings == []
    assert triggered == [1]


def test_semantic_overlap_suppressed_when_subset_or_specificity_resolves():
    status = AttributeSpec(
        name="status",
        type="categorical",
        category="population_specific",
        description="status",
        sampling=SamplingConfig(
            strategy="conditional",
            distribution=CategoricalDistribution(
                type="categorical",
                options=["A", "B"],
                weights=[0.6, 0.4],
            ),
            depends_on=["age", "education_level"],
            modifiers=[
                Modifier(
                    when="age >= 50",
                    weight_overrides={"A": 0.2, "B": 0.8},
                ),
                Modifier(
                    when="age >= 50 and education_level == 'Graduate degree'",
                    weight_overrides={"A": 0.8, "B": 0.2},
                ),
            ],
        ),
        grounding=GroundingInfo(level="low", method="estimated"),
    )
    spec = _spec(
        [
            _int_attr("age", low=18, high=90),
            _categorical_attr("education_level", ["HS", "Graduate degree"]),
            status,
        ],
        ["age", "education_level", "status"],
    )

    issues = run_semantic_checks(spec)
    overlap_issues = [issue for issue in issues if issue.category == "MODIFIER_OVERLAP"]
    assert overlap_issues == []


def test_semantic_overlap_warns_when_only_order_can_resolve():
    identity = AttributeSpec(
        name="identity_segment",
        type="categorical",
        category="population_specific",
        description="identity_segment",
        sampling=SamplingConfig(
            strategy="conditional",
            distribution=CategoricalDistribution(
                type="categorical",
                options=["X", "Y"],
                weights=[0.5, 0.5],
            ),
            depends_on=["us_region", "race_ethnicity", "education_level"],
            modifiers=[
                Modifier(
                    when="us_region == 'West' and race_ethnicity == 'Asian'",
                    weight_overrides={"X": 0.8, "Y": 0.2},
                ),
                Modifier(
                    when="us_region == 'West' and education_level == 'Graduate degree'",
                    weight_overrides={"X": 0.2, "Y": 0.8},
                ),
            ],
        ),
        grounding=GroundingInfo(level="low", method="estimated"),
    )
    spec = _spec(
        [
            _categorical_attr("us_region", ["West", "South"]),
            _categorical_attr("race_ethnicity", ["Asian", "White"]),
            _categorical_attr("education_level", ["HS", "Graduate degree"]),
            identity,
        ],
        ["us_region", "race_ethnicity", "education_level", "identity_segment"],
    )

    issues = run_semantic_checks(spec)
    overlap_issues = [issue for issue in issues if issue.category == "MODIFIER_OVERLAP"]
    assert len(overlap_issues) == 1
