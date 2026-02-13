"""Tests for dependency binding and inference."""

from extropy.core.models.population import (
    AttributeSpec,
    BooleanDistribution,
    GroundingInfo,
    HydratedAttribute,
    Modifier,
    NormalDistribution,
    SamplingConfig,
)
from extropy.population.spec_builder.binder import bind_constraints


def _grounding() -> GroundingInfo:
    return GroundingInfo(level="low", method="estimated")


def _context_attr(name: str) -> AttributeSpec:
    return AttributeSpec(
        name=name,
        type="float",
        category="universal",
        description=f"Context attr {name}",
        sampling=SamplingConfig(
            strategy="independent",
            distribution=NormalDistribution(mean=0.0, std=1.0),
            depends_on=[],
        ),
        grounding=_grounding(),
        constraints=[],
    )


def test_bind_constraints_infers_formula_dependencies_from_context():
    """Derived formulas should auto-add referenced attrs to depends_on."""
    uses_flag = HydratedAttribute(
        name="uses_third_party_app",
        type="boolean",
        category="population_specific",
        description="Whether agent uses a third-party app",
        strategy="conditional",
        depends_on=[],
        sampling=SamplingConfig(
            strategy="independent",
            distribution=BooleanDistribution(probability_true=0.2),
            depends_on=[],
            modifiers=[],
        ),
        grounding=_grounding(),
        constraints=[],
    )

    preferred_app = HydratedAttribute(
        name="preferred_third_party_app",
        type="categorical",
        category="population_specific",
        description="Preferred third-party app",
        strategy="derived",
        depends_on=["uses_third_party_app"],
        sampling=SamplingConfig(
            strategy="derived",
            distribution=None,
            formula="'apollo' if uses_third_party_app and age < 30 else 'none'",
            depends_on=["uses_third_party_app"],
            modifiers=[],
        ),
        grounding=_grounding(),
        constraints=[],
    )

    specs, sampling_order, warnings = bind_constraints(
        [uses_flag, preferred_app],
        context=[_context_attr("age")],
    )

    by_name = {spec.name: spec for spec in specs}
    assert set(by_name["preferred_third_party_app"].sampling.depends_on) == {
        "uses_third_party_app",
        "age",
    }
    assert sampling_order == ["uses_third_party_app", "preferred_third_party_app"]
    assert any("auto-added depends_on" in w for w in warnings)


def test_bind_constraints_infers_modifier_and_mean_formula_dependencies():
    """Conditional refs in when/mean_formula should auto-add depends_on."""
    moderator_count = HydratedAttribute(
        name="moderator_of_subreddits",
        type="int",
        category="population_specific",
        description="Count of moderated subreddits",
        strategy="independent",
        depends_on=[],
        sampling=SamplingConfig(
            strategy="independent",
            distribution=NormalDistribution(mean=1.0, std=1.0, min=0.0, max=10.0),
            depends_on=[],
            modifiers=[],
        ),
        grounding=_grounding(),
        constraints=[],
    )

    tool_dependency = HydratedAttribute(
        name="moderation_tool_dependency",
        type="float",
        category="population_specific",
        description="Reliance on moderation tools",
        strategy="conditional",
        depends_on=["moderator_of_subreddits"],
        sampling=SamplingConfig(
            strategy="conditional",
            distribution=NormalDistribution(
                mean=0.0,
                std=1.0,
                mean_formula="0.2 * annual_income + moderator_of_subreddits",
            ),
            depends_on=["moderator_of_subreddits"],
            modifiers=[
                Modifier(
                    when="weekly_hours_on_reddit > 10 and moderator_of_subreddits > 0",
                    add=0.5,
                )
            ],
        ),
        grounding=_grounding(),
        constraints=[],
    )

    specs, _, _ = bind_constraints(
        [moderator_count, tool_dependency],
        context=[
            _context_attr("annual_income"),
            _context_attr("weekly_hours_on_reddit"),
        ],
    )

    by_name = {spec.name: spec for spec in specs}
    assert set(by_name["moderation_tool_dependency"].sampling.depends_on) == {
        "moderator_of_subreddits",
        "annual_income",
        "weekly_hours_on_reddit",
    }
