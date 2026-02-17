"""Execution-path tests for two-pass reasoning."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

from extropy.core.llm import TokenUsage
from extropy.core.models import ExposureRecord, ReasoningContext, SimulationRunConfig
from extropy.core.models.scenario import (
    Event,
    EventType,
    ExposureChannel,
    ExposureRule,
    InteractionConfig,
    InteractionType,
    OutcomeConfig,
    OutcomeDefinition,
    OutcomeType,
    ScenarioMeta,
    ScenarioSimConfig,
    ScenarioSpec,
    SeedExposure,
    SpreadConfig,
    TimestepUnit,
)
from extropy.simulation.reasoning import _reason_agent_two_pass_async


def _make_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        meta=ScenarioMeta(
            name="test",
            description="Test scenario",
            population_spec="population.v1.yaml",
            study_db="study.db",
            population_id="default",
            network_id="default",
            created_at=datetime(2024, 1, 1),
        ),
        event=Event(
            type=EventType.PRODUCT_LAUNCH,
            content="A new policy is announced.",
            source="City Hall",
            credibility=0.9,
            ambiguity=0.3,
            emotional_valence=-0.1,
        ),
        seed_exposure=SeedExposure(
            channels=[
                ExposureChannel(
                    name="broadcast",
                    description="Broadcast",
                    reach="broadcast",
                    credibility_modifier=1.0,
                )
            ],
            rules=[
                ExposureRule(
                    channel="broadcast",
                    timestep=0,
                    when="true",
                    probability=1.0,
                )
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Observe",
        ),
        spread=SpreadConfig(share_probability=0.3),
        outcomes=OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="adoption",
                    description="Primary stance",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["adopt", "reject", "wait"],
                )
            ]
        ),
        simulation=ScenarioSimConfig(max_timesteps=3),
    )


def _make_context() -> ReasoningContext:
    return ReasoningContext(
        agent_id="a0",
        persona="I'm Alex, a resident in the city.",
        event_content="A new policy is announced.",
        exposure_history=[
            ExposureRecord(
                timestep=0,
                channel="broadcast",
                content="A new policy is announced.",
                credibility=0.9,
            )
        ],
        peer_opinions=[],
        agent_name="Alex",
        timestep=0,
        timestep_unit="day",
    )


def test_two_pass_high_fidelity_classifies_private_and_public_positions_separately():
    scenario = _make_scenario()
    context = _make_context()
    config = SimulationRunConfig(
        scenario_path="test.yaml",
        output_dir="results",
        max_retries=1,
        fidelity="high",
    )

    pass1_response = {
        "reasoning": "I can see both sides, but I'm conflicted.",
        "private_thought": "I privately reject this policy.",
        "public_statement": "Let's stay open-minded and give it a chance.",
        "reasoning_summary": "I'm conflicted.",
        "sentiment": -0.2,
        "conviction": 62,
        "will_share": True,
        "actions": [],
    }

    mocked_call = AsyncMock(
        side_effect=[
            (pass1_response, TokenUsage(input_tokens=10, output_tokens=7)),
            ({"adoption": "reject"}, TokenUsage(input_tokens=4, output_tokens=2)),
            ({"adoption": "adopt"}, TokenUsage(input_tokens=5, output_tokens=3)),
        ]
    )

    with patch("extropy.simulation.reasoning.simple_call_async", mocked_call):
        response = asyncio.run(
            _reason_agent_two_pass_async(
                context=context,
                scenario=scenario,
                config=config,
                rate_limiter=None,
            )
        )

    assert response is not None
    assert response.position == "reject"
    assert response.public_position == "adopt"
    assert response.pass2_input_tokens == 9
    assert response.pass2_output_tokens == 5
    assert mocked_call.await_count == 3
    assert (
        "I privately reject this policy"
        in mocked_call.await_args_list[1].kwargs["prompt"]
    )
    assert "give it a chance" in mocked_call.await_args_list[2].kwargs["prompt"]


def test_two_pass_medium_fidelity_uses_private_classification_for_public_position():
    scenario = _make_scenario()
    context = _make_context()
    config = SimulationRunConfig(
        scenario_path="test.yaml",
        output_dir="results",
        max_retries=1,
        fidelity="medium",
    )

    pass1_response = {
        "reasoning": "I am leaning toward waiting.",
        "private_thought": "I should wait and watch.",
        "public_statement": "I'm not sure yet.",
        "reasoning_summary": "Leaning to wait.",
        "sentiment": 0.0,
        "conviction": 45,
        "will_share": False,
        "actions": [],
    }

    mocked_call = AsyncMock(
        side_effect=[
            (pass1_response, TokenUsage(input_tokens=8, output_tokens=6)),
            ({"adoption": "wait"}, TokenUsage(input_tokens=3, output_tokens=2)),
        ]
    )

    with patch("extropy.simulation.reasoning.simple_call_async", mocked_call):
        response = asyncio.run(
            _reason_agent_two_pass_async(
                context=context,
                scenario=scenario,
                config=config,
                rate_limiter=None,
            )
        )

    assert response is not None
    assert response.position == "wait"
    assert response.public_position == "wait"
    assert mocked_call.await_count == 2


def _make_scenario_no_pass2() -> ScenarioSpec:
    """Scenario with only open-ended outcomes (no categorical pass2)."""
    return ScenarioSpec(
        meta=ScenarioMeta(
            name="reasoning_exec",
            description="Reasoning execution test",
            population_spec="test.yaml",
            study_db="study.db",
            population_id="default",
            network_id="default",
            created_at=datetime(2024, 1, 1),
        ),
        event=Event(
            type=EventType.NEWS,
            content="Policy update",
            source="Board",
            credibility=0.8,
            ambiguity=0.3,
            emotional_valence=0.0,
        ),
        seed_exposure=SeedExposure(
            channels=[
                ExposureChannel(
                    name="broadcast",
                    description="Broadcast",
                    reach="broadcast",
                )
            ],
            rules=[
                ExposureRule(
                    channel="broadcast",
                    timestep=0,
                    when="true",
                    probability=1.0,
                )
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Observe updates",
        ),
        spread=SpreadConfig(share_probability=0.4),
        outcomes=OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="notes",
                    description="Open notes only",
                    type=OutcomeType.OPEN_ENDED,
                    required=False,
                )
            ]
        ),
        simulation=ScenarioSimConfig(max_timesteps=2, timestep_unit=TimestepUnit.DAY),
    )


def _make_context_for_action_intent() -> ReasoningContext:
    return ReasoningContext(
        agent_id="a1",
        persona="I am a parent with limited time.",
        event_content="School board policy update",
        exposure_history=[
            ExposureRecord(
                timestep=0,
                channel="broadcast",
                content="Policy update",
                credibility=0.9,
                source_agent_id=None,
            )
        ],
        agent_name="Taylor",
    )


def test_action_intent_is_captured_without_pass2_outcome():
    context = _make_context_for_action_intent()
    scenario = _make_scenario_no_pass2()
    config = SimulationRunConfig(
        scenario_path="test.yaml",
        output_dir="test/",
        max_retries=1,
    )

    async def _mock_simple_call_async(*args, **kwargs):
        return (
            {
                "reasoning": "I need to show up and speak.",
                "private_thought": "If I don't go, nothing changes.",
                "public_statement": "I should attend the meeting.",
                "reasoning_summary": "I need to attend and speak.",
                "action_intent": "Attend the next board meeting",
                "sentiment": -0.2,
                "conviction": 75,
                "will_share": True,
                "actions": [],
            },
            TokenUsage(),
        )

    with patch(
        "extropy.simulation.reasoning.simple_call_async",
        new=_mock_simple_call_async,
    ):
        response = asyncio.run(_reason_agent_two_pass_async(context, scenario, config))

    assert response is not None
    assert response.action_intent == "Attend the next board meeting"
    assert response.outcomes.get("action_intent") == "Attend the next board meeting"
