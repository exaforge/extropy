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
    ScenarioSpec,
    SeedExposure,
    SimulationConfig,
    SpreadConfig,
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
        simulation=SimulationConfig(max_timesteps=3),
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
    assert "I privately reject this policy" in mocked_call.await_args_list[1].kwargs[
        "prompt"
    ]
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
