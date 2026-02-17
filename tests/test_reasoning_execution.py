"""Execution tests for reasoning internals that affect memory/accountability."""

import asyncio
from datetime import datetime
from unittest.mock import patch

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
    TimestepUnit,
)
from extropy.simulation.reasoning import _reason_agent_two_pass_async


def _make_context() -> ReasoningContext:
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


def _make_scenario_no_pass2() -> ScenarioSpec:
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
        simulation=SimulationConfig(max_timesteps=2, timestep_unit=TimestepUnit.DAY),
    )


def test_action_intent_is_captured_without_pass2_outcome():
    context = _make_context()
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
