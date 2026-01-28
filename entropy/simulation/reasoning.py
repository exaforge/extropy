"""Agent reasoning via LLM for simulation.

Handles the core reasoning loop where agents process information
and form opinions using structured LLM calls.
"""

import json
import logging
import time
from typing import Any

from ..core.llm import simple_call, simple_call_async
from ..core.models import (
    ExposureRecord,
    OutcomeConfig,
    OutcomeType,
    PeerOpinion,
    ReasoningContext,
    ReasoningResponse,
    ScenarioSpec,
    SimulationRunConfig,
)

logger = logging.getLogger(__name__)


def build_reasoning_prompt(
    context: ReasoningContext,
    scenario: ScenarioSpec,
) -> str:
    """Build the agent reasoning prompt.

    Args:
        context: Reasoning context with persona and exposure history
        scenario: Scenario specification

    Returns:
        Complete prompt string for LLM
    """
    prompt_parts = [
        "You ARE the person described below. Respond as yourself - not as an observer or simulator.",
        "Your background, attitudes, and circumstances shape how you interpret and react to information.",
        "People with different characteristics respond very differently to the same news.",
        "",
        context.persona,
        "",
        "## What You Just Learned",
        "",
        scenario.event.content,
        "",
        f"Source: {scenario.event.source}",
        "",
        "## How This Reached You",
        "",
    ]

    # Add exposure history
    for exp in context.exposure_history:
        if exp.source_agent_id:
            prompt_parts.append(
                f"- Someone in your network told you about this"
            )
        else:
            prompt_parts.append(
                f"- You heard about this via {exp.channel}"
            )

    # Add peer opinions if available
    if context.peer_opinions:
        prompt_parts.extend(
            [
                "",
                "## What People Around You Think",
                "",
            ]
        )
        for peer in context.peer_opinions:
            if peer.position:
                prompt_parts.append(
                    f"- A {peer.relationship} of yours: {peer.position}"
                )

    # Add instructions
    prompt_parts.extend(
        [
            "",
            "## Your Authentic Response",
            "",
            "Given YOUR specific background, attitudes, constraints, and priorities:",
            "- What is your genuine, gut reaction?",
            "- How does this actually affect YOUR situation?",
            "- What will YOU realistically do (or not do)?",
            "",
            "Be true to your characteristics. Not everyone reacts the same way.",
            "Someone with your profile might be enthusiastic, skeptical, cautious, opposed, or indifferent.",
        ]
    )

    # Add extraction instructions if provided
    if scenario.outcomes.extraction_instructions:
        prompt_parts.extend(
            [
                "",
                scenario.outcomes.extraction_instructions,
            ]
        )

    return "\n".join(prompt_parts)


def build_response_schema(outcomes: OutcomeConfig) -> dict[str, Any]:
    """Build JSON schema from scenario outcomes.

    Args:
        outcomes: Outcome configuration from scenario

    Returns:
        JSON schema dictionary for structured output
    """
    properties: dict[str, Any] = {
        "reasoning": {
            "type": "string",
            "description": "One sentence: your gut reaction and key reason why",
        },
        "will_share": {
            "type": "boolean",
            "description": "Will you discuss or share this with others?",
        },
    }

    required = ["reasoning", "will_share"]

    for outcome in outcomes.suggested_outcomes:
        outcome_prop: dict[str, Any] = {
            "description": outcome.description,
        }

        if outcome.type == OutcomeType.CATEGORICAL and outcome.options:
            outcome_prop["type"] = "string"
            outcome_prop["enum"] = outcome.options
        elif outcome.type == OutcomeType.BOOLEAN:
            outcome_prop["type"] = "boolean"
        elif outcome.type == OutcomeType.FLOAT and outcome.range:
            outcome_prop["type"] = "number"
            outcome_prop["minimum"] = outcome.range[0]
            outcome_prop["maximum"] = outcome.range[1]
        elif outcome.type == OutcomeType.OPEN_ENDED:
            outcome_prop["type"] = "string"
        else:
            # Default to string
            outcome_prop["type"] = "string"

        properties[outcome.name] = outcome_prop

        required.append(outcome.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _get_primary_position_outcome(scenario: ScenarioSpec) -> str | None:
    """Get the name of the primary position outcome.

    The "position" is the main categorical decision/stance an agent takes.
    Uses the first required categorical outcome, or first categorical if none required.

    Only considers categorical outcomes since position must be a string
    for display (e.g., "A colleague is {position}") and aggregation.

    Args:
        scenario: Scenario specification

    Returns:
        Name of the primary position outcome, or None
    """
    categorical_outcomes = [
        o for o in scenario.outcomes.suggested_outcomes if o.type.value == "categorical"
    ]

    if not categorical_outcomes:
        return None

    # First required categorical, or first categorical if none required
    required = [o for o in categorical_outcomes if o.required]
    return required[0].name if required else categorical_outcomes[0].name


def reason_agent(
    context: ReasoningContext,
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
) -> ReasoningResponse | None:
    """Call LLM to get agent's reasoning and response.

    Args:
        context: Reasoning context with persona, event, exposures
        scenario: Scenario specification
        config: Simulation run configuration

    Returns:
        ReasoningResponse with extracted outcomes, or None if failed
    """
    prompt = build_reasoning_prompt(context, scenario)
    schema = build_response_schema(scenario.outcomes)

    position_outcome = _get_primary_position_outcome(scenario)

    # Heavy logging - payload
    logger.info(f"[REASON] Agent {context.agent_id} - preparing LLM call")
    logger.info(f"[REASON] Agent {context.agent_id} - model: {config.model}")
    logger.info(f"[REASON] Agent {context.agent_id} - prompt length: {len(prompt)} chars")
    logger.debug(f"[REASON] Agent {context.agent_id} - PROMPT:\n{prompt[:500]}...")
    logger.debug(f"[REASON] Agent {context.agent_id} - SCHEMA: {json.dumps(schema, indent=2)}")

    for attempt in range(config.max_retries):
        try:
            logger.info(f"[REASON] Agent {context.agent_id} - attempt {attempt + 1}/{config.max_retries}")

            call_start = time.time()
            response = simple_call(
                prompt=prompt,
                response_schema=schema,
                schema_name="agent_response",
                model=config.model,
                log=True,
            )
            call_elapsed = time.time() - call_start

            logger.info(f"[REASON] Agent {context.agent_id} - API call took {call_elapsed:.2f}s")
            logger.info(f"[REASON] Agent {context.agent_id} - RESPONSE: {json.dumps(response) if response else 'None'}")

            if not response:
                logger.warning(
                    f"[REASON] Agent {context.agent_id} - Empty response, attempt {attempt + 1}"
                )
                continue

            # Extract position from the primary position outcome (always categorical)
            position = None
            if position_outcome and position_outcome in response:
                position = response[position_outcome]

            # Extract sentiment if present
            sentiment = response.get("sentiment")
            if sentiment is not None:
                try:
                    sentiment = float(sentiment)
                except (ValueError, TypeError):
                    sentiment = None

            logger.info(
                f"[REASON] Agent {context.agent_id} - SUCCESS: position={position}, "
                f"sentiment={sentiment}, will_share={response.get('will_share')}"
            )

            return ReasoningResponse(
                position=position,
                sentiment=sentiment,
                action_intent=response.get("action_intent"),
                will_share=response.get("will_share", False),
                reasoning=response.get("reasoning", ""),
                outcomes={k: v for k, v in response.items()},
            )

        except Exception as e:
            logger.warning(
                f"[REASON] Agent {context.agent_id} - EXCEPTION attempt {attempt + 1}: {e}"
            )
            if attempt == config.max_retries - 1:
                logger.error(f"[REASON] Agent {context.agent_id} - All retries exhausted")
                return None

    return None


async def _reason_agent_async(
    context: ReasoningContext,
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
) -> ReasoningResponse | None:
    """Async version of reason_agent for concurrent execution."""
    prompt = build_reasoning_prompt(context, scenario)
    schema = build_response_schema(scenario.outcomes)
    position_outcome = _get_primary_position_outcome(scenario)

    for attempt in range(config.max_retries):
        try:
            call_start = time.time()
            response = await simple_call_async(
                prompt=prompt,
                response_schema=schema,
                schema_name="agent_response",
                model=config.model,
            )
            call_elapsed = time.time() - call_start

            logger.info(f"[ASYNC] Agent {context.agent_id} - API call took {call_elapsed:.2f}s")

            if not response:
                continue

            position = None
            if position_outcome and position_outcome in response:
                position = response[position_outcome]

            sentiment = response.get("sentiment")
            if sentiment is not None:
                try:
                    sentiment = float(sentiment)
                except (ValueError, TypeError):
                    sentiment = None

            return ReasoningResponse(
                position=position,
                sentiment=sentiment,
                action_intent=response.get("action_intent"),
                will_share=response.get("will_share", False),
                reasoning=response.get("reasoning", ""),
                outcomes={k: v for k, v in response.items()},
            )

        except Exception as e:
            logger.warning(f"[ASYNC] Agent {context.agent_id} - attempt {attempt + 1} failed: {e}")
            if attempt == config.max_retries - 1:
                return None

    return None


def batch_reason_agents(
    contexts: list[ReasoningContext],
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
    max_concurrency: int = 50,
) -> list[tuple[str, ReasoningResponse | None]]:
    """Reason multiple agents concurrently using asyncio.

    Args:
        contexts: List of reasoning contexts
        scenario: Scenario specification
        config: Simulation run configuration
        max_concurrency: Max concurrent API calls (default 50)

    Returns:
        List of (agent_id, response) tuples in original order
    """
    import asyncio

    if not contexts:
        return []

    total = len(contexts)
    logger.info(f"[BATCH] Starting async reasoning for {total} agents (max_concurrency={max_concurrency})")

    async def run_all():
        semaphore = asyncio.Semaphore(max_concurrency)
        completed = [0]

        async def reason_with_semaphore(ctx: ReasoningContext) -> tuple[str, ReasoningResponse | None]:
            async with semaphore:
                start = time.time()
                result = await _reason_agent_async(ctx, scenario, config)
                elapsed = time.time() - start
                completed[0] += 1

                if result:
                    logger.info(
                        f"[BATCH] {completed[0]}/{total}: {ctx.agent_id} done in {elapsed:.2f}s "
                        f"(position={result.position})"
                    )
                else:
                    logger.warning(f"[BATCH] {completed[0]}/{total}: {ctx.agent_id} FAILED")

                return (ctx.agent_id, result)

        tasks = [reason_with_semaphore(ctx) for ctx in contexts]
        return await asyncio.gather(*tasks)

    batch_start = time.time()
    results = asyncio.run(run_all())
    batch_elapsed = time.time() - batch_start

    logger.info(f"[BATCH] Completed {total} agents in {batch_elapsed:.2f}s ({batch_elapsed/total:.2f}s/agent avg)")
    return list(results)


def create_reasoning_context(
    agent_id: str,
    agent: dict[str, Any],
    persona: str,
    exposures: list[ExposureRecord],
    scenario: ScenarioSpec,
    peer_opinions: list[PeerOpinion] | None = None,
    current_state: Any = None,
) -> ReasoningContext:
    """Create a reasoning context for an agent.

    Args:
        agent_id: Agent ID
        agent: Agent attributes dictionary
        persona: Generated persona string
        exposures: Exposure history
        scenario: Scenario specification
        peer_opinions: Optional peer opinions for social influence
        current_state: Optional previous state for re-reasoning

    Returns:
        ReasoningContext ready for LLM call
    """
    return ReasoningContext(
        agent_id=agent_id,
        persona=persona,
        event_content=scenario.event.content,
        exposure_history=exposures,
        peer_opinions=peer_opinions or [],
        current_state=current_state,
    )
