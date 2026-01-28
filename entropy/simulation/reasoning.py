"""Agent reasoning via LLM for simulation.

Handles the core reasoning loop where agents process information
and form opinions using structured LLM calls.
"""

import logging
from typing import Any

from ..core.llm import simple_call
from ..core.models import (
    ScenarioSpec,
    OutcomeConfig,
    ReasoningContext,
    ReasoningResponse,
    SimulationRunConfig,
    ExposureRecord,
    PeerOpinion,
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
        "You are simulating how a specific person would react to news/information.",
        "",
        "## Who You Are",
        "",
        context.persona,
        "",
        "## What Happened",
        "",
        scenario.event.content,
        "",
        f"Source: {scenario.event.source}",
        "",
        "## How You Learned About This",
        "",
    ]

    # Add exposure history
    for exp in context.exposure_history:
        if exp.source_agent_id:
            prompt_parts.append(
                f"- A contact in your network told you about this (timestep {exp.timestep})"
            )
        else:
            prompt_parts.append(
                f"- You learned via {exp.channel} (timestep {exp.timestep})"
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
                    f"- A {peer.relationship} of yours is {peer.position}"
                )

    # Add instructions
    prompt_parts.extend(
        [
            "",
            "## Your Response",
            "",
            "Based on who you are and what you've learned, respond naturally. Consider:",
            "- How does this affect you personally?",
            "- How do you feel about this?",
            "- What, if anything, will you do?",
            "- Will you discuss this with others?",
            "",
        ]
    )

    # Add extraction instructions if provided
    if scenario.outcomes.extraction_instructions:
        prompt_parts.extend(
            [
                scenario.outcomes.extraction_instructions,
                "",
            ]
        )

    prompt_parts.append(
        "Respond in character as this person would actually think and react."
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
            "description": "1-3 sentences explaining your reaction in first person",
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

        if outcome.type.value == "categorical" and outcome.options:
            outcome_prop["type"] = "string"
            outcome_prop["enum"] = outcome.options
        elif outcome.type.value == "boolean":
            outcome_prop["type"] = "boolean"
        elif outcome.type.value == "float" and outcome.range:
            outcome_prop["type"] = "number"
            outcome_prop["minimum"] = outcome.range[0]
            outcome_prop["maximum"] = outcome.range[1]
        elif outcome.type.value == "open_ended":
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

    for attempt in range(config.max_retries):
        try:
            response = simple_call(
                prompt=prompt,
                response_schema=schema,
                schema_name="agent_response",
                model=config.model,
                log=True,
            )

            if not response:
                logger.warning(
                    f"Empty response for agent {context.agent_id}, attempt {attempt + 1}"
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
                f"Reasoning failed for agent {context.agent_id}, "
                f"attempt {attempt + 1}: {e}"
            )
            if attempt == config.max_retries - 1:
                logger.error(f"All retries exhausted for agent {context.agent_id}")
                return None

    return None


def batch_reason_agents(
    contexts: list[ReasoningContext],
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
) -> list[tuple[str, ReasoningResponse | None]]:
    """Reason multiple agents (sequentially for now).

    Future: Could be parallelized with async calls.

    Args:
        contexts: List of reasoning contexts
        scenario: Scenario specification
        config: Simulation run configuration

    Returns:
        List of (agent_id, response) tuples
    """
    results = []

    for context in contexts:
        response = reason_agent(context, scenario, config)
        results.append((context.agent_id, response))

    return results


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
