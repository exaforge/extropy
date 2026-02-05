"""Agent reasoning via LLM for simulation.

Implements two-pass reasoning:
- Pass 1 (role-play): Free-text reasoning with no categorical constraints.
  The agent reasons naturally — sentiment, conviction, public statement.
- Pass 2 (classification): A cheap model classifies the free-text into
  scenario-defined categorical/boolean/float outcomes.

This split fixes the central tendency problem where 83% of agents chose
safe middle options when role-play and classification were combined.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from ..core.llm import simple_call, simple_call_async
from ..core.providers import close_simulation_provider
from ..core.models import (
    ExposureRecord,
    MemoryEntry,
    OutcomeConfig,
    OutcomeType,
    PeerOpinion,
    ReasoningContext,
    ReasoningResponse,
    ScenarioSpec,
    SimulationRunConfig,
    float_to_conviction,
    score_to_conviction_float,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pass 1: Free-text role-play prompt
# =============================================================================


def build_pass1_prompt(
    context: ReasoningContext,
    scenario: ScenarioSpec,
) -> str:
    """Build the Pass 1 (role-play) prompt.

    No categorical enums. The agent reasons naturally about the event,
    forms a sentiment, conviction level, and public statement.

    Args:
        context: Reasoning context with persona and exposure history
        scenario: Scenario specification

    Returns:
        Complete prompt string for LLM
    """
    prompt_parts = [
        "You ARE the person described below. Think and respond as yourself — first person, authentic, decisive.",
        "IMPORTANT: Do NOT hedge, equivocate, or give a 'balanced' take. Real people have clear reactions.",
        "Some people love this. Some are indifferent. Some are annoyed. Be honest about YOUR reaction.",
        "Your background and circumstances determine your response — not politeness or caution.",
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
            prompt_parts.append("- Someone in your network told you about this")
        else:
            prompt_parts.append(f"- You heard about this via {exp.channel}")

    # Add memory trace if re-reasoning
    if context.memory_trace:
        prompt_parts.extend(
            [
                "",
                "## Your Previous Thinking",
                "",
            ]
        )
        for memory in context.memory_trace:
            conviction_label = float_to_conviction(memory.conviction) or "uncertain"
            prompt_parts.append(
                f'- Previously, you thought: "{memory.summary}" '
                f"(you felt *{conviction_label}* about this)"
            )

    # Add peer opinions with public statements (NOT position labels)
    if context.peer_opinions:
        prompt_parts.extend(
            [
                "",
                "## What People Around You Are Saying",
                "",
            ]
        )
        for peer in context.peer_opinions:
            if peer.public_statement:
                prompt_parts.append(
                    f'- A {peer.relationship} of yours says: "{peer.public_statement}"'
                )
            elif peer.sentiment is not None:
                # Fallback: describe sentiment tone if no statement
                tone = _sentiment_to_tone(peer.sentiment)
                prompt_parts.append(
                    f"- A {peer.relationship} of yours seems {tone} about this"
                )

    # Add instructions
    if context.memory_trace:
        prompt_parts.extend(
            [
                "",
                "## Your Honest Reaction",
                "",
                "Given YOUR background, YOUR previous thinking, and what you're hearing:",
                "- Has your view changed? If so, own it. If not, say so plainly.",
                "- What are you actually going to DO about this now?",
                "- What would you bluntly tell a friend?",
                "",
                "Commit to where you stand now. Don't hedge just because your opinion changed.",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "## Your Honest Reaction",
                "",
                "React as YOU — given your background, priorities, and constraints:",
                "- What is your immediate, honest reaction? Don't overthink it.",
                "- What are you actually going to DO about this? Decide now.",
                "- What would you bluntly tell a friend?",
                "",
                "Commit to a clear position. Saying 'I'll wait and see' is only valid if you genuinely don't care.",
            ]
        )

    return "\n".join(prompt_parts)


def build_pass1_schema() -> dict[str, Any]:
    """Build the JSON schema for Pass 1 (role-play) response.

    No scenario-specific outcomes here — just the universal fields.
    """
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Your honest first reaction in 2-4 sentences. Be direct — state what you think, not both sides.",
            },
            "public_statement": {
                "type": "string",
                "description": "What would you bluntly tell a friend about this? One strong sentence.",
            },
            "reasoning_summary": {
                "type": "string",
                "description": "A single sentence capturing your core reaction (for your own memory).",
            },
            "sentiment": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Your emotional reaction: -1 = very negative, 0 = neutral, 1 = very positive.",
            },
            "conviction": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "How sure are you? 0 = genuinely no idea what to think, 25 = starting to lean one way, 50 = clear opinion, 75 = quite sure and hard to change your mind, 100 = absolutely certain.",
            },
            "will_share": {
                "type": "boolean",
                "description": "Will you actively discuss or share this with others?",
            },
        },
        "required": [
            "reasoning",
            "public_statement",
            "reasoning_summary",
            "sentiment",
            "conviction",
            "will_share",
        ],
        "additionalProperties": False,
    }


# =============================================================================
# Pass 2: Classification prompt
# =============================================================================


def build_pass2_prompt(reasoning_text: str, scenario: ScenarioSpec) -> str:
    """Build the Pass 2 (classification) prompt.

    Takes the free-text reasoning from Pass 1 and asks a cheap model
    to classify it into scenario-defined outcome categories.

    Args:
        reasoning_text: The agent's reasoning from Pass 1
        scenario: Scenario specification (for outcome definitions)

    Returns:
        Classification prompt string
    """
    parts = [
        "You are a classification assistant. Given a person's reasoning about an event, "
        "extract the structured outcomes below.",
        "",
        "## The Person's Reasoning",
        "",
        reasoning_text,
        "",
        "## Classification Task",
        "",
        "Based on the reasoning above, classify this person's response into the categories below.",
        "Pick the option that BEST matches what they expressed. Do not infer beyond what they said.",
    ]

    if scenario.outcomes.extraction_instructions:
        parts.extend(
            [
                "",
                scenario.outcomes.extraction_instructions,
            ]
        )

    return "\n".join(parts)


def build_pass2_schema(outcomes: OutcomeConfig) -> dict[str, Any] | None:
    """Build JSON schema for Pass 2 (classification) from scenario outcomes.

    Only includes categorical, boolean, and float outcomes —
    these are the ones that need classification.

    Args:
        outcomes: Outcome configuration from scenario

    Returns:
        JSON schema dictionary, or None if no classifiable outcomes
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

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
            outcome_prop["type"] = "string"

        properties[outcome.name] = outcome_prop
        if outcome.required:
            required.append(outcome.name)

    if not properties:
        return None

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# =============================================================================
# Legacy single-pass schema (kept for backwards compatibility)
# =============================================================================


def build_response_schema(outcomes: OutcomeConfig) -> dict[str, Any]:
    """Build JSON schema from scenario outcomes (legacy single-pass).

    Kept for backwards compatibility. New code should use
    build_pass1_schema + build_pass2_schema.
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
            outcome_prop["type"] = "string"

        properties[outcome.name] = outcome_prop
        required.append(outcome.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# =============================================================================
# Prompt building (legacy compatibility wrapper)
# =============================================================================


def build_reasoning_prompt(
    context: ReasoningContext,
    scenario: ScenarioSpec,
) -> str:
    """Build the agent reasoning prompt (delegates to Pass 1).

    Kept as public API for backwards compatibility.
    """
    return build_pass1_prompt(context, scenario)


# =============================================================================
# Primary position outcome extraction
# =============================================================================


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


# =============================================================================
# Helper functions
# =============================================================================


def _sentiment_to_tone(sentiment: float) -> str:
    """Convert sentiment float to a natural language tone descriptor."""
    if sentiment >= 0.6:
        return "enthusiastic"
    elif sentiment >= 0.2:
        return "positive"
    elif sentiment >= -0.2:
        return "neutral"
    elif sentiment >= -0.6:
        return "skeptical"
    else:
        return "strongly opposed"


# =============================================================================
# Two-pass reasoning (async)
# =============================================================================


async def _reason_agent_two_pass_async(
    context: ReasoningContext,
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
    rate_limiter: Any = None,
) -> ReasoningResponse | None:
    """Two-pass async reasoning for an agent.

    Pass 1: Free-text role-play with main model
    Pass 2: Classification with cheap model (if scenario has classifiable outcomes)

    Args:
        context: Reasoning context
        scenario: Scenario specification
        config: Simulation run configuration
        rate_limiter: Optional DualRateLimiter for API pacing
            (uses .pivotal for Pass 1, .routine for Pass 2)

    Returns:
        ReasoningResponse with both passes merged, or None if failed
    """
    pass1_prompt = build_pass1_prompt(context, scenario)
    pass1_schema = build_pass1_schema()
    position_outcome = _get_primary_position_outcome(scenario)

    # Determine models
    main_model = config.model or None  # None = provider default
    classify_model = config.routine_model or None  # None = provider default (cheap)

    # === Pass 1: Role-play ===
    for attempt in range(config.max_retries):
        try:
            if rate_limiter:
                # Dynamic token estimate from prompt length (4 chars ≈ 1 token)
                estimated_input = len(pass1_prompt) // 4
                estimated_output = 300  # structured response
                await rate_limiter.pivotal.acquire(
                    estimated_input_tokens=estimated_input,
                    estimated_output_tokens=estimated_output,
                )

            call_start = time.time()
            pass1_response = await simple_call_async(
                prompt=pass1_prompt,
                response_schema=pass1_schema,
                schema_name="agent_reasoning",
                model=main_model,
            )
            call_elapsed = time.time() - call_start

            logger.info(f"[PASS1] Agent {context.agent_id} - {call_elapsed:.2f}s")

            if not pass1_response:
                continue

            break
        except Exception as e:
            logger.warning(
                f"[PASS1] Agent {context.agent_id} - attempt {attempt + 1} failed: {e}"
            )
            if attempt == config.max_retries - 1:
                return None
    else:
        return None

    # Extract Pass 1 fields
    reasoning = pass1_response.get("reasoning", "")
    public_statement = pass1_response.get("public_statement", "")
    reasoning_summary = pass1_response.get("reasoning_summary", "")
    sentiment = pass1_response.get("sentiment")
    conviction_score = pass1_response.get("conviction")
    will_share = pass1_response.get("will_share", False)

    # Map conviction score (0-100) to float via bucketing
    conviction_float = score_to_conviction_float(conviction_score)

    # === Pass 2: Classification (if needed) ===
    pass2_schema = build_pass2_schema(scenario.outcomes)
    position = None
    outcomes = {}

    if pass2_schema:
        pass2_prompt = build_pass2_prompt(reasoning, scenario)

        for attempt in range(config.max_retries):
            try:
                if rate_limiter:
                    # Dynamic token estimate from prompt length
                    estimated_input = len(pass2_prompt) // 4
                    estimated_output = 80  # classification is small
                    await rate_limiter.routine.acquire(
                        estimated_input_tokens=estimated_input,
                        estimated_output_tokens=estimated_output,
                    )

                call_start = time.time()
                pass2_response = await simple_call_async(
                    prompt=pass2_prompt,
                    response_schema=pass2_schema,
                    schema_name="classification",
                    model=classify_model,
                )
                call_elapsed = time.time() - call_start

                logger.info(f"[PASS2] Agent {context.agent_id} - {call_elapsed:.2f}s")

                if pass2_response:
                    outcomes = dict(pass2_response)
                    # Extract primary position from outcomes
                    if position_outcome and position_outcome in pass2_response:
                        position = pass2_response[position_outcome]
                    break
            except Exception as e:
                logger.warning(
                    f"[PASS2] Agent {context.agent_id} - attempt {attempt + 1} failed: {e}"
                )
                if attempt == config.max_retries - 1:
                    # Pass 2 failure is non-fatal — we still have Pass 1 data
                    logger.warning(
                        f"[PASS2] Agent {context.agent_id} - all retries exhausted, proceeding without classification"
                    )

    # Merge sentiment into outcomes for backwards compat
    if sentiment is not None:
        outcomes["sentiment"] = sentiment

    return ReasoningResponse(
        position=position,
        sentiment=sentiment,
        conviction=conviction_float,
        public_statement=public_statement,
        reasoning_summary=reasoning_summary,
        action_intent=outcomes.get("action_intent"),
        will_share=will_share,
        reasoning=reasoning,
        outcomes=outcomes,
    )


# =============================================================================
# Synchronous reasoning (kept for backwards compatibility / testing)
# =============================================================================


def reason_agent(
    context: ReasoningContext,
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
) -> ReasoningResponse | None:
    """Call LLM to get agent's reasoning and response (synchronous two-pass).

    Args:
        context: Reasoning context with persona, event, exposures
        scenario: Scenario specification
        config: Simulation run configuration

    Returns:
        ReasoningResponse with extracted outcomes, or None if failed
    """
    pass1_prompt = build_pass1_prompt(context, scenario)
    pass1_schema = build_pass1_schema()
    position_outcome = _get_primary_position_outcome(scenario)

    logger.info(f"[REASON] Agent {context.agent_id} - preparing two-pass LLM call")
    logger.info(f"[REASON] Agent {context.agent_id} - model: {config.model}")
    logger.info(
        f"[REASON] Agent {context.agent_id} - prompt length: {len(pass1_prompt)} chars"
    )
    logger.debug(
        f"[REASON] Agent {context.agent_id} - PROMPT:\n{pass1_prompt[:500]}..."
    )

    # === Pass 1: Role-play ===
    pass1_response = None
    for attempt in range(config.max_retries):
        try:
            logger.info(
                f"[PASS1] Agent {context.agent_id} - attempt {attempt + 1}/{config.max_retries}"
            )

            call_start = time.time()
            pass1_response = simple_call(
                prompt=pass1_prompt,
                response_schema=pass1_schema,
                schema_name="agent_reasoning",
                model=config.model or None,
                log=True,
            )
            call_elapsed = time.time() - call_start

            logger.info(
                f"[PASS1] Agent {context.agent_id} - API call took {call_elapsed:.2f}s"
            )

            if not pass1_response:
                logger.warning(
                    f"[PASS1] Agent {context.agent_id} - Empty response, attempt {attempt + 1}"
                )
                continue

            break
        except Exception as e:
            logger.warning(
                f"[PASS1] Agent {context.agent_id} - EXCEPTION attempt {attempt + 1}: {e}"
            )
            if attempt == config.max_retries - 1:
                logger.error(
                    f"[PASS1] Agent {context.agent_id} - All retries exhausted"
                )
                return None

    if not pass1_response:
        return None

    # Extract Pass 1 fields
    reasoning = pass1_response.get("reasoning", "")
    public_statement = pass1_response.get("public_statement", "")
    reasoning_summary = pass1_response.get("reasoning_summary", "")
    sentiment = pass1_response.get("sentiment")
    conviction_score = pass1_response.get("conviction")
    will_share = pass1_response.get("will_share", False)
    conviction_float = score_to_conviction_float(conviction_score)

    # === Pass 2: Classification ===
    pass2_schema = build_pass2_schema(scenario.outcomes)
    position = None
    outcomes = {}

    if pass2_schema:
        pass2_prompt = build_pass2_prompt(reasoning, scenario)
        classify_model = config.routine_model or None

        for attempt in range(config.max_retries):
            try:
                call_start = time.time()
                pass2_response = simple_call(
                    prompt=pass2_prompt,
                    response_schema=pass2_schema,
                    schema_name="classification",
                    model=classify_model,
                    log=True,
                )
                call_elapsed = time.time() - call_start

                logger.info(
                    f"[PASS2] Agent {context.agent_id} - API call took {call_elapsed:.2f}s"
                )

                if pass2_response:
                    outcomes = dict(pass2_response)
                    if position_outcome and position_outcome in pass2_response:
                        position = pass2_response[position_outcome]
                    break
            except Exception as e:
                logger.warning(
                    f"[PASS2] Agent {context.agent_id} - attempt {attempt + 1} failed: {e}"
                )
                if attempt == config.max_retries - 1:
                    logger.warning(
                        f"[PASS2] Agent {context.agent_id} - classification failed, continuing without"
                    )

    if sentiment is not None:
        outcomes["sentiment"] = sentiment

    logger.info(
        f"[REASON] Agent {context.agent_id} - SUCCESS: position={position}, "
        f"sentiment={sentiment}, conviction={conviction_score}→{float_to_conviction(conviction_float)}, will_share={will_share}"
    )

    return ReasoningResponse(
        position=position,
        sentiment=sentiment,
        conviction=conviction_float,
        public_statement=public_statement,
        reasoning_summary=reasoning_summary,
        action_intent=outcomes.get("action_intent"),
        will_share=will_share,
        reasoning=reasoning,
        outcomes=outcomes,
    )


# =============================================================================
# Batch reasoning
# =============================================================================


def batch_reason_agents(
    contexts: list[ReasoningContext],
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
    max_concurrency: int = 50,
    rate_limiter: Any = None,
    on_agent_done: Callable[[str, ReasoningResponse | None], None] | None = None,
) -> list[tuple[str, ReasoningResponse | None]]:
    """Reason multiple agents concurrently using asyncio with two-pass reasoning.

    Args:
        contexts: List of reasoning contexts
        scenario: Scenario specification
        config: Simulation run configuration
        max_concurrency: Max concurrent API calls (default 50, fallback if no rate limiter)
        rate_limiter: Optional DualRateLimiter instance for API pacing
        on_agent_done: Optional callback(agent_id, response) called per agent after reasoning

    Returns:
        List of (agent_id, response) tuples in original order
    """
    import asyncio

    if not contexts:
        return []

    total = len(contexts)
    logger.info(f"[BATCH] Starting two-pass async reasoning for {total} agents")

    async def run_all():
        # Always use a semaphore to cap concurrent tasks.
        # When rate limiter is available, size it from max_safe_concurrent.
        if rate_limiter:
            concurrency = rate_limiter.max_safe_concurrent
            # Stagger interval: spread launches across the RPM window
            # e.g. 500 RPM → 8.3 req/s → 120ms between launches
            stagger_interval = 60.0 / rate_limiter.pivotal.rpm
            logger.info(
                f"[BATCH] Concurrency cap: {concurrency}, "
                f"stagger: {stagger_interval * 1000:.0f}ms between launches"
            )
        else:
            concurrency = max_concurrency
            stagger_interval = 0.0
        semaphore = asyncio.Semaphore(concurrency)
        completed = [0]

        async def reason_with_pacing(
            ctx: ReasoningContext,
        ) -> tuple[str, ReasoningResponse | None]:
            async with semaphore:
                start = time.time()
                result = await _reason_agent_two_pass_async(
                    ctx, scenario, config, rate_limiter
                )
                elapsed = time.time() - start
                completed[0] += 1

                if result:
                    logger.info(
                        f"[BATCH] {completed[0]}/{total}: {ctx.agent_id} done in {elapsed:.2f}s "
                        f"(position={result.position}, sentiment={result.sentiment}, "
                        f"conviction={float_to_conviction(result.conviction)})"
                    )
                else:
                    logger.warning(
                        f"[BATCH] {completed[0]}/{total}: {ctx.agent_id} FAILED"
                    )

                if on_agent_done:
                    on_agent_done(ctx.agent_id, result)

                return (ctx.agent_id, result)

        # Stagger task launches to avoid burst of requests hitting API at once.
        # Each task is created with a small delay so they don't all enter
        # the semaphore simultaneously.
        tasks = []
        for i, ctx in enumerate(contexts):
            tasks.append(asyncio.create_task(reason_with_pacing(ctx)))
            if stagger_interval > 0 and i < concurrency - 1:
                # Only stagger the first batch — after that the semaphore
                # naturally gates as tasks complete and new ones enter
                await asyncio.sleep(stagger_interval)

        results = await asyncio.gather(*tasks)

        # Close the async HTTP client before the event loop shuts down.
        # Without this, orphaned httpx connections produce "Event loop is
        # closed" errors during garbage collection.
        await close_simulation_provider()

        return results

    batch_start = time.time()
    results = asyncio.run(run_all())
    batch_elapsed = time.time() - batch_start

    logger.info(
        f"[BATCH] Completed {total} agents in {batch_elapsed:.2f}s ({batch_elapsed / total:.2f}s/agent avg)"
    )

    if rate_limiter:
        stats = rate_limiter.stats()
        pivotal_stats = stats.get("pivotal", stats)
        routine_stats = stats.get("routine", pivotal_stats)
        total_acquired = pivotal_stats.get("total_acquired", 0) + routine_stats.get(
            "total_acquired", 0
        )
        total_wait = pivotal_stats.get(
            "total_wait_time_seconds", 0
        ) + routine_stats.get("total_wait_time_seconds", 0)
        logger.info(
            f"[BATCH] Rate limiter: {total_acquired} acquired, "
            f"{total_wait:.2f}s total wait"
        )

    return list(results)


# =============================================================================
# Context creation helper
# =============================================================================


def create_reasoning_context(
    agent_id: str,
    agent: dict[str, Any],
    persona: str,
    exposures: list[ExposureRecord],
    scenario: ScenarioSpec,
    peer_opinions: list[PeerOpinion] | None = None,
    current_state: Any = None,
    memory_trace: list[MemoryEntry] | None = None,
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
        memory_trace: Optional memory trace entries

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
        memory_trace=memory_trace or [],
    )
