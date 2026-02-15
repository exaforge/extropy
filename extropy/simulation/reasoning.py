"""Agent reasoning via LLM for simulation.

Implements two-pass reasoning:
- Pass 1 (role-play): Free-text reasoning with no categorical constraints.
  The agent reasons naturally — sentiment, conviction, public statement.
- Pass 2 (classification): A cheap model classifies the free-text into
  scenario-defined categorical/boolean/float outcomes.

This split fixes the central tendency problem where 83% of agents chose
safe middle options when role-play and classification were combined.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.llm import simple_call, simple_call_async, TokenUsage
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


@dataclass
class BatchTokenUsage:
    """Accumulated token usage from a batch of two-pass reasoning calls."""

    pivotal_input_tokens: int = 0
    pivotal_output_tokens: int = 0
    routine_input_tokens: int = 0
    routine_output_tokens: int = 0


# =============================================================================
# Pass 1: Free-text role-play prompt
# =============================================================================


def build_pass1_prompt(
    context: ReasoningContext,
    scenario: ScenarioSpec,
    fidelity: str = "medium",
) -> str:
    """Build the Pass 1 (role-play) prompt.

    Named, temporal, accountable prompt structure. No categorical enums —
    the agent reasons naturally about the event, forms a sentiment,
    conviction level, and public statement.

    Args:
        context: Reasoning context with persona and exposure history
        scenario: Scenario specification
        fidelity: Memory rendering depth — "low", "medium", or "high"

    Returns:
        Complete prompt string for LLM
    """
    is_re_reasoning = bool(context.memory_trace)
    prompt_parts: list[str] = []

    # --- System framing ---
    if context.agent_name:
        prompt_parts.append(
            f"You are going to think as {context.agent_name}. Everything below is from "
            f"{context.agent_name}'s perspective. Respond as {context.agent_name} — "
            "first person, honest, unfiltered."
        )
    else:
        prompt_parts.append(
            "Think as the person described below. Everything is from their perspective. "
            "Respond in first person — honest, unfiltered."
        )
    prompt_parts.extend(
        [
            "IMPORTANT: Do NOT hedge, equivocate, or give a 'balanced' take. Real people have clear reactions.",
            "Your background and circumstances determine your response — not politeness or caution.",
            "",
        ]
    )

    # --- Persona ---
    prompt_parts.extend([context.persona, ""])

    # --- Temporal header ---
    timestep_label = f"{context.timestep_unit} {context.timestep + 1}"
    prompt_parts.extend(
        [
            f"## It's {timestep_label} since {scenario.event.source} announced:",
            "",
            scenario.event.content,
            "",
        ]
    )

    # --- Background context ---
    if context.background_context:
        prompt_parts.extend(
            [
                "## Background",
                "",
                context.background_context,
                "",
            ]
        )

    # --- Exposure history (named + experiential) ---
    prompt_parts.extend(["## How This Reached You", ""])

    # Build channel lookup for experience templates
    channel_map = {ch.name: ch for ch in scenario.seed_exposure.channels}

    for exp in context.exposure_history:
        if exp.source_agent_id:
            # Network exposure — try to find peer name
            peer_name = None
            for peer in context.peer_opinions:
                if peer.agent_id == exp.source_agent_id and peer.peer_name:
                    peer_name = peer.peer_name
                    break
            if peer_name:
                prompt_parts.append(f'- {peer_name} told me: "{exp.content}"')
            else:
                source_name = context.agent_names.get(
                    exp.source_agent_id,
                    f"agent-{exp.source_agent_id[:6]}",
                )
                prompt_parts.append(f'- {source_name} told me: "{exp.content}"')
        else:
            # Seed/channel exposure
            channel = channel_map.get(exp.channel)
            if channel and channel.experience_template:
                text = channel.experience_template.format(
                    channel_name=channel.description or channel.name
                )
                prompt_parts.append(f"- {text}")
            else:
                # Humanize the channel name
                display = (exp.channel or "unknown").replace("_", " ")
                prompt_parts.append(f"- I heard about this via {display}")

    # --- Peer opinions (named) ---
    if context.peer_opinions:
        prompt_parts.extend(["", "## What People Around Me Are Saying", ""])
        for peer in context.peer_opinions:
            name_label = peer.peer_name or f"A {peer.relationship}"
            rel_label = f" (my {peer.relationship})" if peer.peer_name else ""
            if peer.public_statement:
                prompt_parts.append(
                    f'- {name_label}{rel_label}: "{peer.public_statement}"'
                )
            elif peer.sentiment is not None:
                tone = _sentiment_to_tone(peer.sentiment)
                prompt_parts.append(
                    f"- {name_label}{rel_label} seems {tone} about this"
                )

    # --- Local mood ---
    if context.local_mood_summary:
        prompt_parts.extend(["", context.local_mood_summary])

    # --- Macro summary ---
    if context.macro_summary:
        prompt_parts.extend(["", context.macro_summary])

    # --- Memory trace (full, uncapped, fidelity-gated) ---
    if context.memory_trace:
        prompt_parts.extend(["", "## What I've Been Thinking", ""])

        # Determine rendering parameters by fidelity
        if fidelity == "low":
            max_summary_steps = 5
            raw_excerpt_steps = 0
            raw_excerpt_tokens = 0
        elif fidelity == "high":
            max_summary_steps = len(context.memory_trace)
            raw_excerpt_steps = 5
            raw_excerpt_tokens = 200
        else:  # medium (default)
            max_summary_steps = len(context.memory_trace)
            raw_excerpt_steps = 3
            raw_excerpt_tokens = 120

        # Show summaries (possibly limited for low fidelity)
        display_memories = context.memory_trace[-max_summary_steps:]
        recent_cutoff = len(context.memory_trace) - raw_excerpt_steps

        for i, memory in enumerate(display_memories):
            # Compute the original index in the full trace
            original_idx = len(context.memory_trace) - len(display_memories) + i
            conviction_label = float_to_conviction(memory.conviction) or "uncertain"
            unit = context.timestep_unit
            prompt_parts.append(
                f'- {unit} {memory.timestep + 1}: "{memory.summary}" '
                f"(conviction: {conviction_label})"
            )
            # Show raw excerpt for recent steps in medium/high fidelity
            if (
                raw_excerpt_steps > 0
                and original_idx >= recent_cutoff
                and memory.raw_reasoning
            ):
                # Truncate to approximate token budget (1 token ≈ 4 chars)
                max_chars = raw_excerpt_tokens * 4
                excerpt = memory.raw_reasoning[:max_chars]
                if len(memory.raw_reasoning) > max_chars:
                    excerpt = excerpt.rsplit(" ", 1)[0] + "..."
                prompt_parts.append(f'  In my own words: "{excerpt}"')

    # --- Emotional trajectory ---
    if is_re_reasoning and len(context.memory_trace) >= 2:
        sentiments = [
            m.sentiment for m in context.memory_trace if m.sentiment is not None
        ]
        if len(sentiments) >= 2:
            trend = sentiments[-1] - sentiments[0]
            if trend > 0.3:
                trajectory = "increasingly positive"
            elif trend < -0.3:
                trajectory = "increasingly negative"
            elif abs(sentiments[-1]) < 0.2:
                trajectory = "mostly neutral"
            else:
                trajectory = "fairly steady"
            prompt_parts.append(f"\nI've been feeling {trajectory} since this started.")

    # --- Intent accountability ---
    if is_re_reasoning and context.prior_action_intent:
        prompt_parts.extend(
            [
                "",
                f'Last time I said I intended to: "{context.prior_action_intent}". '
                "Has anything changed?",
            ]
        )

    # --- Instructions ---
    if is_re_reasoning:
        prompt_parts.extend(
            [
                "",
                "## Your Honest Reaction",
                "",
                "Consider whether your thinking has changed given new information.",
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
                "Think honestly about how this lands for someone in your exact situation.",
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
    main_model = config.strong or None  # None = provider default
    classify_model = config.fast or None  # None = provider default (cheap)

    # === Pass 1: Role-play ===
    pass1_usage = TokenUsage()
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
            pass1_response, pass1_usage = await asyncio.wait_for(
                simple_call_async(
                    prompt=pass1_prompt,
                    response_schema=pass1_schema,
                    schema_name="agent_reasoning",
                    model=main_model,
                ),
                timeout=30.0,
            )
            call_elapsed = time.time() - call_start

            logger.info(f"[PASS1] Agent {context.agent_id} - {call_elapsed:.2f}s")

            if not pass1_response:
                continue

            break
        except asyncio.TimeoutError:
            logger.warning(
                f"[PASS1] Agent {context.agent_id} - attempt {attempt + 1} timed out after 30s"
            )
            if attempt == config.max_retries - 1:
                return None
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
    if sentiment is not None:
        sentiment = max(-1.0, min(1.0, float(sentiment)))
    conviction_score = pass1_response.get("conviction")
    will_share = pass1_response.get("will_share", False)

    # Map conviction score (0-100) to float via bucketing
    conviction_float = score_to_conviction_float(conviction_score)

    # === Pass 2: Classification (if needed) ===
    pass2_schema = build_pass2_schema(scenario.outcomes)
    position = None
    outcomes = {}
    pass2_usage = TokenUsage()

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
                pass2_response, pass2_usage = await asyncio.wait_for(
                    simple_call_async(
                        prompt=pass2_prompt,
                        response_schema=pass2_schema,
                        schema_name="classification",
                        model=classify_model,
                    ),
                    timeout=20.0,
                )
                call_elapsed = time.time() - call_start

                logger.info(f"[PASS2] Agent {context.agent_id} - {call_elapsed:.2f}s")

                if pass2_response:
                    outcomes = dict(pass2_response)
                    # Extract primary position from outcomes
                    if position_outcome and position_outcome in pass2_response:
                        position = pass2_response[position_outcome]
                    break
            except asyncio.TimeoutError:
                logger.warning(
                    f"[PASS2] Agent {context.agent_id} - attempt {attempt + 1} timed out after 20s"
                )
                if attempt == config.max_retries - 1:
                    logger.warning(
                        f"[PASS2] Agent {context.agent_id} - all retries exhausted, proceeding without classification"
                    )
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
        pass1_input_tokens=pass1_usage.input_tokens,
        pass1_output_tokens=pass1_usage.output_tokens,
        pass2_input_tokens=pass2_usage.input_tokens,
        pass2_output_tokens=pass2_usage.output_tokens,
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
        classify_model = config.fast or None

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


async def batch_reason_agents_async(
    contexts: list[ReasoningContext],
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
    max_concurrency: int = 50,
    rate_limiter: Any = None,
    on_agent_done: Callable[[str, ReasoningResponse | None], None] | None = None,
) -> tuple[list[tuple[str, ReasoningResponse | None]], BatchTokenUsage]:
    """Reason multiple agents concurrently with two-pass reasoning.

    This is an async coroutine — call from within an existing event loop.
    The caller is responsible for provider cleanup when the loop ends.

    Args:
        contexts: List of reasoning contexts
        scenario: Scenario specification
        config: Simulation run configuration
        max_concurrency: Max concurrent API calls (None/0 = auto from rate limiter)
        rate_limiter: Optional DualRateLimiter instance for API pacing
        on_agent_done: Optional callback(agent_id, response) called per agent after reasoning

    Returns:
        Tuple of (results, batch_token_usage) where results is a list of
        (agent_id, response) tuples in original order.
    """
    if not contexts:
        return [], BatchTokenUsage()

    total = len(contexts)
    logger.info(f"[REASONING] Starting two-pass async reasoning for {total} agents")

    if rate_limiter:
        rpm_derived = rate_limiter.max_safe_concurrent
        if max_concurrency:
            target_concurrency = min(max(1, rpm_derived), max(1, max_concurrency))
        else:
            target_concurrency = max(1, rpm_derived)
        stagger_interval = 60.0 / rate_limiter.pivotal.rpm
        logger.info(
            f"[REASONING] Concurrency cap: {target_concurrency} "
            f"(rpm={rate_limiter.pivotal.rpm}, rpm_derived={rpm_derived}), "
            f"stagger: {stagger_interval * 1000:.0f}ms between launches"
        )
    else:
        target_concurrency = max(1, max_concurrency or 50)
        stagger_interval = 0.0
    completed = [0]

    async def reason_with_pacing(
        idx: int,
        ctx: ReasoningContext,
    ) -> tuple[int, str, ReasoningResponse | None, float]:
        start = time.time()
        result = await _reason_agent_two_pass_async(ctx, scenario, config, rate_limiter)
        elapsed = time.time() - start
        completed[0] += 1

        if result:
            logger.info(
                f"[REASONING] {completed[0]}/{total}: {ctx.agent_id} done in {elapsed:.2f}s "
                f"(position={result.position}, sentiment={result.sentiment}, "
                f"conviction={float_to_conviction(result.conviction)})"
            )
        else:
            logger.warning(f"[REASONING] {completed[0]}/{total}: {ctx.agent_id} FAILED")

        if on_agent_done:
            on_agent_done(ctx.agent_id, result)

        return (idx, ctx.agent_id, result, elapsed)

    semaphore = asyncio.Semaphore(target_concurrency)

    async def bounded_reason(idx: int, ctx: ReasoningContext):
        async with semaphore:
            return await reason_with_pacing(idx, ctx)

    results: list[tuple[str, ReasoningResponse | None] | None] = [None] * total
    tasks = []
    for i, ctx in enumerate(contexts):
        tasks.append(asyncio.create_task(bounded_reason(i, ctx)))
        if stagger_interval > 0 and i < total - 1:
            await asyncio.sleep(stagger_interval)

    raw_results = await asyncio.gather(*tasks)
    for idx, agent_id, result, elapsed in raw_results:
        results[idx] = (agent_id, result)

    pair_results = [r for r in results if r is not None]

    logger.info(
        f"[REASONING] Completed {total} agents "
        f"({len(pair_results)} succeeded, {total - len(pair_results)} failed)"
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
            f"[REASONING] Rate limiter: {total_acquired} acquired, "
            f"{total_wait:.2f}s total wait"
        )

    # Accumulate token usage from all successful responses
    batch_usage = BatchTokenUsage()
    for _, response in pair_results:
        if response is not None:
            batch_usage.pivotal_input_tokens += response.pass1_input_tokens
            batch_usage.pivotal_output_tokens += response.pass1_output_tokens
            batch_usage.routine_input_tokens += response.pass2_input_tokens
            batch_usage.routine_output_tokens += response.pass2_output_tokens

    return list(pair_results), batch_usage


def batch_reason_agents(
    contexts: list[ReasoningContext],
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
    max_concurrency: int = 50,
    rate_limiter: Any = None,
    on_agent_done: Callable[[str, ReasoningResponse | None], None] | None = None,
) -> tuple[list[tuple[str, ReasoningResponse | None]], BatchTokenUsage]:
    """Sync wrapper around batch_reason_agents_async.

    Runs a single-use event loop. Prefer batch_reason_agents_async when
    an event loop is already running (e.g. the engine's chunk loop).
    """
    import asyncio

    from ..core.providers import close_simulation_provider

    async def _run():
        try:
            return await batch_reason_agents_async(
                contexts, scenario, config, max_concurrency, rate_limiter, on_agent_done
            )
        finally:
            await close_simulation_provider()

    return asyncio.run(_run())


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
