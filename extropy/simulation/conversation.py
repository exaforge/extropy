"""Agent-agent and agent-NPC conversation system for simulation.

Phase D implements multi-turn LLM-driven conversations between agents.
Conversations can be initiated during reasoning when agents choose to
talk_to someone, and the results override provisional reasoning state.

This module provides:
- ConversationRequest: Request to start a conversation
- ConversationMessage: Single message in a conversation
- ConversationResult: Complete conversation with state changes
- Functions for collecting, prioritizing, and executing conversations
"""

import asyncio
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..core.llm import simple_call_async
from ..core.models import (
    ReasoningContext,
    ReasoningResponse,
    ScenarioSpec,
    SimulationRunConfig,
    score_to_conviction_float,
)
from ..core.models.scenario import DEFAULT_RELATIONSHIP_WEIGHTS

logger = logging.getLogger(__name__)


# =============================================================================
# Conversation Models
# =============================================================================


class ConversationRequest(BaseModel):
    """Request to initiate a conversation."""

    initiator_id: str = Field(description="Agent who wants to talk")
    target_id: str = Field(description="Agent or NPC to talk to")
    target_is_npc: bool = Field(default=False, description="Whether target is NPC")
    target_name: str | None = Field(default=None, description="Target's name")
    topic: str | None = Field(default=None, description="What to discuss")
    priority_score: float = Field(
        default=0.5, description="Priority = edge_weight × relationship_weight"
    )
    relationship: str = Field(default="contact", description="Relationship type")


class ConversationMessage(BaseModel):
    """A single message in a conversation."""

    speaker_id: str = Field(description="Who spoke")
    speaker_name: str = Field(description="Speaker's name")
    content: str = Field(description="Message content")
    turn: int = Field(description="Turn number (0-indexed)")
    is_final: bool = Field(default=False, description="Is this the last message")


class ConversationStateChange(BaseModel):
    """State changes for a participant after conversation."""

    sentiment: float | None = Field(default=None)
    conviction: float | None = Field(default=None)
    position: str | None = Field(default=None)
    internal_reaction: str | None = Field(default=None)


class ConversationResult(BaseModel):
    """Complete result of a conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    initiator_id: str
    target_id: str
    target_is_npc: bool = False
    target_npc_profile: dict[str, Any] | None = None
    messages: list[ConversationMessage] = Field(default_factory=list)
    initiator_state_change: ConversationStateChange | None = None
    target_state_change: ConversationStateChange | None = None
    timestep: int = 0
    priority_score: float = 0.0


# =============================================================================
# Conversation Collection
# =============================================================================


def collect_conversation_requests(
    reasoning_results: list[tuple[str, ReasoningResponse]],
    adjacency: dict[str, list[tuple[str, dict]]],
    agent_map: dict[str, dict[str, Any]],
    relationship_weights: dict[str, float] | None = None,
    households: list[dict[str, Any]] | None = None,
) -> list[ConversationRequest]:
    """Collect talk_to requests from reasoning results.

    Args:
        reasoning_results: List of (agent_id, response) pairs
        adjacency: Network adjacency dict
        agent_map: Agent ID → agent dict mapping
        relationship_weights: Scenario relationship weights (or use defaults)
        households: Optional household data for NPC resolution

    Returns:
        List of conversation requests sorted by priority
    """
    weights = relationship_weights or DEFAULT_RELATIONSHIP_WEIGHTS
    requests: list[ConversationRequest] = []

    # Build name → agent_id lookup for resolving names
    name_to_id: dict[str, str] = {}
    for aid, agent in agent_map.items():
        name = agent.get("first_name", "").lower()
        if name:
            name_to_id[name] = aid

    # Build household NPC lookup
    npc_by_agent: dict[str, list[dict]] = {}
    if households:
        for hh in households:
            for adult_id in hh.get("adult_ids", []):
                if adult_id not in npc_by_agent:
                    npc_by_agent[adult_id] = []
                for dep in hh.get("dependent_data", []):
                    npc_by_agent[adult_id].append(
                        {
                            "name": dep.get("name", "family member"),
                            "relationship": dep.get("relationship", "dependent"),
                            "age": dep.get("age"),
                            "description": dep.get("description", ""),
                        }
                    )

    for agent_id, response in reasoning_results:
        if response is None or not response.actions:
            continue

        agent = agent_map.get(agent_id, {})
        neighbors = adjacency.get(agent_id, [])
        neighbor_map = {nid: edge for nid, edge in neighbors}

        for action in response.actions:
            if action.get("type") != "talk_to":
                continue

            who = action.get("who", "").strip()
            topic = action.get("topic")

            if not who:
                continue

            # Try to resolve who to an agent_id
            target_id = None
            target_is_npc = False
            target_name = who
            relationship = "contact"
            priority = 0.5

            # Check if it's a network neighbor by name
            who_lower = who.lower()
            if who_lower in name_to_id:
                target_id = name_to_id[who_lower]
                if target_id in neighbor_map:
                    edge = neighbor_map[target_id]
                    relationship = edge.get("type", "contact")
                    edge_weight = edge.get("weight", 0.5)
                    rel_weight = weights.get(relationship, 0.3)
                    priority = edge_weight * rel_weight

            # Check if it's an NPC (partner, kid, etc.)
            if target_id is None:
                # Check agent's partner_npc field
                partner_npc = agent.get("partner_npc")
                if partner_npc:
                    partner_name = partner_npc.get("name", "").lower()
                    if partner_name == who_lower or who_lower in ("partner", "spouse"):
                        target_id = f"npc_{agent_id}_partner"
                        target_is_npc = True
                        target_name = partner_npc.get("name", "Partner")
                        relationship = "partner"
                        priority = weights.get("partner", 1.0)

                # Check household NPCs
                if target_id is None and agent_id in npc_by_agent:
                    for npc in npc_by_agent[agent_id]:
                        npc_name = npc.get("name", "").lower()
                        if npc_name == who_lower:
                            target_id = f"npc_{agent_id}_{npc_name}"
                            target_is_npc = True
                            target_name = npc.get("name")
                            relationship = npc.get("relationship", "household")
                            priority = weights.get(relationship, 0.9)
                            break

            # If still not found, maybe it's a generic reference
            if target_id is None:
                # Check if "partner" or "spouse" without specific name
                if who_lower in ("my partner", "partner", "spouse", "my spouse"):
                    partner_npc = agent.get("partner_npc")
                    if partner_npc:
                        target_id = f"npc_{agent_id}_partner"
                        target_is_npc = True
                        target_name = partner_npc.get("name", "Partner")
                        relationship = "partner"
                        priority = weights.get("partner", 1.0)
                    elif agent.get("partner_agent_id"):
                        target_id = agent["partner_agent_id"]
                        relationship = "partner"
                        priority = weights.get("partner", 1.0)

            if target_id is None:
                logger.debug(
                    f"[CONV] Agent {agent_id} wants to talk to '{who}' but couldn't resolve target"
                )
                continue

            requests.append(
                ConversationRequest(
                    initiator_id=agent_id,
                    target_id=target_id,
                    target_is_npc=target_is_npc,
                    target_name=target_name,
                    topic=topic,
                    priority_score=priority,
                    relationship=relationship,
                )
            )

    # Sort by priority descending
    requests.sort(key=lambda r: r.priority_score, reverse=True)
    return requests


def prioritize_and_resolve_conflicts(
    requests: list[ConversationRequest],
    fidelity: str = "medium",
) -> tuple[list[list[ConversationRequest]], list[ConversationRequest]]:
    """Prioritize requests and resolve conflicts.

    Two agents can't both talk to the same third party at the same time.
    Higher priority requests win.

    Args:
        requests: Sorted list of conversation requests
        fidelity: Fidelity level (controls max conversations per agent)

    Returns:
        Tuple of (parallel_batches, deferred_requests)
        - parallel_batches: Lists of non-conflicting requests that can run in parallel
        - deferred_requests: Requests that were skipped due to conflicts
    """
    max_per_agent = 1 if fidelity == "medium" else (2 if fidelity == "high" else 0)

    if max_per_agent == 0:
        return [], requests

    # Track which agents are busy
    busy_agents: set[str] = set()
    agent_conv_count: dict[str, int] = {}
    selected: list[ConversationRequest] = []
    deferred: list[ConversationRequest] = []

    for req in requests:
        # Skip if initiator already has enough conversations
        if agent_conv_count.get(req.initiator_id, 0) >= max_per_agent:
            deferred.append(req)
            continue

        # Skip if target (non-NPC) is busy or has enough conversations
        if not req.target_is_npc:
            if req.target_id in busy_agents:
                deferred.append(req)
                continue
            if agent_conv_count.get(req.target_id, 0) >= max_per_agent:
                deferred.append(req)
                continue

        # Accept this request
        selected.append(req)
        busy_agents.add(req.initiator_id)
        if not req.target_is_npc:
            busy_agents.add(req.target_id)
        agent_conv_count[req.initiator_id] = (
            agent_conv_count.get(req.initiator_id, 0) + 1
        )
        if not req.target_is_npc:
            agent_conv_count[req.target_id] = agent_conv_count.get(req.target_id, 0) + 1

    # All selected can run in parallel (they don't conflict)
    return [selected] if selected else [], deferred


# =============================================================================
# Conversation Execution
# =============================================================================


def _build_conversation_prompt(
    speaker_name: str,
    speaker_persona: str,
    partner_name: str,
    partner_relationship: str,
    topic: str | None,
    prior_messages: list[ConversationMessage],
    scenario: ScenarioSpec,
    is_final: bool,
    is_initiator: bool,
) -> str:
    """Build prompt for a conversation turn."""
    parts = [
        f"You are {speaker_name}. You're having a conversation with {partner_name} "
        f"(your {partner_relationship}) about recent news.",
        "",
        speaker_persona,
        "",
        "## Background",
        "",
        f"{scenario.event.source} announced: {scenario.event.content}",
    ]

    if scenario.background_context:
        parts.extend(["", scenario.background_context])

    if topic:
        parts.extend(
            [
                "",
                "## What you want to discuss",
                "",
                topic if is_initiator else f"{partner_name} wants to discuss: {topic}",
            ]
        )

    if prior_messages:
        parts.extend(["", "## Conversation so far", ""])
        for msg in prior_messages:
            parts.append(f'{msg.speaker_name}: "{msg.content}"')
        parts.append("")

    if is_final:
        parts.extend(
            [
                "## Your turn (final)",
                "",
                "This is your last message. Wrap up the conversation naturally.",
            ]
        )
    else:
        parts.extend(
            [
                "## Your turn",
                "",
                f"Respond naturally as {speaker_name} would.",
            ]
        )

    return "\n".join(parts)


def _build_conversation_schema() -> dict[str, Any]:
    """Build JSON schema for conversation turn response."""
    return {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "What you say in the conversation (1-3 sentences)",
            },
            "internal_reaction": {
                "type": "string",
                "description": "Your private thought about how this conversation is going",
            },
            "updated_sentiment": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "How you feel now: -1 = very negative, 0 = neutral, 1 = very positive",
            },
            "updated_conviction": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "How sure you are now: 0-100",
            },
        },
        "required": [
            "response",
            "internal_reaction",
            "updated_sentiment",
            "updated_conviction",
        ],
        "additionalProperties": False,
    }


def _build_npc_prompt(
    npc_name: str,
    npc_profile: dict[str, Any],
    initiator_name: str,
    topic: str | None,
    prior_messages: list[ConversationMessage],
    scenario: ScenarioSpec,
    is_final: bool,
) -> str:
    """Build prompt for NPC conversation turn."""
    age = npc_profile.get("age", "")
    relationship = npc_profile.get("relationship", "family member")
    description = npc_profile.get("description", "")

    age_str = f", a {age}-year-old" if age else ""

    parts = [
        f"You are {npc_name}{age_str} {relationship}.",
    ]

    if description:
        parts.append(description)

    parts.extend(
        [
            "",
            f"{initiator_name} wants to talk to you about recent news.",
            "",
            "## Background",
            "",
            f"{scenario.event.source} announced: {scenario.event.content}",
        ]
    )

    if topic:
        parts.extend(["", f"Topic: {topic}"])

    if prior_messages:
        parts.extend(["", "## Conversation so far", ""])
        for msg in prior_messages:
            parts.append(f'{msg.speaker_name}: "{msg.content}"')
        parts.append("")

    if is_final:
        parts.append("Respond briefly (1-2 sentences). This wraps up the conversation.")
    else:
        parts.append("Respond naturally (1-2 sentences).")

    return "\n".join(parts)


def _build_npc_schema() -> dict[str, Any]:
    """Build JSON schema for NPC conversation response (simpler than agent)."""
    return {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "What you say (1-2 sentences)",
            },
        },
        "required": ["response"],
        "additionalProperties": False,
    }


def _message_count_for_fidelity(fidelity: str) -> int:
    """Get total message count for a conversation at the given fidelity."""
    if fidelity == "high":
        return 6
    if fidelity == "medium":
        return 4
    return 2


async def execute_conversation_async(
    request: ConversationRequest,
    initiator_context: ReasoningContext,
    target_context: ReasoningContext | None,
    target_npc_profile: dict[str, Any] | None,
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
    rate_limiter: Any = None,
    timestep: int = 0,
) -> ConversationResult:
    """Execute a multi-turn conversation between agents or agent-NPC.

    Args:
        request: The conversation request
        initiator_context: Initiator's reasoning context (for persona)
        target_context: Target's reasoning context (None for NPC)
        target_npc_profile: NPC profile if target is NPC
        scenario: Scenario spec
        config: Simulation config
        rate_limiter: Optional rate limiter
        timestep: Current simulation timestep

    Returns:
        ConversationResult with messages and state changes
    """
    # Determine number of messages based on fidelity
    total_messages = _message_count_for_fidelity(config.fidelity)

    messages: list[ConversationMessage] = []
    initiator_name = initiator_context.agent_name or "Agent"
    target_name = request.target_name or "Partner"

    # Track state changes
    initiator_sentiment = None
    initiator_conviction = None
    target_sentiment = None
    target_conviction = None
    initiator_reaction = None
    target_reaction = None

    schema = _build_conversation_schema()
    npc_schema = _build_npc_schema()
    model = config.fast or None  # Use fast model for conversations

    for turn in range(total_messages):
        # Alternate speakers
        is_initiator_turn = turn % 2 == 0
        is_final = turn == total_messages - 1

        if is_initiator_turn:
            # Initiator speaks
            prompt = _build_conversation_prompt(
                speaker_name=initiator_name,
                speaker_persona=initiator_context.persona,
                partner_name=target_name,
                partner_relationship=request.relationship,
                topic=request.topic,
                prior_messages=messages,
                scenario=scenario,
                is_final=is_final,
                is_initiator=True,
            )

            try:
                if rate_limiter:
                    est_input = len(prompt) // 4
                    await rate_limiter.routine.acquire(
                        estimated_input_tokens=est_input,
                        estimated_output_tokens=100,
                    )

                response, _ = await asyncio.wait_for(
                    simple_call_async(
                        prompt=prompt,
                        response_schema=schema,
                        schema_name="conversation_turn",
                        model=model,
                    ),
                    timeout=15.0,
                )

                if response:
                    content = response.get("response", "...")
                    messages.append(
                        ConversationMessage(
                            speaker_id=request.initiator_id,
                            speaker_name=initiator_name,
                            content=content,
                            turn=turn,
                            is_final=is_final,
                        )
                    )
                    initiator_sentiment = response.get("updated_sentiment")
                    initiator_conviction = score_to_conviction_float(
                        response.get("updated_conviction")
                    )
                    initiator_reaction = response.get("internal_reaction")

            except Exception as e:
                logger.warning(f"[CONV] Initiator turn failed: {e}")
                break

        else:
            # Target speaks (agent or NPC)
            if request.target_is_npc:
                # NPC response
                npc_profile = target_npc_profile or {}
                prompt = _build_npc_prompt(
                    npc_name=target_name,
                    npc_profile=npc_profile,
                    initiator_name=initiator_name,
                    topic=request.topic,
                    prior_messages=messages,
                    scenario=scenario,
                    is_final=is_final,
                )

                try:
                    if rate_limiter:
                        est_input = len(prompt) // 4
                        await rate_limiter.routine.acquire(
                            estimated_input_tokens=est_input,
                            estimated_output_tokens=60,
                        )

                    response, _ = await asyncio.wait_for(
                        simple_call_async(
                            prompt=prompt,
                            response_schema=npc_schema,
                            schema_name="npc_response",
                            model=model,
                        ),
                        timeout=10.0,
                    )

                    if response:
                        content = response.get("response", "...")
                        messages.append(
                            ConversationMessage(
                                speaker_id=request.target_id,
                                speaker_name=target_name,
                                content=content,
                                turn=turn,
                                is_final=is_final,
                            )
                        )

                except Exception as e:
                    logger.warning(f"[CONV] NPC turn failed: {e}")
                    break

            else:
                # Agent target response
                if target_context is None:
                    logger.warning(
                        f"[CONV] No target context for agent {request.target_id}"
                    )
                    break

                prompt = _build_conversation_prompt(
                    speaker_name=target_name,
                    speaker_persona=target_context.persona,
                    partner_name=initiator_name,
                    partner_relationship=request.relationship,
                    topic=request.topic,
                    prior_messages=messages,
                    scenario=scenario,
                    is_final=is_final,
                    is_initiator=False,
                )

                try:
                    if rate_limiter:
                        est_input = len(prompt) // 4
                        await rate_limiter.routine.acquire(
                            estimated_input_tokens=est_input,
                            estimated_output_tokens=100,
                        )

                    response, _ = await asyncio.wait_for(
                        simple_call_async(
                            prompt=prompt,
                            response_schema=schema,
                            schema_name="conversation_turn",
                            model=model,
                        ),
                        timeout=15.0,
                    )

                    if response:
                        content = response.get("response", "...")
                        messages.append(
                            ConversationMessage(
                                speaker_id=request.target_id,
                                speaker_name=target_name,
                                content=content,
                                turn=turn,
                                is_final=is_final,
                            )
                        )
                        target_sentiment = response.get("updated_sentiment")
                        target_conviction = score_to_conviction_float(
                            response.get("updated_conviction")
                        )
                        target_reaction = response.get("internal_reaction")

                except Exception as e:
                    logger.warning(f"[CONV] Target agent turn failed: {e}")
                    break

    # Build result
    initiator_change = None
    if initiator_sentiment is not None or initiator_conviction is not None:
        initiator_change = ConversationStateChange(
            sentiment=initiator_sentiment,
            conviction=initiator_conviction,
            internal_reaction=initiator_reaction,
        )

    target_change = None
    if not request.target_is_npc and (
        target_sentiment is not None or target_conviction is not None
    ):
        target_change = ConversationStateChange(
            sentiment=target_sentiment,
            conviction=target_conviction,
            internal_reaction=target_reaction,
        )

    return ConversationResult(
        initiator_id=request.initiator_id,
        target_id=request.target_id,
        target_is_npc=request.target_is_npc,
        target_npc_profile=target_npc_profile,
        messages=messages,
        initiator_state_change=initiator_change,
        target_state_change=target_change,
        timestep=timestep,
        priority_score=request.priority_score,
    )


async def execute_conversation_batch_async(
    requests: list[ConversationRequest],
    contexts: dict[str, ReasoningContext],
    agent_map: dict[str, dict[str, Any]],
    scenario: ScenarioSpec,
    config: SimulationRunConfig,
    rate_limiter: Any = None,
    timestep: int = 0,
) -> list[ConversationResult]:
    """Execute a batch of non-conflicting conversations in parallel.

    Args:
        requests: List of conversation requests (should be non-conflicting)
        contexts: Agent ID → ReasoningContext mapping
        agent_map: Agent ID → agent dict mapping
        scenario: Scenario spec
        config: Simulation config
        rate_limiter: Optional rate limiter
        timestep: Current simulation timestep

    Returns:
        List of conversation results
    """
    if not requests:
        return []

    async def execute_one(req: ConversationRequest) -> ConversationResult:
        initiator_ctx = contexts.get(req.initiator_id)
        if initiator_ctx is None:
            logger.warning(f"[CONV] No context for initiator {req.initiator_id}")
            return ConversationResult(
                initiator_id=req.initiator_id,
                target_id=req.target_id,
                target_is_npc=req.target_is_npc,
                messages=[],
                timestep=timestep,
            )

        target_ctx = None
        npc_profile = None

        if req.target_is_npc:
            # Build NPC profile from agent data
            agent = agent_map.get(req.initiator_id, {})
            if req.target_id.endswith("_partner"):
                npc_profile = agent.get("partner_npc", {})
            else:
                # Look in dependents or other NPC data
                npc_profile = {
                    "name": req.target_name,
                    "relationship": req.relationship,
                }
        else:
            target_ctx = contexts.get(req.target_id)

        return await execute_conversation_async(
            request=req,
            initiator_context=initiator_ctx,
            target_context=target_ctx,
            target_npc_profile=npc_profile,
            scenario=scenario,
            config=config,
            rate_limiter=rate_limiter,
            timestep=timestep,
        )

    tasks = [execute_one(req) for req in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"[CONV] Conversation failed: {result}")
            # Return empty result for failed conversations
            req = requests[i]
            valid_results.append(
                ConversationResult(
                    initiator_id=req.initiator_id,
                    target_id=req.target_id,
                    target_is_npc=req.target_is_npc,
                    messages=[],
                    timestep=timestep,
                )
            )
        else:
            valid_results.append(result)

    return valid_results
