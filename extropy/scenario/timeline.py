"""Timeline generation for evolving scenarios.

Determines whether a scenario is static (single event) or evolving (multi-event),
and generates timeline events + background context via LLM.
"""

import logging
from typing import Any

from ..core.llm import reasoning_call
from ..core.models import Event, SimulationConfig, TimelineEvent

logger = logging.getLogger(__name__)

TIMELINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenario_type": {
            "type": "string",
            "enum": ["static", "evolving"],
            "description": (
                "static = single event (product change, policy announcement), "
                "evolving = developments over time (crisis, campaign, adoption)"
            ),
        },
        "background_context": {
            "type": "string",
            "description": (
                "Ambient context injected into every prompt (economic conditions, "
                "cultural moment, season). 1-2 sentences."
            ),
        },
        "timeline_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestep": {
                        "type": "integer",
                        "description": "When this development occurs (0 = immediate)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable summary of the development",
                    },
                    "content": {
                        "type": "string",
                        "description": "The announcement/news content for this timestep",
                    },
                    "source": {
                        "type": "string",
                        "description": "Who/what announces this development",
                    },
                    "credibility": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Source credibility (0-1)",
                    },
                    "emotional_valence": {
                        "type": "number",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "Emotional framing (-1 to 1)",
                    },
                },
                "required": [
                    "timestep",
                    "description",
                    "content",
                    "source",
                    "credibility",
                    "emotional_valence",
                ],
            },
            "description": (
                "Only populated if scenario_type=evolving. "
                "3-6 developments at meaningful intervals."
            ),
        },
    },
    "required": ["scenario_type", "background_context"],
}


def _build_timeline_prompt(
    scenario_description: str,
    base_event: Event,
    simulation_config: SimulationConfig,
    timeline_mode: str | None,
) -> str:
    """Build the LLM prompt for timeline generation."""
    parts = [
        "You are designing how a scenario unfolds over time for a population simulation.",
        "",
        "## Scenario Description",
        "",
        scenario_description,
        "",
        "## Initial Event (t=0)",
        "",
        f"Type: {base_event.type.value}",
        f"Source: {base_event.source}",
        f"Content: {base_event.content}",
        f"Credibility: {base_event.credibility}",
        f"Emotional valence: {base_event.emotional_valence}",
        "",
        "## Simulation Parameters",
        "",
        f"Duration: {simulation_config.max_timesteps} {simulation_config.timestep_unit.value}s",
        "",
        "## Your Task",
        "",
    ]

    if timeline_mode == "static":
        parts.extend(
            [
                "This is a STATIC scenario. Generate only background_context.",
                "Set scenario_type to 'static' and leave timeline_events empty.",
            ]
        )
    elif timeline_mode == "evolving":
        parts.extend(
            [
                "This is an EVOLVING scenario. Generate 3-6 timeline events.",
                "Set scenario_type to 'evolving'.",
                "",
                "Timeline event guidelines:",
                "- Space events at meaningful intervals (not every timestep)",
                "- Each event should escalate, complicate, or resolve the situation",
                "- Include reactions, developments, or new information",
                "- Vary sources (officials, media, social, leaked info)",
            ]
        )
    else:
        parts.extend(
            [
                "Determine if this is a STATIC or EVOLVING scenario:",
                "",
                "STATIC scenarios (scenario_type='static'):",
                "- One-time announcements (price changes, policy updates)",
                "- Product launches with no expected developments",
                "- Simple changes with immediate, stable reactions",
                "",
                "EVOLVING scenarios (scenario_type='evolving'):",
                "- Crises that unfold over time (safety issues, scandals)",
                "- Campaigns with multiple phases",
                "- Situations where new information emerges",
                "- Events that trigger reactions, counter-reactions",
                "",
                "For EVOLVING scenarios, generate 3-6 timeline events:",
                "- Space events at meaningful intervals",
                "- Each event should escalate, complicate, or resolve",
                "- Vary sources appropriately",
            ]
        )

    parts.extend(
        [
            "",
            "For ALL scenarios, generate background_context:",
            "- 1-2 sentences of ambient framing",
            "- Economic conditions, cultural moment, season if relevant",
            "- This appears in every agent's reasoning prompt",
        ]
    )

    return "\n".join(parts)


def generate_timeline(
    scenario_description: str,
    base_event: Event,
    simulation_config: SimulationConfig,
    timeline_mode: str | None = None,
) -> tuple[list[TimelineEvent], str | None]:
    """Generate timeline events and background context.

    Args:
        scenario_description: Natural language scenario description
        base_event: Parsed t=0 event
        simulation_config: Simulation parameters (timesteps, unit)
        timeline_mode: Explicit mode override. If None, LLM decides based on scenario.
            - "static": Single event, no timeline (Netflix-style)
            - "evolving": Multi-event timeline (ASI-style)

    Returns:
        Tuple of (timeline_events, background_context)
        timeline_events will be empty for static scenarios
    """
    prompt = _build_timeline_prompt(
        scenario_description,
        base_event,
        simulation_config,
        timeline_mode,
    )

    logger.info("[TIMELINE] Generating timeline and background context...")

    response = reasoning_call(
        prompt=prompt,
        response_schema=TIMELINE_SCHEMA,
        schema_name="timeline_generation",
    )

    if not response:
        logger.warning("[TIMELINE] LLM returned empty response, using defaults")
        return [], None

    scenario_type = response.get("scenario_type", "static")
    background_context = response.get("background_context")
    raw_events = response.get("timeline_events", [])

    # Honor explicit mode override
    if timeline_mode == "static":
        scenario_type = "static"
        raw_events = []
    elif timeline_mode == "evolving" and not raw_events:
        logger.warning("[TIMELINE] Evolving mode requested but no events generated")

    logger.info(
        f"[TIMELINE] Type: {scenario_type}, "
        f"Events: {len(raw_events)}, "
        f"Background: {background_context[:50] + '...' if background_context else 'None'}"
    )

    # Convert raw events to TimelineEvent models
    timeline_events: list[TimelineEvent] = []

    if scenario_type == "evolving" and raw_events:
        for raw in raw_events:
            timestep = raw.get("timestep", 0)
            # Skip t=0 events (that's the base event)
            if timestep == 0:
                continue

            event = Event(
                type=base_event.type,  # Inherit type from base
                content=raw.get("content", ""),
                source=raw.get("source", base_event.source),
                credibility=raw.get("credibility", base_event.credibility),
                ambiguity=base_event.ambiguity,  # Inherit
                emotional_valence=raw.get("emotional_valence", 0.0),
            )

            timeline_event = TimelineEvent(
                timestep=timestep,
                event=event,
                exposure_rules=None,  # Reuse seed exposure rules
                description=raw.get("description"),
            )
            timeline_events.append(timeline_event)

        # Sort by timestep
        timeline_events.sort(key=lambda te: te.timestep)

    return timeline_events, background_context
