"""Timeline and outcome generation for scenarios.

Determines whether a scenario is static (single event) or evolving (multi-event),
generates timeline events + background context, and defines outcome measurements.
Outcomes are generated alongside timeline so they share context about scenario type,
duration, and events.
"""

import logging
from typing import Any

from ..core.llm import reasoning_call
from ..core.models import (
    Event,
    OutcomeConfig,
    OutcomeDefinition,
    OutcomeType,
    ScenarioSimConfig,
    TimelineEvent,
    TimestepUnit,
)

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
        "timestep_unit": {
            "type": "string",
            "enum": ["hour", "day", "week", "month", "year"],
            "description": (
                "The natural time unit for this scenario. Choose based on how "
                "the scenario description frames duration: '6 months' → month, "
                "'12 weeks' → week, '7 days' → day, '48 hours' → hour, "
                "'5 years' → year."
            ),
        },
        "max_timesteps": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Total number of timesteps for the simulation. Should match "
                "the scenario duration in the chosen unit (e.g. 6 months → 6)."
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
                    "re_reasoning_intensity": {
                        "type": "string",
                        "enum": ["normal", "high", "extreme"],
                        "description": (
                            "How broadly this event should trigger committed-agent "
                            "re-reasoning: normal=direct only, high=direct+network, "
                            "extreme=high+all aware."
                        ),
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
        "outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "snake_case outcome name",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["categorical", "boolean", "open_ended"],
                        "description": "Outcome measurement type",
                    },
                    "description": {
                        "type": "string",
                        "description": "What this outcome measures",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "For categorical: mutually exclusive options in snake_case",
                    },
                    "option_friction": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "description": (
                            "For categorical: behavioral friction per option (0-1). "
                            "Higher = harder to sustain. "
                            "Low (0.1-0.3): status quo, inaction. "
                            "Medium (0.4-0.6): partial changes, delays. "
                            "High (0.7-0.9): major changes, adoptions, switches."
                        ),
                    },
                },
                "required": ["name", "type", "description"],
            },
            "minItems": 2,
            "maxItems": 12,
            "description": (
                "Outcome definitions — extract from description if explicit, "
                "otherwise generate appropriate outcomes for this scenario."
            ),
        },
    },
    "required": [
        "scenario_type",
        "timestep_unit",
        "max_timesteps",
        "background_context",
        "outcomes",
    ],
}


def _build_timeline_prompt(
    scenario_description: str,
    base_event: Event,
    simulation_config: ScenarioSimConfig,
    timeline_mode: str | None,
) -> str:
    """Build the LLM prompt for timeline + outcome generation."""
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
        "Determine the appropriate timestep_unit (hour, day, week, month, or year) and "
        "max_timesteps based on the scenario description. If the description mentions a "
        "specific duration (e.g. '6 months', '12 weeks', '5 years'), use the matching "
        "unit and count. Timeline event timesteps must use the same unit.",
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
                "- Set re_reasoning_intensity per event: normal for routine updates, "
                "high for major updates likely to spread rapidly, extreme for systemic shocks",
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
                "- Set re_reasoning_intensity per event: normal/high/extreme",
            ]
        )

    parts.extend(
        [
            "",
            "For ALL scenarios, generate background_context:",
            "- 1-2 sentences of ambient framing",
            "- Economic conditions, cultural moment, season if relevant",
            "- This appears in every agent's reasoning prompt",
            "",
            "## Outcomes",
            "",
            "FIRST, check if the scenario description contains explicit outcome definitions.",
            "If the description specifies outcomes (e.g., 'Outcomes: employment_status, "
            "economic_adaptation...'), extract them exactly as specified — preserve their "
            "names, types, options, and descriptions faithfully.",
            "",
            "If NO explicit outcomes in description, generate 4-8 outcomes appropriate "
            "for this scenario type:",
            "- Evolving scenarios: outcomes that can change over timesteps",
            "- Static scenarios: simpler decision-focused outcomes",
            "- Include open_ended outcomes for capturing emergent insights "
            "(optional, based on scenario complexity)",
            "",
            "CRITICAL RULES for outcomes:",
            "- NEVER use float/numeric type — use categorical with ranges instead",
            "  (e.g., instead of 'sentiment: float', use 'sentiment: categorical' with "
            "  options like 'very_negative', 'negative', 'neutral', 'positive', 'very_positive')",
            "- For categorical outcomes: provide 4-8 mutually exclusive options in snake_case",
            "- For categorical outcomes: include option_friction scores (0.1-0.9) where "
            "  higher means harder to sustain in real behavior",
            "- For boolean outcomes: no options needed",
            "- For open_ended outcomes: no options needed (captures free-text reasoning)",
            "- Each option must represent a DISTINCT behavioral state",
            "- Avoid catch-all middle options like 'undecided' or 'wait_and_see'",
            "- Prefer behavioral options (what people DO) over attitudinal ones (what people FEEL)",
        ]
    )

    return "\n".join(parts)


def generate_timeline_and_outcomes(
    scenario_description: str,
    base_event: Event,
    simulation_config: ScenarioSimConfig,
    timeline_mode: str | None = None,
    timestep_unit_override: str | None = None,
    max_timesteps_override: int | None = None,
) -> tuple[list[TimelineEvent], str | None, ScenarioSimConfig, OutcomeConfig]:
    """Generate timeline events, background context, and outcomes.

    Outcomes are generated in the same LLM call as timeline to ensure they
    share context about scenario type, duration, and events.

    Args:
        scenario_description: Natural language scenario description
        base_event: Parsed t=0 event
        simulation_config: Simulation parameters (timesteps, unit)
        timeline_mode: Explicit mode override. If None, LLM decides based on scenario.
            - "static": Single event, no timeline
            - "evolving": Multi-event timeline
        timestep_unit_override: CLI override for timestep unit (e.g. "month")
        max_timesteps_override: CLI override for max timesteps

    Returns:
        Tuple of (timeline_events, background_context, updated_simulation_config, outcome_config)
        timeline_events will be empty for static scenarios
    """
    prompt = _build_timeline_prompt(
        scenario_description,
        base_event,
        simulation_config,
        timeline_mode,
    )

    logger.info("[TIMELINE] Generating timeline, background context, and outcomes...")

    response = reasoning_call(
        prompt=prompt,
        response_schema=TIMELINE_SCHEMA,
        schema_name="timeline_and_outcomes",
    )

    if not response:
        logger.warning("[TIMELINE] LLM returned empty response, using defaults")
        default_outcomes = OutcomeConfig(
            suggested_outcomes=[], capture_full_reasoning=True
        )
        return [], None, simulation_config, default_outcomes

    scenario_type = response.get("scenario_type", "static")
    background_context = response.get("background_context")
    raw_events = response.get("timeline_events", [])
    raw_outcomes = response.get("outcomes", [])

    # ── Resolve simulation config ────────────────────────────────────────
    # CLI overrides take priority over LLM response
    llm_unit = response.get("timestep_unit")
    llm_max = response.get("max_timesteps")

    unit_map = {
        "hour": TimestepUnit.HOUR,
        "day": TimestepUnit.DAY,
        "week": TimestepUnit.WEEK,
        "month": TimestepUnit.MONTH,
        "year": TimestepUnit.YEAR,
    }

    if timestep_unit_override:
        resolved_unit = unit_map.get(
            timestep_unit_override, simulation_config.timestep_unit
        )
    elif llm_unit:
        resolved_unit = unit_map.get(llm_unit, simulation_config.timestep_unit)
    else:
        resolved_unit = simulation_config.timestep_unit

    if max_timesteps_override:
        resolved_max = max_timesteps_override
    elif llm_max:
        resolved_max = llm_max
    else:
        resolved_max = simulation_config.max_timesteps

    simulation_config = ScenarioSimConfig(
        max_timesteps=resolved_max,
        timestep_unit=resolved_unit,
        stop_conditions=simulation_config.stop_conditions,
        allow_early_convergence=simulation_config.allow_early_convergence,
        seed=simulation_config.seed,
    )

    # Honor explicit mode override
    if timeline_mode == "static":
        scenario_type = "static"
        raw_events = []
    elif timeline_mode == "evolving" and not raw_events:
        logger.warning("[TIMELINE] Evolving mode requested but no events generated")

    logger.info(
        f"[TIMELINE] Type: {scenario_type}, "
        f"Events: {len(raw_events)}, "
        f"Outcomes: {len(raw_outcomes)}, "
        f"Background: {background_context[:50] + '...' if background_context else 'None'}"
    )

    # ── Convert raw events to TimelineEvent models ───────────────────────
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
                re_reasoning_intensity=raw.get("re_reasoning_intensity", "normal"),
            )
            timeline_events.append(timeline_event)

        # Sort by timestep
        timeline_events.sort(key=lambda te: te.timestep)

    # ── Convert raw outcomes to OutcomeConfig ────────────────────────────
    outcome_type_map = {
        "categorical": OutcomeType.CATEGORICAL,
        "boolean": OutcomeType.BOOLEAN,
        "open_ended": OutcomeType.OPEN_ENDED,
    }

    outcome_definitions: list[OutcomeDefinition] = []
    for raw_outcome in raw_outcomes:
        outcome_type_str = raw_outcome.get("type", "categorical")
        outcome_type = outcome_type_map.get(outcome_type_str, OutcomeType.CATEGORICAL)

        options = raw_outcome.get("options")
        raw_friction = raw_outcome.get("option_friction", {})

        # Ensure option_friction has values for all options (default 0.5 if missing)
        option_friction = None
        if outcome_type == OutcomeType.CATEGORICAL and options:
            option_friction = {opt: raw_friction.get(opt, 0.5) for opt in options}

        outcome_definitions.append(
            OutcomeDefinition(
                name=raw_outcome.get("name", "unnamed_outcome"),
                type=outcome_type,
                description=raw_outcome.get("description", ""),
                options=options if outcome_type == OutcomeType.CATEGORICAL else None,
                range=None,  # Never use float ranges
                required=True,
                option_friction=option_friction,
            )
        )

    outcome_config = OutcomeConfig(
        suggested_outcomes=outcome_definitions,
        capture_full_reasoning=True,
        extraction_instructions=None,
    )

    return timeline_events, background_context, simulation_config, outcome_config


# Keep backward-compatible alias
def generate_timeline(
    scenario_description: str,
    base_event: Event,
    simulation_config: ScenarioSimConfig,
    timeline_mode: str | None = None,
) -> tuple[list[TimelineEvent], str | None, ScenarioSimConfig]:
    """Backward-compatible wrapper. Prefer generate_timeline_and_outcomes."""
    events, bg, sim_config, _outcomes = generate_timeline_and_outcomes(
        scenario_description=scenario_description,
        base_event=base_event,
        simulation_config=simulation_config,
        timeline_mode=timeline_mode,
    )
    return events, bg, sim_config
