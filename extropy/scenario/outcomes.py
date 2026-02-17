"""Step 4: Define Outcomes.

Generates minimal structured outcomes for simulation:
1. One categorical outcome (the primary decision)
2. One float outcome (sentiment)

All other insights are captured in the agent's reasoning text and can be
extracted post-hoc during analysis. This approach is:
- More reliable (fewer extraction errors)
- Faster (simpler schema for LLM)
- More powerful (reasoning captures emergent behaviors not pre-defined)
"""

from ..core.llm import simple_call
from ..core.models import (
    PopulationSpec,
    Event,
    OutcomeConfig,
    OutcomeDefinition,
    OutcomeType,
)


# Minimal schema: one categorical decision + sentiment
OUTCOME_DEFINITION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Decision outcome name in snake_case (e.g., adoption_intent, cancel_decision)",
                },
                "description": {
                    "type": "string",
                    "description": "What this decision measures",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 mutually exclusive options in snake_case",
                },
                "option_friction": {
                    "type": "object",
                    "description": "Behavioral friction for each option (0-1). Higher = harder to sustain in real behavior. "
                    "Low friction (0.1-0.3): status quo, inaction, keeping current state. "
                    "Medium friction (0.4-0.6): partial changes, delays, conditional actions. "
                    "High friction (0.7-0.9): major changes, cancellations, adoptions, switches.",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
            "required": ["name", "description", "options", "option_friction"],
            "additionalProperties": False,
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of why these options capture the decision space",
        },
    },
    "required": ["decision", "reasoning"],
    "additionalProperties": False,
}


def define_outcomes(
    event: Event,
    population_spec: PopulationSpec,
    scenario_description: str,
    model: str | None = None,
) -> OutcomeConfig:
    """
    Define minimal outcomes for simulation.

    Generates exactly 2 required outcomes:
    1. A categorical decision (LLM-generated based on scenario)
    2. Sentiment (-1 to 1)

    All nuanced insights come from reasoning text, which is always captured.

    Args:
        event: The parsed event definition
        population_spec: The population spec for context
        scenario_description: Original scenario description
        model: LLM model to use

    Returns:
        OutcomeConfig with 2 outcomes + reasoning capture enabled
    """
    prompt = f"""Define the PRIMARY DECISION agents will make in response to this event.

## Event
{event.content}

## Population
{population_spec.meta.description}

## Scenario
{scenario_description}

## Task
Create ONE categorical decision outcome that captures what agents will decide/do.

Requirements:
- 3-5 mutually exclusive options in snake_case
- Each option must represent a DISTINCT behavioral state — avoid catch-all middle options
- Do NOT include 'undecided' or 'wait_and_see' as options — agents who are unsure will naturally distribute across the more specific options
- Include at least one clearly positive action and one clearly negative/opt-out option
- Prefer behavioral options (what people DO) over attitudinal ones (what people FEEL)
- Options should NOT be orderable on a single scale — each should capture a qualitatively different response

## Option Friction
For EACH option, assign a friction score (0-1) indicating how hard it is to sustain that behavior:
- Low friction (0.1-0.3): Status quo, inaction, keeping current state (e.g., "keep_subscription", "no_change", "stay")
- Medium friction (0.4-0.6): Partial changes, delays, conditional actions (e.g., "reduce_usage", "pause", "wait_for_more_info")
- High friction (0.7-0.9): Major changes requiring effort (e.g., "cancel", "switch_provider", "adopt_immediately", "purchase")

Examples of good decision outcomes with friction:
- adoption_intent: {{adopt_immediately: 0.75, try_pilot_first: 0.5, interested_but_blocked: 0.3, not_relevant_to_me: 0.2}}
- cancel_decision: {{cancel_immediately: 0.8, downgrade_plan: 0.5, stay_but_unhappy: 0.2, satisfied_staying: 0.15}}"""

    data = simple_call(
        prompt=prompt,
        response_schema=OUTCOME_DEFINITION_SCHEMA,
        schema_name="outcome_definition",
        model=model,
    )

    # Build the two outcomes
    decision_data = data.get("decision", {})
    options = decision_data.get("options", ["positive", "neutral", "negative"])
    raw_friction = decision_data.get("option_friction", {})

    # Ensure option_friction has values for all options (default to 0.5 if missing)
    option_friction = {opt: raw_friction.get(opt, 0.5) for opt in options}

    decision_outcome = OutcomeDefinition(
        name=decision_data.get("name", "decision"),
        type=OutcomeType.CATEGORICAL,
        description=decision_data.get("description", "Primary decision"),
        options=options,
        range=None,
        required=True,
        option_friction=option_friction,
    )

    sentiment_outcome = OutcomeDefinition(
        name="sentiment",
        type=OutcomeType.FLOAT,
        description="Overall sentiment toward the event (-1 very negative to 1 very positive)",
        options=None,
        range=(-1.0, 1.0),
        required=True,
    )

    return OutcomeConfig(
        suggested_outcomes=[decision_outcome, sentiment_outcome],
        capture_full_reasoning=True,
        extraction_instructions=None,  # Not needed - schema is simple enough
    )
