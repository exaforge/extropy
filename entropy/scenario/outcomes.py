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
            },
            "required": ["name", "description", "options"],
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

Examples of good decision outcomes:
- adoption_intent: [adopt_immediately, try_pilot_first, interested_but_blocked, not_relevant_to_me]
- cancel_decision: [cancel_immediately, downgrade_plan, stay_but_unhappy, satisfied_staying]
- compliance_stance: [comply_fully, comply_with_workarounds, openly_resist, ignore_entirely]"""

    data = simple_call(
        prompt=prompt,
        response_schema=OUTCOME_DEFINITION_SCHEMA,
        schema_name="outcome_definition",
        model=model,
    )

    # Build the two outcomes
    decision_data = data.get("decision", {})

    decision_outcome = OutcomeDefinition(
        name=decision_data.get("name", "decision"),
        type=OutcomeType.CATEGORICAL,
        description=decision_data.get("description", "Primary decision"),
        options=decision_data.get("options", ["positive", "neutral", "negative"]),
        range=None,
        required=True,
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
