"""Step 4: Define Outcomes.

Determines what outcomes to measure from the simulation based on the
event type, population characteristics, and scenario description.
"""

from ..core.llm import reasoning_call
from ..core.models import (
    PopulationSpec,
    Event,
    OutcomeConfig,
    OutcomeDefinition,
    OutcomeType,
)


# JSON schema for outcome definition response
OUTCOME_DEFINITION_SCHEMA = {
    "type": "object",
    "properties": {
        "outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Outcome identifier in snake_case",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["categorical", "boolean", "float", "open_ended"],
                        "description": "Type of outcome measurement",
                    },
                    "description": {
                        "type": "string",
                        "description": "What this outcome measures",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "For categorical: the possible values",
                    },
                    "range_min": {
                        "type": "number",
                        "description": "For float: minimum value",
                    },
                    "range_max": {
                        "type": "number",
                        "description": "For float: maximum value",
                    },
                    "required": {
                        "type": "boolean",
                        "description": "Whether this outcome must be extracted",
                    },
                },
                "required": ["name", "type", "description", "options", "range_min", "range_max", "required"],
                "additionalProperties": False,
            },
            "minItems": 3,
            "maxItems": 7,
        },
        "extraction_instructions": {
            "type": "string",
            "description": "Hints for Phase 3 outcome extraction",
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of outcome selection",
        },
    },
    "required": ["outcomes", "extraction_instructions", "reasoning"],
    "additionalProperties": False,
}


def define_outcomes(
    event: Event,
    population_spec: PopulationSpec,
    scenario_description: str,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> OutcomeConfig:
    """
    Define what outcomes to measure from the simulation.

    Generates 3-7 outcome measurements appropriate for the scenario:
    - Categorical: Discrete choices (e.g., "will cancel", "staying")
    - Boolean: Yes/no measurements (e.g., will share information)
    - Float: Continuous measurements (e.g., sentiment -1 to 1)
    - Open-ended: Free-form text responses

    Args:
        event: The parsed event definition
        population_spec: The population spec for context
        scenario_description: Original scenario description
        model: LLM model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        OutcomeConfig with suggested outcomes

    Example:
        >>> outcomes = define_outcomes(event, population_spec, "Netflix price increase")
        >>> len(outcomes.suggested_outcomes)
        5
        >>> outcomes.suggested_outcomes[0].name
        'cancel_intent'
    """
    prompt = f"""## Task

Define what outcomes we should measure from agents after this event plays out.
These outcomes will be extracted from agent responses during the simulation.

## Event

Type: {event.type.value}
Content: "{event.content}"
Source: {event.source}
Credibility: {event.credibility:.2f}
Emotional valence: {event.emotional_valence:.2f}

## Population

{population_spec.meta.description}
Size: {population_spec.meta.size} agents
Geography: {population_spec.meta.geography or 'Not specified'}

## Original Scenario Description

"{scenario_description}"

## Outcome Types

### categorical
Discrete choices from predefined options.
- Use for decisions, intentions, classifications
- Must have 2-5 mutually exclusive options
- Example: cancel_intent with options ["will_cancel", "considering", "staying", "undecided"]

### boolean
Simple yes/no measurements.
- Use for binary behaviors or states
- Example: share_behavior (will they share this info?)

### float
Continuous measurements on a scale.
- Use for sentiment, intensity, likelihood
- Must specify range (typically -1 to 1, or 0 to 1)
- Example: sentiment from -1 (very negative) to 1 (very positive)

### open_ended
Free-form text responses.
- Use for capturing nuanced reasoning
- Example: main_concern (what worries them most)

## Outcome Patterns by Scenario Type

| Scenario Type | Typical Outcomes |
|---------------|------------------|
| Price change | cancel_intent, sentiment, share_behavior, downgrade_intent |
| New product/feature | adoption_intent, interest_level, concerns, recommend_likelihood |
| Policy announcement | support_position, compliance_intent, protest_likelihood |
| Technology adoption | adoption_willingness, trust_level, information_seeking |
| Emergency/Crisis | urgency_perception, compliance, information_sharing |
| Rumor/News | belief_level, verification_behavior, share_intent |
| Professional Alignment | exit_intent, voice_intent, compliance_level, burnout_impact |
| Information Warfare | belief_level, amplification_intent, source_skepticism, narrative_alignment |

## Reasoning Capture

**IMPORTANT:** Ensure outcomes capture the **reasoning** behind decisions, not just the decision itself. This enables post-hoc discovery of emergent behaviors. For example, agents might be "rotating subscriptions" rather than just "cancelling"—a nuance only visible when reasoning is captured.

## Guidelines

1. **Mix of types**: Include at least one categorical, one float (sentiment is common), and consider boolean/open-ended
2. **3-7 outcomes**: Focus on what's most meaningful for this scenario
3. **Required vs optional**: Mark sentiment and primary decision as required; secondary measures can be optional
4. **Clear names**: Use snake_case, descriptive names
5. **Population perspective**: Consider what would be meaningful to measure for THIS population

## Output

Define 3-7 outcomes with appropriate types and configurations.
For categorical, provide 2-5 mutually exclusive options.
For float, specify range (typically -1 to 1 for sentiment, 0 to 1 for probabilities)."""

    data = reasoning_call(
        prompt=prompt,
        response_schema=OUTCOME_DEFINITION_SCHEMA,
        schema_name="outcome_definition",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Parse outcomes
    outcomes = []
    for outcome_data in data.get("outcomes", []):
        outcome_type = OutcomeType(outcome_data["type"])

        # Build range for float types
        outcome_range = None
        if outcome_type == OutcomeType.FLOAT:
            range_min = outcome_data.get("range_min", -1.0)
            range_max = outcome_data.get("range_max", 1.0)
            outcome_range = (range_min, range_max)

        # Get options for categorical types
        options = None
        if outcome_type == OutcomeType.CATEGORICAL:
            options = outcome_data.get("options", [])

        outcome = OutcomeDefinition(
            name=outcome_data["name"],
            type=outcome_type,
            description=outcome_data["description"],
            options=options,
            range=outcome_range,
            required=outcome_data.get("required", True),
        )
        outcomes.append(outcome)

    return OutcomeConfig(
        suggested_outcomes=outcomes,
        capture_full_reasoning=True,  # Always capture full reasoning
        extraction_instructions=data.get("extraction_instructions"),
    )
