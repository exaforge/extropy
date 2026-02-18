"""Step 0: Scenario Sufficiency Check.

Verifies that the scenario description has enough information
to proceed with scenario generation. Checks for:
- Duration/timeline (how long does this unfold?)
- Timestep unit (infer from duration or ask)
- Outcomes (are explicit outcomes specified? are they detailed enough?)
- Event type clarity (static vs evolving)

Mirrors the population spec sufficiency check pattern.
"""

from ..core.llm import simple_call
from ..core.models import ClarificationQuestion

from pydantic import BaseModel, Field


class ScenarioSufficiencyResult(BaseModel):
    """Result from scenario sufficiency check."""

    sufficient: bool
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    inferred_duration: str | None = Field(
        default=None,
        description="Inferred duration from description, e.g. '6 months'",
    )
    inferred_timestep_unit: str | None = Field(
        default=None,
        description="Inferred timestep unit: hour, day, week, month, year",
    )
    inferred_scenario_type: str | None = Field(
        default=None,
        description="Inferred scenario type: static or evolving",
    )
    has_explicit_outcomes: bool = Field(
        default=False,
        description="Whether the description contains explicit outcome definitions",
    )


SCENARIO_SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": "Whether the scenario description has enough information to generate a scenario spec",
        },
        "inferred_duration": {
            "type": ["string", "null"],
            "description": "Duration if mentioned or inferable (e.g., '6 months', '2 weeks')",
        },
        "inferred_timestep_unit": {
            "type": ["string", "null"],
            "enum": ["hour", "day", "week", "month", "year", None],
            "description": "Best timestep unit based on duration",
        },
        "inferred_scenario_type": {
            "type": ["string", "null"],
            "enum": ["static", "evolving", None],
            "description": "Whether this is a static or evolving scenario",
        },
        "has_explicit_outcomes": {
            "type": "boolean",
            "description": "Whether the description specifies explicit outcome definitions",
        },
        "questions": {
            "type": "array",
            "description": "Structured clarification questions for missing information",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "snake_case identifier, e.g. 'duration'",
                    },
                    "question": {
                        "type": "string",
                        "description": "Human-readable question text",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["single_choice", "text", "number"],
                        "description": "Question type for UI rendering",
                    },
                    "options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Options for single_choice questions",
                    },
                    "default": {
                        "description": "Default value if user doesn't provide one",
                    },
                },
                "required": ["id", "question", "type"],
            },
        },
    },
    "required": [
        "sufficient",
        "inferred_duration",
        "inferred_timestep_unit",
        "inferred_scenario_type",
        "has_explicit_outcomes",
        "questions",
    ],
    "additionalProperties": False,
}


def check_scenario_sufficiency(
    description: str,
    model: str | None = None,
) -> ScenarioSufficiencyResult:
    """Check if a scenario description is sufficient for spec generation.

    A sufficient description should specify or allow inference of:
    1. What happens (the event/situation)
    2. Duration or timeline (how long it unfolds)
    3. Whether it's static or evolving

    Args:
        description: Natural language scenario description
        model: Model to use

    Returns:
        ScenarioSufficiencyResult with sufficient flag and questions
    """
    prompt = f"""Evaluate if this scenario description is sufficient to create a simulation scenario:

"{description}"

A sufficient scenario description should have:

1. **WHAT happens** — a clear event, situation, or change being introduced
   ✓ Good: "AI replaces 40% of white-collar jobs over 6 months"
   ✗ Bad: "something changes" (too vague)

2. **DURATION** — how long the scenario unfolds (explicit or inferable)
   ✓ Explicit: "over 6 months", "12-week rollout", "within 48 hours"
   ✓ Inferable: "election campaign" (~6 months), "product launch" (~1-2 weeks)
   ✗ Missing: no duration mentioned and not inferable from scenario type

3. **SCENARIO TYPE** — is this a one-time event (static) or something that unfolds (evolving)?
   - Static: price change, policy announcement, product feature
   - Evolving: crisis, campaign, gradual adoption, emerging situation

4. **OUTCOMES** (optional but check) — does the description explicitly define outcome measurements?
   Look for sections like "Outcomes:", numbered outcome lists, or explicit mentions of what to track.

Be LENIENT — if you can reasonably infer duration and type from the scenario, mark as sufficient.
Only mark insufficient if critical information is truly missing or ambiguous.

If insufficient, generate structured questions with:
- id: snake_case identifier (e.g., "duration", "scenario_type")
- question: Human-readable question
- type: "single_choice" (with options), "text" (free text), or "number"
- options: Array of choices for single_choice
- default: Reasonable default value"""

    data = simple_call(
        prompt=prompt,
        response_schema=SCENARIO_SUFFICIENCY_SCHEMA,
        schema_name="scenario_sufficiency_check",
        model=model,
    )

    raw_questions = data.get("questions", [])
    questions = [
        ClarificationQuestion(
            id=q.get("id", f"q{i}"),
            question=q.get("question", ""),
            type=q.get("type", "text"),
            options=q.get("options"),
            default=q.get("default"),
        )
        for i, q in enumerate(raw_questions)
    ]

    return ScenarioSufficiencyResult(
        sufficient=data.get("sufficient", False),
        questions=questions,
        inferred_duration=data.get("inferred_duration"),
        inferred_timestep_unit=data.get("inferred_timestep_unit"),
        inferred_scenario_type=data.get("inferred_scenario_type"),
        has_explicit_outcomes=data.get("has_explicit_outcomes", False),
    )


def check_scenario_sufficiency_with_answers(
    description: str,
    answers: dict[str, str | int],
    model: str | None = None,
) -> ScenarioSufficiencyResult:
    """Re-check sufficiency with pre-supplied answers.

    Enriches the description with answer context and re-runs the check.

    Args:
        description: Original scenario description
        answers: Dict mapping question IDs to answer values
        model: Model to use

    Returns:
        ScenarioSufficiencyResult with updated sufficient flag
    """
    enriched = description
    for key, value in answers.items():
        enriched += f" | {key}: {value}"

    return check_scenario_sufficiency(enriched, model)
