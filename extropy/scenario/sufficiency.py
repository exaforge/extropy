"""Step 0: Scenario Sufficiency Check.

Verifies that the scenario description has enough information
to proceed with scenario generation. Checks for:
- Duration/timeline (how long does this unfold?)
- Timestep unit (infer from duration or ask)
- Outcomes (are explicit outcomes specified? are they detailed enough?)
- Event type clarity (static vs evolving)

Mirrors the population spec sufficiency check pattern.
"""

import re

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
    inferred_agent_focus_mode: str | None = Field(
        default=None,
        description="Inferred household agent scope: primary_only, couples, or all",
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
        "inferred_agent_focus_mode": {
            "type": "string",
            "enum": ["primary_only", "couples", "all"],
            "description": "Who in each household should be a simulated agent: "
            "'primary_only' = one adult per household (most scenarios — professional studies, opinion polling, consumer behavior); "
            "'couples' = both partners are agents (relationship dynamics, dual-income studies, retirement planning); "
            "'all' = all household members including children (family dynamics, community health, school policy). "
            "Default to 'primary_only' unless the scenario clearly involves couple/partner dynamics or whole-family/community dynamics.",
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
        "inferred_agent_focus_mode",
        "questions",
    ],
    "additionalProperties": False,
}

_UNIT_KEYWORD_PATTERNS: dict[str, tuple[str, ...]] = {
    "hour": (
        r"\bhour(?:s)?\b",
        r"\bhr(?:s)?\b",
        r"\bh\b",
    ),
    "day": (
        r"\bday(?:s)?\b",
        r"\bdaily\b",
    ),
    "week": (
        r"\bweek(?:s)?\b",
        r"\bwk(?:s)?\b",
        r"\bweekly\b",
    ),
    "month": (
        r"\bmonth(?:s)?\b",
        r"\bmo(?:s)?\b",
        r"\bmonthly\b",
    ),
    "year": (
        r"\byear(?:s)?\b",
        r"\byr(?:s)?\b",
        r"\bannual(?:ly)?\b",
    ),
}

_TIMELINE_MARKER_PATTERNS: dict[str, tuple[str, ...]] = {
    "hour": (r"\bhour\s*[-_ ]?\d+\b", r"\bhr\s*[-_ ]?\d+\b"),
    "day": (r"\bday\s*[-_ ]?\d+\b",),
    "week": (r"\bweek\s*[-_ ]?\d+\b", r"\bwk\s*[-_ ]?\d+\b"),
    "month": (r"\bmonth\s*[-_ ]?\d+\b", r"\bmo\s*[-_ ]?\d+\b"),
    "year": (r"\byear\s*[-_ ]?\d+\b", r"\byr\s*[-_ ]?\d+\b"),
}


def _extract_unit_candidates(text: str) -> set[str]:
    """Extract timestep unit candidates from free text."""
    lowered = text.lower()
    candidates: set[str] = set()
    for unit, patterns in _UNIT_KEYWORD_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            candidates.add(unit)
    return candidates


def _extract_timeline_marker_unit(text: str) -> str | None:
    """Extract explicit timeline marker unit, e.g. 'month 0', 'week1'."""
    lowered = text.lower()
    matches: set[str] = set()
    for unit, patterns in _TIMELINE_MARKER_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            matches.add(unit)
    if len(matches) == 1:
        return next(iter(matches))
    return None


def _infer_unit_from_duration(duration: str | None) -> str | None:
    """Infer timestep unit from an inferred duration string."""
    if not duration:
        return None
    candidates = _extract_unit_candidates(duration)
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _build_timestep_question(default_unit: str | None = None) -> ClarificationQuestion:
    """Construct deterministic timestep-unit clarification question."""
    default = (
        default_unit
        if default_unit in {"hour", "day", "week", "month", "year"}
        else "day"
    )
    return ClarificationQuestion(
        id="timestep_unit",
        question="What timestep unit should this scenario use?",
        type="single_choice",
        options=["hour", "day", "week", "month", "year"],
        default=default,
    )


def _postprocess_sufficiency(
    *,
    description: str,
    result: ScenarioSufficiencyResult,
) -> ScenarioSufficiencyResult:
    """Apply deterministic guardrails after LLM sufficiency output."""
    questions = list(result.questions)
    explicit_timeline_unit = _extract_timeline_marker_unit(description)
    unit_candidates = _extract_unit_candidates(description)
    explicit_unit = None
    if explicit_timeline_unit:
        explicit_unit = explicit_timeline_unit
    elif len(unit_candidates) == 1:
        explicit_unit = next(iter(unit_candidates))

    inferred_from_duration = _infer_unit_from_duration(result.inferred_duration)
    inferred_unit = (
        result.inferred_timestep_unit or inferred_from_duration or explicit_unit
    )

    inferred_type = result.inferred_scenario_type
    # Hard hint: timeline markers like "month 0, month 1" imply evolving scenarios.
    if explicit_timeline_unit and inferred_type != "evolving":
        inferred_type = "evolving"

    # Static scenarios must explicitly name the timestep unit.
    if inferred_type == "static" and explicit_unit is None:
        if not any(q.id == "timestep_unit" for q in questions):
            questions.append(_build_timestep_question(inferred_unit))
        return ScenarioSufficiencyResult(
            sufficient=False,
            questions=questions,
            inferred_duration=result.inferred_duration,
            inferred_timestep_unit=inferred_unit,
            inferred_scenario_type=inferred_type,
            has_explicit_outcomes=result.has_explicit_outcomes,
            inferred_agent_focus_mode=result.inferred_agent_focus_mode,
        )

    return ScenarioSufficiencyResult(
        sufficient=result.sufficient,
        questions=questions,
        inferred_duration=result.inferred_duration,
        inferred_timestep_unit=inferred_unit,
        inferred_scenario_type=inferred_type,
        has_explicit_outcomes=result.has_explicit_outcomes,
        inferred_agent_focus_mode=result.inferred_agent_focus_mode,
    )


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

4. **TIMESTEP UNIT** — what time unit should simulation steps use (hour/day/week/month/year)?
   - For evolving scenarios with timeline markers like "month 0, month 1", use that marker unit.
   - For static scenarios, still pick a concrete unit so downstream simulation cadence is explicit.
   - If the unit is not clear, ask for clarification.

5. **OUTCOMES** (optional but check) — does the description explicitly define outcome measurements?
   Look for sections like "Outcomes:", numbered outcome lists, or explicit mentions of what to track.

6. **AGENT FOCUS MODE** — who in each household should be a simulated agent?
   - "primary_only": Only one adult per household is an agent (partner/kids are background context).
     Use for: opinion polls, consumer behavior, professional studies, most scenarios.
   - "couples": Both partners are agents (kids are background).
     Use for: relationship dynamics, dual-income household studies, retirement planning.
   - "all": Everyone in the household including children is an agent.
     Use for: family dynamics, community health, school policy, neighborhood studies.
   Default to "primary_only" unless the scenario clearly involves couple/partner or whole-family dynamics.

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

    llm_result = ScenarioSufficiencyResult(
        sufficient=data.get("sufficient", False),
        questions=questions,
        inferred_duration=data.get("inferred_duration"),
        inferred_timestep_unit=data.get("inferred_timestep_unit"),
        inferred_scenario_type=data.get("inferred_scenario_type"),
        has_explicit_outcomes=data.get("has_explicit_outcomes", False),
        inferred_agent_focus_mode=data.get("inferred_agent_focus_mode"),
    )
    return _postprocess_sufficiency(description=description, result=llm_result)


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
