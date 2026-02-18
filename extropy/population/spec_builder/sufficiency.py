"""Step 0: Context Sufficiency Check.

Verifies that the population description has enough information
to proceed with attribute discovery and spec generation.
"""

from ...core.llm import simple_call
from ...core.models import ClarificationQuestion, SufficiencyResult

# JSON schema for sufficiency check response
SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": "Whether the description has enough information to create a population",
        },
        "geography": {
            "type": ["string", "null"],
            "description": "Geographic scope if mentioned (e.g., 'Germany', 'US', 'California')",
        },
        "agent_focus": {
            "type": ["string", "null"],
            "description": "Natural language description of who should be simulated (e.g., 'families', 'surgeons', 'couples'). Used for documentation.",
        },
        "clarifications_needed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of clarifying questions if insufficient (kept for backward compat)",
        },
        "questions": {
            "type": "array",
            "description": "Structured clarification questions with IDs, types, and options",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "snake_case identifier, e.g. 'geography'",
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
        "geography",
        "agent_focus",
        "clarifications_needed",
        "questions",
    ],
    "additionalProperties": False,
}


def check_sufficiency(
    description: str,
    model: str | None = None,
) -> SufficiencyResult:
    """
    Check if a population description is sufficient for spec generation.

    A sufficient description should specify:
    1. Who they are (identity/profession/demographic)
    2. Geographic scope - optional but helpful

    Args:
        description: Natural language population description
        model: Model to use (gpt-5-mini recommended for speed)

    Returns:
        SufficiencyResult with sufficient flag and any clarifications needed
    """
    prompt = f"""Evaluate if this population description is sufficient to create a synthetic population:

"{description}"

A sufficient description should specify:
1. WHO they are - a clear identity, profession, or demographic group
   ✓ Good: "surgeons", "Netflix subscribers", "German farmers", "Tesla owners"
   ✗ Bad: "people", "users" (too vague)

2. GEOGRAPHY (optional) - geographic scope
   - Extract if mentioned (e.g., "German surgeons" → Germany)
   - Can be country, region, or city

3. AGENT FOCUS - who should be simulated as active agents?
   This controls how households are sampled:
   - "families" or "households" → everyone in household gets simulated (both partners, older kids)
     Use for: communities, neighborhoods, social dynamics, local issues
   - "couples" or "partners" → both adults simulated, children are background NPCs
     Use for: relationship studies, retirement planning, couple decisions
   - Specific roles like "surgeons", "subscribers" → one person per household, partner/kids are NPCs
     Use for: professional studies, product research, individual behavior

   Examples:
   - "community reacting to school policy" → "families" (both parents may have opinions)
   - "German surgeons" → "surgeons" (study is about the surgeon, not their family)
   - "retired couples planning travel" → "couples" (both partners matter)

If the description is too vague to create meaningful attributes, mark as insufficient
and provide BOTH:
1. `clarifications_needed`: Simple list of question strings (for backward compat)
2. `questions`: Structured questions with these fields:
   - id: snake_case identifier (e.g., "geography", "agent_focus")
   - question: Human-readable question text
   - type: "single_choice" (with options), "text" (free text), or "number"
   - options: Array of choices for single_choice (null for text/number)
   - default: Reasonable default value if any (helps with --use-defaults flag)

Example structured question:
{{
  "id": "geography",
  "question": "What country or region should these surgeons be from?",
  "type": "single_choice",
  "options": ["United States", "Germany", "United Kingdom", "Other"],
  "default": "United States"
}}

Be lenient - if you can reasonably infer a specific population, mark as sufficient."""

    data = simple_call(
        prompt=prompt,
        response_schema=SUFFICIENCY_SCHEMA,
        schema_name="sufficiency_check",
        model=model,
    )

    # Parse structured questions from response
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

    return SufficiencyResult(
        sufficient=data.get("sufficient", False),
        geography=data.get("geography"),
        agent_focus=data.get("agent_focus"),
        clarifications_needed=data.get("clarifications_needed", []),
        questions=questions,
    )


def check_sufficiency_with_answers(
    description: str,
    answers: dict[str, str | int],
    model: str | None = None,
) -> SufficiencyResult:
    """Re-check sufficiency with pre-supplied answers.

    Enriches the description with answer context and re-runs the sufficiency check.

    Args:
        description: Original natural language population description
        answers: Dict mapping question IDs to answer values
        model: Model to use

    Returns:
        SufficiencyResult with updated sufficient flag
    """
    # Build enriched description with answers
    enriched = description
    for key, value in answers.items():
        enriched += f" | {key}: {value}"

    return check_sufficiency(enriched, model)
