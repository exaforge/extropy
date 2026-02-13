"""Step 0: Context Sufficiency Check.

Verifies that the population description has enough information
to proceed with attribute discovery and spec generation.
"""

from ...core.llm import simple_call
from ...core.models import SufficiencyResult

# JSON schema for sufficiency check response
SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": "Whether the description has enough information to create a population",
        },
        "size": {
            "type": "integer",
            "description": "Extracted population size, or 1000 if not specified",
        },
        "geography": {
            "type": ["string", "null"],
            "description": "Geographic scope if mentioned (e.g., 'Germany', 'US', 'California')",
        },
        "clarifications_needed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of clarifying questions if insufficient",
        },
    },
    "required": ["sufficient", "size", "geography", "clarifications_needed"],
    "additionalProperties": False,
}


def check_sufficiency(
    description: str,
    default_size: int = 1000,
    model: str | None = None,
) -> SufficiencyResult:
    """
    Check if a population description is sufficient for spec generation.

    A sufficient description should specify:
    1. Who they are (identity/profession/demographic)
    2. Size (number of agents) - optional, defaults to 1000
    3. Geographic scope - optional but helpful

    Args:
        description: Natural language population description
        default_size: Default size if not specified
        model: Model to use (gpt-5-mini recommended for speed)

    Returns:
        SufficiencyResult with sufficient flag, size, and any clarifications needed
    """
    prompt = f"""Evaluate if this population description is sufficient to create a synthetic population:

"{description}"

A sufficient description should specify:
1. WHO they are - a clear identity, profession, or demographic group
   ✓ Good: "surgeons", "Netflix subscribers", "German farmers", "Tesla owners"
   ✗ Bad: "people", "users" (too vague)

2. SIZE (optional) - number of agents to create
   - Extract if mentioned (e.g., "500 surgeons" → 500)
   - Default to {default_size} if not specified

3. GEOGRAPHY (optional) - geographic scope
   - Extract if mentioned (e.g., "German surgeons" → Germany)
   - Can be country, region, or city

If the description is too vague to create meaningful attributes, mark as insufficient
and provide specific clarifying questions.

Be lenient - if you can reasonably infer a specific population, mark as sufficient."""

    data = simple_call(
        prompt=prompt,
        response_schema=SUFFICIENCY_SCHEMA,
        schema_name="sufficiency_check",
        model=model,
    )

    return SufficiencyResult(
        sufficient=data.get("sufficient", False),
        size=data.get("size", default_size),
        geography=data.get("geography"),
        clarifications_needed=data.get("clarifications_needed", []),
    )
