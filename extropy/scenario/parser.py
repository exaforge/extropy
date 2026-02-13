"""Step 1: Parse Scenario Description.

Parses a natural language scenario description into a structured Event definition
with type, content, source, credibility, ambiguity, and emotional valence.
"""

from ..core.llm import reasoning_call
from ..core.models import PopulationSpec, Event, EventType


# JSON schema for event parsing response
EVENT_PARSING_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": [
                "announcement",
                "news",
                "rumor",
                "policy_change",
                "product_launch",
                "emergency",
                "observation",
            ],
            "description": "Type of event being introduced",
        },
        "content": {
            "type": "string",
            "description": "The full event content/information that agents will receive",
        },
        "source": {
            "type": "string",
            "description": "Who or what originated this event",
        },
        "credibility": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "How credible is the source (0=not credible, 1=fully credible)",
        },
        "ambiguity": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "How clear/unclear is the information (0=crystal clear, 1=very ambiguous)",
        },
        "emotional_valence": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Emotional framing (-1=very negative, 0=neutral, 1=very positive)",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of classification decisions",
        },
    },
    "required": [
        "event_type",
        "content",
        "source",
        "credibility",
        "ambiguity",
        "emotional_valence",
        "reasoning",
    ],
    "additionalProperties": False,
}


def parse_scenario(
    description: str,
    population_spec: PopulationSpec,
    model: str | None = None,
    reasoning_effort: str = "low",
) -> Event:
    """
    Parse a scenario description into a structured Event.

    Analyzes the scenario description to extract:
    - Event type (announcement, news, rumor, etc.)
    - Full content of what agents will learn
    - Source of the information
    - Credibility, ambiguity, and emotional valence

    Args:
        description: Natural language scenario description
        population_spec: The population spec for context
        model: LLM model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Event model with all fields populated

    Example:
        >>> event = parse_scenario(
        ...     "Netflix announces $3 price increase",
        ...     population_spec
        ... )
        >>> event.type
        <EventType.ANNOUNCEMENT: 'announcement'>
        >>> event.source
        'Netflix'
    """
    population_context = f"""
Population: {population_spec.meta.description}
Size: {population_spec.meta.size} agents
Geography: {population_spec.meta.geography or "Not specified"}

Key attributes:
{chr(10).join(f"- {attr.name}: {attr.description}" for attr in population_spec.attributes[:10])}
"""

    prompt = f"""## Task

Parse the following scenario description into a structured event definition.

## Population Context

{population_context}

## Scenario Description

"{description}"

## Event Types

Choose the most appropriate event type:

1. **announcement**: Official communication from an organization/authority
   - Examples: government mandates, public health guidance, platform-wide service changes
   - High credibility if from known entity

2. **news**: Information reported by media/journalists
   - Examples: news articles, press coverage, investigative reports
   - Medium credibility, depends on source

3. **rumor**: Unverified information spreading informally
   - Examples: gossip, speculation, "I heard that..."
   - Low credibility, high ambiguity

4. **policy_change**: Changes to rules, regulations, or laws
   - Examples: legislation, regulatory shifts, professional board certification changes
   - High credibility if from authority

5. **product_launch**: Market-wide innovation or introduction of transformative product/service
   - Examples: med-tech firm launches AI diagnostic tool, autonomous vehicle rollout, breakthrough treatment approval
   - High credibility from official source

6. **emergency**: Urgent, time-sensitive information
   - Examples: safety alerts, recalls, urgent warnings
   - Very high credibility needed, low ambiguity expected

7. **observation**: Something agents notice/observe
   - Examples: market trends, behavioral changes, environmental changes
   - Credibility varies, often medium ambiguity

## Guidelines

**Content**: Write out the full information as it would be received by agents. Be specific and concrete - this is what agents will actually process.

**Source**: Identify who/what is originating this information. Be specific (e.g., "Netflix" not "the company", "FDA" not "regulators").

**Credibility** (0-1):
- 0.9-1.0: Official, authoritative source (government, established company)
- 0.7-0.9: Reputable media, industry experts
- 0.4-0.7: Mixed or uncertain sources
- 0.0-0.4: Rumors, anonymous sources, unverified

**Ambiguity** (0-1):
- 0.0-0.2: Crystal clear, no room for interpretation
- 0.2-0.5: Mostly clear with some details unclear
- 0.5-0.8: Significant room for interpretation
- 0.8-1.0: Very vague, open to many interpretations

**Emotional Valence** (-1 to 1):
- -1 to -0.5: Clearly negative (bad news, threats, losses)
- -0.5 to 0: Mildly negative or concerning
- 0: Neutral, informational
- 0 to 0.5: Mildly positive or promising
- 0.5 to 1: Clearly positive (good news, benefits, gains)

Consider the perspective of the population when assessing emotional valence. A price increase is negative for consumers but might be neutral or positive for competitors.

## Output

Provide the structured event definition with all required fields."""

    data = reasoning_call(
        prompt=prompt,
        response_schema=EVENT_PARSING_SCHEMA,
        schema_name="event_parsing",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Clamp values to valid ranges (in case LLM returns out-of-range)
    credibility = max(0.0, min(1.0, data.get("credibility", 0.5)))
    ambiguity = max(0.0, min(1.0, data.get("ambiguity", 0.3)))
    emotional_valence = max(-1.0, min(1.0, data.get("emotional_valence", 0.0)))

    return Event(
        type=EventType(data["event_type"]),
        content=data["content"],
        source=data["source"],
        credibility=credibility,
        ambiguity=ambiguity,
        emotional_valence=emotional_valence,
    )
