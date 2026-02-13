"""Step 2: Generate Seed Exposure Rules.

Determines how agents are initially exposed to an event through various channels,
generating exposure rules based on the event type, population characteristics,
and network structure.
"""

from ..core.llm import reasoning_call
from ..core.models import (
    PopulationSpec,
    Event,
    ExposureChannel,
    ExposureRule,
    SeedExposure,
)


# JSON schema for seed exposure response
SEED_EXPOSURE_SCHEMA = {
    "type": "object",
    "properties": {
        "channels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Channel identifier in snake_case",
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description",
                    },
                    "reach": {
                        "type": "string",
                        "enum": ["broadcast", "targeted", "organic"],
                        "description": "How the channel reaches agents",
                    },
                    "credibility_modifier": {
                        "type": "number",
                        "description": "How channel affects perceived credibility (1.0 = no change)",
                    },
                },
                "required": ["name", "description", "reach", "credibility_modifier"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 5,
        },
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Which channel this rule uses",
                    },
                    "when": {
                        "type": "string",
                        "description": "Python expression using agent attributes (use 'true' for all)",
                    },
                    "probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Probability of exposure",
                    },
                    "timestep": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "When this exposure occurs",
                    },
                },
                "required": ["channel", "when", "probability", "timestep"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of exposure strategy",
        },
    },
    "required": ["channels", "rules", "reasoning"],
    "additionalProperties": False,
}


def generate_seed_exposure(
    event: Event,
    population_spec: PopulationSpec,
    network_summary: dict | None = None,
    model: str | None = None,
    reasoning_effort: str = "low",
) -> SeedExposure:
    """
    Generate seed exposure configuration for an event.

    Determines:
    - Which channels will expose agents to the event
    - Which agents are exposed through each channel (via 'when' clauses)
    - When exposures occur (timesteps)
    - Probability of exposure

    Args:
        event: The parsed event definition
        population_spec: The population spec for attribute references
        network_summary: Optional dict with network statistics (edge_types, node_count)
        model: LLM model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        SeedExposure with channels and rules

    Example:
        >>> exposure = generate_seed_exposure(event, population_spec)
        >>> len(exposure.channels)
        3
        >>> exposure.rules[0].when
        'true'
    """
    # Build attribute list for LLM
    attribute_info = "\n".join(
        f"- {attr.name} ({attr.type}): {attr.description}"
        for attr in population_spec.attributes
    )

    # Build network summary if available
    network_info = ""
    if network_summary:
        edge_types = network_summary.get("edge_types", [])
        if edge_types:
            network_info = f"""
Network edge types available: {", ".join(edge_types)}
This means agents can be connected via: {", ".join(edge_types)}
Consider how information might spread through these network connections.
"""

    prompt = f"""## Task

Design how agents in this population will initially be exposed to the event.
Generate exposure channels and rules that determine which agents learn about
the event, when, and through what means.

## Event

Type: {event.type.value}
Content: "{event.content}"
Source: {event.source}
Credibility: {event.credibility:.2f}
Ambiguity: {event.ambiguity:.2f}
Emotional valence: {event.emotional_valence:.2f}

## Population

{population_spec.meta.description} ({population_spec.meta.size} agents)
Geography: {population_spec.meta.geography or "Not specified"}

### Available Attributes (use these in 'when' clauses)

{attribute_info}
{network_info}

## Channels

Define 2-5 exposure channels appropriate for this scenario.

**Channel Types:**

1. **broadcast** - Reaches all agents meeting criteria (email blasts, TV news)
   - Use for official communications, mass media
   - High reach, consistent timing

2. **targeted** - Reaches specific subgroups (targeted ads, professional networks)
   - Use for demographic-specific channels
   - Medium reach, can be very specific

3. **organic** - Spreads through existing connections (word of mouth, social media shares)
   - Use for informal information spread
   - Variable reach, depends on network

**Credibility Modifiers:**
- 1.0 = Channel doesn't affect credibility
- 1.1-1.3 = Channel adds credibility (e.g., official email)
- 0.6-0.9 = Channel reduces credibility (e.g., social media rumor)

## Rules

Create exposure rules that define WHICH agents get exposed WHEN.

**'when' clause syntax:**
- Use Python expression syntax
- Reference attributes exactly as named above
- Use 'true' to match all agents
- Examples:
  - "true" (everyone)
  - "age < 45"
  - "income > 50000"
  - "subscription_tier == 'premium'"

**timestep:**
- 0 = immediately
- 1, 2, 3... = subsequent timesteps
- Stagger exposures for realistic dynamics

**probability:**
- 1.0 = certain exposure if condition met
- 0.0-1.0 = probabilistic exposure

## Exposure Strategy Guidance

Consider how this event would realistically reach this population:

1. **Announcements/Official Communications:**
   - Direct channels (email, app notification) at t=0
   - Media coverage at t=1
   - Social media discussion at t=1-2

2. **News:**
   - Media consumers first (t=0)
   - Social media at t=1
   - Word of mouth at t=2+

3. **Rumors:**
   - Start with small group (low probability, t=0)
   - Spread organically (t=1+)
   - Rumors thrive on **ambiguity**—target agents in "epistemic bubbles" who lack access to authoritative sources
   - For rumors, consider how 'low trust' or 'high anxiety' groups might be exposed first (e.g., when: "institutional_trust < 0.4" or "anxiety > 0.6")

4. **Policy Changes:**
   - Official channels (t=0)
   - Affected groups first (t=0-1)
   - General awareness later (t=2+)

## Output

Provide channels and rules with proper attribute references.
Rules must use exact attribute names from the list above."""

    data = reasoning_call(
        prompt=prompt,
        response_schema=SEED_EXPOSURE_SCHEMA,
        schema_name="seed_exposure",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Parse channels
    channels = []
    for ch_data in data.get("channels", []):
        channel = ExposureChannel(
            name=ch_data["name"],
            description=ch_data["description"],
            reach=ch_data["reach"],
            credibility_modifier=ch_data.get("credibility_modifier", 1.0),
        )
        channels.append(channel)

    # Build set of valid channel names
    valid_channels = {ch.name for ch in channels}

    # Parse rules
    rules = []
    for rule_data in data.get("rules", []):
        # Skip rules that reference undefined channels
        if rule_data["channel"] not in valid_channels:
            continue

        # Clamp probability to valid range
        probability = max(0.0, min(1.0, rule_data.get("probability", 0.5)))

        # Ensure timestep is non-negative
        timestep = max(0, rule_data.get("timestep", 0))

        rule = ExposureRule(
            channel=rule_data["channel"],
            when=rule_data["when"],
            probability=probability,
            timestep=timestep,
        )
        rules.append(rule)

    return SeedExposure(channels=channels, rules=rules)
