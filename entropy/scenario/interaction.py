"""Step 3: Determine Interaction Model.

Selects the appropriate interaction model for how agents will discuss/respond
to the event and configures how information spreads through the network.
"""

from ..core.llm import reasoning_call
from ..core.models import (
    PopulationSpec,
    Event,
    InteractionConfig,
    InteractionType,
    SpreadConfig,
    SpreadModifier,
)


# JSON schema for interaction model response
INTERACTION_MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_model": {
            "type": "string",
            "enum": [
                "passive_observation",
                "direct_conversation",
                "broadcast_response",
                "deliberative",
            ],
            "description": "Primary interaction model",
        },
        "secondary_model": {
            "type": "string",
            "enum": [
                "passive_observation",
                "direct_conversation",
                "broadcast_response",
                "deliberative",
                "none",
            ],
            "description": "Secondary interaction model (or 'none')",
        },
        "interaction_description": {
            "type": "string",
            "description": "How interactions work in this scenario",
        },
        "share_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Base probability of sharing with a neighbor",
        },
        "share_modifiers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": "Condition (can reference agent or edge attrs)",
                    },
                    "multiply": {
                        "type": "number",
                        "description": "Multiplicative adjustment",
                    },
                    "add": {
                        "type": "number",
                        "description": "Additive adjustment",
                    },
                },
                "required": ["when", "multiply", "add"],
                "additionalProperties": False,
            },
        },
        "decay_per_hop": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Information fidelity loss per hop (0 = no decay)",
        },
        "max_hops": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum propagation depth (null for unlimited)",
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of model selection",
        },
    },
    "required": [
        "primary_model",
        "secondary_model",
        "interaction_description",
        "share_probability",
        "share_modifiers",
        "decay_per_hop",
        "max_hops",
        "reasoning",
    ],
    "additionalProperties": False,
}


def determine_interaction_model(
    event: Event,
    population_spec: PopulationSpec,
    network_summary: dict | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> tuple[InteractionConfig, SpreadConfig]:
    """
    Determine how agents will interact about the event.

    Selects:
    - Primary interaction model (how agents communicate)
    - Optional secondary model (for blended scenarios)
    - Spread configuration (how information propagates)

    Args:
        event: The parsed event definition
        population_spec: The population spec for context
        network_summary: Optional dict with network statistics (edge_types, node_count)
        model: LLM model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        Tuple of (InteractionConfig, SpreadConfig)

    Example:
        >>> interaction, spread = determine_interaction_model(event, population_spec)
        >>> interaction.primary_model
        <InteractionType.PASSIVE_OBSERVATION: 'passive_observation'>
        >>> spread.share_probability
        0.35
    """
    # Build attribute list for spread modifiers
    attribute_info = "\n".join(
        f"- {attr.name} ({attr.type})"
        for attr in population_spec.attributes
    )

    # Build edge type info if available
    edge_type_info = ""
    if network_summary and network_summary.get("edge_types"):
        edge_types = network_summary["edge_types"]
        edge_type_info = f"""
### Network Edge Types (use in share_modifiers)

The following edge types exist in this network: {', '.join(edge_types)}

Use these exact edge type names in share_modifier conditions, e.g.:
- {{"when": "edge_type == '{edge_types[0]}'", "multiply": 1.5, "add": 0}}
"""

    prompt = f"""## Task

Determine how agents in this population will interact about the event,
and configure how information spreads through their social network.

## Event

Type: {event.type.value}
Content: "{event.content[:200]}..."
Source: {event.source}
Credibility: {event.credibility:.2f}
Emotional valence: {event.emotional_valence:.2f}

## Population

{population_spec.meta.description}
Geography: {population_spec.meta.geography or 'Not specified'}

### Attributes (use in share_modifiers 'when' clauses)

{attribute_info}
{edge_type_info}

## Interaction Models

Choose the most appropriate interaction model(s):

### 1. passive_observation
Social media style — agents see content/reactions, form opinions passively.
- Low direct engagement
- Influenced by what they observe others doing
- Good for: consumer products, viral content, social media events

### 2. direct_conversation
One-on-one or small group discussions.
- High engagement
- Opinions can change through dialogue
- Good for: professional topics, personal decisions, complex issues

### 3. broadcast_response
Authority/source broadcasts, agents react and discuss reactions.
- Initial broadcast phase, then discussion
- Agents compare reactions with their network
- Good for: official announcements, policy changes, emergencies

### 4. deliberative
Group deliberation with multiple rounds.
- Structured discussion phases
- Opinions evolve through deliberation
- Good for: community decisions, policy debates, consensus-building

## Selection Heuristics

| Population Type | Event Type | Likely Model |
|-----------------|------------|---------------|
| Consumers | Price change | passive_observation + direct_conversation |
| Professionals | New technology | direct_conversation |
| Community | Government policy | broadcast_response |
| Any | Emergency | broadcast_response |
| Civic/Political | Policy debate | deliberative |

## Spread Configuration

### share_probability (0-1)
Base probability an agent shares/discusses with a neighbor.
- 0.1-0.2: Low engagement topics
- 0.2-0.4: Moderate interest
- 0.4-0.6: High interest/controversial
- 0.6-0.8: Urgent/emotional topics
- 0.8-1.0: Critical/emergency

### share_modifiers
Adjust share probability based on conditions:
- Emotional agents share more
- Young people share more on social topics
- Professionals share more on work topics
- Close relationships share more

Format: {{"when": "expression", "multiply": 1.5, "add": 0.0}}

Examples:
- {{"when": "extraversion > 0.7", "multiply": 1.3, "add": 0}}
- {{"when": "age < 30", "multiply": 1.5, "add": 0}}
- {{"when": "edge_type == 'colleague'", "multiply": 1.5, "add": 0}} (use actual edge types from network)
- {{"when": "institutional_trust < 0.3", "multiply": 2.0, "add": 0}} (psychographic: low-trust agents amplify rumors)

### decay_per_hop (0-1)
How much information fidelity is lost each time it's passed on.
- 0: No decay (official, verifiable info)
- 0.05-0.1: Low decay (clear, simple info)
- 0.1-0.2: Medium decay (complex info)
- 0.2+: High decay (rumors, ambiguous info)

### max_hops
Limit how far information can spread (null for unlimited).
- Typically 3-6 for most scenarios
- Lower for time-sensitive or niche topics
- Higher or null for viral content

## Output

Provide interaction model selection and spread configuration.
For secondary_model, use "none" if only one model applies."""

    data = reasoning_call(
        prompt=prompt,
        response_schema=INTERACTION_MODEL_SCHEMA,
        schema_name="interaction_model",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Parse interaction config
    secondary = data.get("secondary_model", "none")
    interaction_config = InteractionConfig(
        primary_model=InteractionType(data["primary_model"]),
        secondary_model=InteractionType(secondary) if secondary != "none" else None,
        description=data.get("interaction_description", ""),
    )

    # Parse spread modifiers
    modifiers = []
    for mod_data in data.get("share_modifiers", []):
        modifier = SpreadModifier(
            when=mod_data["when"],
            multiply=mod_data.get("multiply", 1.0),
            add=mod_data.get("add", 0.0),
        )
        modifiers.append(modifier)

    # Parse spread config
    spread_config = SpreadConfig(
        share_probability=max(0.0, min(1.0, data.get("share_probability", 0.3))),
        share_modifiers=modifiers,
        decay_per_hop=max(0.0, min(1.0, data.get("decay_per_hop", 0.1))),
        max_hops=data.get("max_hops"),
    )

    return interaction_config, spread_config
