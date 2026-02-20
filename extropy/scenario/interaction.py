"""Step 3: Determine Interaction Model.

Selects the appropriate interaction model for how agents will discuss/respond
to the event and configures how information spreads through the network.
"""

from ..core.llm import reasoning_call
from ..core.models import (
    PopulationSpec,
    Event,
    InteractionConfig,
    SpreadConfig,
    SpreadModifier,
)


# JSON schema for interaction model response
INTERACTION_MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "interaction_description": {
            "type": "string",
            "description": "Optional human-readable notes about social dynamics",
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
    model: str | None = None,
    reasoning_effort: str = "low",
) -> tuple[InteractionConfig, SpreadConfig]:
    """
    Determine how agents will interact about the event.

    Selects spread configuration (how information propagates) plus optional
    descriptive interaction notes.

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
        >>> interaction.description
        'Neighbors primarily discuss through informal chats and online posts'
        >>> spread.share_probability
        0.35
    """
    # Build attribute list for spread modifiers
    attribute_info = "\n".join(
        f"- {attr.name} ({attr.type})" for attr in population_spec.attributes
    )

    # Build edge type info if available
    edge_type_info = ""
    if network_summary and network_summary.get("edge_types"):
        edge_types = network_summary["edge_types"]
        edge_type_info = f"""
### Network Edge Types (use in share_modifiers)

The following edge types exist in this network: {", ".join(edge_types)}

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
Geography: {population_spec.meta.geography or "Not specified"}

### Attributes (use in share_modifiers 'when' clauses)

{attribute_info}
{edge_type_info}

## Interaction Notes

Provide a short `interaction_description` describing likely social dynamics.
This field is informational only; runtime behavior is controlled by spread rules below.

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

Provide interaction notes and spread configuration."""

    data = reasoning_call(
        prompt=prompt,
        response_schema=INTERACTION_MODEL_SCHEMA,
        schema_name="interaction_model",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Parse interaction config (informational only)
    interaction_config = InteractionConfig(
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
