"""LLM-based network configuration generator.

Given a PopulationSpec (and optionally sample agents), generates a NetworkConfig
that defines meaningful social structure for the population. This replaces the
need for hardcoded domain-specific configs like the German surgeon defaults.

Usage:
    config = generate_network_config(population_spec)
    config.to_yaml("network-config.yaml")
"""

import logging

from ...core.llm import reasoning_call, ValidatorCallback
from ...core.models import PopulationSpec
from .config import (
    NetworkConfig,
    AttributeWeightConfig,
    DegreeMultiplierConfig,
    EdgeTypeRule,
    InfluenceFactorConfig,
)

logger = logging.getLogger(__name__)


# JSON Schema for the LLM response
NETWORK_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "Explain your choices: why these attributes matter for social connections "
                "in this population, what edge types exist, and what drives influence."
            ),
        },
        "avg_degree": {
            "type": "number",
            "description": (
                "Target average connections per person. "
                "Professional networks: 15-30. Consumer populations: 5-15. "
                "Small tight-knit groups: 30-50."
            ),
        },
        "attribute_weights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {
                        "type": "string",
                        "description": "Attribute name from the population spec",
                    },
                    "weight": {
                        "type": "number",
                        "description": "Importance for forming connections (0.5-5.0). Higher = stronger homophily.",
                    },
                    "match_type": {
                        "type": "string",
                        "enum": ["exact", "numeric_range", "within_n"],
                        "description": (
                            "exact: categorical match. "
                            "numeric_range: closeness within a range. "
                            "within_n: ordinal proximity (needs ordinal_levels)."
                        ),
                    },
                    "range_value": {
                        "type": "number",
                        "description": "For numeric_range: normalization range. For within_n: allowed level distance.",
                    },
                    "ordinal_levels": {
                        "type": "object",
                        "description": (
                            "For within_n only: maps option values to ordinal integers. "
                            'E.g., {"junior": 1, "mid": 2, "senior": 3}. '
                            "Omit for exact/numeric_range."
                        ),
                        "additionalProperties": {"type": "integer"},
                    },
                },
                "required": ["attribute", "weight", "match_type"],
            },
            "description": "6-12 attributes that drive social connections. Pick the most socially relevant ones.",
        },
        "degree_multipliers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {
                        "type": "string",
                        "description": "Attribute name",
                    },
                    "condition_value": {
                        "type": "string",
                        "description": "Value that triggers the multiplier (exact match)",
                    },
                    "multiplier": {
                        "type": "number",
                        "description": "Degree multiplier (1.1-3.0). How many more connections this type has.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this type is more connected",
                    },
                },
                "required": ["attribute", "condition_value", "multiplier", "rationale"],
            },
            "description": "3-8 conditions that make certain people more connected (hubs).",
        },
        "edge_type_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short label for this connection type (e.g., 'coworker', 'neighbor', 'classmate')",
                    },
                    "condition": {
                        "type": "string",
                        "description": (
                            "Python expression using a_{attr} and b_{attr} variables. "
                            "E.g., 'a_company == b_company and a_department == b_department'. "
                            "Supports ==, !=, and, or, not."
                        ),
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Higher = checked first. Use 10-50 range. Most specific rules get highest priority.",
                    },
                    "description": {
                        "type": "string",
                        "description": "When does this connection type occur?",
                    },
                },
                "required": ["name", "condition", "priority", "description"],
            },
            "description": "3-6 edge type rules, from most specific (high priority) to most general (low priority).",
        },
        "influence_factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {
                        "type": "string",
                        "description": "Attribute that drives influence asymmetry",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["ordinal", "boolean", "numeric"],
                        "description": (
                            "ordinal: ranked levels (senior influences junior). "
                            "boolean: presence gives influence bonus. "
                            "numeric: ratio-based influence."
                        ),
                    },
                    "levels": {
                        "type": "object",
                        "description": "For ordinal type: maps values to rank integers. Higher = more influence.",
                        "additionalProperties": {"type": "integer"},
                    },
                    "weight": {
                        "type": "number",
                        "description": "Influence factor weight (0.1-0.5). Higher = more impact.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Why this attribute creates influence asymmetry",
                    },
                },
                "required": ["attribute", "type", "weight", "description"],
            },
            "description": "2-4 factors that create asymmetric influence (A influences B more than B influences A).",
        },
        "default_edge_type": {
            "type": "string",
            "description": "Fallback edge type when no rule matches. Usually 'acquaintance' or 'weak_tie'.",
        },
    },
    "required": [
        "reasoning",
        "avg_degree",
        "attribute_weights",
        "degree_multipliers",
        "edge_type_rules",
        "influence_factors",
        "default_edge_type",
    ],
    "additionalProperties": False,
}


def _build_prompt(
    population_spec: PopulationSpec,
    agents_sample: list[dict] | None = None,
) -> str:
    """Build the prompt for network config generation."""
    # Summarize population attributes
    attr_lines = []
    for attr in population_spec.attributes:
        line = f"  - {attr.name} ({attr.type})"
        if attr.category:
            line += f" [{attr.category}]"
        if hasattr(attr, "sampling") and attr.sampling:
            dist = attr.sampling.distribution
            if dist and dist.options:
                opts = ", ".join(dist.options[:8])
                if len(dist.options) > 8:
                    opts += f", ... ({len(dist.options)} total)"
                line += f": options=[{opts}]"
            elif dist and dist.type:
                line += f": {dist.type}"
                if dist.mean is not None:
                    line += f" (mean={dist.mean})"
        attr_lines.append(line)

    attr_summary = "\n".join(attr_lines)

    # Sample agents if provided
    agent_section = ""
    if agents_sample:
        samples = agents_sample[:5]
        agent_lines = []
        for i, agent in enumerate(samples):
            # Show key attributes, skip _id
            attrs = {k: v for k, v in agent.items() if k != "_id"}
            agent_lines.append(f"  Agent {i}: {attrs}")
        agent_section = f"""

## Sample Agents
{chr(10).join(agent_lines)}
"""

    return f"""You are designing the social network configuration for a population simulation.

## Population
- Description: {population_spec.meta.description}
- Size: {population_spec.meta.size} people
- Location/context: {population_spec.meta.geography or "unspecified"}

## Available Attributes
{attr_summary}
{agent_section}
## Your Task

Design a network configuration that creates a realistic social graph for this population.
Think about HOW these people actually know each other in real life:

1. **Attribute weights**: Which attributes create homophily (people connecting with similar others)?
   - Only pick attributes that genuinely affect who knows whom
   - Geographic proximity, shared workplace/school, shared interests matter most
   - Personality traits and internal attitudes usually DON'T drive connections (skip those)

2. **Degree multipliers**: Who are the natural hubs?
   - Leaders, organizers, popular roles, people in positions that require many contacts
   - Use multipliers between 1.1 and 3.0

3. **Edge type rules**: What kinds of relationships exist?
   - Use conditions like "a_city == b_city" or "a_company == b_company and a_department == b_department"
   - Most specific rules get highest priority
   - Only reference attributes from the list above

4. **Influence factors**: Who influences whom more?
   - Seniority, expertise, authority, or status attributes
   - Only pick 2-4 factors — most attributes don't create influence asymmetry

5. **Average degree**: How many connections per person?
   - Professional networks of specialists: 15-30
   - General consumer populations: 5-15
   - Tight-knit communities: 25-50

IMPORTANT:
- Only reference attribute names that exist in the Available Attributes list above
- For within_n match type, you MUST provide ordinal_levels mapping
- For edge type conditions, use a_{{attribute}} and b_{{attribute}} syntax
- condition_value for degree_multipliers must be a string (use "true"/"false" for booleans)
"""


def _validate_config(data: dict, population_spec: PopulationSpec) -> list[str]:
    """Validate the LLM-generated config against the population spec.

    Returns list of error messages (empty = valid).
    """
    errors = []
    known_attrs = {attr.name for attr in population_spec.attributes}

    # Check attribute weights reference valid attributes
    weights = data.get("attribute_weights", [])
    if len(weights) < 3:
        errors.append(f"Need at least 3 attribute weights, got {len(weights)}")

    for w in weights:
        attr = w.get("attribute", "")
        if attr not in known_attrs:
            errors.append(f"attribute_weights references unknown attribute: '{attr}'")
        if w.get("weight", 0) <= 0:
            errors.append(f"attribute_weights '{attr}' has zero or negative weight")
        if w.get("match_type") == "within_n" and not w.get("ordinal_levels"):
            errors.append(
                f"attribute_weights '{attr}' uses within_n but has no ordinal_levels"
            )

    # Check degree multipliers reference valid attributes
    for dm in data.get("degree_multipliers", []):
        attr = dm.get("attribute", "")
        if attr not in known_attrs:
            errors.append(f"degree_multipliers references unknown attribute: '{attr}'")

    # Check edge type rules reference valid attributes
    for rule in data.get("edge_type_rules", []):
        condition = rule.get("condition", "")
        # Extract referenced attributes from condition
        for part in condition.replace("(", " ").replace(")", " ").split():
            if part.startswith("a_") or part.startswith("b_"):
                attr_name = part[2:]  # Strip a_/b_ prefix
                if attr_name not in known_attrs:
                    errors.append(
                        f"edge_type_rules '{rule.get('name')}' references "
                        f"unknown attribute: '{attr_name}'"
                    )

    # Check influence factors reference valid attributes
    for factor in data.get("influence_factors", []):
        attr = factor.get("attribute", "")
        if attr not in known_attrs:
            errors.append(f"influence_factors references unknown attribute: '{attr}'")

    return errors


def _make_validator(population_spec: PopulationSpec) -> ValidatorCallback:
    """Create a validator callback for reasoning_call retry."""

    def validator(data: dict) -> list[str]:
        return _validate_config(data, population_spec)

    return validator


def _convert_to_network_config(
    data: dict, population_spec: PopulationSpec
) -> NetworkConfig:
    """Convert validated LLM response dict to a NetworkConfig."""
    # Build attribute weights
    attribute_weights: dict[str, AttributeWeightConfig] = {}
    ordinal_levels: dict[str, dict[str, int]] = {}

    for w in data.get("attribute_weights", []):
        attr_name = w["attribute"]
        attr_ordinal = w.get("ordinal_levels")
        attribute_weights[attr_name] = AttributeWeightConfig(
            weight=w["weight"],
            match_type=w["match_type"],
            range_value=w.get("range_value"),
            ordinal_levels=attr_ordinal,
        )
        if attr_ordinal:
            ordinal_levels[attr_name] = attr_ordinal

    # Build degree multipliers
    degree_multipliers = [
        DegreeMultiplierConfig(
            attribute=dm["attribute"],
            condition=dm["condition_value"],
            multiplier=dm["multiplier"],
            rationale=dm["rationale"],
        )
        for dm in data.get("degree_multipliers", [])
    ]

    # Build edge type rules
    edge_type_rules = [
        EdgeTypeRule(
            name=rule["name"],
            condition=rule["condition"],
            priority=rule["priority"],
            description=rule.get("description", ""),
        )
        for rule in data.get("edge_type_rules", [])
    ]

    # Build influence factors
    influence_factors = [
        InfluenceFactorConfig(
            attribute=f["attribute"],
            type=f["type"],
            levels=f.get("levels"),
            weight=f["weight"],
            description=f.get("description", ""),
        )
        for f in data.get("influence_factors", [])
    ]

    return NetworkConfig(
        avg_degree=data.get("avg_degree", 20.0),
        attribute_weights=attribute_weights,
        degree_multipliers=degree_multipliers,
        edge_type_rules=edge_type_rules,
        influence_factors=influence_factors,
        default_edge_type=data.get("default_edge_type", "acquaintance"),
        ordinal_levels=ordinal_levels,
        generated_from=population_spec.meta.description,
        generation_rationale=data.get("reasoning"),
    )


def generate_network_config(
    population_spec: PopulationSpec,
    agents_sample: list[dict] | None = None,
    model: str | None = None,
) -> NetworkConfig:
    """Generate a NetworkConfig from a population spec using LLM reasoning.

    The LLM analyzes the population's attributes and determines:
    - Which attributes create social connections (homophily weights)
    - Who are the natural hubs (degree multipliers)
    - What types of relationships exist (edge type rules)
    - Who influences whom (influence factors)

    Args:
        population_spec: The population specification with attribute definitions
        agents_sample: Optional sample agents for additional context
        model: Optional model override

    Returns:
        NetworkConfig ready for use with generate_network()
    """
    prompt = _build_prompt(population_spec, agents_sample)
    validator = _make_validator(population_spec)

    logger.info("[NetworkConfigGen] Generating network config via LLM")

    data = reasoning_call(
        prompt=prompt,
        response_schema=NETWORK_CONFIG_SCHEMA,
        schema_name="network_config",
        model=model,
        reasoning_effort="low",
        validator=validator,
        max_retries=2,
    )

    config = _convert_to_network_config(data, population_spec)

    logger.info(
        f"[NetworkConfigGen] Generated config: "
        f"{len(config.attribute_weights)} weights, "
        f"{len(config.degree_multipliers)} multipliers, "
        f"{len(config.edge_type_rules)} edge rules, "
        f"{len(config.influence_factors)} influence factors"
    )

    return config
