"""LLM-based network configuration generator.

Given a PopulationSpec (and optionally sample agents), generates a NetworkConfig
that defines meaningful social structure for the population. This replaces the
need for hardcoded domain-specific presets.

Usage:
    config = generate_network_config(population_spec)
    config.to_yaml("network-config.yaml")
"""

import json
import logging

from ...core.llm import reasoning_call, ValidatorCallback
from ...core.models import PopulationSpec
from .config import (
    NetworkConfig,
    AttributeWeightConfig,
    DegreeMultiplierConfig,
    EdgeTypeRule,
    InfluenceFactorConfig,
    StructuralAttributeRoles,
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
                        "type": ["number", "null"],
                        "description": "For numeric_range: normalization range. For within_n: allowed level distance. null if not applicable.",
                    },
                    "ordinal_levels": {
                        "type": ["string", "null"],
                        "description": (
                            "For within_n only: JSON string mapping option values to ordinal integers. "
                            'E.g., \'{"junior": 1, "mid": 2, "senior": 3}\'. '
                            "null for exact/numeric_range."
                        ),
                    },
                },
                "required": [
                    "attribute",
                    "weight",
                    "match_type",
                    "range_value",
                    "ordinal_levels",
                ],
                "additionalProperties": False,
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
                        "type": ["string", "number", "boolean"],
                        "description": (
                            "Value that triggers the multiplier (exact match). "
                            "Use native booleans/numbers for boolean/numeric attributes."
                        ),
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
                "additionalProperties": False,
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
                "additionalProperties": False,
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
                        "type": ["string", "null"],
                        "description": (
                            "For ordinal type: JSON string mapping values to rank integers. Higher = more influence. "
                            'E.g., \'{"low": 1, "medium": 2, "high": 3}\'. '
                            "null for boolean/numeric types."
                        ),
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
                "required": ["attribute", "type", "levels", "weight", "description"],
                "additionalProperties": False,
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

STRUCTURAL_ROLE_SCHEMA = {
    "type": "object",
    "properties": {
        "household_id": {"type": ["string", "null"]},
        "partner_id": {"type": ["string", "null"]},
        "age": {"type": ["string", "null"]},
        "sector": {"type": ["string", "null"]},
        "region": {"type": ["string", "null"]},
        "urbanicity": {"type": ["string", "null"]},
        "religion": {"type": ["string", "null"]},
        "dependents": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
    },
    "required": [
        "household_id",
        "partner_id",
        "age",
        "sector",
        "region",
        "urbanicity",
        "religion",
        "dependents",
        "reasoning",
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
    max_prompt_attrs = 50
    for attr in population_spec.attributes[:max_prompt_attrs]:
        line = f"  - {attr.name} ({attr.type})"
        if attr.category:
            line += f" [{attr.category}]"
        if hasattr(attr, "sampling") and attr.sampling:
            dist = attr.sampling.distribution
            dist_options = getattr(dist, "options", None) if dist else None
            if dist_options:
                opts = ", ".join(dist_options[:8])
                if len(dist_options) > 8:
                    opts += f", ... ({len(dist_options)} total)"
                line += f": options=[{opts}]"
            elif dist and dist.type:
                line += f": {dist.type}"
                dist_mean = getattr(dist, "mean", None)
                if dist_mean is not None:
                    line += f" (mean={dist_mean})"
        attr_lines.append(line)
    if len(population_spec.attributes) > max_prompt_attrs:
        attr_lines.append(
            f"  - ... ({len(population_spec.attributes) - max_prompt_attrs} more attributes omitted)"
        )

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
- For degree_multipliers condition_value, use the same type as the attribute (boolean/number/string)
    """


def _extract_sample_value(
    agents_sample: list[dict] | None, attr_name: str
) -> object | None:
    """Return a representative non-null sample value for an attribute."""
    if not agents_sample:
        return None
    for agent in agents_sample:
        value = agent.get(attr_name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _score_structural_role_candidate(
    *,
    role_name: str,
    attr,
    sample_value: object | None,
) -> float:
    """Deterministic score for assigning an attribute to a structural role."""
    name = attr.name.lower()
    desc = (attr.description or "").lower()
    semantic = (attr.semantic_type or "").lower()
    identity = (attr.identity_type or "").lower()
    scope = (attr.scope or "").lower()
    attr_type = (attr.type or "").lower()
    text = f"{name} {desc}"

    def has_any(*tokens: str) -> bool:
        return any(token in text for token in tokens)

    score = 0.0
    if role_name == "household_id":
        if "household" in name:
            score += 3.0
        if name.endswith("_id"):
            score += 1.5
        if scope == "household":
            score += 1.5
        if has_any("household", "co-resid", "home"):
            score += 1.0
        if attr_type == "boolean":
            score -= 2.0

    elif role_name == "partner_id":
        if has_any("partner", "spouse", "husband", "wife"):
            score += 3.0
        if name.endswith("_id"):
            score += 2.0
        if has_any("agent id", "spouse id", "partner id"):
            score += 1.5
        if attr_type == "boolean":
            score -= 2.0
        if isinstance(sample_value, str) and (
            sample_value.startswith("agent_") or sample_value.startswith("a")
        ):
            score += 1.0

    elif role_name == "age":
        if name == "age":
            score += 6.0
        if semantic == "age":
            score += 3.0
        if has_any("age", "years old", "years_old", "current age"):
            score += 2.0
        if attr_type in {"int", "float"}:
            score += 1.0

    elif role_name == "sector":
        if semantic in {"employment", "occupation"}:
            score += 3.0
        if has_any("employment", "sector", "occupation", "industry", "profession"):
            score += 2.5
        if identity == "professional_identity":
            score += 1.5
        if attr_type == "boolean":
            score -= 1.5

    elif role_name == "region":
        if has_any("region", "state", "province", "county", "city", "country"):
            score += 3.0
        if has_any("geography", "location", "district", "zip", "postal"):
            score += 1.5
        if scope == "household":
            score += 0.5

    elif role_name == "urbanicity":
        if has_any("urban", "rural", "suburban", "metro", "urbanicity"):
            score += 4.0
        if has_any("population density", "city type"):
            score += 1.0

    elif role_name == "religion":
        if identity == "religious_affiliation":
            score += 3.0
        if has_any("relig", "faith", "church", "denomination", "congregation"):
            score += 2.5
        if semantic == "occupation":
            score -= 1.0

    elif role_name == "dependents":
        if has_any("dependents", "children", "child", "household_members"):
            score += 3.0
        if has_any("has_children", "child_present"):
            score += 0.5
        if isinstance(sample_value, list):
            score += 4.0
        elif isinstance(sample_value, bool):
            # Boolean parental flags are not structural dependent records.
            score -= 4.0
        if attr_type == "boolean":
            score -= 2.0

    return score


def _resolve_structural_roles_deterministic(
    population_spec: PopulationSpec,
    agents_sample: list[dict] | None = None,
) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    """Resolve role mapping deterministically; return ambiguous slots for tie-break."""
    roles = [
        "household_id",
        "partner_id",
        "age",
        "sector",
        "region",
        "urbanicity",
        "religion",
        "dependents",
    ]
    resolved: dict[str, str | None] = {role: None for role in roles}
    ambiguous: dict[str, list[str]] = {}

    for role in roles:
        scored: list[tuple[float, str]] = []
        for attr in population_spec.attributes:
            sample_value = _extract_sample_value(agents_sample, attr.name)
            score = _score_structural_role_candidate(
                role_name=role,
                attr=attr,
                sample_value=sample_value,
            )
            if score <= 0:
                continue
            scored.append((score, attr.name))

        if not scored:
            resolved[role] = None
            continue

        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_attr = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -999.0
        margin = best_score - second_score

        # Confident deterministic pick.
        if best_score >= 3.5 and margin >= 0.8:
            resolved[role] = best_attr
            continue

        # Ambiguous: keep deterministic suggestion but ask LLM to tie-break.
        resolved[role] = best_attr
        ambiguous[role] = [name for _, name in scored[:3]]

    return resolved, ambiguous


def _build_structural_role_prompt(
    population_spec: PopulationSpec,
    *,
    locked_roles: dict[str, str | None] | None = None,
    ambiguous_candidates: dict[str, list[str]] | None = None,
) -> str:
    """Build structural-role prompt with full-schema visibility."""
    attribute_names = [attr.name for attr in population_spec.attributes]
    attr_lines = "\n".join(f"- {name}" for name in attribute_names)

    locked_lines = ""
    if locked_roles:
        fixed = [
            f"- {role}: {value}"
            for role, value in locked_roles.items()
            if value is not None and role not in (ambiguous_candidates or {})
        ]
        if fixed:
            locked_lines = "\nLocked role assignments (do not change):\n" + "\n".join(
                fixed
            )

    ambiguous_lines = ""
    if ambiguous_candidates:
        parts = []
        for role, candidates in ambiguous_candidates.items():
            options = ", ".join(candidates) if candidates else "null"
            parts.append(f"- {role}: choose one of [{options}] or null")
        ambiguous_lines = (
            "\nAmbiguous roles to decide (only choose from listed options):\n"
            + "\n".join(parts)
        )

    return f"""Map structural network roles to exact attribute names from this population.

Population: {population_spec.meta.description}
Geography/context: {population_spec.meta.geography or "unspecified"}

Available attribute names:
{attr_lines}
{locked_lines}
{ambiguous_lines}

Roles to map (return exact attribute name or null):
- household_id: household grouping identifier
- partner_id: partner/spouse agent id reference
- age: person's age
- sector: occupation/employment sector
- region: location bucket (state/province/country/region)
- urbanicity: urban/rural style location bucket
- religion: faith/religious affiliation
- dependents: list of children/dependents

Rules:
- NEVER invent names; choose only from Available attribute names.
- If role is not represented, return null.
- Keep this mapping pragmatic for structural edges: partner, household, coworker, neighbor, congregation, school_parent.
- For roles listed as ambiguous above, choose only from those candidate options or null.
- Keep locked roles unchanged.
"""


def _generate_structural_attribute_roles(
    population_spec: PopulationSpec,
    model: str | None = None,
    agents_sample: list[dict] | None = None,
) -> StructuralAttributeRoles:
    """Resolve structural-role mapping deterministically, with LLM tie-break fallback."""
    role_names = (
        "household_id",
        "partner_id",
        "age",
        "sector",
        "region",
        "urbanicity",
        "religion",
        "dependents",
    )
    deterministic_roles, ambiguous = _resolve_structural_roles_deterministic(
        population_spec=population_spec,
        agents_sample=agents_sample,
    )

    # If deterministic resolution is confident, skip LLM entirely.
    if not ambiguous:
        return StructuralAttributeRoles.model_validate(deterministic_roles)

    prompt = _build_structural_role_prompt(
        population_spec,
        locked_roles=deterministic_roles,
        ambiguous_candidates=ambiguous,
    )
    data = reasoning_call(
        prompt=prompt,
        response_schema=STRUCTURAL_ROLE_SCHEMA,
        schema_name="structural_roles",
        model=model,
        reasoning_effort="low",
        max_retries=1,
    )

    known_attrs = {attr.name for attr in population_spec.attributes}
    roles: dict[str, str | None] = {}
    for role_name in role_names:
        locked = deterministic_roles.get(role_name)
        value = data.get(role_name)
        if role_name in ambiguous:
            allowed = set(ambiguous[role_name])
            if (
                isinstance(value, str)
                and value in known_attrs
                and (not allowed or value in allowed)
            ):
                roles[role_name] = value
            else:
                roles[role_name] = locked
        else:
            roles[role_name] = locked
    return StructuralAttributeRoles.model_validate(roles)


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
        if w.get("match_type") == "within_n":
            raw_ordinal = w.get("ordinal_levels")
            has_ordinal = False
            if isinstance(raw_ordinal, dict) and raw_ordinal:
                has_ordinal = True
            elif isinstance(raw_ordinal, str) and raw_ordinal:
                try:
                    parsed = json.loads(raw_ordinal)
                    has_ordinal = isinstance(parsed, dict) and len(parsed) > 0
                except (json.JSONDecodeError, TypeError):
                    pass
            if not has_ordinal:
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
    """Create a validator callback for reasoning_call retry.

    Returns a validator that conforms to ValidatorCallback signature:
    (data: dict) -> tuple[bool, str] where bool is validity and str is error message.
    """

    def validator(data: dict) -> tuple[bool, str]:
        errors = _validate_config(data, population_spec)
        if not errors:
            return True, ""
        # Format errors for retry prompt - include all errors so LLM can fix them
        error_msg = "VALIDATION FAILED. Fix these errors:\n" + "\n".join(
            f"- {e}" for e in errors
        )
        return False, error_msg

    return validator


def _convert_to_network_config(
    data: dict, population_spec: PopulationSpec
) -> NetworkConfig:
    """Convert validated LLM response dict to a NetworkConfig."""
    attribute_types = {attr.name: attr.type for attr in population_spec.attributes}

    def coerce_condition_value(attribute: str, value: object) -> object:
        """Coerce degree multiplier condition to the runtime attribute type."""
        attr_type = attribute_types.get(attribute)
        if attr_type is None:
            return value

        if attr_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered == "true":
                    return True
                if lowered == "false":
                    return False
            return value

        if attr_type == "int":
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                text = value.strip()
                try:
                    return int(text)
                except ValueError:
                    return value
            return value

        if attr_type == "float":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                text = value.strip()
                try:
                    return float(text)
                except ValueError:
                    return value
            return value

        return value

    # Build attribute weights
    attribute_weights: dict[str, AttributeWeightConfig] = {}
    ordinal_levels: dict[str, dict[str, int]] = {}

    for w in data.get("attribute_weights", []):
        attr_name = w["attribute"]
        # Parse ordinal_levels from JSON string if present
        raw_ordinal = w.get("ordinal_levels")
        attr_ordinal = None
        if isinstance(raw_ordinal, str):
            try:
                attr_ordinal = json.loads(raw_ordinal)
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(raw_ordinal, dict):
            attr_ordinal = raw_ordinal
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
            condition=coerce_condition_value(dm["attribute"], dm["condition_value"]),
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
    influence_factors = []
    for f in data.get("influence_factors", []):
        # Parse levels from JSON string if present
        raw_levels = f.get("levels")
        parsed_levels = None
        if isinstance(raw_levels, str):
            try:
                parsed_levels = json.loads(raw_levels)
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(raw_levels, dict):
            parsed_levels = raw_levels
        influence_factors.append(
            InfluenceFactorConfig(
                attribute=f["attribute"],
                type=f["type"],
                levels=parsed_levels,
                weight=f["weight"],
                description=f.get("description", ""),
            )
        )

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

    Uses two focused LLM calls:
    1) core network config (weights, multipliers, edge rules, influence, avg degree)
    2) structural attribute role mapping (household/partner/region/etc.)

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
    config = config.model_copy(
        update={
            "structural_attribute_roles": _generate_structural_attribute_roles(
                population_spec,
                model=model,
                agents_sample=agents_sample,
            )
        }
    )

    logger.info(
        f"[NetworkConfigGen] Generated config: "
        f"{len(config.attribute_weights)} weights, "
        f"{len(config.degree_multipliers)} multipliers, "
        f"{len(config.edge_type_rules)} edge rules, "
        f"{len(config.influence_factors)} influence factors, "
        f"structural roles="
        f"{sum(1 for v in config.structural_attribute_roles.model_dump().values() if v)}"
    )

    return config
