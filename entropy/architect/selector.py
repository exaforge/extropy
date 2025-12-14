"""Step 1: Attribute Selection.

Discovers all relevant attributes for a population across four categories:
- Universal: Human attributes everyone has (age, gender, income, etc.)
- Population-specific: What makes this population distinct
- Context-specific: Only if a product/service is mentioned
- Personality: Behavioral/psychological traits when relevant
"""

from ..llm import reasoning_call
from ..models import AttributeSpec, DiscoveredAttribute


# JSON schema for attribute selection response
ATTRIBUTE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Attribute name in snake_case",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["int", "float", "categorical", "boolean"],
                        "description": "Data type of the attribute",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "universal",
                            "population_specific",
                            "context_specific",
                            "personality",
                        ],
                        "description": "Category of the attribute",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description of what this attribute represents",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["independent", "derived", "conditional"],
                        "description": "Sampling strategy: independent (no dependencies), derived (zero-variance formula), conditional (probabilistic dependency)",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of other attributes this depends on (empty for independent)",
                    },
                },
                "required": ["name", "type", "category", "description", "strategy", "depends_on"],
                "additionalProperties": False,
            },
        },
        "include_big_five": {
            "type": "boolean",
            "description": "Whether Big Five personality traits should be included",
        },
        "notes": {
            "type": "string",
            "description": "Any notes about attribute selection decisions",
        },
    },
    "required": ["attributes", "include_big_five", "notes"],
    "additionalProperties": False,
}


def select_attributes(
    description: str,
    size: int,
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> list[DiscoveredAttribute]:
    """
    Discover all relevant attributes for a population.

    Uses GPT-5 with reasoning to analyze the population and identify
    attributes across all applicable categories. The model considers:
    - What makes this population unique
    - Geographic/cultural context
    - Dependencies between attributes

    When context is provided (overlay mode), only discovers NEW attributes
    not already in the base population. Can reference context attributes
    in dependencies.

    Args:
        description: Natural language population description
        size: Number of agents (for context)
        geography: Geographic scope if known
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        List of DiscoveredAttribute objects

    Example:
        >>> attrs = select_attributes("German surgeons", 500, "Germany")
        >>> [a.name for a in attrs[:3]]
        ['age', 'gender', 'specialty']

        # Overlay mode
        >>> overlay_attrs = select_attributes(
        ...     "AI device adoption scenario", 500, "Germany",
        ...     context=base_spec.attributes
        ... )
    """
    geo_context = f" in {geography}" if geography else ""
    geo_label = geography or "the relevant region"

    # Build context section if we have existing attributes
    context_section = ""
    if context:
        context_items = "\n".join(
            f"- {attr.name} ({attr.type}): {attr.description}" for attr in context
        )
        context_section = f"""
## EXISTING CONTEXT (DO NOT REDISCOVER)

The following {len(context)} attributes ALREADY EXIST in the base population.
**DO NOT include these in your output.** You may reference them in depends_on.

{context_items}

---

"""
        # In overlay mode, adjust constraints
        constraint_note = f"""## Constraints (OVERLAY MODE)

- **Only discover NEW attributes for this scenario** (5-15 attributes typical)
- You may reference existing attributes in depends_on
- Focus on behavioral/situational attributes relevant to: "{description}"
- Max 3 dependencies per attribute (can include base attributes)
- NO duplicates with existing attributes above"""
    else:
        constraint_note = """## Constraints (STRICT)

- **Total: 25-40 attributes maximum**
- Universal demographics: 8-12 attributes
- Population-specific: 10-18 attributes
- Personality/behavioral: 5-8 attributes
- Max 3 dependencies per attribute
- NO duplicates
- Only attributes where real statistical data likely exists"""

    prompt = f"""{context_section}## Intent

We are building a synthetic population for **agent-based simulation**. These agents will:
- Have scenarios injected (events, information, decisions)
- Respond based on their attributes
- Interact with each other in a social network

The goal is realistic variance, not exhaustive detail. We need attributes that:
1. Create meaningful differences in how agents BEHAVE
2. Can be grounded in real data (researchable distributions)
3. Matter for simulation outcomes

## Population

"{description}" ({size} agents{geo_context})

{constraint_note}

## Geography & Context

- Use {geo_label} administrative divisions (states/provinces/districts)
- Use {geo_label} currency (EUR, USD, INR, etc.)
- Use {geo_label} education/certification systems
- Consider local context (don't assume Western structures like credit scores, formal employment, nuclear families)

## Categories

### 1. UNIVERSAL (8-12)
Core demographics that everyone has:
- age, gender, location, income, education, marital_status, household_size
- Adapt to local context (e.g., region/state/province names appropriate for the geography)

### 2. POPULATION-SPECIFIC (10-18)
What defines THIS population's identity and work/life:
- For professionals: role, experience, employer type, workload
- For farmers: land size, crop type, irrigation, market access
- For consumers: usage patterns, preferences
- Focus on attributes with REAL VARIANCE in this population

### 3. CONTEXT-SPECIFIC (0-5)
Only if a product/service/brand is mentioned:
- Relationship tenure, satisfaction, usage frequency
- Skip if no context entity in description

### 4. PERSONALITY (5-8)
Traits that affect behavior in simulations:

**Big Five (EXACTLY these 5 names, no variations):**
- openness
- conscientiousness
- extraversion
- agreeableness
- neuroticism

Do NOT use alternative names like big_five_openness, o_openness, trait_openness, etc.
Do NOT include Big Five twice under different names.

**Domain-specific traits (1-3):**
- e.g., risk_tolerance, trust_in_institutions
- These should be specific to the population context

## Sampling Strategy

For each attribute, determine the sampling strategy:

**independent**: Attribute stands alone, sampled directly from a distribution.
- Most base attributes: age, gender, surgical_specialty, employer_type
- Has NO dependencies (depends_on is empty)

**derived**: ONLY for ZERO-VARIANCE, definitional transformations:
- age_bracket derived from age (categorical binning)
- is_senior derived from years_experience (boolean flag)
- bmi derived from height and weight (physics formula)
- ⚠️ DERIVED IS RARE. If two people with the same inputs could have different outputs, use CONDITIONAL.

**conditional**: Any probabilistic relationship where variance exists:
- years_experience depends on age (correlated, but two 50-year-olds can have different experience)
- income depends on role (chiefs earn more on average, but there's variance)
- research_activity depends on employer_type (university hospitals skew toward research)

## Dependencies

Only mark depends_on if there's a LOGICAL relationship:
- depends_on must only reference attributes that will be sampled BEFORE this one
- No circular dependencies (A→B→A)
- Independent attributes MUST have empty depends_on
- Derived and conditional MUST have non-empty depends_on
- Max 3 dependencies per attribute

## Output Format

For each attribute:
- name: snake_case
- type: int, float, categorical, boolean
- category: universal, population_specific, context_specific, personality
- description: One clear sentence
- strategy: independent, derived, or conditional
- depends_on: List of attribute names (max 3, empty if independent)"""

    data = reasoning_call(
        prompt=prompt,
        response_schema=ATTRIBUTE_SELECTION_SCHEMA,
        schema_name="attribute_selection",
        model=model,
        reasoning_effort=reasoning_effort,
    )

    attributes = []
    for attr_data in data.get("attributes", []):
        # Determine strategy - default to independent if not specified
        strategy = attr_data.get("strategy", "independent")
        depends_on = attr_data.get("depends_on", [])

        # Validate strategy/depends_on consistency
        if strategy == "independent" and depends_on:
            # Independent can't have dependencies - likely LLM error, switch to conditional
            strategy = "conditional"
        elif strategy in ("derived", "conditional") and not depends_on:
            # Derived/conditional need dependencies - default to independent
            strategy = "independent"

        attr = DiscoveredAttribute(
            name=attr_data["name"],
            type=attr_data["type"],
            category=attr_data["category"],
            description=attr_data["description"],
            strategy=strategy,
            depends_on=depends_on,
        )
        attributes.append(attr)

    # Add Big Five if recommended and not already present
    if data.get("include_big_five", False):
        big_five_names = {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }
        existing_names = {a.name for a in attributes}

        # Check for both exact matches and partial matches (e.g., big_five_openness)
        def trait_already_exists(trait: str, existing: set[str]) -> bool:
            """Check if trait exists exactly or as part of another name."""
            if trait in existing:
                return True
            # Check for common variations like big_five_openness, o_openness, etc.
            for name in existing:
                if trait in name.lower():
                    return True
            return False

        for trait in big_five_names:
            if not trait_already_exists(trait, existing_names):
                attributes.append(
                    DiscoveredAttribute(
                        name=trait,
                        type="float",
                        category="personality",
                        description=f"Big Five personality trait: {trait} (0-1 scale)",
                        strategy="independent",  # Personality traits are independent
                        depends_on=[],
                    )
                )

    return attributes
