"""Step 1: Attribute Selection.

Discovers all relevant attributes for a population across four categories:
- Universal: Human attributes everyone has (age, gender, income, etc.)
- Population-specific: What makes this population distinct
- Context-specific: Only if a product/service is mentioned
- Personality: Behavioral/psychological traits when relevant
"""

from ...core.llm import reasoning_call
from ...core.models import AttributeSpec, DiscoveredAttribute


# Multi-country geography patterns that should trigger country attribute injection
MULTI_COUNTRY_PATTERNS = [
    # Global
    "world",
    "global",
    "globe",
    "international",
    "worldwide",
    # Continents
    "africa",
    "asia",
    "europe",
    "north america",
    "south america",
    "latin america",
    "oceania",
    "antarctica",
    # Regions spanning multiple countries
    "east asia",
    "southeast asia",
    "south asia",
    "central asia",
    "middle east",
    "western europe",
    "eastern europe",
    "central europe",
    "northern europe",
    "southern europe",
    "sub-saharan africa",
    "north africa",
    "central america",
    "caribbean",
    "pacific",
    "nordic",
    "scandinavian",
    "balkan",
    "mediterranean",
    "gulf",
    "apac",
    "emea",
    "latam",
    "amer",
]


def _is_multi_country_geography(geography: str | None, description: str) -> bool:
    """Detect if the population spans multiple countries."""
    if not geography and not description:
        return False

    text = f"{geography or ''} {description}".lower()

    for pattern in MULTI_COUNTRY_PATTERNS:
        if pattern in text:
            return True

    return False


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
                    "scope": {
                        "type": "string",
                        "enum": ["individual", "household", "partner_correlated"],
                        "description": "individual: varies per person; household: shared across household members; partner_correlated: correlated between partners (e.g., age, education, religion, politics)",
                    },
                    "correlation_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "For partner_correlated scope only: probability (0-1) that partner has same value. Use ~0.6-0.7 for education, ~0.7-0.8 for religion, ~0.5-0.6 for politics. Omit for age (uses gaussian offset) and race/ethnicity (uses per-group rates).",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of other attributes this depends on (empty for independent)",
                    },
                    "semantic_type": {
                        "type": "string",
                        "enum": [
                            "age",
                            "income",
                            "education",
                            "employment",
                            "occupation",
                        ],
                        "description": "Semantic meaning for special handling. Set ONLY for: age (person's age), income (monetary earnings/salary), education (education level/degree), employment (employment status like employed/unemployed/student), occupation (job title/profession). Leave unset for all other attributes.",
                    },
                    "identity_type": {
                        "type": "string",
                        "enum": [
                            "political_orientation",
                            "religious_affiliation",
                            "race_ethnicity",
                            "gender_identity",
                            "sexual_orientation",
                            "parental_status",
                            "citizenship",
                            "socioeconomic_class",
                            "professional_identity",
                            "generational_identity",
                        ],
                        "description": "Identity dimension this attribute represents. Set for attributes that capture identity: political_orientation (voting, ideology), religious_affiliation (faith, religion), race_ethnicity (race, ethnicity), gender_identity (gender), sexual_orientation, parental_status (has children, parent), citizenship (nationality, immigrant status), socioeconomic_class (class, wealth bracket), professional_identity (occupation, career), generational_identity (generation, age group). Leave unset for attributes that don't represent identity dimensions.",
                    },
                    "display_format": {
                        "type": "string",
                        "enum": [
                            "time_12h",
                            "time_24h",
                            "currency",
                            "percentage",
                            "number",
                        ],
                        "description": "Display format for persona rendering. Set for: time_12h (clock times displayed as '7:30 AM', e.g. departure_time, wake_time, shift_start), time_24h (clock times as '19:30'), currency (monetary values with $ symbol, e.g. monthly_rent, salary), percentage (values with % symbol, e.g. savings_rate). Leave unset for regular numbers.",
                    },
                },
                "required": [
                    "name",
                    "type",
                    "category",
                    "description",
                    "strategy",
                    "scope",
                    "depends_on",
                ],
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
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    model: str | None = None,
    reasoning_effort: str = "low",
) -> list[DiscoveredAttribute]:
    """
    Discover all relevant attributes for a population.

    Uses GPT-5 with reasoning to analyze the population and identify
    attributes across all applicable categories. The model considers:
    - What makes this population unique
    - Geographic/cultural context
    - Dependencies between attributes

    When context is provided (extend mode), only discovers NEW attributes
    not already in the base population. Can reference context attributes
    in dependencies.

    Args:
        description: Natural language population description
        geography: Geographic scope if known
        context: Existing attributes from base population (for extend mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"

    Returns:
        List of DiscoveredAttribute objects

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
        # In extend mode, adjust constraints
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

    prompt = f"""
    
    {context_section}
    
    ## Intent

    We are building a synthetic population for agent-based simulation. These agents will:
    - Have scenarios injected (events, information, decisions)
    - Respond based on their attributes
    - Interact with each other in a social network

    The goal is realistic variance, not exhaustive detail. We need attributes that:
    1. Create meaningful differences in how agents BEHAVE
    2. Can be grounded in real data (researchable distributions)
    3. Matter for simulation outcomes

    ## Population

    "{description}"{geo_context}

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
    - DERIVED IS RARE. If two people with the same inputs could have different outputs, use CONDITIONAL.

    **conditional**: Any probabilistic relationship where variance exists:
    - years_experience depends on age (correlated, but two 50-year-olds can have different experience)
    - income depends on role (chiefs earn more on average, but there's variance)
    - research_activity depends on employer_type (university hospitals skew toward research)

    ## Attribute Scope

    Each attribute has one of three scopes:

    - `household`: Shared by ALL household members (copied, not sampled separately)
      - Examples: state, urban_rural, household_income, household_size, housing_type
      - Use for anything describing WHERE or HOW the household lives together

    - `partner_correlated`: Correlated between partners (assortative mating)
      - Examples: age, education_level, religious_affiliation, political views, race/ethnicity, country
      - Partners tend to have similar values but NOT identical
      - Provide `correlation_rate` (0-1) for probability of same value:
        - age: OMIT (uses gaussian offset automatically)
        - race/ethnicity: OMIT (uses per-group rates automatically)
        - education: ~0.6-0.7
        - religion: ~0.7-0.8
        - politics: ~0.5-0.6
        - country: ~0.95 (most people marry within their country)

    - `individual`: Varies per person, sampled independently
      - Examples: gender, occupation, personality traits, personal attitudes
      - Default if unsure

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
    - scope: individual, household, or partner_correlated
    - correlation_rate: (only for partner_correlated scope, omit for age/race)
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
        strategy = attr_data.get("strategy", "independent")
        depends_on = attr_data.get("depends_on", [])

        if strategy == "independent" and depends_on:
            strategy = "conditional"
        elif strategy in ("derived", "conditional") and not depends_on:
            strategy = "independent"

        attr = DiscoveredAttribute(
            name=attr_data["name"],
            type=attr_data["type"],
            category=attr_data["category"],
            description=attr_data["description"],
            strategy=strategy,
            scope=attr_data.get("scope", "individual"),
            correlation_rate=attr_data.get("correlation_rate"),
            semantic_type=attr_data.get("semantic_type"),
            identity_type=attr_data.get("identity_type"),
            display_format=attr_data.get("display_format"),
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

    # Inject country attribute for multi-country geographies
    if _is_multi_country_geography(geography, description):
        existing_names = {a.name for a in attributes}
        if "country" not in existing_names:
            # Insert at the beginning (universal demographic)
            attributes.insert(
                0,
                DiscoveredAttribute(
                    name="country",
                    type="categorical",
                    category="universal",
                    description="Country of residence",
                    strategy="independent",
                    scope="household",
                    depends_on=[],
                ),
            )

    return attributes
