"""Step 1: Attribute Selection.

Discovers all relevant attributes for a population across four categories:
- Universal: Human attributes everyone has (age, gender, income, etc.)
- Population-specific: What makes this population distinct
- Context-specific: Only if a product/service is mentioned
- Personality: Behavioral/psychological traits when relevant
"""

from ..llm import reasoning_call
from ..spec import DiscoveredAttribute


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
                        "enum": ["universal", "population_specific", "context_specific", "personality"],
                        "description": "Category of the attribute",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description of what this attribute represents",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of other attributes this depends on (for derived values)",
                    },
                },
                "required": ["name", "type", "category", "description", "depends_on"],
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
    
    Args:
        description: Natural language population description
        size: Number of agents (for context)
        geography: Geographic scope if known
        model: Model to use
        reasoning_effort: "low", "medium", or "high"
    
    Returns:
        List of DiscoveredAttribute objects
    
    Example:
        >>> attrs = select_attributes("German surgeons", 500, "Germany")
        >>> [a.name for a in attrs[:3]]
        ['age', 'gender', 'specialty']
    """
    geo_context = f" in {geography}" if geography else ""
    geo_label = geography or "the relevant region"
    
    prompt = f"""## Intent

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

## Constraints (STRICT)

- **Total: 25-40 attributes maximum**
- Universal demographics: 8-12 attributes
- Population-specific: 10-18 attributes
- Personality/behavioral: 5-8 attributes
- Max 3 dependencies per attribute
- NO duplicates
- Only attributes where real statistical data likely exists

## Geography & Context

- Use {geo_label} administrative divisions (states/provinces/districts)
- Use {geo_label} currency (EUR, USD, INR, etc.)
- Use {geo_label} education/certification systems
- Consider local context (don't assume Western structures like credit scores, formal employment, nuclear families)

## Categories

### 1. UNIVERSAL (8-12)
Core demographics that everyone has:
- age, gender, location, income, education, marital_status, household_size
- Adapt to local context (e.g., caste for India, Bundesland for Germany)

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
- Big Five (openness, conscientiousness, extraversion, agreeableness, neuroticism)
- 1-3 domain-specific traits (e.g., risk_tolerance, trust_in_institutions)

## Dependencies

Only mark depends_on if there's a LOGICAL constraint:
- years_experience depends on age (can't exceed age - training_years)
- income may depend on experience, role
- Max 3 dependencies. Leave empty if independent.

## Output Format

For each attribute:
- name: snake_case
- type: int, float, categorical, boolean
- category: universal, population_specific, context_specific, personality
- description: One clear sentence
- depends_on: List of attribute names (max 3, empty if none)"""

    data = reasoning_call(
        prompt=prompt,
        response_schema=ATTRIBUTE_SELECTION_SCHEMA,
        schema_name="attribute_selection",
        model=model,
        reasoning_effort=reasoning_effort,
    )
    
    attributes = []
    for attr_data in data.get("attributes", []):
        attr = DiscoveredAttribute(
            name=attr_data["name"],
            type=attr_data["type"],
            category=attr_data["category"],
            description=attr_data["description"],
            depends_on=attr_data.get("depends_on", []),
        )
        attributes.append(attr)
    
    # Add Big Five if recommended and not already present
    if data.get("include_big_five", False):
        big_five_names = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
        existing_names = {a.name for a in attributes}
        
        for trait in big_five_names:
            if trait not in existing_names:
                attributes.append(DiscoveredAttribute(
                    name=trait,
                    type="float",
                    category="personality",
                    description=f"Big Five personality trait: {trait} (0-1 scale)",
                    depends_on=[],
                ))
    
    return attributes

