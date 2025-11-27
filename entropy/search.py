"""Web search and research functions for Entropy.

Uses GPT-5 agentic search - the model decides what to search,
searches the web, reasons about results, and returns structured data.
All in ONE API call.
"""

from .llm import agentic_research, validate_situation_attributes
from .models import ParsedContext, ResearchData, SituationSchema


# =============================================================================
# JSON Schemas for Structured Output
# =============================================================================

# Schema for demographic research
DEMOGRAPHICS_SCHEMA = {
    "type": "object",
    "properties": {
        "age_distribution": {
            "type": "array",
            "description": "Age ranges with percentages",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Age range like '18-24', '25-34'"},
                    "percentage": {"type": "number", "description": "Decimal 0-1, e.g., 0.25 for 25%"},
                },
                "required": ["category", "percentage"],
                "additionalProperties": False,
            },
        },
        "gender_distribution": {
            "type": "array",
            "description": "Gender with percentages",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Gender like 'male', 'female'"},
                    "percentage": {"type": "number", "description": "Decimal 0-1"},
                },
                "required": ["category", "percentage"],
                "additionalProperties": False,
            },
        },
        "income_distribution": {
            "type": "array",
            "description": "Income ranges with percentages",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Income range like '<30k', '30k-50k', '100k+'"},
                    "percentage": {"type": "number", "description": "Decimal 0-1"},
                },
                "required": ["category", "percentage"],
                "additionalProperties": False,
            },
        },
        "education_distribution": {
            "type": "array",
            "description": "Education levels with percentages",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Level like 'high_school', 'bachelors', 'masters'"},
                    "percentage": {"type": "number", "description": "Decimal 0-1"},
                },
                "required": ["category", "percentage"],
                "additionalProperties": False,
            },
        },
        "location_distribution": {
            "type": "array",
            "description": "Geographic distribution (urban/suburban/rural or states)",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Location type"},
                    "percentage": {"type": "number", "description": "Decimal 0-1"},
                },
                "required": ["category", "percentage"],
                "additionalProperties": False,
            },
        },
        "ethnicity_distribution": {
            "type": "array",
            "description": "Ethnicity/race distribution",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Ethnicity"},
                    "percentage": {"type": "number", "description": "Decimal 0-1"},
                },
                "required": ["category", "percentage"],
                "additionalProperties": False,
            },
        },
        "situation_attributes": {
            "type": "array",
            "description": "Context-specific attributes discovered from research",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Attribute name in snake_case"},
                    "field_type": {"type": "string", "description": "Type: int, float, str, or list"},
                    "description": {"type": "string", "description": "What this attribute represents"},
                    "min_value": {"type": ["number", "null"], "description": "Min value for numeric types, null if not applicable"},
                    "max_value": {"type": ["number", "null"], "description": "Max value for numeric types, null if not applicable"},
                    "mean": {"type": ["number", "null"], "description": "Mean value for numeric types, null if not applicable"},
                    "options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Possible values for categorical types, null if not applicable",
                    },
                },
                "required": ["name", "field_type", "description", "min_value", "max_value", "mean", "options"],
                "additionalProperties": False,
            },
        },
        "grounding_quality": {
            "type": "string",
            "description": "'strong' if data from multiple reliable sources, 'medium' if some data, 'low' if mostly estimated",
        },
    },
    "required": [
        "age_distribution",
        "gender_distribution",
        "income_distribution",
        "education_distribution",
        "location_distribution",
        "ethnicity_distribution",
        "situation_attributes",
        "grounding_quality",
    ],
    "additionalProperties": False,
}


# =============================================================================
# Main Research Function
# =============================================================================


def conduct_research(
    context: ParsedContext,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
) -> ResearchData:
    """
    Conduct research for a population context using agentic search.
    
    ONE API call that:
    1. Searches the web for demographics and behavioral data
    2. Reasons about what's relevant
    3. Returns structured data
    
    Args:
        context: Parsed population context
        model: Model to use (gpt-5, gpt-5.1)
        reasoning_effort: "low", "medium", or "high"
    
    Returns:
        ResearchData with demographics and situation schema
    """
    # Build the research prompt
    entity = context.context_entity or context.base_population
    geo = f" in {context.geography}" if context.geography else ""
    
    prompt = f"""Research the demographics and behavioral characteristics of {entity} customers/users{geo}.

Find and return:
1. **Demographics**: Age, gender, income, education, location, and ethnicity distributions.
   - Provide percentages that sum to approximately 1.0 for each distribution.
   - Use standard ranges (e.g., age: 18-24, 25-34, 35-44, etc.)

2. **Situation Attributes**: Key behavioral/contextual attributes specific to {entity} customers.
   
   IMPORTANT: Only include attributes that VARY BETWEEN INDIVIDUAL PEOPLE:
   ✅ Per-person attributes (include these):
      - Their personal tenure/time as customer
      - Their specific plan/tier
      - Their individual satisfaction score
      - Their usage frequency
      - Their engagement level
   
   ❌ Aggregate statistics (DO NOT include):
      - Market churn rates
      - Industry averages
      - Population-level statistics
      - Company metrics
   
   For each attribute, provide type (int/float/str/list) and realistic value ranges.

3. **Grounding Quality**: Assess how well the data is grounded in real sources.

Focus on recent data from reliable sources like industry reports, surveys, and official statistics."""

    # Make ONE agentic research call
    data, sources = agentic_research(
        prompt=prompt,
        response_schema=DEMOGRAPHICS_SCHEMA,
        schema_name="population_research",
        model=model,
        reasoning_effort=reasoning_effort,
    )
    
    # Validate situation attributes - filter out aggregate stats
    raw_attributes = data.get("situation_attributes", [])
    context_desc = f"{entity} customers/users{geo}"
    
    if raw_attributes:
        validated_attributes = validate_situation_attributes(
            attributes=raw_attributes,
            context_description=context_desc,
            model="gpt-5-mini",
        )
    else:
        validated_attributes = []
    
    # Convert to internal format
    def entries_to_dict(entries: list[dict] | None) -> dict[str, float]:
        if not entries:
            return {}
        return {entry["category"]: entry["percentage"] for entry in entries}
    
    demographics_dict = {
        "age_distribution": entries_to_dict(data.get("age_distribution")),
        "gender_distribution": entries_to_dict(data.get("gender_distribution")),
        "income_distribution": entries_to_dict(data.get("income_distribution")),
        "education_distribution": entries_to_dict(data.get("education_distribution")),
        "location_distribution": entries_to_dict(data.get("location_distribution")),
        "ethnicity_distribution": entries_to_dict(data.get("ethnicity_distribution")),
    }
    
    # Convert situation attributes to schema format (using validated attributes)
    situation_schema = []
    situation_distributions = {}
    
    for attr in validated_attributes:
        schema = SituationSchema(
            name=attr["name"],
            field_type=attr["field_type"],
            description=attr["description"],
            min_value=attr.get("min_value"),
            max_value=attr.get("max_value"),
            options=attr.get("options"),
        )
        situation_schema.append(schema)
        
        # Build distribution dict
        dist = {}
        if attr.get("min_value") is not None:
            dist["min"] = attr["min_value"]
        if attr.get("max_value") is not None:
            dist["max"] = attr["max_value"]
        if attr.get("mean") is not None:
            dist["mean"] = attr["mean"]
        if attr.get("options"):
            dist["options"] = attr["options"]
        situation_distributions[attr["name"]] = dist
    
    return ResearchData(
        demographics=demographics_dict,
        psychographics={},  # Big Five uses standard distributions, not researched
        situation_schema=situation_schema,
        situation_distributions=situation_distributions,
        sources=sources,
        grounding_level=data.get("grounding_quality", "medium"),
    )
