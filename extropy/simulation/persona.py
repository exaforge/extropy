"""Persona rendering for agent reasoning.

Converts agent attributes into natural language personas using a hybrid approach:
1. Narrative intro from template (core identity attributes)
2. Structured list of ALL remaining attributes (nothing filtered)

This ensures the LLM sees everything while maintaining readability.
"""

import re
from typing import Any

from ..core.models import PopulationSpec, AttributeSpec


def is_narrative_safe(attr: AttributeSpec) -> bool:
    """Determine if an attribute is safe for narrative prose.

    Attributes NOT safe for narrative:
    - Booleans (awkward: "am True", "has False")
    - Floats with 0-1 range (scores/scales: "digital_literacy is 0.0")

    These work better in structured list format where "Yes/No" and "Very low" are natural.
    """
    if attr.type == "boolean":
        return False

    if attr.type == "float":
        min_val, max_val = get_value_range(attr)
        # If range is 0-1 (or close to it), treat as a scale
        if min_val is not None and max_val is not None:
            if min_val >= -0.1 and max_val <= 1.1:
                return False
        # No range info but float - check if it's likely a score by name
        if any(
            term in attr.name.lower()
            for term in ("score", "index", "level", "concern", "perception", "literacy")
        ):
            return False

    return True


def get_value_range(attr: AttributeSpec) -> tuple[float | None, float | None]:
    """Extract min/max range from attribute constraints or distribution."""
    min_val = None
    max_val = None

    # Check constraints first (most explicit)
    for constraint in attr.constraints:
        if constraint.type in ("hard_min", "min") and constraint.value is not None:
            min_val = constraint.value
        elif constraint.type in ("hard_max", "max") and constraint.value is not None:
            max_val = constraint.value

    # Fall back to distribution params
    if attr.sampling.distribution:
        dist = attr.sampling.distribution
        if hasattr(dist, "min") and dist.min is not None and min_val is None:
            min_val = dist.min
        if hasattr(dist, "max") and dist.max is not None and max_val is None:
            max_val = dist.max

    return min_val, max_val


def _format_time(value: float, use_12hr: bool = True) -> str:
    """Format a numeric hour value as clock time."""
    hours = int(value)
    minutes = int((value - hours) * 60)
    if use_12hr:
        suffix = "AM" if hours < 12 else "PM"
        display_hour = hours % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minutes:02d} {suffix}"
    return f"{hours:02d}:{minutes:02d}"


def format_value(value: Any, attr: AttributeSpec) -> str:
    """Convert raw value to human-readable string.

    Uses display_format from AttributeSpec (set by LLM during spec creation)
    for time, currency, percentage formatting.

    - Booleans → "Yes" / "No"
    - Floats 0-1 → "Very low" / "Low" / "Moderate" / "High" / "Very high"
    - Categoricals → Title case, underscores to spaces
    - Integers → str(int(value))
    - None → "Unknown"
    """
    if value is None:
        return "Unknown"

    # Check LLM-classified display_format first (for time, currency, percentage)
    if attr.display_format and isinstance(value, (int, float)):
        if attr.display_format == "time_12h" and 0 <= float(value) <= 24:
            return _format_time(float(value), use_12hr=True)
        elif attr.display_format == "time_24h" and 0 <= float(value) <= 24:
            return _format_time(float(value), use_12hr=False)
        elif attr.display_format == "currency":
            if value >= 1000:
                return f"${value:,.0f}"
            return f"${value:.2f}"
        elif attr.display_format == "percentage":
            # If value is 0-1, convert to percentage
            if 0 <= value <= 1:
                return f"{value * 100:.0f}%"
            return f"{value:.0f}%"

    # Booleans
    if isinstance(value, bool) or attr.type == "boolean":
        return "Yes" if value else "No"

    # Floats
    if attr.type == "float" or isinstance(value, float):
        min_val, max_val = get_value_range(attr)

        # Normalize to 0-1 scale if we have range info
        if min_val is not None and max_val is not None and max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        elif 0 <= value <= 1:
            # Assume 0-1 scale
            normalized = value
        else:
            # Can't normalize - return formatted number
            if value == int(value):
                return str(int(value))
            return f"{value:.1f}"

        # Map to labels
        if normalized <= 0.2:
            return "Very low"
        elif normalized <= 0.4:
            return "Low"
        elif normalized <= 0.6:
            return "Moderate"
        elif normalized <= 0.8:
            return "High"
        else:
            return "Very high"

    # Integers
    if attr.type == "int" or isinstance(value, int):
        return str(int(value))

    # Categoricals / strings
    if isinstance(value, str):
        # Handle snake_case and format nicely
        formatted = value.replace("_", " ")
        # Title case but preserve acronyms/abbreviations
        words = formatted.split()
        result = []
        for word in words:
            if word.isupper() and len(word) > 1:
                result.append(word)  # Keep acronyms
            else:
                result.append(word.capitalize())
        return " ".join(result)

    return str(value)


def format_agent(agent: dict[str, Any], spec: PopulationSpec) -> dict[str, str]:
    """Format all agent values to human-readable strings."""
    formatted = {}
    for attr in spec.attributes:
        value = agent.get(attr.name)
        formatted[attr.name] = format_value(value, attr)
    return formatted


def extract_template_attrs(template: str) -> set[str]:
    """Extract attribute names from template placeholders like {attr_name}."""
    return set(re.findall(r"\{(\w+)\}", template))


def build_characteristics_list(
    spec: PopulationSpec,
    formatted_agent: dict[str, str],
    exclude: set[str],
    decision_relevant_attributes: list[str] | None = None,
) -> str:
    """Build grouped list of all attributes not in the narrative.

    If decision_relevant_attributes is provided, those attributes are listed
    first under a dedicated "Most Relevant to This Decision" section,
    regardless of their original category. This ensures the LLM attends
    to the attributes that matter most for the scenario outcome.

    Groups by attribute category - puts personality/attitudes FIRST since
    they most strongly influence decision-making.
    """
    # Order: personality first (most decision-relevant), then attitudes, then demographics, then professional
    category_order = [
        ("personality", "Mindset & Values"),
        ("context_specific", "Attitudes & Concerns"),
        ("universal", "Background"),
        ("population_specific", "Professional Context"),
    ]

    decision_set = set(decision_relevant_attributes or [])
    decision_items: list[str] = []
    groups: dict[str, list[str]] = {cat: [] for cat, _ in category_order}

    for attr in spec.attributes:
        if attr.name in exclude:
            continue

        value = formatted_agent.get(attr.name, "Unknown")
        label = attr.name.replace("_", " ").title()
        line = f"- {label}: {value}"

        if attr.name in decision_set:
            decision_items.append(line)
        else:
            groups[attr.category].append(line)

    # Build output — decision-relevant first, then standard categories
    sections = []
    if decision_items:
        sections.append(
            "**Most Relevant to This Decision**\n" + "\n".join(decision_items)
        )

    for category, category_label in category_order:
        items = groups.get(category, [])
        if items:
            sections.append(f"**{category_label}**\n" + "\n".join(items))

    return "\n\n".join(sections)


def render_persona(agent: dict[str, Any], template: str) -> str:
    """Render persona string from template and agent attributes.

    Uses Python's str.format() for simple {attribute} placeholder substitution.
    """
    try:
        return template.format(**agent)
    except KeyError as e:
        # Missing attribute - return template with missing placeholder noted
        return f"[Template error: missing {e}]"
    except Exception:
        return "[Template rendering error]"


def generate_persona(
    agent: dict[str, Any],
    population_spec: PopulationSpec | None = None,
    persona_config: Any | None = None,
    decision_relevant_attributes: list[str] | None = None,
) -> str:
    """Generate a natural language persona from agent attributes.

    If persona_config is provided, uses the new first-person embodied rendering.
    Otherwise, uses the legacy hybrid approach:
    1. If persona_template exists: render narrative intro from template
    2. Append ALL remaining attributes as structured list

    If decision_relevant_attributes is provided, those attributes are grouped
    first under a dedicated section so the LLM attends to them for outcome reasoning.

    Args:
        agent: Agent attribute dictionary
        population_spec: Population specification (for legacy rendering)
        persona_config: PersonaConfig for new embodied rendering (optional)
        decision_relevant_attributes: Attributes most relevant to scenario outcome

    Returns:
        Complete persona as string
    """
    # Use new PersonaConfig rendering if available
    if persona_config is not None:
        from ..population.persona import render_persona as render_new_persona

        # Build display_format_map from population_spec if available
        display_format_map = None
        semantic_type_map = None
        if population_spec:
            display_format_map = {
                attr.name: attr.display_format
                for attr in population_spec.attributes
                if attr.display_format
            }
            semantic_type_map = {
                attr.name: attr.semantic_type
                for attr in population_spec.attributes
                if attr.semantic_type
            }

        return render_new_persona(
            agent,
            persona_config,
            decision_relevant_attributes,
            display_format_map,
            semantic_type_map,
        )

    # Legacy rendering below
    if not population_spec:
        return _fallback_persona(agent)

    # Format all values to human-readable strings
    formatted = format_agent(agent, population_spec)

    # Build the persona
    parts = ["## Who I Am", ""]

    if population_spec.meta.persona_template:
        # Render narrative intro from template using formatted values
        narrative = render_persona(formatted, population_spec.meta.persona_template)
        parts.append(narrative)

        # Get attributes used in template
        used_attrs = extract_template_attrs(population_spec.meta.persona_template)
    else:
        # No template - generate simple intro from safe attrs
        narrative = _build_simple_intro(agent, formatted, population_spec)
        parts.append(narrative)

        # Mark intro attrs as used
        used_attrs = _get_intro_attrs(population_spec)

    # Add all remaining attributes as structured list
    parts.append("")
    parts.append("## My Characteristics")
    parts.append("")

    characteristics = build_characteristics_list(
        population_spec, formatted, used_attrs, decision_relevant_attributes
    )
    parts.append(characteristics)

    return "\n".join(parts)


def _build_simple_intro(
    agent: dict[str, Any],
    formatted: dict[str, str],
    spec: PopulationSpec,
) -> str:
    """Build a simple narrative intro when no template exists.

    Uses only narrative-safe attributes.
    """
    # Try to build: "I'm a {age}-year-old {gender} {role/specialty}..."
    intro_parts = []

    age = agent.get("age")
    gender = formatted.get("gender", "").lower()

    if age is not None:
        if gender:
            intro_parts.append(f"I'm a {int(age)}-year-old {gender}")
        else:
            intro_parts.append(f"I'm {int(age)} years old")
    elif gender:
        intro_parts.append(f"I'm {gender}")
    else:
        intro_parts.append("I'm a member of this population")

    # Add role/occupation/specialty if available
    for attr_name in (
        "surgical_specialty",
        "specialty",
        "occupation",
        "role",
        "role_rank",
    ):
        if attr_name in formatted and agent.get(attr_name):
            value = formatted[attr_name].lower()
            intro_parts[0] += f" {value}"
            break

    # Add employer/location
    employer = formatted.get("employer_type")
    location = formatted.get("location_state") or formatted.get("location")
    experience = agent.get("years_experience")

    if employer or location:
        location_part = "working"
        if employer:
            location_part += f" at a {employer.lower()}"
        if location:
            location_part += f" in {location}"
        intro_parts.append(location_part)

    if experience is not None:
        intro_parts.append(f"with {int(experience)} years of experience")

    return ", ".join(intro_parts) + "."


def _get_intro_attrs(spec: PopulationSpec) -> set[str]:
    """Get attribute names typically used in simple intro."""
    intro_attrs = {
        "age",
        "gender",
        "surgical_specialty",
        "specialty",
        "occupation",
        "role",
        "role_rank",
        "employer_type",
        "location_state",
        "location",
        "years_experience",
    }
    # Only return attrs that exist in spec
    spec_attrs = {attr.name for attr in spec.attributes}
    return intro_attrs & spec_attrs


def _fallback_persona(agent: dict[str, Any]) -> str:
    """Generate a basic fallback persona when no spec is available."""
    parts = []

    age = agent.get("age")
    gender = agent.get("gender", "person")

    if age:
        parts.append(f"I'm a {int(age)}-year-old {gender}.")
    else:
        parts.append(f"I'm a {gender}.")

    # Add a few key attributes
    for key in ("role", "occupation", "specialty", "employer_type", "years_experience"):
        if key in agent and agent[key]:
            parts.append(f"My {key.replace('_', ' ')} is {agent[key]}.")
            if len(parts) >= 4:
                break

    return " ".join(parts)
