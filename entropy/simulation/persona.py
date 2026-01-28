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


def format_value(value: Any, attr: AttributeSpec) -> str:
    """Convert raw value to human-readable string.

    - Booleans → "Yes" / "No"
    - Floats 0-1 → "Very low" / "Low" / "Moderate" / "High" / "Very high"
    - Categoricals → Title case, underscores to spaces
    - Integers → str(int(value))
    - None → "Unknown"
    """
    if value is None:
        return "Unknown"

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
) -> str:
    """Build grouped list of all attributes not in the narrative.

    Groups by attribute category for readability.
    """
    # Group attributes by category
    category_labels = {
        "universal": "Demographics",
        "population_specific": "Professional",
        "context_specific": "Attitudes & Behaviors",
        "personality": "Personality",
    }

    groups: dict[str, list[str]] = {cat: [] for cat in category_labels}

    for attr in spec.attributes:
        if attr.name in exclude:
            continue

        value = formatted_agent.get(attr.name, "Unknown")
        label = attr.name.replace("_", " ").title()
        groups[attr.category].append(f"- {label}: {value}")

    # Build output
    sections = []
    for category, category_label in category_labels.items():
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
) -> str:
    """Generate a natural language persona from agent attributes.

    Uses hybrid approach:
    1. If persona_template exists: render narrative intro from template
    2. Append ALL remaining attributes as structured list

    This ensures nothing is filtered out while maintaining readability.
    """
    if not population_spec:
        return _fallback_persona(agent)

    # Format all values to human-readable strings
    formatted = format_agent(agent, population_spec)

    # Build the persona
    parts = ["## Who You Are", ""]

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
    parts.append("## Your Characteristics")
    parts.append("")

    characteristics = build_characteristics_list(population_spec, formatted, used_attrs)
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
    # Try to build: "You are a {age}-year-old {gender} {role/specialty}..."
    intro_parts = []

    age = agent.get("age")
    gender = formatted.get("gender", "").lower()

    if age is not None:
        if gender:
            intro_parts.append(f"You are a {int(age)}-year-old {gender}")
        else:
            intro_parts.append(f"You are {int(age)} years old")
    elif gender:
        intro_parts.append(f"You are {gender}")
    else:
        intro_parts.append("You are a member of this population")

    # Add role/occupation/specialty if available
    for attr_name in ("surgical_specialty", "specialty", "occupation", "role", "role_rank"):
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
        parts.append(f"You are a {int(age)}-year-old {gender}.")
    else:
        parts.append(f"You are a {gender}.")

    # Add a few key attributes
    for key in ("role", "occupation", "specialty", "employer_type", "years_experience"):
        if key in agent and agent[key]:
            parts.append(f"Your {key.replace('_', ' ')} is {agent[key]}.")
            if len(parts) >= 4:
                break

    return " ".join(parts)
