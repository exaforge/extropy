"""Template-based persona generation for agent reasoning.

Converts agent attributes into natural language personas that can
be used as context for LLM reasoning calls.

Supports two modes:
1. Template mode (preferred): Uses a Jinja2 template stored in the PopulationSpec
2. Fallback mode: Uses hardcoded logic for backwards compatibility
"""

from typing import Any

from jinja2 import Environment, BaseLoader, TemplateSyntaxError, UndefinedError

from ..core.models import PopulationSpec, AttributeSpec


# =============================================================================
# Jinja2 Template Rendering
# =============================================================================


def _create_jinja_environment() -> Environment:
    """Create a Jinja2 environment with custom filters for persona generation."""
    env = Environment(loader=BaseLoader())

    # Add custom filters for personality trait descriptions
    def describe_trait(value: float, trait_name: str) -> str:
        """Describe a 0-1 scaled personality trait in natural language."""
        if value is None:
            return ""

        descriptors = {
            "openness": {
                (0.0, 0.3): "prefer routine and familiarity",
                (0.3, 0.5): "am moderately open to new experiences",
                (0.5, 0.7): "enjoy exploring new ideas",
                (0.7, 1.0): "am highly curious and creative",
            },
            "conscientiousness": {
                (0.0, 0.3): "am flexible and spontaneous",
                (0.3, 0.5): "am moderately organized",
                (0.5, 0.7): "am reliable and methodical",
                (0.7, 1.0): "am highly disciplined and detail-oriented",
            },
            "extraversion": {
                (0.0, 0.3): "prefer solitude and quiet reflection",
                (0.3, 0.5): "am moderately social",
                (0.5, 0.7): "enjoy social interaction",
                (0.7, 1.0): "am highly outgoing and energized by others",
            },
            "agreeableness": {
                (0.0, 0.3): "am skeptical and competitive",
                (0.3, 0.5): "balance cooperation with assertiveness",
                (0.5, 0.7): "am cooperative and trusting",
                (0.7, 1.0): "am highly empathetic and accommodating",
            },
            "neuroticism": {
                (0.0, 0.3): "am emotionally stable and calm",
                (0.3, 0.5): "experience moderate emotional variability",
                (0.5, 0.7): "can be sensitive to stress",
                (0.7, 1.0): "experience emotions intensely",
            },
            "risk_tolerance": {
                (0.0, 0.4): "prefer caution over risk",
                (0.4, 0.6): "have moderate risk tolerance",
                (0.6, 1.0): "am comfortable taking risks",
            },
        }

        trait_descriptors = descriptors.get(trait_name.lower(), {})
        for (low, high), desc in trait_descriptors.items():
            if low <= value < high:
                return desc
        return ""

    def format_age(age: int | float | None) -> str:
        """Format age for display."""
        if age is None:
            return ""
        return f"{int(age)}-year-old"

    def format_gender(gender: str | None) -> str:
        """Format gender for display."""
        if not gender:
            return "person"
        gender_lower = gender.lower()
        if gender_lower in ("male", "m", "man"):
            return "man"
        elif gender_lower in ("female", "f", "woman"):
            return "woman"
        return "person"

    def humanize(name: str) -> str:
        """Convert snake_case to human readable text."""
        return name.replace("_", " ").replace("-", " ").lower()

    env.filters["describe_trait"] = describe_trait
    env.filters["format_age"] = format_age
    env.filters["format_gender"] = format_gender
    env.filters["humanize"] = humanize

    return env


def render_persona_from_template(
    template: str,
    agent: dict[str, Any],
) -> str | None:
    """Render a persona from a Jinja2 template.

    Args:
        template: Jinja2 template string
        agent: Agent dictionary with attributes

    Returns:
        Rendered persona string, or None if rendering failed
    """
    try:
        env = _create_jinja_environment()
        tmpl = env.from_string(template)
        result = tmpl.render(**agent)
        # Clean up whitespace
        lines = [line.strip() for line in result.split("\n")]
        cleaned = " ".join(line for line in lines if line)
        return cleaned.strip()
    except (TemplateSyntaxError, UndefinedError, Exception):
        return None


# =============================================================================
# Fallback: Hardcoded Logic (for backwards compatibility)
# =============================================================================


# Personality trait descriptors for Big Five (fallback mode)
OPENNESS_DESCRIPTORS = {
    (0.0, 0.3): "prefer routine and familiarity",
    (0.3, 0.5): "are moderately open to new experiences",
    (0.5, 0.7): "enjoy exploring new ideas",
    (0.7, 1.0): "are highly curious and creative",
}

CONSCIENTIOUSNESS_DESCRIPTORS = {
    (0.0, 0.3): "are flexible and spontaneous",
    (0.3, 0.5): "are moderately organized",
    (0.5, 0.7): "are reliable and methodical",
    (0.7, 1.0): "are highly disciplined and detail-oriented",
}

EXTRAVERSION_DESCRIPTORS = {
    (0.0, 0.3): "prefer solitude and quiet reflection",
    (0.3, 0.5): "are moderately social",
    (0.5, 0.7): "enjoy social interaction",
    (0.7, 1.0): "are highly outgoing and energized by others",
}

AGREEABLENESS_DESCRIPTORS = {
    (0.0, 0.3): "are skeptical and competitive",
    (0.3, 0.5): "balance cooperation with assertiveness",
    (0.5, 0.7): "are cooperative and trusting",
    (0.7, 1.0): "are highly empathetic and accommodating",
}

NEUROTICISM_DESCRIPTORS = {
    (0.0, 0.3): "are emotionally stable and calm",
    (0.3, 0.5): "experience moderate emotional variability",
    (0.5, 0.7): "can be sensitive to stress",
    (0.7, 1.0): "experience emotions intensely",
}


def _get_descriptor(value: float, descriptors: dict) -> str:
    """Get descriptor for a personality value."""
    for (low, high), desc in descriptors.items():
        if low <= value < high:
            return desc
    return descriptors.get((0.7, 1.0), "")


def _format_age(age: int | float) -> str:
    """Format age for persona."""
    return f"{int(age)}-year-old"


def _format_gender(gender: str | None) -> str:
    """Format gender for persona, defaulting to neutral."""
    if not gender:
        return "person"
    gender_lower = gender.lower()
    if gender_lower in ("male", "m", "man"):
        return "man"
    elif gender_lower in ("female", "f", "woman"):
        return "woman"
    return "person"


def _format_role(agent: dict[str, Any]) -> str:
    """Format professional role from agent attributes."""
    parts = []

    # Check common role attributes
    role = agent.get("role_seniority") or agent.get("role") or agent.get("job_title") or agent.get("professional_rank")
    if role:
        parts.append(role)

    specialty = (
        agent.get("surgical_specialty")
        or agent.get("specialty")
        or agent.get("field")
    )
    if specialty and specialty not in parts:
        if parts:
            parts.append("specializing in")
        parts.append(specialty)

    return " ".join(parts) if parts else ""


def _format_employer(agent: dict[str, Any]) -> str:
    """Format employer/workplace information."""
    employer_type = (
        agent.get("employer_type")
        or agent.get("workplace_type")
        or agent.get("organization_type")
    )
    if employer_type:
        return f"at a {employer_type}"
    return ""


def _format_experience(agent: dict[str, Any]) -> str:
    """Format experience/tenure information."""
    years = (
        agent.get("years_experience")
        or agent.get("years_in_role")
        or agent.get("tenure")
    )
    if years is not None:
        return f"with {int(years)} years of experience"
    return ""


def _format_location(agent: dict[str, Any]) -> str:
    """Format location information."""
    location = (
        agent.get("federal_state")
        or agent.get("state")
        or agent.get("region")
        or agent.get("city")
        or agent.get("location")
    )
    if location:
        return f"based in {location}"
    return ""


def _format_household(agent: dict[str, Any]) -> str:
    """Format household information."""
    parts = []

    household_size = agent.get("household_size")
    if household_size and household_size > 1:
        parts.append(f"living in a household of {int(household_size)}")

    has_children = agent.get("has_children") or agent.get("children") or (agent.get("children_count", 0) > 0)
    if has_children:
        parts.append("with children")

    return ", ".join(parts)


def _generate_personality_summary(
    agent: dict[str, Any], personality_attrs: list[AttributeSpec]
) -> str:
    """Generate natural language personality description.

    Args:
        agent: Agent dictionary
        personality_attrs: Personality attribute specs

    Returns:
        Personality summary string
    """
    descriptors = []

    for attr in personality_attrs:
        value = agent.get(attr.name)
        if value is None:
            continue

        name_lower = attr.name.lower()

        # Map to Big Five descriptors
        if "openness" in name_lower:
            desc = _get_descriptor(value, OPENNESS_DESCRIPTORS)
            if desc:
                descriptors.append(desc)
        elif "conscientiousness" in name_lower:
            desc = _get_descriptor(value, CONSCIENTIOUSNESS_DESCRIPTORS)
            if desc:
                descriptors.append(desc)
        elif "extraversion" in name_lower or "extroversion" in name_lower:
            desc = _get_descriptor(value, EXTRAVERSION_DESCRIPTORS)
            if desc:
                descriptors.append(desc)
        elif "agreeableness" in name_lower:
            desc = _get_descriptor(value, AGREEABLENESS_DESCRIPTORS)
            if desc:
                descriptors.append(desc)
        elif "neuroticism" in name_lower or "emotional_stability" in name_lower:
            desc = _get_descriptor(value, NEUROTICISM_DESCRIPTORS)
            if desc:
                descriptors.append(desc)
        elif "risk" in name_lower:
            if value > 0.6:
                descriptors.append("are comfortable taking risks")
            elif value < 0.4:
                descriptors.append("prefer caution over risk")

    if not descriptors:
        return ""

    # Limit to 3 descriptors for brevity
    descriptors = descriptors[:3]

    if len(descriptors) == 1:
        return f"You {descriptors[0]}."
    elif len(descriptors) == 2:
        return f"You {descriptors[0]} and {descriptors[1]}."
    else:
        return f"You {', '.join(descriptors[:-1])}, and {descriptors[-1]}."


def _format_context_attributes(
    agent: dict[str, Any], context_attrs: list[AttributeSpec]
) -> str:
    """Format context-specific attributes (relationship to product/service)."""
    parts = []

    for attr in context_attrs:
        value = agent.get(attr.name)
        if value is None:
            continue

        name = attr.name.replace("_", " ").lower()

        if isinstance(value, bool):
            if value:
                parts.append(f"You have {name}.")
        elif isinstance(value, (int, float)):
            if "satisfaction" in name:
                if value > 0.7:
                    parts.append(f"You are generally satisfied with {name.replace('satisfaction', '').strip() or 'the service'}.")
                elif value < 0.3:
                    parts.append(f"You have been dissatisfied with {name.replace('satisfaction', '').strip() or 'the service'}.")
            elif "tenure" in name or "months" in name or "years" in name:
                unit = "years" if value > 12 else "months"
                val = int(value) if value > 12 else int(value)
                parts.append(f"You have been a customer for about {val} {unit}.")
            else:
                parts.append(f"Your {name} is {value:.1f}.")
        elif isinstance(value, str):
            parts.append(f"Your {name} is {value}.")

    return " ".join(parts[:3])  # Limit context info


def _generate_persona_fallback(
    agent: dict[str, Any],
    population_spec: PopulationSpec | None = None,
) -> str:
    """Generate a persona using hardcoded logic (fallback mode).

    Args:
        agent: Agent dictionary with attributes
        population_spec: Optional population spec for attribute categorization

    Returns:
        Natural language persona string
    """
    parts = []

    # Core identity (age, gender, role)
    age = agent.get("age")
    gender = agent.get("gender")

    identity_parts = []
    if age:
        identity_parts.append(_format_age(age))
    identity_parts.append(_format_gender(gender))

    role = _format_role(agent)
    if role:
        identity_parts.append(f"working as a {role}")

    employer = _format_employer(agent)
    if employer:
        identity_parts.append(employer)

    if identity_parts:
        parts.append("You are a " + " ".join(identity_parts) + ".")

    # Experience
    experience = _format_experience(agent)
    if experience:
        parts.append(f"You have {experience}.")

    # Location
    location = _format_location(agent)
    if location:
        parts.append(f"You are {location}.")

    # Household
    household = _format_household(agent)
    if household:
        parts.append(f"You are {household}.")

    # Additional professional attributes
    additional_prof = []
    if agent.get("participation_in_research") or agent.get("research_activity_level") in ["regular", "leading/PI"]:
        additional_prof.append("actively involved in research")
    if agent.get("teaching_responsibility"):
        additional_prof.append("have teaching responsibilities")
    if agent.get("professional_society_membership"):
        additional_prof.append("a member of professional societies")

    if additional_prof:
        parts.append("You are " + ", ".join(additional_prof) + ".")

    # Personality summary (if spec provided)
    if population_spec:
        personality_attrs = [
            attr for attr in population_spec.attributes
            if attr.category == "personality"
        ]
        if personality_attrs:
            personality = _generate_personality_summary(agent, personality_attrs)
            if personality:
                parts.append(personality)

        # Context-specific attributes
        context_attrs = [
            attr for attr in population_spec.attributes
            if attr.category == "context_specific"
        ]
        if context_attrs:
            context_info = _format_context_attributes(agent, context_attrs)
            if context_info:
                parts.append(context_info)

    # Fallback: include any other notable attributes
    if not parts:
        # Generic fallback using available attributes
        parts.append("You are a member of this population.")
        for key, value in list(agent.items())[:5]:
            if key.startswith("_") or key == "id":
                continue
            if isinstance(value, (bool, int, float, str)):
                parts.append(f"Your {key.replace('_', ' ')} is {value}.")

    return " ".join(parts)


# =============================================================================
# Main Public API
# =============================================================================


def generate_persona(
    agent: dict[str, Any],
    population_spec: PopulationSpec | None = None,
) -> str:
    """Generate a natural language persona from agent attributes.

    Uses template-based rendering if a persona_template is available in the
    population spec. Falls back to hardcoded logic otherwise.

    Args:
        agent: Agent dictionary with attributes
        population_spec: Optional population spec for attribute categorization and template

    Returns:
        Natural language persona string
    """
    # Try template-based rendering first if available
    if population_spec and population_spec.meta.persona_template:
        rendered = render_persona_from_template(
            population_spec.meta.persona_template,
            agent,
        )
        if rendered:
            return rendered

    # Fall back to hardcoded logic
    return _generate_persona_fallback(agent, population_spec)


def generate_persona_for_reasoning(
    agent: dict[str, Any],
    population_spec: PopulationSpec,
    event_context: str | None = None,
) -> str:
    """Generate an enhanced persona for reasoning context.

    This version includes additional framing for the reasoning task.

    Args:
        agent: Agent dictionary
        population_spec: Population specification
        event_context: Optional context about the event being reasoned about

    Returns:
        Enhanced persona string for LLM reasoning
    """
    base_persona = generate_persona(agent, population_spec)

    # Add framing for simulation
    framing = (
        "You are simulating how this person would react to news/information. "
        "Think and respond as this person would, based on their background and personality."
    )

    return f"{base_persona}\n\n{framing}"
