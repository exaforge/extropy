"""Persona template generation via LLM.

Generates templates for converting agent attributes into natural language
personas. Templates use Python's {attribute} format placeholders.

The template is used for the narrative intro only - all remaining attributes
are automatically appended as a structured list by persona.py.
"""

from ...core.llm import simple_call
from ...core.models import PopulationSpec, AttributeSpec
from ...simulation.persona import is_narrative_safe


# JSON Schema for persona template generation response
PERSONA_TEMPLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "template": {
            "type": "string",
            "description": "A second-person narrative intro using {attribute_name} placeholders",
        },
    },
    "required": ["template"],
    "additionalProperties": False,
}


def _get_narrative_safe_attrs(attributes: list[AttributeSpec]) -> list[AttributeSpec]:
    """Filter to only narrative-safe attributes.

    Excludes booleans and 0-1 floats which read awkwardly in prose.
    """
    return [attr for attr in attributes if is_narrative_safe(attr)]


def _build_attribute_summary(attributes: list[AttributeSpec]) -> str:
    """Build a summary of available attributes for the prompt."""
    lines = []

    for attr in attributes:
        type_info = attr.type
        if attr.sampling and attr.sampling.distribution:
            dist = attr.sampling.distribution
            if hasattr(dist, "options") and dist.options:
                type_info = f"categorical: {', '.join(dist.options[:5])}"
                if len(dist.options) > 5:
                    type_info += ", ..."
        lines.append(f"- {attr.name} ({type_info}): {attr.description}")

    return "\n".join(lines)


class PersonaTemplateError(Exception):
    """Raised when persona template generation fails after retries."""

    pass


def generate_persona_template(
    spec: PopulationSpec,
    log: bool = True,
    max_retries: int = 3,
) -> str:
    """Generate a persona template for a population spec.

    Uses LLM to create a natural, population-appropriate template that
    serves as the narrative INTRO for persona generation. All remaining
    attributes are automatically appended as a structured list.

    Only narrative-safe attributes (no booleans, no 0-1 floats) are
    included - these work better in prose form.

    Args:
        spec: Population specification with attributes
        log: Whether to log the LLM call
        max_retries: Maximum retry attempts on failure

    Returns:
        Template string with {attribute} placeholders

    Raises:
        PersonaTemplateError: If generation fails after all retries
    """
    # Only include narrative-safe attributes
    safe_attrs = _get_narrative_safe_attrs(spec.attributes)
    safe_attr_names = {a.name for a in safe_attrs}
    attribute_summary = _build_attribute_summary(safe_attrs)

    base_prompt = f"""Generate a narrative intro template for agents in this population.

Population: {spec.meta.description}
Geography: {spec.meta.geography or "Not specified"}

Available attributes for narrative (values will be pre-formatted as readable strings):
{attribute_summary}

IMPORTANT: This template is ONLY for the narrative intro. All other attributes
(booleans, scores, etc.) will be automatically appended as a structured list.
Don't try to include everything - focus on core identity.

Write 2-3 natural sentences using {{attribute_name}} placeholders.

Rules:
- Use ONLY attributes from the list above (they are pre-filtered for narrative use)
- Focus on: age, gender, role, specialty, location, experience
- Values are pre-formatted: "University Hospital" not "university_hospital"
- Start with "You are a..."
- Write naturally, like introducing someone
- Use {{attribute_name}} syntax for placeholders (curly braces)

Example output:
"You are a {{age}}-year-old {{gender}} surgeon specializing in {{surgical_specialty}}, working as a {{role_rank}} at a {{employer_type}} in {{location_state}} with {{years_experience}} years of experience."

Output only the template string, no explanation."""

    last_error = None

    for attempt in range(max_retries):
        prompt = base_prompt

        # Add error feedback for retries
        if last_error:
            prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nPlease fix and try again."

        try:
            response = simple_call(
                prompt=prompt,
                response_schema=PERSONA_TEMPLATE_SCHEMA,
                schema_name="persona_template",
                log=log,
            )
            template = response.get("template", "")

            if not template:
                last_error = "Empty template returned"
                continue

            # Validate: check all placeholders exist in safe attrs
            import re

            used_attrs = set(re.findall(r"\{(\w+)\}", template))
            invalid_attrs = used_attrs - safe_attr_names

            if invalid_attrs:
                last_error = f"Invalid attributes used: {invalid_attrs}. Only use: {', '.join(sorted(safe_attr_names))}"
                continue

            # Success
            return template

        except Exception as e:
            last_error = str(e)
            continue

    raise PersonaTemplateError(
        f"Failed to generate persona template after {max_retries} attempts. Last error: {last_error}"
    )




def validate_persona_template(template: str, sample_agent: dict) -> tuple[bool, str]:
    """Validate a persona template by rendering it with a sample agent.

    Args:
        template: Template string with {attribute} placeholders
        sample_agent: Sample agent dictionary with attributes

    Returns:
        Tuple of (is_valid, error_message_or_rendered_result)
    """
    try:
        result = template.format(**sample_agent)
        return True, result.strip()
    except KeyError as e:
        return False, f"Missing attribute: {e}"
    except Exception as e:
        return False, f"Template error: {str(e)}"


def refine_persona_template(
    current_template: str,
    spec: PopulationSpec,
    feedback: str,
    log: bool = True,
    max_retries: int = 3,
) -> str:
    """Refine a persona template based on user feedback.

    Args:
        current_template: The current template to refine
        spec: Population specification
        feedback: User feedback on what to change
        log: Whether to log the LLM call
        max_retries: Maximum retry attempts on failure

    Returns:
        Refined template string

    Raises:
        PersonaTemplateError: If refinement fails after all retries
    """
    import re

    # Only include narrative-safe attributes
    safe_attrs = _get_narrative_safe_attrs(spec.attributes)
    safe_attr_names = {a.name for a in safe_attrs}
    attribute_summary = _build_attribute_summary(safe_attrs)

    base_prompt = f"""Refine this narrative intro template based on user feedback.

Population: {spec.meta.description}

Available attributes (narrative-safe only, values will be pre-formatted):
{attribute_summary}

Current template:
{current_template}

User feedback: {feedback}

Rules:
- Use ONLY attributes from the list above
- Use {{attribute_name}} syntax for placeholders
- Keep it 2-3 sentences focused on core identity
- Start with "You are a..."
- This is just the intro - other attributes are appended as a list

Output only the refined template string, no explanation."""

    last_error = None

    for _ in range(max_retries):
        prompt = base_prompt

        if last_error:
            prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nPlease fix and try again."

        try:
            response = simple_call(
                prompt=prompt,
                response_schema=PERSONA_TEMPLATE_SCHEMA,
                schema_name="persona_template",
                log=log,
            )
            template = response.get("template", "")

            if not template:
                last_error = "Empty template returned"
                continue

            # Validate placeholders
            used_attrs = set(re.findall(r"\{(\w+)\}", template))
            invalid_attrs = used_attrs - safe_attr_names

            if invalid_attrs:
                last_error = f"Invalid attributes used: {invalid_attrs}. Only use: {', '.join(sorted(safe_attr_names))}"
                continue

            return template

        except Exception as e:
            last_error = str(e)
            continue

    raise PersonaTemplateError(
        f"Failed to refine persona template after {max_retries} attempts. Last error: {last_error}"
    )
