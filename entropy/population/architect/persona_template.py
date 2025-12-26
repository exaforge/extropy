"""Persona template generation via LLM.

Generates Jinja2 templates for converting agent attributes into
natural language personas. Templates are specific to each population
and can be customized by the user.
"""

from typing import Any

from ...core.llm import reasoning_call
from ...core.models import PopulationSpec, AttributeSpec


# JSON Schema for persona template generation response
PERSONA_TEMPLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "template": {
            "type": "string",
            "description": "Jinja2 template that converts agent attributes into a first-person persona narrative",
        },
        "explanation": {
            "type": "string",
            "description": "Brief explanation of how the template uses the attributes",
        },
    },
    "required": ["template", "explanation"],
    "additionalProperties": False,
}


def _build_attribute_summary(attributes: list[AttributeSpec]) -> str:
    """Build a summary of available attributes for the prompt."""
    lines = []

    # Group by category
    by_category: dict[str, list[AttributeSpec]] = {
        "universal": [],
        "population_specific": [],
        "context_specific": [],
        "personality": [],
    }

    for attr in attributes:
        by_category[attr.category].append(attr)

    category_labels = {
        "universal": "Universal (demographics)",
        "population_specific": "Population-specific",
        "context_specific": "Context/scenario-specific",
        "personality": "Personality traits",
    }

    for category, label in category_labels.items():
        attrs = by_category[category]
        if not attrs:
            continue

        lines.append(f"\n**{label}:**")
        for attr in attrs:
            type_info = attr.type
            if attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "options") and dist.options:
                    type_info = f"categorical: {', '.join(dist.options[:5])}"
                    if len(dist.options) > 5:
                        type_info += ", ..."
            lines.append(f"- `{attr.name}` ({type_info}): {attr.description}")

    return "\n".join(lines)


def generate_persona_template(
    spec: PopulationSpec,
    log: bool = True,
) -> tuple[str, str]:
    """Generate a Jinja2 persona template for a population spec.

    Uses LLM reasoning to create a natural, population-appropriate template
    that converts agent attributes into first-person persona narratives.

    Args:
        spec: Population specification with attributes
        log: Whether to log the LLM call

    Returns:
        Tuple of (template_string, explanation)
    """
    attribute_summary = _build_attribute_summary(spec.attributes)

    prompt = f"""You are generating a Jinja2 template that will convert agent attribute data into a natural language first-person persona.

## Population Description
{spec.meta.description}
{f"Geography: {spec.meta.geography}" if spec.meta.geography else ""}

## Available Attributes
{attribute_summary}

## Task
Write a Jinja2 template that creates a natural, engaging first-person persona narrative using these attributes. The persona should:

1. **Be written in first person** ("I am a...", "I work at...", "I believe...")
2. **Sound natural and human** - not robotic or listing attributes mechanically
3. **Use conditional blocks** (`{{% if ... %}}`) to only include relevant information
4. **Handle missing values gracefully** (use `{{% if attr %}}` checks)
5. **Include personality descriptors** that translate 0-1 scaled traits into natural language
6. **Be concise** - aim for 3-5 sentences that capture the essential identity

## Example Patterns

For age/gender:
```jinja2
{{% if age %}}I am a {{{{ age }}}}-year-old{{% endif %}} {{% if gender == 'male' %}}man{{% elif gender == 'female' %}}woman{{% else %}}person{{% endif %}}
```

For personality traits (0-1 scale):
```jinja2
{{% if openness is defined %}}{{% if openness > 0.7 %}}I am quite curious and open to new ideas.{{% elif openness < 0.3 %}}I prefer familiar approaches and routines.{{% endif %}}{{% endif %}}
```

For categorical attributes:
```jinja2
{{% if employer_type %}}I work at a {{{{ employer_type }}}}.{{% endif %}}
```

## Requirements
- Template must be valid Jinja2 syntax
- Use `{{{{ variable }}}}` for value substitution
- Use `{{% if condition %}}...{{% endif %}}` for conditionals
- Attributes are accessed directly by name (e.g., `age`, `gender`, `employer_type`)
- For boolean checks, use `{{% if attr %}}` or `{{% if attr == True %}}`

Generate a template tailored to this specific population that brings agents to life as believable personas."""

    response = reasoning_call(
        prompt=prompt,
        response_schema=PERSONA_TEMPLATE_SCHEMA,
        schema_name="persona_template",
        reasoning_effort="medium",
        log=log,
    )

    template = response.get("template", "")
    explanation = response.get("explanation", "")

    return template, explanation


def validate_persona_template(template: str, sample_agent: dict[str, Any]) -> tuple[bool, str]:
    """Validate a persona template by rendering it with a sample agent.

    Args:
        template: Jinja2 template string
        sample_agent: Sample agent dictionary with attributes

    Returns:
        Tuple of (is_valid, error_message_or_rendered_result)
    """
    from jinja2 import Environment, BaseLoader, TemplateSyntaxError, UndefinedError

    try:
        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(template)
        result = tmpl.render(**sample_agent)
        return True, result.strip()
    except TemplateSyntaxError as e:
        return False, f"Template syntax error at line {e.lineno}: {e.message}"
    except UndefinedError as e:
        return False, f"Undefined variable: {e.message}"
    except Exception as e:
        return False, f"Template error: {str(e)}"


def refine_persona_template(
    current_template: str,
    spec: PopulationSpec,
    feedback: str,
    log: bool = True,
) -> tuple[str, str]:
    """Refine a persona template based on user feedback.

    Args:
        current_template: The current template to refine
        spec: Population specification
        feedback: User feedback on what to change
        log: Whether to log the LLM call

    Returns:
        Tuple of (refined_template, explanation)
    """
    attribute_summary = _build_attribute_summary(spec.attributes)

    prompt = f"""You are refining a Jinja2 persona template based on user feedback.

## Population Description
{spec.meta.description}

## Available Attributes
{attribute_summary}

## Current Template
```jinja2
{current_template}
```

## User Feedback
{feedback}

## Task
Modify the template to address the user's feedback while maintaining:
- Valid Jinja2 syntax
- Natural, first-person language
- Proper handling of missing/optional attributes
- Concise but descriptive personas

Generate the improved template."""

    response = reasoning_call(
        prompt=prompt,
        response_schema=PERSONA_TEMPLATE_SCHEMA,
        schema_name="persona_template",
        reasoning_effort="medium",
        log=log,
    )

    template = response.get("template", "")
    explanation = response.get("explanation", "")

    return template, explanation


# Default fallback template for when LLM generation fails
DEFAULT_FALLBACK_TEMPLATE = """{% if age %}I am a {{ age }}-year-old{% endif %}{% if gender == 'male' %} man{% elif gender == 'female' %} woman{% else %} person{% endif %}.
{% if occupation or role %}I work as a {{ occupation or role }}.{% endif %}
{% if location or city or region %}I am based in {{ location or city or region }}.{% endif %}
{% for key, value in _extra_attrs.items() %}{% if loop.index <= 3 %}My {{ key | replace('_', ' ') }} is {{ value }}.{% endif %}{% endfor %}"""
