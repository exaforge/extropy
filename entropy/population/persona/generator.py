"""Persona configuration generator.

Uses LLM to analyze population attributes and generate rendering configuration.
This is called once per population, not per agent.

Pipeline:
    Step 1: generate_structure() - Classify attributes and create groups
    Step 2: generate_boolean_phrasings() - First-person phrases for booleans
    Step 3: generate_categorical_phrasings() - First-person phrases for categoricals
    Step 4: generate_relative_phrasings() - Z-score bucket phrases for psychological traits
    Step 5: generate_concrete_phrasings() - Templates for numeric values
"""

from typing import Any

from ...core.llm import reasoning_call
from ...core.models import PopulationSpec, AttributeSpec
from ...utils.callbacks import StepProgressCallback
from .config import (
    PersonaConfig,
    AttributeTreatment,
    TreatmentType,
    AttributeGroup,
    AttributePhrasing,
    BooleanPhrasing,
    CategoricalPhrasing,
    RelativePhrasing,
    RelativeLabels,
    ConcretePhrasing,
    PopulationStats,
)
from .stats import compute_population_stats


class PersonaConfigError(Exception):
    """Raised when persona config generation fails."""

    pass


ProgressCallback = StepProgressCallback


# =============================================================================
# Step 1: Structure (Treatments + Groups + Intro)
# =============================================================================

STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "intro_template": {
            "type": "string",
            "description": "First-person narrative intro (2-4 sentences) with {attribute_name} placeholders",
        },
        "attribute_treatments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "treatment": {"type": "string", "enum": ["concrete", "relative"]},
                    "group": {"type": "string"},
                },
                "required": ["attribute", "treatment", "group"],
                "additionalProperties": False,
            },
        },
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "label": {"type": "string"},
                    "order": {"type": "integer"},
                },
                "required": ["name", "label", "order"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["intro_template", "attribute_treatments", "groups"],
    "additionalProperties": False,
}


def _build_attribute_summary(attributes: list[AttributeSpec]) -> str:
    """Build a summary of all attributes for the prompt."""
    lines = []
    for attr in attributes:
        type_info = attr.type
        extra = ""
        if attr.sampling and attr.sampling.distribution:
            dist = attr.sampling.distribution
            if hasattr(dist, "options") and dist.options:
                type_info = "categorical"
                extra = f" [{len(dist.options)} options]"
            elif hasattr(dist, "probability_true"):
                type_info = "boolean"
        lines.append(f"- {attr.name} ({type_info}{extra}): {attr.description}")
    return "\n".join(lines)


def generate_structure(
    spec: PopulationSpec,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[AttributeTreatment], list[AttributeGroup], str]:
    """Step 1: Generate attribute treatments, groups, and intro template."""

    if on_progress:
        on_progress("1", "Classifying attributes and creating groups...")

    attr_summary = _build_attribute_summary(spec.attributes)

    prompt = f"""You are configuring how agent personas will be rendered.

POPULATION: {spec.meta.description}

ATTRIBUTES ({len(spec.attributes)}):
{attr_summary}

Generate:

1. INTRO_TEMPLATE: A first-person narrative intro (2-4 sentences) using {{attribute_name}} placeholders.
   Focus on core identity. Example: "I'm a {{age}}-year-old {{gender}} living in {{home_zip_code}}..."
   
   IMPORTANT for booleans: Do NOT include boolean attributes directly like "...my awareness: {{aware}}".
   Either skip them in the intro, OR weave them naturally: "I've been following the new policy closely" (if aware=true is common).
   The intro should read naturally without "yes/no" or "true/false" appearing.

2. ATTRIBUTE_TREATMENTS: For EACH attribute:
   - treatment: "concrete" (keep numbers) or "relative" (express relative to population)
   - group: which group it belongs to
   
   Use "concrete" for: age, income, distance, time, costs, counts
   Use "relative" for: trust, sensitivity, personality traits, attitudes

3. GROUPS: Create 6-10 logical groups with name (snake_case), label (like "My Commute"), and order (1=first)."""

    response = reasoning_call(
        prompt=prompt,
        response_schema=STRUCTURE_SCHEMA,
        schema_name="persona_structure",
        log=True,
    )

    if not response:
        raise PersonaConfigError("Empty response for structure generation")

    # Parse treatments
    treatments = []
    for t in response.get("attribute_treatments", []):
        treatments.append(
            AttributeTreatment(
                attribute=t["attribute"],
                treatment=TreatmentType(t["treatment"]),
                group=t["group"],
            )
        )

    # Parse groups
    group_data = sorted(response.get("groups", []), key=lambda g: g.get("order", 999))
    group_attrs: dict[str, list[str]] = {}
    for t in treatments:
        group_attrs.setdefault(t.group, []).append(t.attribute)

    groups = []
    for g in group_data:
        groups.append(
            AttributeGroup(
                name=g["name"],
                label=g["label"],
                attributes=group_attrs.get(g["name"], []),
            )
        )

    intro = response.get("intro_template", "")

    if on_progress:
        on_progress(
            "1",
            f"Created {len(groups)} groups, classified {len(treatments)} attributes",
        )

    return treatments, groups, intro


# =============================================================================
# Step 2: Boolean Phrasings
# =============================================================================

BOOLEAN_SCHEMA = {
    "type": "object",
    "properties": {
        "phrasings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "true_phrase": {"type": "string"},
                    "false_phrase": {"type": "string"},
                },
                "required": ["attribute", "true_phrase", "false_phrase"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["phrasings"],
    "additionalProperties": False,
}


def generate_boolean_phrasings(
    spec: PopulationSpec,
    on_progress: ProgressCallback | None = None,
) -> list[BooleanPhrasing]:
    """Step 2: Generate first-person phrases for boolean attributes."""

    bool_attrs = [a for a in spec.attributes if a.type == "boolean"]

    if not bool_attrs:
        return []

    if on_progress:
        on_progress(
            "2", f"Generating phrases for {len(bool_attrs)} boolean attributes..."
        )

    attr_list = "\n".join(f"- {a.name}: {a.description}" for a in bool_attrs)

    prompt = f"""Generate first-person phrases for these boolean attributes.

POPULATION: {spec.meta.description}

BOOLEAN ATTRIBUTES:
{attr_list}

For each attribute, provide:
- true_phrase: What to say when True (e.g., "I own a bike")
- false_phrase: What to say when False (e.g., "I don't own a bike")

All phrases must be first-person ("I", "my", "me")."""

    response = reasoning_call(
        prompt=prompt,
        response_schema=BOOLEAN_SCHEMA,
        schema_name="boolean_phrasings",
        log=True,
    )

    if not response:
        raise PersonaConfigError("Empty response for boolean phrasings")

    phrasings = []
    for p in response.get("phrasings", []):
        phrasings.append(
            BooleanPhrasing(
                attribute=p["attribute"],
                true_phrase=p["true_phrase"],
                false_phrase=p["false_phrase"],
            )
        )

    if on_progress:
        on_progress("2", f"Generated {len(phrasings)} boolean phrasings")

    return phrasings


# =============================================================================
# Step 3: Categorical Phrasings
# =============================================================================

CATEGORICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "phrasings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "option_phrases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "option": {"type": "string"},
                                "phrase": {"type": "string"},
                            },
                            "required": ["option", "phrase"],
                            "additionalProperties": False,
                        },
                    },
                    "null_options": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "null_phrase": {"type": "string"},
                },
                "required": [
                    "attribute",
                    "option_phrases",
                    "null_options",
                    "null_phrase",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["phrasings"],
    "additionalProperties": False,
}


def generate_categorical_phrasings(
    spec: PopulationSpec,
    on_progress: ProgressCallback | None = None,
) -> list[CategoricalPhrasing]:
    """Step 3: Generate first-person phrases for categorical attributes."""

    cat_attrs = []
    for a in spec.attributes:
        if a.type == "categorical" and a.sampling and a.sampling.distribution:
            dist = a.sampling.distribution
            if hasattr(dist, "options") and dist.options:
                cat_attrs.append((a, dist.options))

    if not cat_attrs:
        return []

    if on_progress:
        on_progress(
            "3", f"Generating phrases for {len(cat_attrs)} categorical attributes..."
        )

    attr_lines = []
    for a, options in cat_attrs:
        opts_str = ", ".join(options[:8])
        if len(options) > 8:
            opts_str += f", ... ({len(options)} total)"
        attr_lines.append(f"- {a.name}: {a.description}\n  Options: {opts_str}")

    prompt = f"""Generate first-person phrases for these categorical attributes.

POPULATION: {spec.meta.description}

CATEGORICAL ATTRIBUTES:
{chr(10).join(attr_lines)}

For each attribute, provide option_phrases - an array with one entry per option:
- option: The option value (e.g., "cannot_shift")
- phrase: First-person phrase (e.g., "My schedule is fixed — I can't easily avoid peak hours")

Also provide:
- null_options: option values that represent explicit absence/not-applicable/no-preference states for THIS attribute (subset of options, or [] if none)
- null_phrase: natural first-person phrase to use whenever value is in null_options.
  If null_options is [], return an empty string for null_phrase.

Rules:
- Every option must appear exactly once in option_phrases.
- Keep phrases natural and conversational; never repeat raw option tokens verbatim as standalone wording.
- null_phrase must be natural language, not a copied enum token.

All phrases must be first-person ("I", "my", "me") and sound natural."""

    response = reasoning_call(
        prompt=prompt,
        response_schema=CATEGORICAL_SCHEMA,
        schema_name="categorical_phrasings",
        log=True,
    )

    if not response:
        raise PersonaConfigError("Empty response for categorical phrasings")

    option_map = {attr.name: set(options) for attr, options in cat_attrs}
    phrasings = []
    for p in response.get("phrasings", []):
        attr_name = p["attribute"]
        valid_options = option_map.get(attr_name)
        if valid_options is None:
            raise PersonaConfigError(
                f"Unknown categorical attribute in persona phrasing output: {attr_name}"
            )

        phrases_dict = {}
        for op in p.get("option_phrases", []):
            phrases_dict[op["option"]] = op["phrase"]

        actual_options = set(phrases_dict.keys())
        missing = valid_options - actual_options
        extras = actual_options - valid_options
        if missing or extras:
            parts = []
            if missing:
                parts.append(f"missing options: {sorted(missing)}")
            if extras:
                parts.append(f"unknown options: {sorted(extras)}")
            raise PersonaConfigError(
                f"Invalid categorical phrasing coverage for {attr_name} ({'; '.join(parts)})"
            )

        null_options = list(p.get("null_options", []))
        unknown_null_options = set(null_options) - valid_options
        if unknown_null_options:
            raise PersonaConfigError(
                f"Invalid null_options for {attr_name}: {sorted(unknown_null_options)}"
            )

        null_phrase_raw = (p.get("null_phrase") or "").strip()
        null_phrase = null_phrase_raw if null_phrase_raw else None
        if null_options and not null_phrase:
            raise PersonaConfigError(
                f"Missing null_phrase for {attr_name} with null_options={null_options}"
            )

        phrasings.append(
            CategoricalPhrasing(
                attribute=attr_name,
                phrases=phrases_dict,
                null_options=null_options,
                null_phrase=null_phrase,
            )
        )

    if on_progress:
        on_progress("3", f"Generated {len(phrasings)} categorical phrasings")

    return phrasings


# =============================================================================
# Step 4: Relative Phrasings
# =============================================================================

RELATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "phrasings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "much_below": {"type": "string"},
                    "below": {"type": "string"},
                    "average": {"type": "string"},
                    "above": {"type": "string"},
                    "much_above": {"type": "string"},
                },
                "required": [
                    "attribute",
                    "much_below",
                    "below",
                    "average",
                    "above",
                    "much_above",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["phrasings"],
    "additionalProperties": False,
}


def generate_relative_phrasings(
    spec: PopulationSpec,
    treatments: list[AttributeTreatment],
    on_progress: ProgressCallback | None = None,
) -> list[RelativePhrasing]:
    """Step 4: Generate z-score bucket phrases for relative attributes."""

    relative_attrs = [
        t.attribute for t in treatments if t.treatment == TreatmentType.RELATIVE
    ]
    attr_map = {a.name: a for a in spec.attributes}
    rel_specs = [attr_map[name] for name in relative_attrs if name in attr_map]

    if not rel_specs:
        return []

    if on_progress:
        on_progress(
            "4", f"Generating phrases for {len(rel_specs)} relative attributes..."
        )

    attr_list = "\n".join(f"- {a.name}: {a.description}" for a in rel_specs)

    prompt = f"""Generate relative positioning phrases for these psychological/attitudinal attributes.

POPULATION: {spec.meta.description}

RELATIVE ATTRIBUTES:
{attr_list}

For each attribute, provide 5 phrases for different z-score buckets:
- much_below (z < -1): Very low compared to population
- below (-1 ≤ z < -0.3): Somewhat below average
- average (-0.3 ≤ z ≤ 0.3): About average
- above (0.3 < z ≤ 1): Somewhat above average
- much_above (z > 1): Very high compared to population

Example for trust_in_institutions:
- much_below: "I trust institutions far less than most people"
- below: "I'm more skeptical of institutions than most"
- average: "I'm about as trusting of institutions as the average person"
- above: "I generally trust institutions more than most"
- much_above: "I have much more faith in institutions than most people"

All phrases must be first-person and compare to "most people" or "average"."""

    response = reasoning_call(
        prompt=prompt,
        response_schema=RELATIVE_SCHEMA,
        schema_name="relative_phrasings",
        log=True,
    )

    if not response:
        raise PersonaConfigError("Empty response for relative phrasings")

    phrasings = []
    for p in response.get("phrasings", []):
        phrasings.append(
            RelativePhrasing(
                attribute=p["attribute"],
                labels=RelativeLabels(
                    much_below=p["much_below"],
                    below=p["below"],
                    average=p["average"],
                    above=p["above"],
                    much_above=p["much_above"],
                ),
            )
        )

    if on_progress:
        on_progress("4", f"Generated {len(phrasings)} relative phrasings")

    return phrasings


# =============================================================================
# Step 5: Concrete Phrasings
# =============================================================================

CONCRETE_SCHEMA = {
    "type": "object",
    "properties": {
        "phrasings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "template": {"type": "string"},
                    "format_spec": {"type": "string"},
                    "prefix": {"type": "string"},
                    "suffix": {"type": "string"},
                },
                "required": [
                    "attribute",
                    "template",
                    "format_spec",
                    "prefix",
                    "suffix",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["phrasings"],
    "additionalProperties": False,
}


def generate_concrete_phrasings(
    spec: PopulationSpec,
    treatments: list[AttributeTreatment],
    on_progress: ProgressCallback | None = None,
) -> list[ConcretePhrasing]:
    """Step 5: Generate templates for concrete (numeric) attributes."""

    concrete_attrs = [
        t.attribute for t in treatments if t.treatment == TreatmentType.CONCRETE
    ]
    attr_map = {a.name: a for a in spec.attributes}

    # Only include int/float attributes that are concrete
    conc_specs = []
    for name in concrete_attrs:
        if name in attr_map:
            a = attr_map[name]
            if a.type in ("int", "float"):
                conc_specs.append(a)

    if not conc_specs:
        return []

    if on_progress:
        on_progress(
            "5", f"Generating templates for {len(conc_specs)} concrete attributes..."
        )

    attr_list = "\n".join(f"- {a.name} ({a.type}): {a.description}" for a in conc_specs)

    prompt = f"""Generate first-person templates for these numeric attributes.

POPULATION: {spec.meta.description}

NUMERIC ATTRIBUTES:
{attr_list}

For each attribute, provide:
- template: A sentence with {{value}} placeholder (e.g., "I drive {{value}} to downtown")
- format_spec: How to format the number. Use one of these:
  - ".0f" for whole numbers (age, counts, years)
  - ".1f" for 1 decimal place (distances, durations)
  - ".2f" for 2 decimal places (money - combine with "$" prefix)
  - "time12" for decimal hours converted to 12-hour time (e.g., 8.5 → "8:30 AM")
  - "time24" for decimal hours converted to 24-hour time (e.g., 14.5 → "14:30")
- prefix: Text before the number (e.g., "$" for money), empty string if none
- suffix: Text after the number (e.g., " miles"), empty string if none

Examples:
- commute_distance_miles: template="I drive {{value}} to downtown", format_spec=".1f", prefix="", suffix=" miles"
- parking_cost_daily_usd: template="I pay {{value}} a day for parking", format_spec=".2f", prefix="$", suffix=""
- age: template="I'm {{value}} years old", format_spec=".0f", prefix="", suffix=""
- work_start_time: template="I usually start work around {{value}}", format_spec="time12", prefix="", suffix=""
- household_size: template="There are {{value}} people in my household", format_spec=".0f", prefix="", suffix=""

Choose the appropriate format_spec based on what makes sense for each attribute:
- Times (work_start_time, departure_time, etc.): use "time12"
- Money (income, costs, prices): use ".2f" with "$" prefix
- Whole-number counts (household_size, vehicle_count, years): use ".0f"
- Measurements with decimals (miles, minutes, hours): use ".1f"

All templates must be first-person and read naturally with the number inserted."""

    response = reasoning_call(
        prompt=prompt,
        response_schema=CONCRETE_SCHEMA,
        schema_name="concrete_phrasings",
        log=True,
    )

    if not response:
        raise PersonaConfigError("Empty response for concrete phrasings")

    phrasings = []
    for p in response.get("phrasings", []):
        phrasings.append(
            ConcretePhrasing(
                attribute=p["attribute"],
                template=p["template"],
                format_spec=p.get("format_spec"),
                prefix=p.get("prefix", ""),
                suffix=p.get("suffix", ""),
            )
        )

    if on_progress:
        on_progress("5", f"Generated {len(phrasings)} concrete phrasings")

    return phrasings


# =============================================================================
# Main Orchestrator
# =============================================================================


def generate_persona_config(
    spec: PopulationSpec,
    agents: list[dict[str, Any]] | None = None,
    log: bool = True,
    on_progress: ProgressCallback | None = None,
) -> PersonaConfig:
    """Generate persona configuration for a population.

    Orchestrates 5 steps:
    - Step 1: Classify attributes and create groups
    - Step 2: Generate boolean phrasings
    - Step 3: Generate categorical phrasings
    - Step 4: Generate relative phrasings
    - Step 5: Generate concrete phrasings

    Args:
        spec: Population specification with attributes
        agents: Optional sampled agents for computing population stats
        log: Whether to log LLM calls
        on_progress: Optional callback (step, status) for progress updates

    Returns:
        PersonaConfig ready for rendering

    Raises:
        PersonaConfigError: If generation fails
    """

    def report(step: str, status: str):
        if on_progress:
            on_progress(step, status)

    # Step 1: Structure
    treatments, groups, intro = generate_structure(spec, on_progress)

    # Step 2: Boolean phrasings
    boolean_phrasings = generate_boolean_phrasings(spec, on_progress)

    # Step 3: Categorical phrasings
    categorical_phrasings = generate_categorical_phrasings(spec, on_progress)

    # Step 4: Relative phrasings
    relative_phrasings = generate_relative_phrasings(spec, treatments, on_progress)

    # Step 5: Concrete phrasings
    concrete_phrasings = generate_concrete_phrasings(spec, treatments, on_progress)

    # Combine all phrasings
    phrasings = AttributePhrasing(
        boolean=boolean_phrasings,
        categorical=categorical_phrasings,
        relative=relative_phrasings,
        concrete=concrete_phrasings,
    )

    # Compute population stats if agents provided
    population_stats = PopulationStats()
    if agents:
        report("stats", "Computing population statistics...")
        population_stats = compute_population_stats(agents)
        report("stats", f"Computed stats for {len(population_stats.stats)} attributes")

    return PersonaConfig(
        population_description=spec.meta.description,
        intro_template=intro,
        treatments=treatments,
        groups=groups,
        phrasings=phrasings,
        population_stats=population_stats,
    )
