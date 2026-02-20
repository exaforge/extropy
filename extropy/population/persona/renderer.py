"""Persona renderer.

Applies PersonaConfig to individual agents to produce first-person persona text.
Pure computation — no LLM calls.
"""

import re
from typing import Any

from .config import (
    PersonaConfig,
    BooleanPhrasing,
    CategoricalPhrasing,
    RelativePhrasing,
    ConcretePhrasing,
)


def _normalize_value(value: str) -> str:
    """Normalize categorical option values for comparison."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _contains_raw_option_token(phrase: str, option: str) -> bool:
    """Detect whether phrase contains the raw categorical token text."""
    normalized_phrase = phrase.lower().replace("_", " ")
    normalized_option = option.lower().replace("_", " ")
    # Word-boundary match avoids false positives inside other words.
    pattern = r"\b" + re.escape(normalized_option) + r"\b"
    return re.search(pattern, normalized_phrase) is not None


def _format_time(decimal_hours: float, use_12hr: bool = True) -> str:
    """Convert decimal hours to human-readable time.

    Args:
        decimal_hours: Time as decimal (e.g., 8.5 for 8:30)
        use_12hr: If True, use 12-hour format with AM/PM

    Returns:
        Formatted time string
    """
    hours = int(decimal_hours)
    minutes = int((decimal_hours - hours) * 60)

    if use_12hr:
        period = "AM" if hours < 12 else "PM"
        display_hour = hours % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minutes:02d} {period}"
    else:
        return f"{hours}:{minutes:02d}"


def _format_concrete_value(value: Any, phrasing: ConcretePhrasing) -> str:
    """Format a concrete value using the phrasing template."""
    if value is None:
        return ""

    # Get format spec (default to sensible formatting)
    fmt = phrasing.format_spec

    # Handle special format specs
    if fmt == "time12" and isinstance(value, (int, float)):
        formatted = _format_time(float(value), use_12hr=True)
    elif fmt == "time24" and isinstance(value, (int, float)):
        formatted = _format_time(float(value), use_12hr=False)
    elif fmt and isinstance(value, (int, float)):
        # Standard Python format spec
        try:
            formatted = format(value, fmt)
            # Add thousands separator for large numbers
            if value >= 1000 and "," not in fmt:
                formatted = format(value, "," + fmt)
        except (ValueError, TypeError):
            formatted = str(value)
    elif isinstance(value, (int, float)):
        # Default formatting based on value characteristics
        if isinstance(value, float) and value == int(value):
            formatted = f"{int(value):,}"
        elif isinstance(value, float):
            formatted = f"{value:,.1f}"
        else:
            formatted = f"{value:,}"
    else:
        formatted = str(value)

    # Apply prefix/suffix
    if phrasing.prefix or phrasing.suffix:
        formatted = f"{phrasing.prefix}{formatted}{phrasing.suffix}"

    # Fill template
    try:
        return phrasing.template.format(value=formatted)
    except (KeyError, ValueError):
        return phrasing.template.replace("{value}", formatted)


def _format_relative_value(
    value: Any, phrasing: RelativePhrasing, config: PersonaConfig
) -> str:
    """Format a relative value using z-score positioning."""
    if value is None:
        return ""

    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return ""

    # Get z-score from population stats
    z_score = config.population_stats.get_z_score(phrasing.attribute, numeric_value)

    if z_score is None:
        # No stats available, default to average
        return phrasing.labels.average

    return phrasing.labels.get_label(z_score)


def _format_boolean_value(value: Any, phrasing: BooleanPhrasing) -> str:
    """Format a boolean value."""
    if value is None:
        return ""

    # Handle various boolean representations
    if isinstance(value, bool):
        phrase = phrasing.true_phrase if value else phrasing.false_phrase
        return _sanitize_boolean_phrase(phrase, value)
    if isinstance(value, str):
        truthy = value.lower() in ("true", "yes", "1")
        phrase = phrasing.true_phrase if truthy else phrasing.false_phrase
        return _sanitize_boolean_phrase(phrase, truthy)
    if isinstance(value, (int, float)):
        truthy = bool(value)
        phrase = phrasing.true_phrase if truthy else phrasing.false_phrase
        return _sanitize_boolean_phrase(phrase, truthy)

    return _sanitize_boolean_phrase(phrasing.false_phrase, False)


def _sanitize_boolean_phrase(phrase: str, truthy: bool) -> str:
    """Deterministically sanitize boolean phrases for rendering."""
    clean = phrase.strip()
    if not clean:
        return "I do." if truthy else "I do not."

    raw_token_map = {
        "true": "I do.",
        "yes": "I do.",
        "false": "I do not.",
        "no": "I do not.",
    }
    token_key = clean.lower()
    if token_key in raw_token_map:
        return raw_token_map[token_key]

    lowered = clean.lower()
    if not lowered.startswith(("i ", "i'", "i’m", "i'm", "my ")):
        clean = f"I {clean[0].lower()}{clean[1:]}" if len(clean) > 1 else f"I {clean}"
    return clean


def _format_categorical_value(value: Any, phrasing: CategoricalPhrasing) -> str:
    """Format a categorical value."""
    if value is None:
        return phrasing.null_phrase or phrasing.fallback or ""

    str_value = str(value)
    normalized_value = _normalize_value(str_value)
    null_option_norms = {_normalize_value(o) for o in phrasing.null_options}

    selected_key = None
    selected_phrase = None
    if str_value in phrasing.phrases:
        selected_key = str_value
        selected_phrase = phrasing.phrases[str_value]
    else:
        for key, phrase in phrasing.phrases.items():
            if _normalize_value(key) == normalized_value:
                selected_key = key
                selected_phrase = phrase
                break

    if normalized_value in null_option_norms:
        if phrasing.null_phrase:
            return phrasing.null_phrase
        if (
            selected_key
            and selected_phrase
            and not _contains_raw_option_token(selected_phrase, selected_key)
        ):
            return selected_phrase
        if phrasing.fallback:
            return phrasing.fallback
        return ""

    if selected_phrase:
        return selected_phrase

    if phrasing.fallback:
        return phrasing.fallback

    # Last resort: just format the value nicely
    return str_value.replace("_", " ").title()


def render_attribute(attr_name: str, value: Any, config: PersonaConfig) -> str:
    """Render a single attribute value to first-person phrase."""
    phrasing = config.phrasings.get_phrasing(attr_name)

    if phrasing is None:
        # No phrasing defined, return empty
        return ""

    if isinstance(phrasing, BooleanPhrasing):
        return _format_boolean_value(value, phrasing)
    elif isinstance(phrasing, CategoricalPhrasing):
        return _format_categorical_value(value, phrasing)
    elif isinstance(phrasing, RelativePhrasing):
        relative = _format_relative_value(value, phrasing, config)
        concrete = next(
            (p for p in config.phrasings.concrete if p.attribute == attr_name),
            None,
        )
        if concrete and value is not None:
            concrete_text = _format_concrete_value(value, concrete)
            if concrete_text:
                return f"{relative} ({concrete_text})"
        return relative
    elif isinstance(phrasing, ConcretePhrasing):
        return _format_concrete_value(value, phrasing)

    return ""


def render_persona_section(
    group_name: str, agent: dict[str, Any], config: PersonaConfig
) -> str:
    """Render a single section of the persona.

    Args:
        group_name: Name of the attribute group
        agent: Agent attribute dictionary
        config: Persona configuration

    Returns:
        Rendered section as markdown string
    """
    group = config.get_group(group_name)
    if not group:
        return ""

    lines = [f"## {group.label}", ""]

    phrases = []
    for attr_name in group.attributes:
        value = agent.get(attr_name)
        phrase = render_attribute(attr_name, value, config)
        if phrase:
            phrases.append(phrase)

    if not phrases:
        return ""

    # Join phrases into paragraphs
    # Group related phrases together (simple heuristic: every 3-4 phrases)
    current_para = []
    paragraphs = []

    for phrase in phrases:
        current_para.append(phrase)
        # Start new paragraph after 3-4 phrases or if phrase ends with period
        if len(current_para) >= 3 or phrase.endswith("."):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []

    if current_para:
        paragraphs.append(" ".join(current_para))

    lines.extend(paragraphs)

    return "\n\n".join(lines)


def extract_intro_attributes(intro_template: str) -> set[str]:
    """Extract attribute names used in the intro template.

    Args:
        intro_template: Template string with {attribute} placeholders

    Returns:
        Set of attribute names found in template
    """
    # Match {attribute_name} patterns, excluding format specs like {value:.2f}
    pattern = r"\{(\w+)(?::[^}]*)?\}"
    matches = re.findall(pattern, intro_template)
    # Filter out 'value' which is a concrete phrasing placeholder
    return {m for m in matches if m != "value"}


def _ensure_period(phrase: str) -> str:
    """Ensure phrase ends with sentence-ending punctuation."""
    phrase = phrase.strip()
    if not phrase:
        return phrase
    if phrase[-1] not in ".!?":
        return phrase + "."
    return phrase


def render_intro(
    agent: dict[str, Any],
    config: PersonaConfig,
    display_format_map: dict[str, str] | None = None,
) -> str:
    """Render the narrative intro section.

    Args:
        agent: Agent attribute dictionary
        config: Persona configuration
        display_format_map: Optional mapping of attr_name -> display_format
            (from AttributeSpec.display_format, set by LLM during spec creation)
    """
    if display_format_map is None:
        display_format_map = {}

    try:
        # Format values for template
        formatted = {}
        for key, value in agent.items():
            if value is None:
                formatted[key] = "unknown"
            elif isinstance(value, bool):
                bp = config.phrasings.get_phrasing(key)
                if isinstance(bp, BooleanPhrasing):
                    formatted[key] = bp.true_phrase if value else bp.false_phrase
                else:
                    formatted[key] = "yes" if value else "no"
            elif isinstance(value, (int, float)):
                # Check LLM-classified display_format first
                display_fmt = display_format_map.get(key)
                if display_fmt == "time_12h" and 0 <= float(value) <= 24:
                    formatted[key] = _format_time(float(value), use_12hr=True)
                elif display_fmt == "time_24h" and 0 <= float(value) <= 24:
                    formatted[key] = _format_time(float(value), use_12hr=False)
                elif display_fmt == "currency":
                    if value >= 1000:
                        formatted[key] = f"${value:,.0f}"
                    else:
                        formatted[key] = f"${value:.2f}"
                elif display_fmt == "percentage":
                    if 0 <= value <= 1:
                        formatted[key] = f"{value * 100:.0f}%"
                    else:
                        formatted[key] = f"{value:.0f}%"
                elif isinstance(value, float) and value == int(value):
                    formatted[key] = (
                        f"{int(value):,}" if value >= 1000 else str(int(value))
                    )
                elif isinstance(value, float):
                    # Round to 1 decimal for most floats, 2 for small values
                    if value >= 100:
                        formatted[key] = f"{value:,.0f}"
                    elif value >= 1:
                        formatted[key] = f"{value:.1f}"
                    else:
                        formatted[key] = f"{value:.2f}"
                else:
                    formatted[key] = f"{value:,}" if value >= 1000 else str(value)
            elif isinstance(value, str):
                # Make categorical values readable
                formatted[key] = value.replace("_", " ")
            else:
                formatted[key] = str(value)

        intro = config.intro_template.format(**formatted)
        return f"## Who I Am\n\n{intro}"
    except (KeyError, ValueError) as e:
        return f"## Who I Am\n\n[Error rendering intro: {e}]"


# =============================================================================
# Household Section Rendering
# =============================================================================

# Templates for partner phrases, keyed by (has_kids, partner_gender)
_PARTNER_TEMPLATES = [
    "My {title} {name} is {age}.",
    "{name} and I have been together for a while now — {pronoun}'s {age}.",
    "I live with {name} ({age}), my {title}.",
]

# Templates for kids, keyed by count
_KIDS_TEMPLATES_SINGLE = [
    "Our {relationship} {name} is {age}{school_phrase}.",
    "We have a {relationship}, {name}, who's {age}{school_phrase}.",
]

_KIDS_TEMPLATES_MULTI = [
    "We have {count} kids: {kid_list}.",
    "Our children are {kid_list}.",
]

# Templates for elderly dependents
_ELDERLY_TEMPLATES = [
    "My {relationship} {name} ({age}) also lives with us.",
    "{name}, my {relationship}, lives with us at {age}.",
]

# Single parent templates
_SINGLE_PARENT_TEMPLATES = [
    "It's just me and {kid_summary}.",
    "I'm raising {kid_summary} on my own.",
]


def _format_age(age: int | str) -> str:
    """Format age, handling babies specially."""
    try:
        age_int = int(age)
        if age_int == 0:
            return "less than a year old"
        if age_int == 1:
            return "1 year old"
        return str(age_int)
    except (ValueError, TypeError):
        return str(age)


def _format_kid(dep: dict[str, Any]) -> str:
    """Format a single kid for listing."""
    name = dep.get("name", "")
    age = dep.get("age", "")
    school = dep.get("school_status")

    age_str = _format_age(age)

    if school and school not in ("adult", "working_adult", "home"):
        return f"{name} ({age_str}, {school.replace('_', ' ')})"
    return f"{name} ({age_str})"


def _get_school_phrase(dep: dict[str, Any]) -> str:
    """Get school status phrase for a dependent."""
    school = dep.get("school_status")
    if not school or school in ("adult", "working_adult", "home"):
        return ""
    return f", in {school.replace('_', ' ')}"


def render_household_section(agent: dict[str, Any], rng: Any = None) -> str:
    """Render the household section with partner and dependents.

    Args:
        agent: Agent dict with optional partner_npc and dependents
        rng: Optional random source for template variation (defaults to hash-based)

    Returns:
        Rendered household section, or empty string if no household context
    """
    partner = agent.get("partner_npc")
    dependents = agent.get("dependents", [])
    partner_id = agent.get("partner_id")  # If partner is also an agent

    # Skip if no household context
    if not partner and not dependents and not partner_id:
        return ""

    # Use agent ID for deterministic randomness
    if rng is None:
        import random

        seed = hash(agent.get("_id", "")) % (2**31)
        rng = random.Random(seed)

    phrases = []

    # Separate kids from elderly
    kids = [d for d in dependents if d.get("relationship") in ("son", "daughter")]
    elderly = [
        d
        for d in dependents
        if d.get("relationship") in ("mother", "father", "grandmother", "grandfather")
    ]

    # Partner phrase
    if partner:
        name = partner.get("first_name", "my partner")
        age = partner.get("age", "")
        gender = partner.get("gender", "")

        # Title based on gender
        title = (
            "husband"
            if gender == "male"
            else "wife"
            if gender == "female"
            else "partner"
        )
        pronoun = "he" if gender == "male" else "she" if gender == "female" else "they"

        template = rng.choice(_PARTNER_TEMPLATES)
        phrase = template.format(name=name, age=age, title=title, pronoun=pronoun)
        phrases.append(phrase)
    elif partner_id:
        # Partner is an agent, just note we have one
        phrases.append("I live with my partner.")

    # Kids phrase
    if kids:
        is_single_parent = not partner and not partner_id

        if len(kids) == 1:
            kid = kids[0]
            name = kid.get("name", "my child")
            age = _format_age(kid.get("age", ""))
            rel = kid.get("relationship", "child")
            school_phrase = _get_school_phrase(kid)

            if is_single_parent:
                template = rng.choice(_SINGLE_PARENT_TEMPLATES)
                phrase = template.format(kid_summary=f"my {rel} {name} ({age})")
            else:
                template = rng.choice(_KIDS_TEMPLATES_SINGLE)
                phrase = template.format(
                    name=name, age=age, relationship=rel, school_phrase=school_phrase
                )
            phrases.append(phrase)
        else:
            kid_list = ", ".join(_format_kid(k) for k in kids[:-1])
            kid_list += f" and {_format_kid(kids[-1])}"

            if is_single_parent:
                template = rng.choice(_SINGLE_PARENT_TEMPLATES)
                phrases.append(template.format(kid_summary=f"my {len(kids)} kids"))
                phrases.append(f"That's {kid_list}.")
            else:
                template = rng.choice(_KIDS_TEMPLATES_MULTI)
                phrases.append(template.format(count=len(kids), kid_list=kid_list))

    # Elderly phrase
    for dep in elderly:
        name = dep.get("name", "")
        age = dep.get("age", "")
        rel = dep.get("relationship", "parent")

        template = rng.choice(_ELDERLY_TEMPLATES)
        phrases.append(template.format(name=name, age=age, relationship=rel))

    if not phrases:
        return ""

    return "## My Household\n\n" + " ".join(phrases)


def render_persona(
    agent: dict[str, Any],
    config: PersonaConfig,
    decision_relevant_attributes: list[str] | None = None,
    display_format_map: dict[str, str] | None = None,
) -> str:
    """Render complete first-person persona for an agent.

    Args:
        agent: Agent attribute dictionary
        config: Persona configuration
        decision_relevant_attributes: Attributes most relevant to scenario outcome.
            If provided, these are pulled out and rendered first under a dedicated
            "Most Relevant to This Decision" section.
        display_format_map: Optional mapping of attr_name -> display_format
            (from AttributeSpec.display_format, set by LLM during spec creation)

    Returns:
        Complete persona as markdown string
    """
    sections = []

    # Render intro
    intro = render_intro(agent, config, display_format_map)
    if intro:
        sections.append(intro)

    # Render household section (partner, kids, elderly)
    household = render_household_section(agent)
    if household:
        sections.append(household)

    # Track attributes to exclude from group rendering
    # 1. Attributes already rendered in intro_template
    intro_attrs = extract_intro_attributes(config.intro_template)
    # 2. Decision-relevant attributes (rendered in separate section)
    decision_set = set(decision_relevant_attributes or [])
    # Combined exclusion set
    exclude_set = intro_attrs | decision_set

    # Render decision-relevant attributes first if specified
    if decision_set:
        decision_phrases = []
        for attr_name in decision_relevant_attributes:
            value = agent.get(attr_name)
            phrase = render_attribute(attr_name, value, config)
            if phrase:
                decision_phrases.append(_ensure_period(phrase))
        if decision_phrases:
            sections.append(
                "## Most Relevant to This Decision\n\n" + " ".join(decision_phrases)
            )

    # Render each group in order, excluding already-rendered attrs
    for group in config.groups:
        # Filter out attributes already shown in intro or decision section
        remaining_attrs = [a for a in group.attributes if a not in exclude_set]
        if not remaining_attrs:
            continue

        # Render manually with filtered attributes
        group_obj = config.get_group(group.name)
        if not group_obj:
            continue

        # render_intro() already emits "## Who I Am" — skip duplicate entirely
        if group_obj.label == "Who I Am":
            continue

        lines = [f"## {group_obj.label}", ""]
        phrases = []
        for attr_name in remaining_attrs:
            value = agent.get(attr_name)
            phrase = render_attribute(attr_name, value, config)
            if phrase:
                phrases.append(_ensure_period(phrase))

        if not phrases:
            continue

        # Group phrases into paragraphs (3 phrases per paragraph)
        current_para = []
        paragraphs = []
        for phrase in phrases:
            current_para.append(phrase)
            if len(current_para) >= 3:
                paragraphs.append(" ".join(current_para))
                current_para = []
        if current_para:
            paragraphs.append(" ".join(current_para))

        lines.extend(paragraphs)
        section = "\n\n".join(lines)
        if section:
            sections.append(section)

    return "\n\n".join(sections)


def preview_persona(
    agent: dict[str, Any], config: PersonaConfig, max_width: int = 80
) -> str:
    """Render persona with word wrapping for terminal preview.

    Args:
        agent: Agent attribute dictionary
        config: Persona configuration
        max_width: Maximum line width

    Returns:
        Word-wrapped persona string
    """
    import textwrap

    raw = render_persona(agent, config)

    # Wrap each paragraph
    lines = []
    for line in raw.split("\n"):
        if line.startswith("##"):
            lines.append("")
            lines.append(line)
            lines.append("")
        elif line.strip():
            wrapped = textwrap.fill(line, width=max_width)
            lines.append(wrapped)
        else:
            lines.append("")

    return "\n".join(lines)
