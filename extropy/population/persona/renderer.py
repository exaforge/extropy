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
        return phrasing.true_phrase if value else phrasing.false_phrase
    if isinstance(value, str):
        return (
            phrasing.true_phrase
            if value.lower() in ("true", "yes", "1")
            else phrasing.false_phrase
        )
    if isinstance(value, (int, float)):
        return phrasing.true_phrase if value else phrasing.false_phrase

    return phrasing.false_phrase


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
        return _format_relative_value(value, phrasing, config)
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


def render_intro(agent: dict[str, Any], config: PersonaConfig) -> str:
    """Render the narrative intro section."""
    # Time-related attribute name patterns
    time_attrs = {"start_time", "departure_time", "arrival_time", "end_time"}

    try:
        # Format values for template
        formatted = {}
        for key, value in agent.items():
            if value is None:
                formatted[key] = "unknown"
            elif isinstance(value, bool):
                formatted[key] = "yes" if value else "no"
            elif isinstance(value, (int, float)):
                # Check if this looks like a time attribute
                is_time = any(t in key.lower() for t in time_attrs)
                if is_time and 0 <= float(value) <= 24:
                    formatted[key] = _format_time(float(value), use_12hr=True)
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


def render_persona(
    agent: dict[str, Any],
    config: PersonaConfig,
    decision_relevant_attributes: list[str] | None = None,
) -> str:
    """Render complete first-person persona for an agent.

    Args:
        agent: Agent attribute dictionary
        config: Persona configuration
        decision_relevant_attributes: Attributes most relevant to scenario outcome.
            If provided, these are pulled out and rendered first under a dedicated
            "Most Relevant to This Decision" section.

    Returns:
        Complete persona as markdown string
    """
    sections = []

    # Render intro
    intro = render_intro(agent, config)
    if intro:
        sections.append(intro)

    decision_set = set(decision_relevant_attributes or [])

    # Render decision-relevant attributes first if specified
    if decision_set:
        decision_phrases = []
        for attr_name in decision_relevant_attributes:
            value = agent.get(attr_name)
            phrase = render_attribute(attr_name, value, config)
            if phrase:
                decision_phrases.append(phrase)
        if decision_phrases:
            sections.append(
                "## Most Relevant to This Decision\n\n" + " ".join(decision_phrases)
            )

    # Render each group in order, excluding decision-relevant attrs already shown
    for group in config.groups:
        if not decision_set:
            section = render_persona_section(group.name, agent, config)
        else:
            # Filter out decision-relevant attributes from regular groups
            remaining_attrs = [a for a in group.attributes if a not in decision_set]
            if not remaining_attrs:
                continue
            # Render manually with filtered attributes
            group_obj = config.get_group(group.name)
            if not group_obj:
                continue
            lines = [f"## {group_obj.label}", ""]
            phrases = []
            for attr_name in remaining_attrs:
                value = agent.get(attr_name)
                phrase = render_attribute(attr_name, value, config)
                if phrase:
                    phrases.append(phrase)
            if not phrases:
                continue
            current_para = []
            paragraphs = []
            for phrase in phrases:
                current_para.append(phrase)
                if len(current_para) >= 3 or phrase.endswith("."):
                    if current_para:
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
