"""Step 2f: Name Config Hydration.

Research culturally-appropriate name frequency tables for the target population.
US populations skip this step (bundled SSA/Census CSVs are higher quality).
"""

import logging
import re

from ....core.llm import agentic_research, RetryCallback
from ....core.models.population import NameConfig, NameEntry
from ..schemas import build_name_config_schema

logger = logging.getLogger(__name__)

# Heuristic patterns for detecting US populations
_US_PATTERNS = re.compile(
    r"\b("
    r"united\s+states|(?<!\w)u\.?s\.?a?\.?(?!\w)|american|americans"
    r")\b",
    re.IGNORECASE,
)

_US_STATES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
}


def _is_us_population(population: str, geography: str | None) -> bool:
    """Heuristic: does this description clearly indicate a US population?"""
    text = f"{population} {geography or ''}".lower()
    if _US_PATTERNS.search(text):
        return True
    # Check for US state names
    for state in _US_STATES:
        if state in text:
            return True
    return False


def hydrate_name_config(
    population: str,
    geography: str | None = None,
    model: str | None = None,
    reasoning_effort: str = "low",
    on_retry: RetryCallback | None = None,
) -> tuple[NameConfig | None, list[str]]:
    """
    Research culturally-appropriate names for the target population (Step 2f).

    For US populations, returns (None, []) — bundled CSVs are higher quality.

    Args:
        population: Population description
        geography: Geographic scope
        model: Model to use
        reasoning_effort: "low", "medium", or "high"
        on_retry: Callback for retry notifications

    Returns:
        Tuple of (NameConfig or None, list of source URLs)
    """
    if _is_us_population(population, geography):
        logger.debug("US population detected — using bundled CSV names")
        return None, []

    geo_context = f" in {geography}" if geography else ""

    prompt = f"""Research culturally-appropriate personal names for {population}{geo_context}.

Provide name frequency tables that reflect the real naming patterns of this population:

1. **male_first_names**: 150+ male first names with frequency weights.
   - Span generations (older traditional names AND modern/younger names).
   - Include regional variation where relevant.
   - Weights should reflect actual popularity (higher = more common).

2. **female_first_names**: 150+ female first names with frequency weights.
   - Same guidelines as male names.

3. **last_names**: 150+ surnames/family names with frequency weights.
   - Reflect the actual surname distribution of this population.
   - Include common AND less-common names for long-tail diversity.

4. **sources**: List of URLs or citations used.

## Important Guidelines

- Do NOT just list the top-10 stereotypical names. Include diverse, realistic names.
- Weights should create a realistic long-tail distribution (a few common names, many less-common ones).
- Include generational variation: names popular with older people vs younger people.
- For populations with multiple ethnic/linguistic groups, include names from all major groups proportionally.
- The total number of names in each list should be at least 150 for realistic diversity.

Return a single JSON object with all fields."""

    try:
        data, sources = agentic_research(
            prompt=prompt,
            response_schema=build_name_config_schema(),
            schema_name="name_config",
            model=model,
            reasoning_effort=reasoning_effort,
            on_retry=on_retry,
        )
    except Exception:
        logger.warning(
            "Name config hydration failed, falling back to CSV names", exc_info=True
        )
        return None, []

    try:
        config = _parse_name_config(data)
        response_sources = data.get("sources", [])
        all_sources = list(set(sources + [s for s in response_sources if s]))
        return config, all_sources
    except Exception:
        logger.warning(
            "Failed to parse name config response, falling back to CSV names",
            exc_info=True,
        )
        return None, sources


def _parse_name_config(data: dict) -> NameConfig:
    """Parse LLM response into a NameConfig with diversity guardrails."""
    male = _parse_name_entries(data.get("male_first_names", []))
    female = _parse_name_entries(data.get("female_first_names", []))
    last = _parse_name_entries(data.get("last_names", []))

    # Diversity guardrail: warn if top-5 names exceed 40% of total weight
    for label, entries in [
        ("male_first_names", male),
        ("female_first_names", female),
        ("last_names", last),
    ]:
        if len(entries) >= 5:
            total_weight = sum(e.weight for e in entries)
            if total_weight > 0:
                top5_weight = sum(
                    e.weight
                    for e in sorted(entries, key=lambda x: x.weight, reverse=True)[:5]
                )
                if top5_weight / total_weight > 0.40:
                    logger.warning(
                        f"Name config: top-5 {label} account for "
                        f"{top5_weight / total_weight:.0%} of total weight "
                        f"(>40%% threshold)"
                    )

    return NameConfig(
        male_first_names=male,
        female_first_names=female,
        last_names=last,
    )


def _parse_name_entries(raw: list) -> list[NameEntry]:
    """Parse a list of {name, weight} dicts into NameEntry objects."""
    entries = []
    for item in raw:
        if isinstance(item, dict) and "name" in item:
            weight = float(item.get("weight", 1.0))
            if weight > 0:
                entries.append(NameEntry(name=str(item["name"]), weight=weight))
    return entries
