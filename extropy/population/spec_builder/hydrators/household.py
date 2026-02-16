"""Step 2e: Household Config Hydration.

Research household composition parameters for the target population using web search.
"""

import logging

from ....core.llm import agentic_research, RetryCallback
from ....core.models.population import HouseholdConfig, LifeStageThreshold
from ..schemas import build_household_config_schema

logger = logging.getLogger(__name__)


def hydrate_household_config(
    population: str,
    geography: str | None = None,
    model: str | None = None,
    reasoning_effort: str = "low",
    on_retry: RetryCallback | None = None,
) -> tuple[HouseholdConfig, list[str]]:
    """
    Research household composition for the target population (Step 2e).

    Uses web search to find population-appropriate household composition data
    including household type distributions, partner correlation rates,
    and dependent generation parameters.

    Args:
        population: Population description (e.g., "Japanese retirees in Osaka")
        geography: Geographic scope (e.g., "Japan")
        model: Model to use
        reasoning_effort: "low", "medium", or "high"
        on_retry: Callback for retry notifications

    Returns:
        Tuple of (HouseholdConfig, list of source URLs)
    """
    geo_context = f" in {geography}" if geography else ""

    prompt = f"""Research household composition data for {population}{geo_context}.

For this population, provide statistically grounded values for:

1. **Age brackets**: Array of objects with `upper_bound` (exclusive integer) and `label` (string).
   Example: [{{"upper_bound": 30, "label": "18-29"}}, {{"upper_bound": 45, "label": "30-44"}}, {{"upper_bound": 65, "label": "45-64"}}, {{"upper_bound": 999, "label": "65+"}}]

2. **Household type weights**: Array of objects, each with `bracket` (label from age_brackets) and `types` (array of {{"type": string, "weight": number}}).
   Valid types: "single", "couple", "single_parent", "couple_with_kids", "multi_generational". Weights must sum to ~1.0 per bracket.
   Example: [{{"bracket": "18-29", "types": [{{"type": "single", "weight": 0.4}}, {{"type": "couple", "weight": 0.3}}, ...]}}]

3. **Same-group partner rates**: Array of objects with `group` (ethnicity name) and `rate` (probability 0-1).
   Example: [{{"group": "white", "rate": 0.90}}, {{"group": "black", "rate": 0.82}}]

4. **Default same-group rate**: Fallback rate for groups not explicitly listed.

5. **Assortative mating**: Array of objects with `attribute` (attribute name) and `correlation` (probability 0-1).
   Example: [{{"attribute": "education_level", "correlation": 0.65}}, {{"attribute": "religious_affiliation", "correlation": 0.70}}]

6. **Partner age gap**: Mean offset (partner_age - primary_age, negative means younger) and standard deviation.

7. **Minimum adult age**: Legal/cultural age of adulthood for this population.

8. **Child generation parameters**:
   - child_min_parent_offset: Minimum typical age gap between parent and child (e.g. 20)
   - child_max_parent_offset: Maximum typical age gap between parent and child (e.g. 40)
   - max_dependent_child_age: Maximum age for a child to be classified as dependent (e.g. 17)

9. **Elderly dependent offsets**:
   - elderly_min_offset: Minimum age gap between primary adult and elderly dependent
   - elderly_max_offset: Maximum age gap between primary adult and elderly dependent

10. **Life stages**: Age thresholds for dependent life stages based on the education system. Each entry has max_age (exclusive) and label.
    Example: [{{"max_age": 6, "label": "preschool"}}, {{"max_age": 12, "label": "primary"}}]

11. **Adult stage label**: Label for post-school adults (e.g. "adult").

12. **Average household size**: Mean number of persons per household.

13. **Sources**: List of URLs or citations used.

## Research Guidelines

- Use government census data, demographic surveys, or academic studies specific to {geography or "this population"}.
- Adapt household types and education stages to the local context (e.g., Japanese education system stages differ from US).
- Be specific to the population described, not generic global averages.
- Ensure household_type_weights sum to ~1.0 for each age bracket.
- If data is unavailable for a specific field, use the best available regional or national data.

Return a single JSON object with all fields."""

    try:
        data, sources = agentic_research(
            prompt=prompt,
            response_schema=build_household_config_schema(),
            schema_name="household_config",
            model=model,
            reasoning_effort=reasoning_effort,
            on_retry=on_retry,
        )
    except Exception:
        logger.warning(
            "Household config hydration failed, using US defaults", exc_info=True
        )
        return HouseholdConfig(), []

    try:
        config = _parse_household_config(data)
        # Merge any sources from the response into the source list
        response_sources = data.get("sources", [])
        all_sources = list(set(sources + [s for s in response_sources if s]))
        return config, all_sources
    except Exception:
        logger.warning(
            "Failed to parse household config response, using US defaults",
            exc_info=True,
        )
        return HouseholdConfig(), sources


def _parse_household_config(data: dict) -> HouseholdConfig:
    """Parse LLM response into a HouseholdConfig, falling back to defaults for bad fields.

    Converts array-of-objects LLM output back to the dict/tuple structures that
    HouseholdConfig expects:
    - age_brackets: [{upper_bound, label}] -> [(int, str)]
    - household_type_weights: [{bracket, types: [{type, weight}]}] -> nested dict
    - same_group_rates: [{group, rate}] -> {str: float}
    - assortative_mating: [{attribute, correlation}] -> {str: float}
    """
    kwargs: dict = {}

    # Age brackets: [{upper_bound, label}] -> [(int, str)]
    if "age_brackets" in data and isinstance(data["age_brackets"], list):
        brackets = []
        for item in data["age_brackets"]:
            if isinstance(item, dict) and "upper_bound" in item and "label" in item:
                brackets.append((int(item["upper_bound"]), str(item["label"])))
        if brackets:
            kwargs["age_brackets"] = brackets

    # Household type weights: [{bracket, types: [{type, weight}]}] -> nested dict
    if "household_type_weights" in data and isinstance(
        data["household_type_weights"], list
    ):
        weights = {}
        for entry in data["household_type_weights"]:
            if isinstance(entry, dict) and "bracket" in entry and "types" in entry:
                bracket_label = str(entry["bracket"])
                type_weights = {}
                for tw in entry.get("types", []):
                    if isinstance(tw, dict) and "type" in tw and "weight" in tw:
                        type_weights[str(tw["type"])] = float(tw["weight"])
                if type_weights:
                    weights[bracket_label] = type_weights
        if weights:
            kwargs["household_type_weights"] = weights

    # Same-group rates: [{group, rate}] -> {str: float}
    if "same_group_rates" in data and isinstance(data["same_group_rates"], list):
        rates = {}
        for item in data["same_group_rates"]:
            if isinstance(item, dict) and "group" in item and "rate" in item:
                rates[str(item["group"])] = float(item["rate"])
        if rates:
            kwargs["same_group_rates"] = rates

    # Scalar fields
    for field in (
        "default_same_group_rate",
        "partner_age_gap_mean",
        "partner_age_gap_std",
        "avg_household_size",
    ):
        if field in data and data[field] is not None:
            kwargs[field] = float(data[field])

    for field in (
        "min_adult_age",
        "child_min_parent_offset",
        "child_max_parent_offset",
        "max_dependent_child_age",
        "elderly_min_offset",
        "elderly_max_offset",
    ):
        if field in data and data[field] is not None:
            kwargs[field] = int(data[field])

    # Assortative mating: [{attribute, correlation}] -> {str: float}
    if "assortative_mating" in data and isinstance(data["assortative_mating"], list):
        mating = {}
        for item in data["assortative_mating"]:
            if isinstance(item, dict) and "attribute" in item and "correlation" in item:
                mating[str(item["attribute"])] = float(item["correlation"])
        if mating:
            kwargs["assortative_mating"] = mating

    # Life stages
    if "life_stages" in data and isinstance(data["life_stages"], list):
        stages = []
        for item in data["life_stages"]:
            if isinstance(item, dict) and "max_age" in item and "label" in item:
                stages.append(
                    LifeStageThreshold(
                        max_age=int(item["max_age"]), label=str(item["label"])
                    )
                )
        if stages:
            kwargs["life_stages"] = stages

    # Adult stage label
    if "adult_stage_label" in data and data["adult_stage_label"]:
        kwargs["adult_stage_label"] = str(data["adult_stage_label"])

    return HouseholdConfig(**kwargs)
