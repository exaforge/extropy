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

1. **Age brackets**: List of [upper_bound_exclusive, bracket_label] pairs that partition the adult age range. E.g. [[30, "18-29"], [45, "30-44"], [65, "45-64"], [999, "65+"]]

2. **Household type weights**: For each age bracket, the probability distribution over household types: "single", "couple", "single_parent", "couple_with_kids", "multi_generational". Weights must sum to ~1.0 per bracket.

3. **Same-group partner rates**: Probability that a partner shares the same race/ethnicity group. Provide rates by group name (e.g. {{"white": 0.90, "black": 0.82}}).

4. **Default same-group rate**: Fallback rate for groups not explicitly listed.

5. **Assortative mating coefficients**: Probability that partners share the same value for correlated attributes. Keys are attribute names like "education_level", "religious_affiliation", "political_orientation".

6. **Partner age gap**: Mean offset (partner_age - primary_age, negative means younger) and standard deviation.

7. **Minimum adult age**: Legal/cultural age of adulthood for this population.

8. **Child generation parameters**:
   - child_min_parent_offset: Minimum typical age gap between parent and child (e.g. 20)
   - child_max_parent_offset: Maximum typical age gap between parent and child (e.g. 40)
   - max_dependent_child_age: Maximum age for a child to be classified as dependent (e.g. 17)

9. **Elderly dependent offsets**:
   - elderly_min_offset: Minimum age gap between primary adult and elderly dependent
   - elderly_max_offset: Maximum age gap between primary adult and elderly dependent

10. **Life stages**: Age thresholds for dependent life stages based on the education system. Each entry has max_age (exclusive) and label. E.g. [{{"max_age": 6, "label": "preschool"}}, {{"max_age": 12, "label": "primary"}}]

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
    """Parse LLM response into a HouseholdConfig, falling back to defaults for bad fields."""
    kwargs: dict = {}

    # Age brackets: list of [int, str] tuples
    if "age_brackets" in data and isinstance(data["age_brackets"], list):
        brackets = []
        for item in data["age_brackets"]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                brackets.append((int(item[0]), str(item[1])))
        if brackets:
            kwargs["age_brackets"] = brackets

    # Household type weights
    if "household_type_weights" in data and isinstance(
        data["household_type_weights"], dict
    ):
        weights = {}
        for bracket_label, type_weights in data["household_type_weights"].items():
            if isinstance(type_weights, dict):
                weights[str(bracket_label)] = {
                    str(k): float(v) for k, v in type_weights.items()
                }
        if weights:
            kwargs["household_type_weights"] = weights

    # Same-group rates
    if "same_group_rates" in data and isinstance(data["same_group_rates"], dict):
        kwargs["same_group_rates"] = {
            str(k): float(v) for k, v in data["same_group_rates"].items()
        }

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

    # Assortative mating
    if "assortative_mating" in data and isinstance(data["assortative_mating"], dict):
        kwargs["assortative_mating"] = {
            str(k): float(v) for k, v in data["assortative_mating"].items()
        }

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
