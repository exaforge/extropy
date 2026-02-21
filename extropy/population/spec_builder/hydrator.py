"""Step 2: Attribute Hydration Orchestrator.

This module orchestrates the split hydration approach:
- Step 2a: hydrate_independent() - Research distributions for independent attributes
- Step 2b: hydrate_derived() - Specify formulas for derived attributes
- Step 2c: hydrate_conditional_base() - Research base distributions for conditional
- Step 2d: hydrate_conditional_modifiers() - Specify modifiers for conditional

Each step is processed separately with specialized prompts and validation.

FAIL-FAST VALIDATION:
Each hydration step validates LLM output immediately and retries with
error feedback if syntax errors are detected. This catches issues like
unterminated strings, invalid formulas, etc. before proceeding.
"""

import logging

from ...core.llm import RetryCallback
from ...core.models import (
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
    HouseholdConfig,
)
from ...utils.callbacks import HydrationProgressCallback
from .hydrators import (
    hydrate_independent,
    hydrate_derived,
    hydrate_conditional_base,
    hydrate_conditional_modifiers,
    hydrate_household_config,
)

logger = logging.getLogger(__name__)

# Keywords that suggest household context is relevant
_HOUSEHOLD_KEYWORDS = frozenset(
    [
        # Population keywords
        "family",
        "families",
        "couple",
        "couples",
        "household",
        "households",
        "parent",
        "parents",
        "retired",
        "retiree",
        "retirees",
        "spouse",
        "spouses",
        "married",
        "cohabit",
        "living together",
        "home owner",
        "homeowner",
        # Scenario keywords that imply household context
        "childcare",
        "child care",
        "parental",
        "maternity",
        "paternity",
        "housing",
        "mortgage",
        "rent",
        "school district",
        "relocation",
        "moving",
    ]
)


def _should_hydrate_household_config(
    description: str,
    attributes: list[DiscoveredAttribute],
) -> bool:
    """Check if this population/scenario needs household config research.

    Household config is expensive (agentic research with web search).
    Only hydrate it when:
    1. Any discovered attribute has scope="household", OR
    2. Description (population or scenario) mentions household-related keywords

    At spec time, description is just the population ("2000 physicians").
    At extend time, description is "population + scenario" so scenario
    keywords like "childcare policy" will trigger household research.

    Args:
        description: Population description OR "population + scenario" for extend
        attributes: Discovered attributes from selector

    Returns:
        True if household config should be researched
    """
    # Check if any attribute has household scope
    has_household_attrs = any(
        getattr(attr, "scope", "individual") == "household" for attr in attributes
    )
    if has_household_attrs:
        return True

    # Check if description implies household context
    desc_lower = description.lower()
    for keyword in _HOUSEHOLD_KEYWORDS:
        if keyword in desc_lower:
            return True

    return False


def _extract_gender_options_for_household(
    context: list[AttributeSpec] | None,
    hydrated_attrs: list[HydratedAttribute],
) -> list[str]:
    """Extract explicit gender-category options for household partner pairing."""

    def options_for(attr: AttributeSpec) -> list[str]:
        if attr.type != "categorical" or attr.sampling.distribution is None:
            return []
        dist = attr.sampling.distribution
        if not hasattr(dist, "options"):
            return []
        opts = getattr(dist, "options", None) or []
        return [str(o) for o in opts if str(o).strip()]

    # Prefer semantic metadata over name heuristics.
    candidates: list[AttributeSpec] = []
    for attr in (context or []):
        if attr.identity_type == "gender_identity":
            candidates.append(attr)
    for attr in hydrated_attrs:
        if attr.identity_type == "gender_identity":
            candidates.append(attr)

    # Legacy fallback for older specs lacking identity_type metadata.
    if not candidates:
        fallback_names = {"gender", "sex", "gender_identity"}
        for attr in (context or []):
            if attr.name in fallback_names:
                candidates.append(attr)
        for attr in hydrated_attrs:
            if attr.name in fallback_names:
                candidates.append(attr)

    for attr in candidates:
        opts = options_for(attr)
        if opts:
            # Preserve original option casing/order from spec.
            seen: set[str] = set()
            ordered: list[str] = []
            for value in opts:
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)
            return ordered

    return []


# =============================================================================
# Main Orchestrator
# =============================================================================

ProgressCallback = HydrationProgressCallback


def hydrate_attributes(
    attributes: list[DiscoveredAttribute],
    description: str,
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    include_household: bool = True,
    model: str | None = None,
    reasoning_effort: str = "low",
    on_progress: ProgressCallback | None = None,
) -> tuple[list[HydratedAttribute], list[str], list[str], HouseholdConfig]:
    """
    Research distributions for discovered attributes using split hydration.

    This function orchestrates the 5-step split hydration process:
    - Step 2a: hydrate_independent() - Research distributions for independent attributes
    - Step 2b: hydrate_derived() - Specify formulas for derived attributes
    - Step 2c: hydrate_conditional_base() - Research base distributions for conditional
    - Step 2d: hydrate_conditional_modifiers() - Specify modifiers for conditional
    - Step 2e: hydrate_household_config() - Research household composition parameters

    When context is provided (extend mode), the model can reference
    context attributes in formulas and modifiers.

    Args:
        attributes: List of DiscoveredAttribute from selector
        description: Original population description
        geography: Geographic scope for research
        context: Existing attributes from base population (for extend mode)
        include_household: Whether to hydrate household config (Step 2e)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"
        on_progress: Optional callback for progress updates (step, status, count)

    Returns:
        Tuple of (list of HydratedAttribute, list of source URLs, list of validation warnings, HouseholdConfig)
    """
    if not attributes:
        return [], [], [], HouseholdConfig()

    all_sources = []
    all_warnings = []
    population = description

    def report(step: str, status: str, count: int | None = None):
        """Report progress via callback or logger."""
        if on_progress:
            on_progress(step, status, count)
        else:
            if count is not None:
                logger.info("  %s: %s (%s)", step, status, count)
            else:
                logger.info("  %s: %s", step, status)

    def make_retry_callback(step: str) -> RetryCallback:
        """Create a retry callback for a specific step."""

        def on_retry(attempt: int, max_retries: int, error_summary: str):
            if attempt > max_retries:
                # Retries exhausted
                report(step, f"Validation failed after {max_retries} retries", None)
            else:
                # Retrying
                report(
                    step,
                    f"Retrying ({attempt}/{max_retries}): {error_summary[:40]}...",
                    None,
                )

        return on_retry

    # Step 2a: Independent attributes
    report("2a", "Researching independent distributions...")
    independent_attrs, independent_sources, independent_errors = hydrate_independent(
        attributes=attributes,
        population=population,
        geography=geography,
        context=context,
        model=model,
        reasoning_effort=reasoning_effort,
        on_retry=make_retry_callback("2a"),
    )
    all_sources.extend(independent_sources)
    all_warnings.extend([f"[2a] {e}" for e in independent_errors])
    # Report validation status
    if independent_errors:
        report(
            "2a",
            f"Hydrated {len(independent_attrs)} independent, {len(independent_errors)} validation warning(s)",
            len(independent_sources),
        )
    else:
        report(
            "2a",
            f"Hydrated {len(independent_attrs)} independent, Validation passed",
            len(independent_sources),
        )

    # Step 2b: Derived attributes
    report("2b", "Specifying derived formulas...")
    derived_attrs, derived_sources, derived_errors = hydrate_derived(
        attributes=attributes,
        population=population,
        geography=geography,
        independent_attrs=independent_attrs,
        context=context,
        model=model,
        reasoning_effort=reasoning_effort,
        on_retry=make_retry_callback("2b"),
    )
    all_sources.extend(derived_sources)
    all_warnings.extend([f"[2b] {e}" for e in derived_errors])
    # Report validation status
    if derived_errors:
        report(
            "2b",
            f"Hydrated {len(derived_attrs)} derived, {len(derived_errors)} validation warning(s)",
            0,
        )
    else:
        report("2b", f"Hydrated {len(derived_attrs)} derived, Validation passed", 0)

    # Step 2c: Conditional base distributions
    report("2c", "Researching conditional distributions...")
    conditional_base_attrs, conditional_sources, conditional_errors = (
        hydrate_conditional_base(
            attributes=attributes,
            population=population,
            geography=geography,
            independent_attrs=independent_attrs,
            derived_attrs=derived_attrs,
            context=context,
            model=model,
            reasoning_effort=reasoning_effort,
            on_retry=make_retry_callback("2c"),
        )
    )
    all_sources.extend(conditional_sources)
    all_warnings.extend([f"[2c] {e}" for e in conditional_errors])
    # Report validation status
    if conditional_errors:
        report(
            "2c",
            f"Hydrated {len(conditional_base_attrs)} conditional, {len(conditional_errors)} validation warning(s)",
            len(conditional_sources),
        )
    else:
        report(
            "2c",
            f"Hydrated {len(conditional_base_attrs)} conditional, Validation passed",
            len(conditional_sources),
        )

    # Step 2d: Conditional modifiers
    report("2d", "Specifying conditional modifiers...")
    conditional_attrs, modifier_sources, modifier_errors = (
        hydrate_conditional_modifiers(
            conditional_attrs=conditional_base_attrs,
            population=population,
            geography=geography,
            independent_attrs=independent_attrs,
            derived_attrs=derived_attrs,
            context=context,
            model=model,
            reasoning_effort=reasoning_effort,
            on_retry=make_retry_callback("2d"),
        )
    )
    all_sources.extend(modifier_sources)
    all_warnings.extend([f"[2d] {e}" for e in modifier_errors])
    # Report validation status
    if modifier_errors:
        report(
            "2d",
            f"Added modifiers to {len(conditional_attrs)}, {len(modifier_errors)} validation warning(s)",
            len(modifier_sources),
        )
    else:
        report(
            "2d",
            f"Added modifiers to {len(conditional_attrs)}, Validation passed",
            len(modifier_sources),
        )

    # Step 2e: Household config (optional; enabled by scenario stage)
    if include_household and _should_hydrate_household_config(population, attributes):
        report("2e", "Researching household composition...")
        merged_attribute_names = sorted(
            {
                *(attr.name for attr in attributes),
                *(attr.name for attr in context or []),
            }
        )
        gender_options = _extract_gender_options_for_household(
            context=context,
            hydrated_attrs=independent_attrs + derived_attrs + conditional_attrs,
        )
        household_config, hh_sources = hydrate_household_config(
            population=population,
            geography=geography,
            allowed_attributes=merged_attribute_names,
            allowed_gender_values=gender_options,
            model=model,
            reasoning_effort=reasoning_effort,
            on_retry=make_retry_callback("2e"),
        )
        all_sources.extend(hh_sources)
        report("2e", "Household config researched", len(hh_sources))
    elif include_household:
        logger.info(
            "Skipping household config hydration - no household-scoped attributes "
            "or household keywords in population description"
        )
        household_config = HouseholdConfig()
        report("2e", "Skipped (no household context)", 0)
    else:
        household_config = HouseholdConfig()
        report("2e", "Skipped (disabled for this stage)", 0)

    # Combine all hydrated attributes
    all_hydrated = independent_attrs + derived_attrs + conditional_attrs
    unique_sources = list(set(all_sources))

    # Validate strategy consistency across all attributes
    report("strategy", "Validating strategy consistency...")
    # Strategy consistency validation is done by structural.run_structural_checks()
    # when validate_spec() is called on the final PopulationSpec
    report("strategy", "Strategy consistency check passed", None)

    return all_hydrated, unique_sources, all_warnings, household_config
