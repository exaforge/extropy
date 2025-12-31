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

from typing import Callable

from ...core.llm import RetryCallback
from ...core.models import (
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
)
from .hydrators import (
    hydrate_independent,
    hydrate_derived,
    hydrate_conditional_base,
    hydrate_conditional_modifiers,
)


# =============================================================================
# Main Orchestrator
# =============================================================================

# Type alias for progress callback: (step: str, status: str, count: int | None) -> None
ProgressCallback = Callable[[str, str, int | None], None]


def hydrate_attributes(
    attributes: list[DiscoveredAttribute],
    description: str,
    geography: str | None = None,
    context: list[AttributeSpec] | None = None,
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    on_progress: ProgressCallback | None = None,
) -> tuple[list[HydratedAttribute], list[str], list[str]]:
    """
    Research distributions for discovered attributes using split hydration.

    This function orchestrates the 4-step split hydration process:
    - Step 2a: hydrate_independent() - Research distributions for independent attributes
    - Step 2b: hydrate_derived() - Specify formulas for derived attributes
    - Step 2c: hydrate_conditional_base() - Research base distributions for conditional
    - Step 2d: hydrate_conditional_modifiers() - Specify modifiers for conditional

    When context is provided (overlay mode), the model can reference
    context attributes in formulas and modifiers.

    Args:
        attributes: List of DiscoveredAttribute from selector
        description: Original population description
        geography: Geographic scope for research
        context: Existing attributes from base population (for overlay mode)
        model: Model to use
        reasoning_effort: "low", "medium", or "high"
        on_progress: Optional callback for progress updates (step, status, count)

    Returns:
        Tuple of (list of HydratedAttribute, list of source URLs, list of validation warnings)
    """
    if not attributes:
        return [], [], []

    all_sources = []
    all_warnings = []
    population = description

    def report(step: str, status: str, count: int | None = None):
        """Report progress via callback or print."""
        if on_progress:
            on_progress(step, status, count)
        else:
            if count is not None:
                print(f"  {step}: {status} ({count})")
            else:
                print(f"  {step}: {status}")

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

    # Combine all hydrated attributes
    all_hydrated = independent_attrs + derived_attrs + conditional_attrs
    unique_sources = list(set(all_sources))

    # Validate strategy consistency across all attributes
    report("strategy", "Validating strategy consistency...")
    # Strategy consistency validation is done by structural.run_structural_checks()
    # when validate_spec() is called on the final PopulationSpec
    report("strategy", "Strategy consistency check passed", None)

    return all_hydrated, unique_sources, all_warnings
