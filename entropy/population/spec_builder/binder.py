"""Step 3: Constraint Binding and Spec Assembly.

Builds the dependency graph, determines sampling order via topological sort,
and assembles the final PopulationSpec.
"""

from datetime import datetime

from ...core.models import (
    HydratedAttribute,
    AttributeSpec,
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    SamplingConfig,
)
from ...utils import topological_sort


def bind_constraints(
    attributes: list[HydratedAttribute],
    context: list[AttributeSpec] | None = None,
) -> tuple[list[AttributeSpec], list[str], list[str]]:
    """
    Build dependency graph and determine sampling order.

    This step:
    1. Validates all dependencies reference existing or context attributes
    2. Checks for circular dependencies
    3. Computes topological sort for sampling order
    4. Converts HydratedAttribute to final AttributeSpec

    When context is provided (extend mode), dependencies on context
    attributes are valid - they're treated as already-sampled.

    Args:
        attributes: List of HydratedAttribute from hydrator
        context: Existing attributes from base population (for extend mode)

    Returns:
        Tuple of (list of AttributeSpec, sampling_order, warnings)

    Raises:
        CircularDependencyError: If circular dependencies exist
        ValueError: If dependencies reference unknown attributes

    Example:
        # Base mode
        >>> specs, order, warnings = bind_constraints(hydrated_attrs)

        # Overlay mode - allows dependencies on context attrs
        >>> specs, order, warnings = bind_constraints(extend_attrs, context=base_spec.attributes)
    """
    attr_names = {a.name for a in attributes}
    context_names = {a.name for a in context} if context else set()
    known_names = attr_names | context_names
    warnings = []

    # Filter unknown dependencies and create specs
    # Note: we don't mutate input objects - we filter during spec creation
    specs = []
    for attr in attributes:
        # Collect unknown dependencies before filtering
        unknown_deps = [d for d in attr.depends_on if d not in known_names]
        if unknown_deps:
            for dep in unknown_deps:
                warnings.append(f"{attr.name}: removed unknown dependency '{dep}'")

        # Filter depends_on to only known attributes
        filtered_depends_on = [d for d in attr.depends_on if d in known_names]

        # Create new SamplingConfig with filtered depends_on
        filtered_sampling = SamplingConfig(
            strategy=attr.sampling.strategy,
            distribution=attr.sampling.distribution,
            formula=attr.sampling.formula,
            depends_on=filtered_depends_on,
            modifiers=attr.sampling.modifiers,
        )

        spec = AttributeSpec(
            name=attr.name,
            type=attr.type,
            category=attr.category,
            description=attr.description,
            sampling=filtered_sampling,
            grounding=attr.grounding,
            constraints=attr.constraints,
        )
        specs.append(spec)

    # Compute sampling order using specs (which have filtered depends_on)
    # Context attributes are already sampled, so they don't need ordering
    deps = {s.name: s.sampling.depends_on for s in specs}
    sampling_order = topological_sort(deps)

    return specs, sampling_order, warnings


def _compute_grounding_summary(
    attributes: list[AttributeSpec],
    sources: list[str],
) -> GroundingSummary:
    """Compute overall grounding summary from individual attribute grounding."""
    strong_count = sum(1 for a in attributes if a.grounding.level == "strong")
    medium_count = sum(1 for a in attributes if a.grounding.level == "medium")
    low_count = sum(1 for a in attributes if a.grounding.level == "low")

    total = len(attributes)

    # Determine overall level
    if total == 0:
        overall = "low"
    elif strong_count / total >= 0.6:
        overall = "strong"
    elif (strong_count + medium_count) / total >= 0.5:
        overall = "medium"
    else:
        overall = "low"

    return GroundingSummary(
        overall=overall,
        sources_count=len(sources),
        strong_count=strong_count,
        medium_count=medium_count,
        low_count=low_count,
        sources=sources,
    )


def build_spec(
    description: str,
    size: int,
    geography: str | None,
    attributes: list[AttributeSpec],
    sampling_order: list[str],
    sources: list[str],
) -> PopulationSpec:
    """
    Assemble the final PopulationSpec from all components.

    Args:
        description: Original population description
        size: Number of agents
        geography: Geographic scope
        attributes: List of AttributeSpec
        sampling_order: Order for sampling
        sources: List of source URLs from research

    Returns:
        Complete PopulationSpec ready for YAML export
    """
    meta = SpecMeta(
        description=description,
        size=size,
        geography=geography,
        created_at=datetime.now(),
    )

    grounding = _compute_grounding_summary(attributes, sources)

    return PopulationSpec(
        meta=meta,
        grounding=grounding,
        attributes=attributes,
        sampling_order=sampling_order,
    )
