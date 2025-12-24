"""Step 3: Constraint Binding and Spec Assembly.

Builds the dependency graph, determines sampling order via topological sort,
and assembles the final PopulationSpec.
"""

from collections import defaultdict
from datetime import datetime

from ...core.models import (
    HydratedAttribute,
    AttributeSpec,
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    SamplingConfig,
)


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected in attributes."""

    pass


def _topological_sort_specs(attributes: list[AttributeSpec]) -> list[str]:
    """
    Topological sort of AttributeSpec based on dependencies.

    Uses Kahn's algorithm to determine a valid sampling order
    where all dependencies are sampled before dependents.

    Args:
        attributes: List of AttributeSpec with sampling.depends_on fields

    Returns:
        List of attribute names in sampling order

    Raises:
        CircularDependencyError: If circular dependencies exist
    """
    # Build adjacency list and in-degree count
    graph = defaultdict(list)  # attr -> list of attrs that depend on it
    in_degree = {a.name: 0 for a in attributes}
    attr_names = {a.name for a in attributes}

    for attr in attributes:
        for dep in attr.sampling.depends_on:
            # Only count dependencies on attributes we're tracking
            if dep in attr_names:
                graph[dep].append(attr.name)
                in_degree[attr.name] += 1

    # Start with nodes that have no dependencies
    queue = [name for name, degree in in_degree.items() if degree == 0]
    order = []

    while queue:
        # Sort for deterministic ordering
        queue.sort()
        node = queue.pop(0)
        order.append(node)

        # Reduce in-degree for dependents
        for dependent in graph[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Check for cycles
    if len(order) != len(attributes):
        remaining = [a.name for a in attributes if a.name not in order]
        raise CircularDependencyError(
            f"Circular dependency detected involving: {remaining}"
        )

    return order


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

    When context is provided (overlay mode), dependencies on context
    attributes are valid - they're treated as already-sampled.

    Args:
        attributes: List of HydratedAttribute from hydrator
        context: Existing attributes from base population (for overlay mode)

    Returns:
        Tuple of (list of AttributeSpec, sampling_order, warnings)

    Raises:
        CircularDependencyError: If circular dependencies exist
        ValueError: If dependencies reference unknown attributes

    Example:
        # Base mode
        >>> specs, order, warnings = bind_constraints(hydrated_attrs)

        # Overlay mode - allows dependencies on context attrs
        >>> specs, order, warnings = bind_constraints(overlay_attrs, context=base_spec.attributes)
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
                warnings.append(
                    f"{attr.name}: removed unknown dependency '{dep}'"
                )

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
    sampling_order = _topological_sort_specs(specs)

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
