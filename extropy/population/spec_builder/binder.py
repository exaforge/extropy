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
    HouseholdConfig,
)
from ...utils import topological_sort, extract_names_from_expression


def _infer_expression_dependencies(
    attr: HydratedAttribute, known_names: set[str]
) -> tuple[list[str], list[str], list[str]]:
    """Infer dependencies referenced in formulas/conditions.

    Returns:
        - merged_depends_on: declared + inferred refs (known names only)
        - inferred_refs: known references that were auto-added
        - unknown_refs: references not present in known_names
    """
    declared = [d for d in attr.depends_on if d in known_names]
    declared_set = set(declared)

    referenced: set[str] = set()

    # Derived formulas
    if attr.sampling.formula:
        referenced.update(extract_names_from_expression(attr.sampling.formula))

    # Distribution-level formulas
    dist = attr.sampling.distribution
    if dist is not None:
        for field in ("mean_formula", "std_formula", "min_formula", "max_formula"):
            expr = getattr(dist, field, None)
            if expr:
                referenced.update(extract_names_from_expression(expr))

    # Conditional modifier expressions
    for mod in attr.sampling.modifiers:
        if mod.when:
            referenced.update(extract_names_from_expression(mod.when))

    inferred_refs = sorted(
        name for name in referenced if name in known_names and name not in declared_set
    )
    unknown_refs = sorted(name for name in referenced if name not in known_names)

    merged_depends_on = declared + inferred_refs
    return merged_depends_on, inferred_refs, unknown_refs


def bind_constraints(
    attributes: list[HydratedAttribute],
    context: list[AttributeSpec] | None = None,
) -> tuple[list[AttributeSpec], list[str], list[str]]:
    """
    Build dependency graph and determine sampling order.

    This step:
    1. Validates all dependencies reference existing or context attributes
    2. Auto-infers dependencies from formulas/conditions
    3. Checks for circular dependencies
    4. Computes topological sort for sampling order
    5. Converts HydratedAttribute to final AttributeSpec

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

        # Infer dependencies from formulas/conditions and merge with declared deps
        filtered_depends_on, inferred_refs, unknown_expr_refs = (
            _infer_expression_dependencies(attr, known_names)
        )

        if inferred_refs:
            warnings.append(
                f"{attr.name}: auto-added depends_on from expressions: {', '.join(inferred_refs)}"
            )

        if unknown_expr_refs:
            warnings.append(
                f"{attr.name}: expression references unknown attributes: {', '.join(unknown_expr_refs)}"
            )

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
            scope=attr.scope,
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
    agent_focus: str | None = None,
    household_config: HouseholdConfig | None = None,
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
        agent_focus: Who the study agents represent (determines agent vs NPC)
        household_config: LLM-researched household composition (defaults to US Census)

    Returns:
        Complete PopulationSpec ready for YAML export
    """
    meta = SpecMeta(
        description=description,
        size=size,
        geography=geography,
        agent_focus=agent_focus,
        created_at=datetime.now(),
        household_config=household_config or HouseholdConfig(),
    )

    grounding = _compute_grounding_summary(attributes, sources)

    return PopulationSpec(
        meta=meta,
        grounding=grounding,
        attributes=attributes,
        sampling_order=sampling_order,
    )
