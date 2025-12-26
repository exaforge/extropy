"""Validation functions for hydrated attributes.

These functions validate HydratedAttribute objects after they've been
populated by the hydration pipeline. Used to catch structural and
semantic issues before sampling.
"""

from ...core.models import (
    HydratedAttribute,
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
)
from .common import (
    BUILTIN_NAMES,
    extract_names_from_expression as extract_names_from_formula,
    extract_names_from_expression as extract_names_from_condition,
)
from .llm_response import (
    is_spec_level_constraint,
    extract_bound_from_constraint,
)


# =============================================================================
# Validation Functions
# =============================================================================


def validate_independent_hydration(attributes: list[HydratedAttribute]) -> list[str]:
    """Validate hydrated independent attributes."""
    errors = []

    for attr in attributes:
        dist = attr.sampling.distribution

        if dist is None:
            errors.append(f"{attr.name}: independent attribute missing distribution")
            continue

        # Validate based on distribution type (not attribute type)
        # This catches mismatches between declared type and actual distribution

        # Categorical distribution validation
        if isinstance(dist, CategoricalDistribution):
            # Check for empty options list
            if not dist.options:
                errors.append(f"{attr.name}: categorical distribution has no options")
            # Check for mismatched options/weights array lengths
            elif dist.weights and len(dist.options) != len(dist.weights):
                errors.append(
                    f"{attr.name}: options ({len(dist.options)}) and weights ({len(dist.weights)}) length mismatch"
                )
            # Check weights sum to ~1.0
            elif dist.weights and abs(sum(dist.weights) - 1.0) > 0.02:
                errors.append(
                    f"{attr.name}: weights sum to {sum(dist.weights):.2f}, should be ~1.0"
                )

        # Boolean distribution validation
        elif isinstance(dist, BooleanDistribution):
            if dist.probability_true is not None:
                if dist.probability_true < 0 or dist.probability_true > 1:
                    errors.append(
                        f"{attr.name}: probability_true ({dist.probability_true}) must be between 0 and 1"
                    )

        # Normal/Lognormal distribution validation
        elif isinstance(dist, (NormalDistribution, LognormalDistribution)):
            # Check for negative standard deviation
            if dist.std is not None and dist.std < 0:
                errors.append(f"{attr.name}: std ({dist.std}) cannot be negative")
            # Check for zero standard deviation (should use derived strategy)
            elif dist.std is not None and dist.std == 0:
                errors.append(
                    f"{attr.name}: std is 0 (no variance) — use derived strategy instead"
                )
            # Check min/max validity
            if dist.min is not None and dist.max is not None:
                if dist.min >= dist.max:
                    errors.append(f"{attr.name}: min ({dist.min}) >= max ({dist.max})")

        # Beta distribution validation
        elif isinstance(dist, BetaDistribution):
            # Check for missing or non-positive alpha
            if dist.alpha is None or dist.alpha <= 0:
                errors.append(f"{attr.name}: beta distribution alpha must be positive")
            # Check for missing or non-positive beta
            if dist.beta is None or dist.beta <= 0:
                errors.append(f"{attr.name}: beta distribution beta must be positive")

        # Uniform distribution validation
        elif isinstance(dist, UniformDistribution):
            if dist.min is not None and dist.max is not None:
                if dist.min >= dist.max:
                    errors.append(f"{attr.name}: min ({dist.min}) >= max ({dist.max})")

        # Check constraints for spec-level expressions using wrong type
        for constraint in attr.constraints:
            if constraint.type == "expression" and constraint.expression:
                if is_spec_level_constraint(constraint.expression):
                    errors.append(
                        f"{attr.name}: constraint '{constraint.expression}' references spec-level variables "
                        f"(weights/options) but uses type='expression'. "
                        f"Change to type='spec_expression' — this validates the YAML spec itself, not individual agents."
                    )

    return errors


def validate_derived_hydration(
    attributes: list[HydratedAttribute], all_attribute_names: set[str]
) -> list[str]:
    """Validate hydrated derived attributes."""
    errors = []

    for attr in attributes:
        if not attr.sampling.formula:
            errors.append(f"{attr.name}: derived attribute missing formula")
            continue

        try:
            ast.parse(attr.sampling.formula, mode="eval")
        except SyntaxError as e:
            errors.append(f"{attr.name}: invalid formula syntax: {e}")
            continue

        used_names = extract_names_from_formula(attr.sampling.formula)
        for name in used_names:
            if name in BUILTIN_NAMES:
                continue
            if name not in attr.depends_on:
                errors.append(
                    f"{attr.name}: formula references '{name}' not in depends_on"
                )
            elif name not in all_attribute_names:
                errors.append(
                    f"{attr.name}: formula references unknown attribute '{name}'"
                )

    return errors


def validate_conditional_base(attributes: list[HydratedAttribute]) -> list[str]:
    """Validate conditional attributes with base distributions."""
    errors = []

    for attr in attributes:
        dist = attr.sampling.distribution

        if dist is None:
            errors.append(f"{attr.name}: conditional attribute missing distribution")
            continue

        if (
            isinstance(dist, (NormalDistribution, LognormalDistribution))
            and dist.mean_formula
        ):
            used_names = extract_names_from_formula(dist.mean_formula)
            for name in used_names:
                if name not in attr.depends_on and name not in BUILTIN_NAMES:
                    errors.append(
                        f"{attr.name}: mean_formula references '{name}' not in depends_on"
                    )

        if isinstance(dist, (NormalDistribution, LognormalDistribution)):
            if dist.mean_formula and dist.std is None:
                errors.append(
                    f"{attr.name}: has mean_formula but no std — this makes it derived, not conditional"
                )

        # Check for expression constraints that need corresponding formula bounds
        # Only for numeric distributions that support min/max formulas
        if isinstance(
            dist, (NormalDistribution, LognormalDistribution, BetaDistribution)
        ):
            for constraint in attr.constraints:
                if constraint.type == "expression" and constraint.expression:
                    # Skip spec-level constraints
                    if is_spec_level_constraint(constraint.expression):
                        errors.append(
                            f"{attr.name}: constraint '{constraint.expression}' references spec-level variables "
                            f"(weights/options) but uses type='expression'. "
                            f"Change to type='spec_expression'."
                        )
                        continue

                    # Try to extract bound from constraint
                    bound_type, bound_expr, is_strict = extract_bound_from_constraint(
                        constraint.expression, attr.name
                    )

                    if bound_type == "max" and bound_expr:
                        # Check if distribution has max_formula or static max
                        has_max_formula = getattr(dist, "max_formula", None) is not None
                        has_static_max = getattr(dist, "max", None) is not None
                        if not has_max_formula and not has_static_max:
                            errors.append(
                                f"{attr.name}: constraint '{constraint.expression}' exists but distribution has no max_formula. "
                                f"Add to distribution: max_formula: '{bound_expr}'"
                            )
                    elif bound_type == "min" and bound_expr:
                        # Check if distribution has min_formula
                        has_min_formula = getattr(dist, "min_formula", None) is not None
                        if not has_min_formula:
                            # Only error if there's no static min either
                            has_static_min = getattr(dist, "min", None) is not None
                            if not has_static_min:
                                errors.append(
                                    f"{attr.name}: constraint '{constraint.expression}' exists but distribution has no min_formula. "
                                    f"Add to distribution: min_formula: '{bound_expr}'"
                                )

    return errors


def validate_modifiers(
    attributes: list[HydratedAttribute], all_attributes: dict[str, HydratedAttribute]
) -> tuple[list[str], list[str]]:
    """Validate modifiers for conditional attributes.

    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []

    for attr in attributes:
        dist = attr.sampling.distribution

        for i, mod in enumerate(attr.sampling.modifiers):
            # Validate 'when' clause references
            referenced = extract_names_from_condition(mod.when)
            for name in referenced:
                if name not in attr.depends_on:
                    errors.append(
                        f"{attr.name} modifier {i}: 'when' references '{name}' not in depends_on"
                    )

            # Distribution type/modifier compatibility validation
            # Check the actual distribution type, not just the attribute type
            is_numeric_dist = isinstance(
                dist,
                (
                    NormalDistribution,
                    LognormalDistribution,
                    UniformDistribution,
                    BetaDistribution,
                ),
            )
            is_categorical_dist = isinstance(dist, CategoricalDistribution)
            is_boolean_dist = isinstance(dist, BooleanDistribution)

            if is_numeric_dist:
                # Numeric distributions: can only use multiply/add
                if mod.weight_overrides is not None:
                    errors.append(
                        f"{attr.name} modifier {i}: numeric distribution cannot use weight_overrides"
                    )
                if mod.probability_override is not None:
                    errors.append(
                        f"{attr.name} modifier {i}: numeric distribution cannot use probability_override"
                    )
            elif is_categorical_dist:
                # Categorical distributions: can only use weight_overrides
                if mod.multiply is not None and mod.multiply != 1.0:
                    errors.append(
                        f"{attr.name} modifier {i}: categorical distribution cannot use multiply"
                    )
                if mod.add is not None and mod.add != 0:
                    errors.append(
                        f"{attr.name} modifier {i}: categorical distribution cannot use add"
                    )
                if mod.probability_override is not None:
                    errors.append(
                        f"{attr.name} modifier {i}: categorical distribution cannot use probability_override"
                    )
                # Validate weight_override keys match distribution options
                if mod.weight_overrides and dist.options:
                    valid_options = set(dist.options)
                    for key in mod.weight_overrides.keys():
                        if key not in valid_options:
                            errors.append(
                                f"{attr.name} modifier {i}: weight_override key '{key}' not in options"
                            )
            elif is_boolean_dist:
                # Boolean distributions: can only use probability_override
                if mod.multiply is not None and mod.multiply != 1.0:
                    errors.append(
                        f"{attr.name} modifier {i}: boolean distribution cannot use multiply"
                    )
                if mod.add is not None and mod.add != 0:
                    errors.append(
                        f"{attr.name} modifier {i}: boolean distribution cannot use add"
                    )
                if mod.weight_overrides is not None:
                    errors.append(
                        f"{attr.name} modifier {i}: boolean distribution cannot use weight_overrides"
                    )

            # P2 Warning: Check for no-op modifiers
            is_noop = True
            if mod.multiply is not None and mod.multiply != 1.0:
                is_noop = False
            if mod.add is not None and mod.add != 0:
                is_noop = False
            if mod.weight_overrides:
                is_noop = False
            if mod.probability_override is not None:
                is_noop = False

            if is_noop:
                warnings.append(
                    f"{attr.name} modifier {i}: no-op modifier (multiply=1.0, add=0, no overrides)"
                )

    return errors, warnings


def validate_strategy_consistency(attributes: list[HydratedAttribute]) -> list[str]:
    """Validate that attributes have correct fields for their declared strategy.

    Rules:
    - Independent strategy: Must have distribution. Must not have formula, modifiers, or depends_on.
    - Derived strategy: Must have formula and depends_on. Must not have distribution or modifiers.
    - Conditional strategy: Must have distribution and depends_on. Must not have formula. Modifiers optional.

    Args:
        attributes: List of HydratedAttribute to validate

    Returns:
        List of error messages
    """
    errors = []

    for attr in attributes:
        strategy = attr.sampling.strategy
        has_dist = attr.sampling.distribution is not None
        has_formula = bool(attr.sampling.formula)
        has_depends = bool(attr.sampling.depends_on)
        has_modifiers = bool(attr.sampling.modifiers)

        if strategy == "independent":
            # Must have distribution
            if not has_dist:
                errors.append(
                    f"{attr.name}: independent strategy requires distribution"
                )
            # Must not have formula
            if has_formula:
                errors.append(f"{attr.name}: independent strategy cannot have formula")
            # Must not have modifiers
            if has_modifiers:
                errors.append(
                    f"{attr.name}: independent strategy cannot have modifiers"
                )
            # Must not have depends_on
            if has_depends:
                errors.append(
                    f"{attr.name}: independent strategy cannot have depends_on"
                )

        elif strategy == "derived":
            # Must have formula
            if not has_formula:
                errors.append(f"{attr.name}: derived strategy requires formula")
            # Must have depends_on
            if not has_depends:
                errors.append(f"{attr.name}: derived strategy requires depends_on")
            # Must not have distribution
            if has_dist:
                errors.append(f"{attr.name}: derived strategy cannot have distribution")
            # Must not have modifiers
            if has_modifiers:
                errors.append(f"{attr.name}: derived strategy cannot have modifiers")

        elif strategy == "conditional":
            # Must have distribution
            if not has_dist:
                errors.append(
                    f"{attr.name}: conditional strategy requires distribution"
                )
            # Must have depends_on
            if not has_depends:
                errors.append(f"{attr.name}: conditional strategy requires depends_on")
            # Must not have formula
            if has_formula:
                errors.append(f"{attr.name}: conditional strategy cannot have formula")
            # Modifiers are optional for conditional

    return errors
