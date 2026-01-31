"""Structural validation checks (Categories 1-9).

These checks produce ERROR severity issues that block sampling.
All checks are structural - no sampling required.
"""

import ast

from ...core.models.validation import Severity, ValidationIssue
from ...core.models import (
    PopulationSpec,
    AttributeSpec,
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
)
from ...utils.expressions import extract_names_from_expression


# =============================================================================
# Main Entry Point
# =============================================================================


def run_structural_checks(spec: PopulationSpec) -> list[ValidationIssue]:
    """Run all structural (ERROR) checks on a spec.

    Categories:
    1. Type/Modifier Compatibility
    2. Range Violations
    3. Weight Validity
    4. Distribution Parameters
    5. Dependency Validation
    6. Condition Syntax & References
    7. Formula Validation
    8. Duplicate Detection
    9. Strategy Consistency
    """
    issues: list[ValidationIssue] = []

    # Build lookup structures
    attr_names = {a.name for a in spec.attributes}

    # Category 8: Duplicate Detection (check first)
    issues.extend(_check_duplicates(spec.attributes))

    # Run per-attribute checks
    for attr in spec.attributes:
        # Category 1: Type/Modifier Compatibility
        issues.extend(_check_type_modifier_compatibility(attr))

        # Category 2: Range Violations
        issues.extend(_check_range_violations(attr))

        # Category 3: Weight Validity
        issues.extend(_check_weight_validity(attr))

        # Category 4: Distribution Parameters
        issues.extend(_check_distribution_params(attr))

        # Category 5: Dependency Validation
        issues.extend(_check_dependencies(attr, attr_names, set(spec.sampling_order)))

        # Category 6: Condition Syntax & References
        issues.extend(_check_conditions(attr))

        # Category 7: Formula Validation
        issues.extend(_check_formulas(attr))

        # Category 9: Strategy Consistency
        issues.extend(_check_strategy_consistency(attr))

    # Category 5: Sampling order validation
    issues.extend(_check_sampling_order(spec.attributes, spec.sampling_order))

    return issues


# =============================================================================
# Category 1: Type/Modifier Compatibility
# =============================================================================


def _check_type_modifier_compatibility(attr: AttributeSpec) -> list[ValidationIssue]:
    """Check that modifiers use operations compatible with distribution type."""
    issues = []
    dist = attr.sampling.distribution

    if not dist or not attr.sampling.modifiers:
        return issues

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

    for i, mod in enumerate(attr.sampling.modifiers):
        if is_numeric_dist:
            if mod.weight_overrides is not None:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="numeric distribution cannot use weight_overrides",
                        suggestion="Use multiply/add instead",
                    )
                )
            if mod.probability_override is not None:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="numeric distribution cannot use probability_override",
                        suggestion="Use multiply/add instead",
                    )
                )

        elif is_categorical_dist:
            if mod.multiply is not None and mod.multiply != 1.0:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="categorical distribution cannot use multiply",
                        suggestion="Use weight_overrides instead",
                    )
                )
            if mod.add is not None and mod.add != 0:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="categorical distribution cannot use add",
                        suggestion="Use weight_overrides instead",
                    )
                )
            if mod.probability_override is not None:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="categorical distribution cannot use probability_override",
                        suggestion="Use weight_overrides instead",
                    )
                )

        elif is_boolean_dist:
            if mod.multiply is not None and mod.multiply != 1.0:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="boolean distribution cannot use multiply",
                        suggestion="Use probability_override instead",
                    )
                )
            if mod.add is not None and mod.add != 0:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="boolean distribution cannot use add",
                        suggestion="Use probability_override instead",
                    )
                )
            if mod.weight_overrides is not None:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="TYPE_MISMATCH",
                        location=attr.name,
                        modifier_index=i,
                        message="boolean distribution cannot use weight_overrides",
                        suggestion="Use probability_override instead",
                    )
                )

    return issues


# =============================================================================
# Category 2: Range Violations
# =============================================================================


def _check_range_violations(attr: AttributeSpec) -> list[ValidationIssue]:
    """Check for obvious range violations in modifiers."""
    issues = []
    dist = attr.sampling.distribution

    if not dist or not attr.sampling.modifiers:
        return issues

    for i, mod in enumerate(attr.sampling.modifiers):
        # Beta distribution with large add values
        if isinstance(dist, BetaDistribution):
            if mod.add is not None and abs(mod.add) > 0.5:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="RANGE_VIOLATION",
                        location=attr.name,
                        modifier_index=i,
                        message=f"beta distribution add={mod.add} is too large (outputs 0-1 scale)",
                        suggestion="Use small add values (±0.05 to ±0.15) for beta distributions",
                    )
                )

        # Boolean probability_override out of bounds
        if isinstance(dist, BooleanDistribution):
            if mod.probability_override is not None:
                if mod.probability_override < 0 or mod.probability_override > 1:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="RANGE_VIOLATION",
                            location=attr.name,
                            modifier_index=i,
                            message=f"probability_override={mod.probability_override} must be between 0 and 1",
                        )
                    )

    return issues


# =============================================================================
# Category 3: Weight Validity
# =============================================================================


def _check_weight_validity(attr: AttributeSpec) -> list[ValidationIssue]:
    """Check categorical weights sum to ~1.0 and reference valid options."""
    issues = []
    dist = attr.sampling.distribution

    if not isinstance(dist, CategoricalDistribution):
        return issues

    # Check base distribution
    if not dist.options:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                category="WEIGHT_INVALID",
                location=attr.name,
                message="categorical distribution has no options",
            )
        )
    elif dist.weights:
        if len(dist.options) != len(dist.weights):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="WEIGHT_INVALID",
                    location=attr.name,
                    message=f"options ({len(dist.options)}) and weights ({len(dist.weights)}) length mismatch",
                )
            )
        elif abs(sum(dist.weights) - 1.0) > 0.02:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="WEIGHT_INVALID",
                    location=attr.name,
                    message=f"weights sum to {sum(dist.weights):.2f}, should be ~1.0",
                )
            )

    # Check modifier weight_overrides
    valid_options = set(dist.options) if dist.options else set()

    for i, mod in enumerate(attr.sampling.modifiers):
        if mod.weight_overrides:
            # Check for unknown options (ERROR - likely typo)
            for key in mod.weight_overrides.keys():
                if key not in valid_options:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="WEIGHT_INVALID",
                            location=attr.name,
                            modifier_index=i,
                            message=f"weight_override key '{key}' not in distribution options",
                            suggestion=f"Valid options: {', '.join(sorted(valid_options))}",
                        )
                    )

            # Check weights sum to 1.0 (ERROR)
            weight_sum = sum(mod.weight_overrides.values())
            if abs(weight_sum - 1.0) > 0.02:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="WEIGHT_INVALID",
                        location=attr.name,
                        modifier_index=i,
                        message=f"weight_overrides sum to {weight_sum:.2f}, should be ~1.0",
                    )
                )

            # Check for missing options (WARNING - partial override may be intentional)
            missing = valid_options - set(mod.weight_overrides.keys())
            if missing and len(mod.weight_overrides) > 0:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category="WEIGHT_INCOMPLETE",
                        location=attr.name,
                        modifier_index=i,
                        message=f"weight_overrides missing options: {', '.join(sorted(missing))}",
                    )
                )

    return issues


# =============================================================================
# Category 4: Distribution Parameters
# =============================================================================


def _check_distribution_params(attr: AttributeSpec) -> list[ValidationIssue]:
    """Check distribution parameters are mathematically valid."""
    issues = []
    dist = attr.sampling.distribution

    if dist is None:
        return issues

    if isinstance(dist, (NormalDistribution, LognormalDistribution)):
        if dist.std is not None and dist.std < 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DIST_PARAM_INVALID",
                    location=attr.name,
                    message=f"std ({dist.std}) cannot be negative",
                )
            )
        elif dist.std is not None and dist.std == 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DIST_PARAM_INVALID",
                    location=attr.name,
                    message="std is 0 (no variance) - use derived strategy instead",
                )
            )
        if dist.min is not None and dist.max is not None and dist.min >= dist.max:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DIST_PARAM_INVALID",
                    location=attr.name,
                    message=f"min ({dist.min}) >= max ({dist.max})",
                )
            )

    elif isinstance(dist, BetaDistribution):
        if dist.alpha is None or dist.alpha <= 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DIST_PARAM_INVALID",
                    location=attr.name,
                    message="beta distribution alpha must be positive",
                )
            )
        if dist.beta is None or dist.beta <= 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DIST_PARAM_INVALID",
                    location=attr.name,
                    message="beta distribution beta must be positive",
                )
            )

    elif isinstance(dist, UniformDistribution):
        if dist.min is not None and dist.max is not None and dist.min >= dist.max:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DIST_PARAM_INVALID",
                    location=attr.name,
                    message=f"min ({dist.min}) >= max ({dist.max})",
                )
            )

    elif isinstance(dist, BooleanDistribution):
        if dist.probability_true is not None:
            if dist.probability_true < 0 or dist.probability_true > 1:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="DIST_PARAM_INVALID",
                        location=attr.name,
                        message=f"probability_true ({dist.probability_true}) must be between 0 and 1",
                    )
                )

    return issues


# =============================================================================
# Category 5: Dependency Validation
# =============================================================================


def _check_dependencies(
    attr: AttributeSpec,
    all_attr_names: set[str],
    sampling_order_set: set[str],
) -> list[ValidationIssue]:
    """Check dependencies reference existing attributes."""
    issues = []

    for dep in attr.sampling.depends_on:
        if dep not in all_attr_names:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DEPENDENCY_INVALID",
                    location=attr.name,
                    message=f"depends_on references non-existent attribute '{dep}'",
                )
            )

    return issues


def _check_sampling_order(
    attributes: list[AttributeSpec],
    sampling_order: list[str],
) -> list[ValidationIssue]:
    """Check sampling order respects dependencies and includes all attributes."""
    issues = []
    attr_names = {a.name for a in attributes}
    order_set = set(sampling_order)
    order_index = {name: i for i, name in enumerate(sampling_order)}

    # Check all attributes are in sampling order
    missing = attr_names - order_set
    for name in missing:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                category="SAMPLING_ORDER_INVALID",
                location=name,
                message="attribute missing from sampling_order",
            )
        )

    # Check dependencies are sampled before dependents
    for attr in attributes:
        attr_idx = order_index.get(attr.name)
        if attr_idx is None:
            continue  # Already reported as missing

        for dep in attr.sampling.depends_on:
            dep_idx = order_index.get(dep)
            if dep_idx is not None and dep_idx >= attr_idx:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="SAMPLING_ORDER_INVALID",
                        location=attr.name,
                        message=f"sampled before dependency '{dep}' (positions {attr_idx} vs {dep_idx})",
                    )
                )

    return issues


# =============================================================================
# Category 6: Condition Syntax & References
# =============================================================================


def _check_conditions(attr: AttributeSpec) -> list[ValidationIssue]:
    """Check modifier conditions are valid Python and reference declared dependencies."""
    issues = []

    for i, mod in enumerate(attr.sampling.modifiers):
        if not mod.when:
            continue

        # Try to parse as Python expression
        try:
            ast.parse(mod.when, mode="eval")
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="CONDITION_SYNTAX",
                    location=attr.name,
                    modifier_index=i,
                    message=f"invalid condition syntax: {e}",
                    value=mod.when,
                    suggestion="Fix the Python expression syntax in the 'when' condition",
                )
            )
            continue

        # Check referenced attributes are in depends_on
        referenced = extract_names_from_expression(mod.when)
        depends_on_set = set(attr.sampling.depends_on)

        for name in referenced:
            if name not in depends_on_set:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="CONDITION_REFERENCE",
                        location=attr.name,
                        modifier_index=i,
                        message=f"condition references '{name}' not in depends_on",
                        suggestion=f"Add '{name}' to depends_on or remove from condition",
                    )
                )

    return issues


# =============================================================================
# Category 7: Formula Validation
# =============================================================================


def _check_formulas(attr: AttributeSpec) -> list[ValidationIssue]:
    """Check derived formulas and mean_formulas are valid."""
    issues = []
    depends_on_set = set(attr.sampling.depends_on)

    # Check sampling formula (for derived strategy)
    if attr.sampling.formula:
        try:
            ast.parse(attr.sampling.formula, mode="eval")
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="FORMULA_SYNTAX",
                    location=attr.name,
                    message=f"invalid formula syntax: {e}",
                )
            )
        else:
            referenced = extract_names_from_expression(attr.sampling.formula)
            for name in referenced:
                if name not in depends_on_set:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="FORMULA_REFERENCE",
                            location=attr.name,
                            message=f"formula references '{name}' not in depends_on",
                        )
                    )

    # Check mean_formula in distribution
    dist = attr.sampling.distribution
    if (
        isinstance(dist, (NormalDistribution, LognormalDistribution))
        and dist.mean_formula
    ):
        try:
            ast.parse(dist.mean_formula, mode="eval")
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="FORMULA_SYNTAX",
                    location=attr.name,
                    message=f"invalid mean_formula syntax: {e}",
                )
            )
        else:
            referenced = extract_names_from_expression(dist.mean_formula)
            for name in referenced:
                if name not in depends_on_set:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="FORMULA_REFERENCE",
                            location=attr.name,
                            message=f"mean_formula references '{name}' not in depends_on",
                        )
                    )

    return issues


# =============================================================================
# Category 8: Duplicate Detection
# =============================================================================


def _check_duplicates(attributes: list[AttributeSpec]) -> list[ValidationIssue]:
    """Check for duplicate attribute names."""
    issues = []
    seen = set()

    for attr in attributes:
        if attr.name in seen:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="DUPLICATE",
                    location=attr.name,
                    message="duplicate attribute name",
                )
            )
        seen.add(attr.name)

    return issues


# =============================================================================
# Category 9: Strategy Consistency
# =============================================================================


def _check_strategy_consistency(attr: AttributeSpec) -> list[ValidationIssue]:
    """Check strategy, distribution, and formula are consistent."""
    issues = []
    strategy = attr.sampling.strategy
    has_dist = attr.sampling.distribution is not None
    has_formula = bool(attr.sampling.formula)
    has_depends = bool(attr.sampling.depends_on)
    has_modifiers = bool(attr.sampling.modifiers)

    if strategy == "independent":
        if not has_dist:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="independent strategy requires distribution",
                )
            )
        if has_formula:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category="STRATEGY_INCONSISTENT",
                    location=attr.name,
                    message="independent strategy has formula (will be ignored)",
                )
            )
        if has_modifiers:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="independent strategy cannot have modifiers",
                )
            )
        if has_depends:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="independent strategy cannot have depends_on",
                )
            )

    elif strategy == "derived":
        if not has_formula:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="derived strategy requires formula",
                )
            )
        if not has_depends:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="derived strategy requires depends_on",
                )
            )
        if has_dist:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category="STRATEGY_INCONSISTENT",
                    location=attr.name,
                    message="derived strategy has distribution (will be ignored)",
                )
            )
        if has_modifiers:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="derived strategy cannot have modifiers",
                )
            )

    elif strategy == "conditional":
        if not has_dist:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="conditional strategy requires distribution",
                )
            )
        if not has_depends:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category="STRATEGY_INCONSISTENT",
                    location=attr.name,
                    message="conditional strategy has no depends_on (should be independent?)",
                )
            )
        if has_formula:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="STRATEGY_INVALID",
                    location=attr.name,
                    message="conditional strategy cannot have formula",
                )
            )

    return issues
