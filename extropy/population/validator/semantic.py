"""Semantic validation checks (Categories 10-12).

These checks produce WARNING severity issues that don't block sampling.
They help identify potential issues but don't indicate structural problems.
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
)
from ...utils import extract_comparisons_from_expression


# =============================================================================
# Main Entry Point
# =============================================================================


def run_semantic_checks(spec: PopulationSpec) -> list[ValidationIssue]:
    """Run all semantic (WARNING) checks on a spec.

    Categories:
    10. No-Op Detection
    11. Modifier Stacking Analysis
    12. Condition Value Validity
    13. Partner-Correlation Policy Completeness
    14. Modifier Overlap Ambiguity
    """
    issues: list[ValidationIssue] = []

    # Build lookup for categorical options
    attr_lookup = {a.name: a for a in spec.attributes}

    for attr in spec.attributes:
        # Category 10: No-Op Detection
        issues.extend(_check_noop_modifiers(attr))

        # Category 11: Modifier Stacking Analysis
        issues.extend(_check_modifier_stacking(attr))

        # Category 12: Condition Value Validity
        issues.extend(_check_condition_values(attr, attr_lookup))

        # Category 13: Partner-correlation policy completeness
        issues.extend(_check_partner_correlation_policy(attr))

        # Category 14: Modifier overlap ambiguity
        issues.extend(_check_modifier_overlap(attr))

    return issues


# =============================================================================
# Category 10: No-Op Detection
# =============================================================================


def _check_noop_modifiers(attr: AttributeSpec) -> list[ValidationIssue]:
    """Detect modifiers that have no effect."""
    issues = []

    for i, mod in enumerate(attr.sampling.modifiers):
        is_noop = True

        # Check if any field has a meaningful value
        if mod.multiply is not None and mod.multiply != 1.0:
            is_noop = False
        if mod.add is not None and mod.add != 0:
            is_noop = False
        if mod.weight_overrides:
            is_noop = False
        if mod.probability_override is not None:
            is_noop = False

        if is_noop:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category="NO_OP",
                    location=attr.name,
                    modifier_index=i,
                    message="modifier has no effect (multiply=1.0, add=0, no overrides)",
                    suggestion="Remove this modifier or add meaningful values",
                )
            )

    return issues


# =============================================================================
# Category 11: Modifier Stacking Analysis
# =============================================================================


def _check_modifier_stacking(attr: AttributeSpec) -> list[ValidationIssue]:
    """Analyze if stacked modifiers could push values far out of bounds."""
    issues = []
    dist = attr.sampling.distribution

    if not dist or not attr.sampling.modifiers:
        return issues

    # Only analyze numeric distributions with hard constraints
    if not isinstance(
        dist,
        (
            NormalDistribution,
            LognormalDistribution,
            UniformDistribution,
            BetaDistribution,
        ),
    ):
        return issues

    # Find hard constraints
    hard_min = None
    hard_max = None

    for c in attr.constraints:
        if c.type == "hard_min" and c.value is not None:
            hard_min = c.value
        elif c.type == "hard_max" and c.value is not None:
            hard_max = c.value

    if hard_min is None and hard_max is None:
        return issues

    # Calculate worst-case modifier effects
    # Assume all modifiers could apply simultaneously (conservative analysis)
    total_multiply = 1.0
    total_add = 0.0

    for mod in attr.sampling.modifiers:
        if mod.multiply is not None:
            # Take the most extreme multiplier
            if mod.multiply > 1.0:
                total_multiply *= mod.multiply
            elif mod.multiply < 1.0 and mod.multiply > 0:
                # Track minimum multiplier separately for low-end analysis
                pass
        if mod.add is not None:
            total_add += mod.add

    # Estimate base value
    base_value = None
    if isinstance(dist, (NormalDistribution, LognormalDistribution)):
        base_value = dist.mean
    elif isinstance(dist, UniformDistribution):
        if dist.min is not None and dist.max is not None:
            base_value = (dist.min + dist.max) / 2
    elif isinstance(dist, BetaDistribution):
        # Beta mean is alpha / (alpha + beta)
        if dist.alpha and dist.beta:
            base_value = dist.alpha / (dist.alpha + dist.beta)

    if base_value is None:
        return issues

    # Calculate worst-case high value
    worst_high = base_value * total_multiply + total_add

    # Check against constraints
    if hard_max is not None and worst_high > hard_max * 1.5:
        issues.append(
            ValidationIssue(
                severity=Severity.WARNING,
                category="MODIFIER_STACKING",
                location=attr.name,
                message=f"stacked modifiers could push value to {worst_high:.1f} (hard_max={hard_max})",
                suggestion="Review modifier values or add clamping logic",
            )
        )

    # Calculate worst-case low value (all negative adds, smallest multipliers)
    min_multiply = 1.0
    min_add = 0.0
    for mod in attr.sampling.modifiers:
        if mod.multiply is not None and mod.multiply < min_multiply:
            min_multiply = mod.multiply
        if mod.add is not None and mod.add < 0:
            min_add += mod.add

    worst_low = base_value * min_multiply + min_add

    if hard_min is not None and worst_low < hard_min * 0.5:
        issues.append(
            ValidationIssue(
                severity=Severity.WARNING,
                category="MODIFIER_STACKING",
                location=attr.name,
                message=f"stacked modifiers could push value to {worst_low:.1f} (hard_min={hard_min})",
                suggestion="Review modifier values or add clamping logic",
            )
        )

    return issues


# =============================================================================
# Category 12: Condition Value Validity
# =============================================================================


def _check_condition_values(
    attr: AttributeSpec,
    attr_lookup: dict[str, AttributeSpec],
) -> list[ValidationIssue]:
    """Check that condition comparisons use valid categorical options.

    Uses AST parsing to correctly identify which values are compared to which
    attributes, even in compound conditions like:
        employer_type == 'university_hospital' and job_title in ['senior_Oberarzt']
    """
    issues = []

    for i, mod in enumerate(attr.sampling.modifiers):
        if not mod.when:
            continue

        # Parse condition with AST to get (attr_name, values) pairs
        comparisons = extract_comparisons_from_expression(mod.when)

        for compared_attr, values in comparisons:
            if compared_attr not in attr_lookup:
                continue

            ref_attr = attr_lookup[compared_attr]
            ref_dist = ref_attr.sampling.distribution

            if not isinstance(ref_dist, CategoricalDistribution):
                continue

            if not ref_dist.options:
                continue

            valid_options = set(ref_dist.options)

            # Check each value compared to this specific attribute
            for value in values:
                if value not in valid_options:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            category="CONDITION_VALUE",
                            location=attr.name,
                            modifier_index=i,
                            message=f"condition compares {compared_attr} to '{value}' which is not in its options",
                            suggestion=f"Valid options for {compared_attr}: {', '.join(sorted(valid_options))}",
                        )
                    )

    return issues


# =============================================================================
# Category 13: Partner-Correlation Policy Completeness
# =============================================================================


def _check_partner_correlation_policy(attr: AttributeSpec) -> list[ValidationIssue]:
    """Warn when partner-correlated attributes lack explicit policy metadata."""
    if attr.scope != "partner_correlated":
        return []

    has_policy = attr.partner_correlation_policy is not None
    has_semantics = attr.semantic_type is not None or attr.identity_type is not None
    has_explicit_rate = attr.correlation_rate is not None

    if has_policy or has_semantics or has_explicit_rate:
        return []

    return [
        ValidationIssue(
            severity=Severity.WARNING,
            category="PARTNER_POLICY",
            location=attr.name,
            message=(
                "partner_correlated attribute has no explicit correlation policy or "
                "semantic metadata; behavior will fall back to legacy defaults"
            ),
            suggestion=(
                "Set partner_correlation_policy, semantic_type/identity_type, "
                "or correlation_rate"
            ),
        )
    ]


# =============================================================================
# Category 14: Modifier Overlap Ambiguity
# =============================================================================


def _extract_literal_sets(expr: str) -> dict[str, set[str]]:
    """Extract literal comparison sets per attribute from an expression."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return {}

    extracted: dict[str, set[str]] = {}

    def _add(attr_name: str, value: str) -> None:
        extracted.setdefault(attr_name, set()).add(value)

    def _extract_values(node: ast.AST) -> list[str]:
        values: list[str] = []
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, bool, int, float)):
                values.append(str(node.value))
        elif isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(
                    elt.value, (str, bool, int, float)
                ):
                    values.append(str(elt.value))
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(
                    elt.value, (str, bool, int, float)
                ):
                    values.append(str(elt.value))
        return values

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if not isinstance(node.left, ast.Name):
            continue
        attr_name = node.left.id
        for comparator in node.comparators:
            for value in _extract_values(comparator):
                _add(attr_name, value)

    return extracted


def _conditions_could_overlap(expr_a: str, expr_b: str) -> bool:
    """Conservative overlap check for modifier conditions.

    Returns True only when overlap is plausible from literal comparisons.
    """
    if expr_a.strip() == expr_b.strip():
        return True

    literals_a = _extract_literal_sets(expr_a)
    literals_b = _extract_literal_sets(expr_b)
    if not literals_a or not literals_b:
        return False

    shared_attrs = set(literals_a) & set(literals_b)
    if not shared_attrs:
        return False

    for attr_name in shared_attrs:
        if literals_a[attr_name].isdisjoint(literals_b[attr_name]):
            return False

    return True


def _check_modifier_overlap(attr: AttributeSpec) -> list[ValidationIssue]:
    """Warn when categorical/boolean modifiers appear to overlap ambiguously."""
    if attr.type not in {"categorical", "boolean"}:
        return []
    if len(attr.sampling.modifiers) < 2:
        return []
    if attr.sampling.modifier_overlap_policy == "ordered_override":
        return []

    issues: list[ValidationIssue] = []
    for i in range(len(attr.sampling.modifiers)):
        for j in range(i + 1, len(attr.sampling.modifiers)):
            mod_i = attr.sampling.modifiers[i]
            mod_j = attr.sampling.modifiers[j]
            if not mod_i.when or not mod_j.when:
                continue
            if not _conditions_could_overlap(mod_i.when, mod_j.when):
                continue

            policy = attr.sampling.modifier_overlap_policy
            if policy == "exclusive":
                message = (
                    f"modifiers[{i}] and modifiers[{j}] overlap despite "
                    "modifier_overlap_policy='exclusive'"
                )
                suggestion = (
                    "Make the conditions mutually exclusive, or set "
                    "modifier_overlap_policy: ordered_override if precedence is intended"
                )
                category = "MODIFIER_OVERLAP_EXCLUSIVE"
            else:
                message = (
                    f"modifiers[{i}] and modifiers[{j}] may overlap; categorical/boolean "
                    "modifiers use last-wins semantics"
                )
                suggestion = (
                    "Set sampling.modifier_overlap_policy to 'ordered_override' "
                    "if intentional, otherwise make conditions mutually exclusive"
                )
                category = "MODIFIER_OVERLAP"

            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category=category,
                    location=attr.name,
                    message=message,
                    suggestion=suggestion,
                )
            )

    return issues
