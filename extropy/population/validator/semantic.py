"""Semantic validation checks (Categories 10-12).

These checks produce WARNING severity issues that don't block sampling.
They help identify potential issues but don't indicate structural problems.
"""

import ast
import itertools
import math

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
from ..modifier_precedence import choose_modifier_precedence
from ...utils import extract_comparisons_from_expression, extract_names_from_expression


# =============================================================================
# Main Entry Point
# =============================================================================


def run_semantic_checks(spec: PopulationSpec) -> list[ValidationIssue]:
    """Run all semantic (WARNING) checks on a spec.

    Categories:
    10. No-Op Detection
    11. Modifier Stacking Analysis
    12. Condition Value Validity
    13. Last-Wins Overlap Detection
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

        # Category 13: Last-Wins overlap detection for categorical/boolean
        issues.extend(_check_last_wins_overlaps(attr, attr_lookup))

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
# Category 13: Last-Wins Overlap Detection
# =============================================================================


def _check_last_wins_overlaps(
    attr: AttributeSpec,
    attr_lookup: dict[str, AttributeSpec],
) -> list[ValidationIssue]:
    """Warn when categorical/boolean modifiers overlap ambiguously.

    Overlap itself is not always problematic when precedence is deterministic
    by condition specificity/subset. We only warn when overlap would still
    resolve by declaration order.
    """
    issues: list[ValidationIssue] = []
    dist = attr.sampling.distribution
    modifiers = attr.sampling.modifiers
    if not dist or not modifiers or len(modifiers) < 2:
        return issues

    is_last_wins_categorical = isinstance(dist, CategoricalDistribution)
    is_last_wins_boolean = isinstance(dist, BooleanDistribution)
    if not (is_last_wins_categorical or is_last_wins_boolean):
        return issues

    def normalize_when(expr: str) -> str:
        return expr.strip().lower().replace(" ", "")

    def is_unconditional(expr: str) -> bool:
        normalized = normalize_when(expr)
        return normalized in {"true", "1==1", "(true)"}

    for i in range(len(modifiers)):
        for j in range(i + 1, len(modifiers)):
            left = modifiers[i].when
            right = modifiers[j].when
            if not left or not right:
                continue

            may_overlap = False
            if is_unconditional(left) or is_unconditional(right):
                may_overlap = True
            elif normalize_when(left) == normalize_when(right):
                may_overlap = True
            else:
                may_overlap = _conditions_can_both_be_true(
                    left,
                    right,
                    attr_lookup,
                )

            if may_overlap:
                decision = choose_modifier_precedence([(i, left), (j, right)])
                if decision and decision.reason in {"subset", "specificity"}:
                    continue

                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category="MODIFIER_OVERLAP",
                        location=attr.name,
                        message=(
                            f"modifiers {i} and {j} overlap and currently rely on declaration order"
                        ),
                        suggestion=(
                            "Make conditions mutually exclusive or add specificity so one "
                            "rule deterministically dominates"
                        ),
                    )
                )

    return issues


def _conditions_can_both_be_true(
    left: str,
    right: str,
    attr_lookup: dict[str, AttributeSpec],
) -> bool:
    """Return True if both condition expressions appear jointly satisfiable."""
    referenced = sorted(
        (
            extract_names_from_expression(left) | extract_names_from_expression(right)
        ).intersection(set(attr_lookup.keys()))
    )

    # If no referenced attrs, evaluate directly.
    if not referenced:
        return _eval_condition_bool(left, {}) and _eval_condition_bool(right, {})

    # If expressions reference unknown names, keep conservative warning behavior.
    all_referenced = extract_names_from_expression(
        left
    ) | extract_names_from_expression(right)
    unknown = [name for name in all_referenced if name not in attr_lookup]
    if unknown:
        return True

    domains: dict[str, list[object]] = {}
    for name in referenced:
        domains[name] = _candidate_values_for_attr(attr_lookup[name], [left, right])
        if not domains[name]:
            return True

    domain_lists = [domains[name] for name in referenced]
    total = 1
    for values in domain_lists:
        total *= max(1, len(values))

    max_combinations = 120_000
    if total > max_combinations:
        trimmed = []
        for values in domain_lists:
            if len(values) <= 18:
                trimmed.append(values)
                continue
            step = max(1, len(values) // 18)
            trimmed.append(values[::step][:18])
        domain_lists = trimmed

    for combo in itertools.product(*domain_lists):
        env = dict(zip(referenced, combo, strict=False))
        if not _eval_condition_bool(left, env):
            continue
        if _eval_condition_bool(right, env):
            return True

    return False


def _eval_condition_bool(expr: str, env: dict[str, object]) -> bool:
    """Evaluate condition expression; return False on evaluation errors."""
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def _candidate_values_for_attr(
    attr: AttributeSpec,
    expressions: list[str],
) -> list[object]:
    """Build a bounded candidate set for condition satisfiability checks."""
    dist = attr.sampling.distribution

    if attr.type == "boolean":
        return [True, False]

    if attr.type == "categorical" and isinstance(dist, CategoricalDistribution):
        return list(dist.options or [])

    if attr.type in {"int", "float"}:
        values: set[float] = set()
        min_val, max_val = _numeric_bounds(attr)

        if min_val is not None:
            values.add(min_val)
        if max_val is not None:
            values.add(max_val)
        if min_val is not None and max_val is not None and min_val <= max_val:
            values.add((min_val + max_val) / 2.0)

        for expr in expressions:
            for constant in _extract_numeric_constants(expr):
                values.add(constant)
                values.add(constant - 1)
                values.add(constant + 1)
                values.add(constant - 0.5)
                values.add(constant + 0.5)

        if not values:
            values = {0.0, 1.0, 10.0}

        bounded: list[float] = []
        for value in values:
            if min_val is not None and value < min_val:
                continue
            if max_val is not None and value > max_val:
                continue
            bounded.append(value)

        if attr.type == "int":
            ints: list[int] = []
            for value in bounded:
                ints.extend(
                    [
                        int(math.floor(value)),
                        int(round(value)),
                        int(math.ceil(value)),
                    ]
                )
            ints = sorted(set(ints))
            if len(ints) > 80:
                step = max(1, len(ints) // 80)
                ints = ints[::step]
            return ints

        floats = sorted(set(round(value, 6) for value in bounded))
        if len(floats) > 80:
            step = max(1, len(floats) // 80)
            floats = floats[::step]
        return floats

    return []


def _numeric_bounds(attr: AttributeSpec) -> tuple[float | None, float | None]:
    """Infer numeric min/max from distribution bounds and hard constraints."""
    min_val = None
    max_val = None

    dist = attr.sampling.distribution
    if isinstance(
        dist,
        (
            NormalDistribution,
            LognormalDistribution,
            UniformDistribution,
            BetaDistribution,
        ),
    ):
        min_val = dist.min
        max_val = dist.max

    for constraint in attr.constraints:
        if constraint.type == "hard_min" and constraint.value is not None:
            if min_val is None:
                min_val = float(constraint.value)
            else:
                min_val = max(min_val, float(constraint.value))
        elif constraint.type == "hard_max" and constraint.value is not None:
            if max_val is None:
                max_val = float(constraint.value)
            else:
                max_val = min(max_val, float(constraint.value))

    return min_val, max_val


def _extract_numeric_constants(expr: str) -> set[float]:
    """Extract numeric constants from an expression AST."""
    out: set[float] = set()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                continue
            out.add(float(node.value))
    return out
