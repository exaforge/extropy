"""LLM response validation for fail-fast error detection.

This module provides immediate syntax validation after each LLM call,
catching errors early so we can retry in-place instead of failing
after 10+ minutes of pipeline execution.

The philosophy is FAIL FAST:
- Validate immediately after each LLM response
- Feed errors back to LLM for self-correction
- Never proceed with invalid data
"""

import re
from typing import Any

from ...core.models.validation import (
    Severity,
    ValidationIssue,
    ValidationResult,
)
from ...utils.expressions import validate_expression_syntax


def _make_error(
    field: str,
    value: str,
    error: str,
    suggestion: str | None = None,
) -> ValidationIssue:
    """Create a ValidationIssue for LLM response validation."""
    return ValidationIssue(
        severity=Severity.ERROR,
        category="LLM_RESPONSE",
        location=field,
        message=error,
        suggestion=suggestion,
        value=value,
    )


# Spec-level variable patterns that should use spec_expression, not expression
SPEC_LEVEL_PATTERNS = {"weights", "options"}


def is_spec_level_constraint(expression: str) -> bool:
    """Check if a constraint expression references spec-level variables.

    Spec-level constraints validate the YAML spec itself (e.g., weights sum to 1),
    not individual sampled agents. These should use type='spec_expression'.
    """
    if "sum(weights)" in expression:
        return True
    if "len(options)" in expression:
        return True
    if "weights[" in expression:
        return True
    if "options[" in expression:
        return True

    # Check if expression references spec-level variable names directly
    # Simple token extraction for the most common cases
    tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression))
    if tokens & SPEC_LEVEL_PATTERNS:
        return True

    return False


def extract_bound_from_constraint(
    expression: str,
    attr_name: str,
) -> tuple[str | None, str | None, bool]:
    """Extract bound expression from a constraint.

    Parses simple inequality constraints to extract the bound expression.

    Returns:
        Tuple of (bound_type, bound_expr, is_strict) where:
        - bound_type is "max" or "min" or None if not a simple bound
        - bound_expr is the expression for the bound
        - is_strict is True for < or > (strict inequality)
    """
    expr = expression.strip()
    escaped_name = re.escape(attr_name)

    # Upper bound patterns
    upper_patterns = [
        (rf"^{escaped_name}\s*<=\s*(.+)$", False),
        (rf"^{escaped_name}\s*<\s*(.+)$", True),
        (rf"^(.+)\s*>=\s*{escaped_name}$", False),
        (rf"^(.+)\s*>\s*{escaped_name}$", True),
    ]

    # Lower bound patterns
    lower_patterns = [
        (rf"^{escaped_name}\s*>=\s*(.+)$", False),
        (rf"^{escaped_name}\s*>\s*(.+)$", True),
        (rf"^(.+)\s*<=\s*{escaped_name}$", False),
        (rf"^(.+)\s*<\s*{escaped_name}$", True),
    ]

    for pattern, is_strict in upper_patterns:
        match = re.match(pattern, expr)
        if match:
            return ("max", match.group(1).strip(), is_strict)

    for pattern, is_strict in lower_patterns:
        match = re.match(pattern, expr)
        if match:
            return ("min", match.group(1).strip(), is_strict)

    return (None, None, False)


# =============================================================================
# Formula Validation
# =============================================================================


def validate_formula_syntax(
    formula: str | None, field_name: str = "formula"
) -> ValidationIssue | None:
    """Validate a Python expression/formula for syntax errors.

    Returns None if valid, ValidationIssue if invalid.
    """
    if not formula:
        return None

    error_msg = validate_expression_syntax(formula)
    if error_msg:
        return _make_error(
            field=field_name,
            value=formula,
            error=error_msg,
            suggestion="Ensure the formula is a valid Python expression",
        )

    return None


def validate_condition_syntax(
    condition: str | None, field_name: str = "when"
) -> ValidationIssue | None:
    """Validate a 'when' condition for syntax errors.

    Conditions are Python boolean expressions like:
    - age > 50
    - specialty == 'cardiology'
    - role in ['senior', 'chief']

    Returns None if valid, ValidationIssue if invalid.
    """
    return validate_formula_syntax(condition, field_name)


# =============================================================================
# Distribution Validation
# =============================================================================


def validate_distribution_data(
    dist_data: dict[str, Any],
    attr_name: str,
    attr_type: str,
) -> list[ValidationIssue]:
    """Validate distribution data from LLM response.

    Checks for:
    - Valid distribution type
    - Required parameters present
    - Parameter values in valid ranges
    - Formula syntax (for mean_formula, std_formula)
    """
    errors = []

    if not dist_data:
        errors.append(
            _make_error(
                field=f"{attr_name}.distribution",
                value="null",
                error="distribution is missing",
                suggestion=f"Provide a distribution object for {attr_name}",
            )
        )
        return errors

    dist_type = dist_data.get("type")

    if dist_type is None:
        errors.append(
            _make_error(
                field=f"{attr_name}.distribution.type",
                value="null",
                error="distribution type is missing",
                suggestion="Specify type: normal, lognormal, uniform, beta, categorical, or boolean",
            )
        )
        return errors

    valid_types = {"normal", "lognormal", "uniform", "beta", "categorical", "boolean"}
    if dist_type not in valid_types:
        errors.append(
            _make_error(
                field=f"{attr_name}.distribution.type",
                value=dist_type,
                error="unknown distribution type",
                suggestion=f"Use one of: {', '.join(sorted(valid_types))}",
            )
        )
        return errors

    # Type-specific validation
    if dist_type in ("normal", "lognormal"):
        # Check mean_formula syntax
        mean_formula = dist_data.get("mean_formula")
        if mean_formula:
            err = validate_formula_syntax(
                mean_formula, f"{attr_name}.distribution.mean_formula"
            )
            if err:
                errors.append(err)

        # Check std_formula syntax
        std_formula = dist_data.get("std_formula")
        if std_formula:
            err = validate_formula_syntax(
                std_formula, f"{attr_name}.distribution.std_formula"
            )
            if err:
                errors.append(err)

        # Check std is positive if present
        std = dist_data.get("std")
        if std is not None and std < 0:
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.std",
                    value=str(std),
                    error="standard deviation cannot be negative",
                    suggestion="Use a positive value for std",
                )
            )

        # Check min < max if both present
        min_val = dist_data.get("min")
        max_val = dist_data.get("max")
        if min_val is not None and max_val is not None and min_val >= max_val:
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.min/max",
                    value=f"min={min_val}, max={max_val}",
                    error="min must be less than max",
                    suggestion="Swap min and max values",
                )
            )

    elif dist_type == "beta":
        alpha = dist_data.get("alpha")
        beta = dist_data.get("beta")

        if alpha is None or alpha <= 0:
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.alpha",
                    value=str(alpha),
                    error="alpha must be positive",
                    suggestion="Use a positive value like 2.0",
                )
            )

        if beta is None or beta <= 0:
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.beta",
                    value=str(beta),
                    error="beta must be positive",
                    suggestion="Use a positive value like 5.0",
                )
            )

    elif dist_type == "uniform":
        min_val = dist_data.get("min")
        max_val = dist_data.get("max")

        if min_val is not None and max_val is not None and min_val >= max_val:
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.min/max",
                    value=f"min={min_val}, max={max_val}",
                    error="min must be less than max",
                    suggestion="Swap min and max values",
                )
            )

    elif dist_type == "categorical":
        options = dist_data.get("options")
        weights = dist_data.get("weights")

        if not options:
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.options",
                    value="null or empty",
                    error="categorical distribution requires options",
                    suggestion="Provide an array of string options",
                )
            )
        elif weights and len(weights) != len(options):
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.weights",
                    value=f"{len(weights)} weights, {len(options)} options",
                    error="weights and options arrays must have same length",
                    suggestion="Ensure one weight per option",
                )
            )
        elif weights:
            weight_sum = sum(weights)
            if abs(weight_sum - 1.0) > 0.02:
                errors.append(
                    _make_error(
                        field=f"{attr_name}.distribution.weights",
                        value=f"sum={weight_sum:.3f}",
                        error="weights must sum to 1.0",
                        suggestion="Normalize weights to sum to 1.0",
                    )
                )

    elif dist_type == "boolean":
        prob = dist_data.get("probability_true")
        if prob is not None and (prob < 0 or prob > 1):
            errors.append(
                _make_error(
                    field=f"{attr_name}.distribution.probability_true",
                    value=str(prob),
                    error="probability must be between 0 and 1",
                    suggestion="Use a value like 0.5 or 0.75",
                )
            )

    return errors


# =============================================================================
# Modifier Validation
# =============================================================================


def validate_modifier_data(
    modifier_data: dict[str, Any],
    attr_name: str,
    modifier_index: int,
    dist_type: str | None = None,
) -> list[ValidationIssue]:
    """Validate a single modifier from LLM response.

    Checks for:
    - Valid 'when' condition syntax
    - Appropriate modifier fields for distribution type
    - Valid value ranges
    """
    errors = []

    # Validate 'when' condition
    when = modifier_data.get("when")
    if when:
        err = validate_condition_syntax(
            when, f"{attr_name}.modifiers[{modifier_index}].when"
        )
        if err:
            errors.append(err)
    else:
        errors.append(
            _make_error(
                field=f"{attr_name}.modifiers[{modifier_index}].when",
                value="null",
                error="modifier missing 'when' condition",
                suggestion='Provide a condition like "age > 50" or "role == \'senior\'"',
            )
        )

    # Check for type/field compatibility if we know the distribution type
    if dist_type:
        numeric_types = {"normal", "lognormal", "uniform", "beta"}

        if dist_type in numeric_types:
            # Numeric: should use multiply/add, not weight_overrides
            if modifier_data.get("weight_overrides"):
                errors.append(
                    _make_error(
                        field=f"{attr_name}.modifiers[{modifier_index}].weight_overrides",
                        value="present",
                        error=f"cannot use weight_overrides with {dist_type} distribution",
                        suggestion="Use multiply and/or add instead",
                    )
                )
            if modifier_data.get("probability_override") is not None:
                errors.append(
                    _make_error(
                        field=f"{attr_name}.modifiers[{modifier_index}].probability_override",
                        value="present",
                        error=f"cannot use probability_override with {dist_type} distribution",
                        suggestion="Use multiply and/or add instead",
                    )
                )

        elif dist_type == "categorical":
            # Categorical: should use weight_overrides, not multiply/add
            multiply = modifier_data.get("multiply")
            add = modifier_data.get("add")
            if (multiply is not None and multiply != 1.0) or (
                add is not None and add != 0
            ):
                errors.append(
                    _make_error(
                        field=f"{attr_name}.modifiers[{modifier_index}]",
                        value=f"multiply={multiply}, add={add}",
                        error="cannot use multiply/add with categorical distribution",
                        suggestion="Use weight_overrides instead",
                    )
                )

        elif dist_type == "boolean":
            # Boolean: should use probability_override
            multiply = modifier_data.get("multiply")
            add = modifier_data.get("add")
            if (multiply is not None and multiply != 1.0) or (
                add is not None and add != 0
            ):
                errors.append(
                    _make_error(
                        field=f"{attr_name}.modifiers[{modifier_index}]",
                        value=f"multiply={multiply}, add={add}",
                        error="cannot use multiply/add with boolean distribution",
                        suggestion="Use probability_override instead",
                    )
                )

    # Validate probability_override range
    prob_override = modifier_data.get("probability_override")
    if prob_override is not None and (prob_override < 0 or prob_override > 1):
        errors.append(
            _make_error(
                field=f"{attr_name}.modifiers[{modifier_index}].probability_override",
                value=str(prob_override),
                error="probability_override must be between 0 and 1",
                suggestion="Use a value like 0.75",
            )
        )

    # Validate weight_overrides sum to 1.0
    weight_overrides = modifier_data.get("weight_overrides")
    if weight_overrides and isinstance(weight_overrides, dict):
        weight_sum = sum(weight_overrides.values())
        if abs(weight_sum - 1.0) > 0.02:
            errors.append(
                _make_error(
                    field=f"{attr_name}.modifiers[{modifier_index}].weight_overrides",
                    value=f"sum={weight_sum:.3f}",
                    error="weight_overrides must sum to 1.0",
                    suggestion="Normalize weights to sum to 1.0",
                )
            )

    return errors


# =============================================================================
# Full Response Validation
# =============================================================================


def validate_independent_response(
    data: dict[str, Any],
    expected_attrs: list[str],
) -> ValidationResult:
    """Validate LLM response for independent attribute hydration."""
    errors = []

    attributes = data.get("attributes", [])

    for attr_data in attributes:
        name = attr_data.get("name", "unknown")
        dist_data = attr_data.get("distribution", {})

        # Skip if name not in expected (will be filtered out anyway)
        if name not in expected_attrs:
            continue

        # Validate distribution
        dist_errors = validate_distribution_data(dist_data, name, "numeric")
        errors.extend(dist_errors)

        # Validate constraints for spec-level expressions with wrong type
        constraints = attr_data.get("constraints", [])
        for constraint in constraints:
            c_type = constraint.get("type")
            c_expr = constraint.get("expression")
            if c_type == "expression" and c_expr:
                if is_spec_level_constraint(c_expr):
                    errors.append(
                        _make_error(
                            field=f"{name}.constraints",
                            value=c_expr,
                            error="constraint references spec-level variables (weights/options) but uses type='expression'",
                            suggestion="Change to type='spec_expression' — this validates the YAML spec itself, not individual agents",
                        )
                    )

    return ValidationResult(issues=errors)


def validate_derived_response(
    data: dict[str, Any],
    expected_attrs: list[str],
) -> ValidationResult:
    """Validate LLM response for derived attribute hydration."""
    errors = []

    attributes = data.get("attributes", [])

    for attr_data in attributes:
        name = attr_data.get("name", "unknown")

        if name not in expected_attrs:
            continue

        formula = attr_data.get("formula")

        if not formula:
            errors.append(
                _make_error(
                    field=f"{name}.formula",
                    value="null",
                    error="derived attribute requires formula",
                    suggestion='Provide a Python expression like "age - 28" or "income * 0.3"',
                )
            )
        else:
            err = validate_formula_syntax(formula, f"{name}.formula")
            if err:
                errors.append(err)

    return ValidationResult(issues=errors)


def validate_conditional_base_response(
    data: dict[str, Any],
    expected_attrs: list[str],
) -> ValidationResult:
    """Validate LLM response for conditional base distribution hydration."""
    errors = []

    attributes = data.get("attributes", [])

    for attr_data in attributes:
        name = attr_data.get("name", "unknown")

        if name not in expected_attrs:
            continue

        dist_data = attr_data.get("distribution", {})
        dist_errors = validate_distribution_data(dist_data, name, "numeric")
        errors.extend(dist_errors)

        # Check constraints for issues
        constraints = attr_data.get("constraints", [])
        dist_type = dist_data.get("type") if dist_data else None

        for constraint in constraints:
            c_type = constraint.get("type")
            c_expr = constraint.get("expression")

            if c_type == "expression" and c_expr:
                # Check for spec-level constraints with wrong type
                if is_spec_level_constraint(c_expr):
                    errors.append(
                        _make_error(
                            field=f"{name}.constraints",
                            value=c_expr,
                            error="constraint references spec-level variables (weights/options) but uses type='expression'",
                            suggestion="Change to type='spec_expression'",
                        )
                    )
                    continue

                # Check for missing formula bounds (only for numeric distributions)
                if dist_type in ("normal", "lognormal", "beta"):
                    bound_type, bound_expr, is_strict = extract_bound_from_constraint(
                        c_expr, name
                    )

                    if bound_type == "max" and bound_expr:
                        has_max_formula = dist_data.get("max_formula") is not None
                        has_static_max = dist_data.get("max") is not None
                        if not has_max_formula and not has_static_max:
                            errors.append(
                                _make_error(
                                    field=f"{name}.distribution",
                                    value=f"constraint '{c_expr}'",
                                    error="constraint exists but distribution has no max_formula to enforce it during sampling",
                                    suggestion=f"Add to distribution: max_formula: '{bound_expr}'",
                                )
                            )
                    elif bound_type == "min" and bound_expr:
                        has_min_formula = dist_data.get("min_formula") is not None
                        has_static_min = dist_data.get("min") is not None
                        if not has_min_formula and not has_static_min:
                            errors.append(
                                _make_error(
                                    field=f"{name}.distribution",
                                    value=f"constraint '{c_expr}'",
                                    error="constraint exists but distribution has no min_formula to enforce it during sampling",
                                    suggestion=f"Add to distribution: min_formula: '{bound_expr}'",
                                )
                            )

    return ValidationResult(issues=errors)


def validate_modifiers_response(
    data: dict[str, Any],
    attr_dist_types: dict[str, str],
) -> ValidationResult:
    """Validate LLM response for conditional modifiers hydration.

    Args:
        data: LLM response data
        attr_dist_types: Mapping of attribute name to distribution type
    """
    errors = []

    attributes = data.get("attributes", [])

    for attr_data in attributes:
        name = attr_data.get("name", "unknown")
        modifiers = attr_data.get("modifiers", [])

        dist_type = attr_dist_types.get(name)

        for i, mod_data in enumerate(modifiers):
            mod_errors = validate_modifier_data(mod_data, name, i, dist_type)
            errors.extend(mod_errors)

    return ValidationResult(issues=errors)
