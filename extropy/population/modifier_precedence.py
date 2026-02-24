"""Deterministic precedence rules for overlapping conditional modifiers.

Precedence policy:
1. Subset wins (stricter condition wins)
2. Higher specificity wins
3. Declaration order wins (higher index)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Literal


PrecedenceReason = Literal["single", "subset", "specificity", "order"]


@dataclass(frozen=True)
class ModifierPrecedenceDecision:
    """Winner selection result for overlapping modifiers."""

    winner_index: int
    reason: PrecedenceReason


@dataclass
class _VarConstraint:
    """Normalized constraints for one variable in a condition expression."""

    allowed: set[Any] | None = None
    disallowed: set[Any] = field(default_factory=set)
    lower: tuple[float, bool] | None = None  # (value, inclusive)
    upper: tuple[float, bool] | None = None  # (value, inclusive)

    def apply_eq(self, value: Any) -> None:
        values = {value}
        self.allowed = values if self.allowed is None else self.allowed & values

    def apply_in(self, values: set[Any]) -> None:
        self.allowed = values if self.allowed is None else self.allowed & values

    def apply_neq(self, value: Any) -> None:
        self.disallowed.add(value)

    def apply_not_in(self, values: set[Any]) -> None:
        self.disallowed.update(values)

    def apply_lower(self, value: float, inclusive: bool) -> None:
        if self.lower is None:
            self.lower = (value, inclusive)
            return
        current_value, current_inclusive = self.lower
        if value > current_value:
            self.lower = (value, inclusive)
        elif value == current_value and not inclusive and current_inclusive:
            self.lower = (value, inclusive)

    def apply_upper(self, value: float, inclusive: bool) -> None:
        if self.upper is None:
            self.upper = (value, inclusive)
            return
        current_value, current_inclusive = self.upper
        if value < current_value:
            self.upper = (value, inclusive)
        elif value == current_value and not inclusive and current_inclusive:
            self.upper = (value, inclusive)

    def effective_allowed(self) -> set[Any] | None:
        if self.allowed is None:
            return None
        return {value for value in self.allowed if value not in self.disallowed}

    def numeric_range_width(self) -> float | None:
        if self.lower is None or self.upper is None:
            return None
        return self.upper[0] - self.lower[0]


@dataclass
class _WhenSummary:
    raw: str
    parseable: bool
    unconditional: bool
    always_false: bool
    clause_count: int
    constraints: dict[str, _VarConstraint]


def choose_modifier_precedence(
    conditions: list[tuple[int, str]],
) -> ModifierPrecedenceDecision | None:
    """Choose one winning modifier index from matching conditions."""
    if not conditions:
        return None
    if len(conditions) == 1:
        return ModifierPrecedenceDecision(
            winner_index=conditions[0][0], reason="single"
        )

    summaries = {idx: _parse_when(expr) for idx, expr in conditions}
    subset_balance: dict[int, int] = {idx: 0 for idx, _ in conditions}

    for a, b in _iter_pairs([idx for idx, _ in conditions]):
        a_implies_b = _when_implies(summaries[a], summaries[b])
        b_implies_a = _when_implies(summaries[b], summaries[a])
        if a_implies_b and not b_implies_a:
            subset_balance[a] += 1
            subset_balance[b] -= 1
        elif b_implies_a and not a_implies_b:
            subset_balance[b] += 1
            subset_balance[a] -= 1

    specificity_scores = {
        idx: _specificity_score(summaries[idx]) for idx, _ in conditions
    }
    ranked = sorted(
        [idx for idx, _ in conditions],
        key=lambda idx: (
            subset_balance[idx],
            specificity_scores[idx],
            idx,
        ),
    )
    winner = ranked[-1]
    runner_up = ranked[-2]

    if subset_balance[winner] > subset_balance[runner_up]:
        reason: PrecedenceReason = "subset"
    elif specificity_scores[winner] > specificity_scores[runner_up]:
        reason = "specificity"
    else:
        reason = "order"

    return ModifierPrecedenceDecision(winner_index=winner, reason=reason)


def _iter_pairs(indices: list[int]):
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            yield indices[i], indices[j]


def _parse_when(expr: str) -> _WhenSummary:
    text = (expr or "").strip()
    if not text:
        return _WhenSummary(
            raw=text,
            parseable=False,
            unconditional=False,
            always_false=False,
            clause_count=0,
            constraints={},
        )

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return _WhenSummary(
            raw=text,
            parseable=False,
            unconditional=False,
            always_false=False,
            clause_count=0,
            constraints={},
        )

    constraints: dict[str, _VarConstraint] = {}
    clause_count = 0
    parseable = True
    always_false = False

    def add_clause() -> None:
        nonlocal clause_count
        clause_count += 1

    def get_constraint(name: str) -> _VarConstraint:
        return constraints.setdefault(name, _VarConstraint())

    def visit(node: ast.AST) -> bool:
        nonlocal always_false

        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, ast.And):
                return False
            return all(visit(value) for value in node.values)

        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            if node.value is False:
                always_false = True
            return True

        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for op, left_node, right_node in zip(
                node.ops, operands[:-1], operands[1:], strict=False
            ):
                if not _apply_comparison(
                    left_node,
                    op,
                    right_node,
                    get_constraint,
                    add_clause,
                ):
                    return False
            return True

        return False

    parseable = visit(tree.body)
    if not parseable:
        constraints = {}
        clause_count = 0

    unconditional = parseable and clause_count == 0 and not always_false

    return _WhenSummary(
        raw=text,
        parseable=parseable,
        unconditional=unconditional,
        always_false=always_false,
        clause_count=clause_count,
        constraints=constraints,
    )


def _apply_comparison(
    left_node: ast.AST,
    op: ast.cmpop,
    right_node: ast.AST,
    get_constraint,
    add_clause,
) -> bool:
    left_name = _name_from_node(left_node)
    right_name = _name_from_node(right_node)

    if left_name and not right_name:
        return _apply_name_to_value_comparison(
            left_name, op, right_node, get_constraint, add_clause
        )

    if right_name and not left_name:
        reversed_op = _reverse_compare_op(op)
        if reversed_op is None:
            return False
        return _apply_name_to_value_comparison(
            right_name,
            reversed_op,
            left_node,
            get_constraint,
            add_clause,
        )

    return False


def _apply_name_to_value_comparison(
    name: str,
    op: ast.cmpop,
    value_node: ast.AST,
    get_constraint,
    add_clause,
) -> bool:
    constraint = get_constraint(name)

    if isinstance(op, ast.In):
        values = _literal_set_from_node(value_node)
        if values is None:
            return False
        constraint.apply_in(values)
        add_clause()
        return True

    if isinstance(op, ast.NotIn):
        values = _literal_set_from_node(value_node)
        if values is None:
            return False
        constraint.apply_not_in(values)
        add_clause()
        return True

    value = _literal_from_node(value_node)
    if value is None:
        return False

    if isinstance(op, ast.Eq):
        constraint.apply_eq(value)
        add_clause()
        return True
    if isinstance(op, ast.NotEq):
        constraint.apply_neq(value)
        add_clause()
        return True

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False

    numeric = float(value)
    if isinstance(op, ast.Gt):
        constraint.apply_lower(numeric, inclusive=False)
        add_clause()
        return True
    if isinstance(op, ast.GtE):
        constraint.apply_lower(numeric, inclusive=True)
        add_clause()
        return True
    if isinstance(op, ast.Lt):
        constraint.apply_upper(numeric, inclusive=False)
        add_clause()
        return True
    if isinstance(op, ast.LtE):
        constraint.apply_upper(numeric, inclusive=True)
        add_clause()
        return True

    return False


def _name_from_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _literal_from_node(node: ast.AST) -> Any | None:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _literal_set_from_node(node: ast.AST) -> set[Any] | None:
    if isinstance(node, ast.Constant):
        return {node.value}

    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None

    values: set[Any] = set()
    for elt in node.elts:
        if not isinstance(elt, ast.Constant):
            return None
        values.add(elt.value)
    return values


def _reverse_compare_op(op: ast.cmpop) -> ast.cmpop | None:
    mapping: dict[type[ast.cmpop], ast.cmpop] = {
        ast.Lt: ast.Gt(),
        ast.LtE: ast.GtE(),
        ast.Gt: ast.Lt(),
        ast.GtE: ast.LtE(),
        ast.Eq: ast.Eq(),
        ast.NotEq: ast.NotEq(),
        ast.In: ast.In(),
        ast.NotIn: ast.NotIn(),
    }
    return mapping.get(type(op))


def _when_implies(left: _WhenSummary, right: _WhenSummary) -> bool:
    if right.unconditional:
        return True
    if left.always_false:
        return True
    if right.always_false:
        return left.always_false
    if left.unconditional:
        return right.unconditional
    if not left.parseable or not right.parseable:
        return False

    for var_name, right_constraint in right.constraints.items():
        left_constraint = left.constraints.get(var_name)
        if left_constraint is None:
            return False
        if not _constraint_implies(left_constraint, right_constraint):
            return False

    return True


def _constraint_implies(left: _VarConstraint, right: _VarConstraint) -> bool:
    left_allowed = left.effective_allowed()
    right_allowed = right.effective_allowed()

    if right_allowed is not None:
        if left_allowed is None:
            return False
        if not left_allowed.issubset(right_allowed):
            return False

    if right.lower is not None or right.upper is not None:
        if left_allowed is not None:
            if not left_allowed:
                return True
            for value in left_allowed:
                if not _value_satisfies_bounds(value, right.lower, right.upper):
                    return False
        elif not _bounds_imply(
            left.lower,
            left.upper,
            right.lower,
            right.upper,
        ):
            return False

    for disallowed in right.disallowed:
        if _value_possible_under_constraint(disallowed, left):
            return False

    return True


def _value_possible_under_constraint(value: Any, constraint: _VarConstraint) -> bool:
    allowed = constraint.effective_allowed()
    if allowed is not None and value not in allowed:
        return False
    if value in constraint.disallowed:
        return False
    if not _value_satisfies_bounds(value, constraint.lower, constraint.upper):
        return False
    return True


def _value_satisfies_bounds(
    value: Any,
    lower: tuple[float, bool] | None,
    upper: tuple[float, bool] | None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return lower is None and upper is None

    numeric = float(value)
    if lower is not None:
        lower_value, lower_inclusive = lower
        if lower_inclusive:
            if numeric < lower_value:
                return False
        elif numeric <= lower_value:
            return False
    if upper is not None:
        upper_value, upper_inclusive = upper
        if upper_inclusive:
            if numeric > upper_value:
                return False
        elif numeric >= upper_value:
            return False
    return True


def _bounds_imply(
    left_lower: tuple[float, bool] | None,
    left_upper: tuple[float, bool] | None,
    right_lower: tuple[float, bool] | None,
    right_upper: tuple[float, bool] | None,
) -> bool:
    if right_lower is not None:
        if left_lower is None:
            return False
        if not _lower_bound_implies(left_lower, right_lower):
            return False
    if right_upper is not None:
        if left_upper is None:
            return False
        if not _upper_bound_implies(left_upper, right_upper):
            return False
    return True


def _lower_bound_implies(
    left: tuple[float, bool],
    right: tuple[float, bool],
) -> bool:
    left_value, left_inclusive = left
    right_value, right_inclusive = right

    if left_value > right_value:
        return True
    if left_value < right_value:
        return False

    if right_inclusive:
        return True
    return not left_inclusive


def _upper_bound_implies(
    left: tuple[float, bool],
    right: tuple[float, bool],
) -> bool:
    left_value, left_inclusive = left
    right_value, right_inclusive = right

    if left_value < right_value:
        return True
    if left_value > right_value:
        return False

    if right_inclusive:
        return True
    return not left_inclusive


def _specificity_score(summary: _WhenSummary) -> int:
    if summary.always_false:
        return 1_000_000

    if not summary.parseable:
        return _fallback_specificity_score(summary.raw)

    score = summary.clause_count * 20
    for constraint in summary.constraints.values():
        allowed = constraint.effective_allowed()
        if allowed is not None:
            size = max(1, len(allowed))
            score += max(0, 60 - min(size, 10) * 6)
        score += len(constraint.disallowed) * 4
        if constraint.lower is not None:
            score += 9
        if constraint.upper is not None:
            score += 9
        if constraint.lower is not None and constraint.upper is not None:
            width = constraint.numeric_range_width()
            if width is not None:
                if width <= 0:
                    score += 30
                elif width <= 1:
                    score += 20
                elif width <= 5:
                    score += 14
                elif width <= 20:
                    score += 9
                elif width <= 100:
                    score += 5
    return score


def _fallback_specificity_score(expr: str) -> int:
    text = expr.strip().lower()
    if not text:
        return 0

    score = 0
    score += text.count(" and ") * 6
    score += text.count("==") * 4
    score += text.count(" in ") * 4
    score += text.count("!=") * 3
    score += text.count(">=") * 3
    score += text.count("<=") * 3
    score += text.count(">") * 2
    score += text.count("<") * 2

    # Light lexical fallback so longer, denser expressions rank as more specific.
    token_count = len(re.findall(r"[a-z_][a-z0-9_]*", text))
    score += token_count
    return score
