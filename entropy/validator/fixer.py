"""Auto-fix utility for modifier condition option references.

This module fixes a common LLM error where modifier `when` conditions
reference categorical options with inconsistent naming conventions
(e.g., 'University hospital' instead of 'University_hospital').

The fixer:
1. Builds a map of all valid categorical options across attributes
2. Extracts string literals from `when` clauses using AST
3. Fuzzy-matches literals against valid options
4. Rewrites conditions with correct option names
"""

import ast
import re
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..models import (
    PopulationSpec,
    AttributeSpec,
    CategoricalDistribution,
    Modifier,
)


@dataclass
class ConditionFix:
    """A single fix applied to a modifier condition."""

    attribute: str
    modifier_index: int
    original_value: str
    fixed_value: str
    original_condition: str
    fixed_condition: str
    confidence: float  # 0-1, based on fuzzy match score


@dataclass
class FixResult:
    """Result of fixing a spec's modifier conditions."""

    spec: PopulationSpec
    fixes: list[ConditionFix]
    unfixable: list[str]  # Conditions that couldn't be confidently fixed

    @property
    def fix_count(self) -> int:
        return len(self.fixes)

    def summary(self) -> str:
        """Get a human-readable summary of fixes."""
        if not self.fixes:
            return "No fixes needed"

        lines = [f"Applied {len(self.fixes)} fix(es):"]
        for fix in self.fixes:
            lines.append(
                f"  - {fix.attribute}[{fix.modifier_index}]: "
                f"'{fix.original_value}' → '{fix.fixed_value}' "
                f"(confidence: {fix.confidence:.0%})"
            )

        if self.unfixable:
            lines.append(f"\n{len(self.unfixable)} unfixable issue(s):")
            for issue in self.unfixable:
                lines.append(f"  - {issue}")

        return "\n".join(lines)


def fix_modifier_conditions(
    spec: PopulationSpec,
    min_confidence: float = 0.6,
) -> FixResult:
    """
    Fix modifier condition option references to match actual option values.

    This function:
    1. Builds a map of attribute_name -> set of valid options
    2. For each modifier `when` clause, extracts string literals
    3. Fuzzy-matches literals against valid options for referenced attributes
    4. Rewrites the condition with correct option names

    Args:
        spec: The PopulationSpec to fix
        min_confidence: Minimum fuzzy match confidence (0-1) to apply fix

    Returns:
        FixResult with the fixed spec and list of applied fixes
    """
    # Build map of attribute -> valid options
    option_map = _build_option_map(spec.attributes)

    # Deep copy spec to avoid mutation
    fixed_spec = deepcopy(spec)

    fixes: list[ConditionFix] = []
    unfixable: list[str] = []

    # Process each attribute's modifiers
    for attr in fixed_spec.attributes:
        if not attr.sampling.modifiers:
            continue

        for i, mod in enumerate(attr.sampling.modifiers):
            if not mod.when:
                continue

            # Get the set of valid options for attributes in depends_on
            valid_options = set()
            for dep_name in attr.sampling.depends_on:
                if dep_name in option_map:
                    valid_options.update(option_map[dep_name])

            if not valid_options:
                # No categorical dependencies, skip
                continue

            # Extract string literals from condition and try to fix them
            fixed_condition, condition_fixes = _fix_condition(
                mod.when,
                valid_options,
                min_confidence,
            )

            if condition_fixes:
                # Apply the fix
                mod.when = fixed_condition
                for orig, fixed, conf in condition_fixes:
                    fixes.append(
                        ConditionFix(
                            attribute=attr.name,
                            modifier_index=i,
                            original_value=orig,
                            fixed_value=fixed,
                            original_condition=mod.when,
                            fixed_condition=fixed_condition,
                            confidence=conf,
                        )
                    )

            # Check for any remaining potential issues
            literals = _extract_string_literals(fixed_condition)
            for lit in literals:
                if lit not in valid_options and not _is_likely_non_option(lit):
                    # Check if it's close but below threshold
                    best_match, score = _find_best_match(lit, valid_options)
                    if best_match and score < min_confidence:
                        unfixable.append(
                            f"{attr.name}[{i}]: '{lit}' might be '{best_match}' "
                            f"(score: {score:.0%} < {min_confidence:.0%})"
                        )

    return FixResult(
        spec=fixed_spec,
        fixes=fixes,
        unfixable=unfixable,
    )


def _build_option_map(attributes: list[AttributeSpec]) -> dict[str, set[str]]:
    """Build a map of attribute name -> set of valid option values."""
    option_map = {}
    for attr in attributes:
        dist = attr.sampling.distribution
        if isinstance(dist, CategoricalDistribution) and dist.options:
            option_map[attr.name] = set(dist.options)
    return option_map


def _extract_string_literals(condition: str) -> list[str]:
    """Extract all string literals from a Python expression."""
    literals = []
    try:
        tree = ast.parse(condition, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.append(node.value)
    except SyntaxError:
        # Fallback: regex extraction
        # Match single or double quoted strings
        for match in re.finditer(r"['\"]([^'\"]+)['\"]", condition):
            literals.append(match.group(1))
    return literals


def _normalize_for_matching(s: str) -> str:
    """Normalize a string for fuzzy matching.

    Converts to lowercase and replaces separators with spaces.
    """
    # Replace underscores, slashes, and hyphens with spaces
    normalized = re.sub(r"[_/\-]", " ", s)
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower().strip()


def _fuzzy_match_score(s1: str, s2: str) -> float:
    """Calculate fuzzy match score between two strings.

    Uses normalized comparison for better matching across
    different naming conventions.
    """
    # Direct match
    if s1 == s2:
        return 1.0

    # Normalize and compare
    n1 = _normalize_for_matching(s1)
    n2 = _normalize_for_matching(s2)

    if n1 == n2:
        return 0.95  # Very high confidence for normalized match

    # Use SequenceMatcher on normalized strings
    return SequenceMatcher(None, n1, n2).ratio()


def _find_best_match(
    target: str,
    candidates: set[str],
) -> tuple[str | None, float]:
    """Find the best matching candidate for a target string."""
    if not candidates:
        return None, 0.0

    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = _fuzzy_match_score(target, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match, best_score


def _is_likely_non_option(literal: str) -> bool:
    """Check if a string literal is likely NOT an option reference.

    Returns True for things like operators, common values, etc.
    """
    # Common non-option literals
    non_options = {
        "true",
        "false",
        "none",
        "and",
        "or",
        "not",
        "in",
        "is",
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
    }
    return literal.lower() in non_options


def _fix_condition(
    condition: str,
    valid_options: set[str],
    min_confidence: float,
) -> tuple[str, list[tuple[str, str, float]]]:
    """Fix string literals in a condition to match valid options.

    Returns:
        Tuple of (fixed_condition, list of (original, fixed, confidence))
    """
    literals = _extract_string_literals(condition)
    fixes: list[tuple[str, str, float]] = []
    fixed_condition = condition

    for lit in literals:
        # Skip if already a valid option
        if lit in valid_options:
            continue

        # Skip likely non-option literals
        if _is_likely_non_option(lit):
            continue

        # Find best match
        best_match, score = _find_best_match(lit, valid_options)

        if best_match and score >= min_confidence:
            # Apply the fix - replace the literal in the condition
            # We need to be careful to replace the exact string including quotes
            for quote in ['"', "'"]:
                old_str = f"{quote}{lit}{quote}"
                new_str = f"{quote}{best_match}{quote}"
                if old_str in fixed_condition:
                    fixed_condition = fixed_condition.replace(old_str, new_str, 1)
                    fixes.append((lit, best_match, score))
                    break

    return fixed_condition, fixes


def fix_spec_file(
    input_path: str,
    output_path: str | None = None,
    min_confidence: float = 0.6,
    dry_run: bool = False,
) -> FixResult:
    """
    Fix a spec YAML file and optionally write the result.

    Args:
        input_path: Path to input YAML spec
        output_path: Path to output (defaults to input_path if not provided)
        min_confidence: Minimum fuzzy match confidence to apply fix
        dry_run: If True, don't write output file

    Returns:
        FixResult with the fixed spec and list of applied fixes
    """
    from pathlib import Path

    spec = PopulationSpec.from_yaml(input_path)
    result = fix_modifier_conditions(spec, min_confidence)

    if not dry_run and result.fix_count > 0:
        out_path = output_path or input_path
        result.spec.to_yaml(Path(out_path))

    return result
