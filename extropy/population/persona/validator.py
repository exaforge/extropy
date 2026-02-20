"""Persona config validation against merged population attributes."""

from __future__ import annotations

import re

from ...core.models import PopulationSpec, CategoricalDistribution
from ...core.models.validation import ValidationResult
from .config import PersonaConfig
from .renderer import extract_intro_attributes


def _normalize_token(value: str) -> str:
    token = value.strip().lower()
    token = re.sub(r"[\s\-]+", "_", token)
    token = re.sub(r"[^a-z0-9_]", "", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def validate_persona_config(
    spec: PopulationSpec, config: PersonaConfig
) -> ValidationResult:
    """Validate persona config coverage and phrase completeness."""
    result = ValidationResult()
    attr_map = {attr.name: attr for attr in spec.attributes}
    attr_names = set(attr_map.keys())

    treatment_attrs = [t.attribute for t in config.treatments]
    treatment_set = set(treatment_attrs)

    # Treatments must cover merged attributes exactly once.
    missing_treatments = sorted(attr_names - treatment_set)
    unknown_treatments = sorted(treatment_set - attr_names)
    duplicate_treatments = sorted(
        name for name in treatment_set if treatment_attrs.count(name) > 1
    )

    for name in missing_treatments:
        result.add_error(
            category="PERSONA_COVERAGE",
            location="treatments",
            message=f"missing treatment for attribute '{name}'",
            suggestion="add treatment entry for this attribute",
        )
    for name in unknown_treatments:
        result.add_error(
            category="PERSONA_COVERAGE",
            location="treatments",
            message=f"treatment references unknown attribute '{name}'",
        )
    for name in duplicate_treatments:
        result.add_error(
            category="PERSONA_COVERAGE",
            location="treatments",
            message=f"duplicate treatment entries for '{name}'",
        )

    # Group membership completeness and uniqueness.
    group_member_counts: dict[str, int] = {}
    for group in config.groups:
        for attr_name in group.attributes:
            if attr_name not in attr_names:
                result.add_error(
                    category="PERSONA_GROUP",
                    location=f"groups.{group.name}",
                    message=f"group references unknown attribute '{attr_name}'",
                )
                continue
            group_member_counts[attr_name] = group_member_counts.get(attr_name, 0) + 1

    for attr_name in sorted(attr_names):
        count = group_member_counts.get(attr_name, 0)
        if count == 0:
            result.add_error(
                category="PERSONA_GROUP",
                location="groups",
                message=f"attribute '{attr_name}' is not assigned to any group",
            )
        elif count > 1:
            result.add_error(
                category="PERSONA_GROUP",
                location="groups",
                message=f"attribute '{attr_name}' appears in multiple groups",
            )

    # Intro template references must resolve to known attributes.
    intro_refs = extract_intro_attributes(config.intro_template)
    unknown_intro_refs = sorted(intro_refs - attr_names)
    for ref in unknown_intro_refs:
        result.add_error(
            category="PERSONA_INTRO",
            location="intro_template",
            message=f"intro_template references unknown attribute '{ref}'",
        )

    # Build phrasing lookup maps.
    bool_map = {p.attribute: p for p in config.phrasings.boolean}
    cat_map = {p.attribute: p for p in config.phrasings.categorical}
    rel_map = {p.attribute: p for p in config.phrasings.relative}
    conc_map = {p.attribute: p for p in config.phrasings.concrete}

    treatment_lookup = {t.attribute: t.treatment.value for t in config.treatments}

    for attr in spec.attributes:
        attr_name = attr.name

        if attr.type == "boolean":
            phr = bool_map.get(attr_name)
            if phr is None:
                result.add_error(
                    category="PERSONA_PHRASING",
                    location="phrasings.boolean",
                    message=f"missing boolean phrasing for '{attr_name}'",
                )
            else:
                if not phr.true_phrase.strip() or not phr.false_phrase.strip():
                    result.add_error(
                        category="PERSONA_PHRASING",
                        location=f"phrasings.boolean.{attr_name}",
                        message="boolean phrasing must define non-empty true/false phrases",
                    )
                raw_tokens = {"true", "false", "yes", "no"}
                if phr.true_phrase.strip().lower() in raw_tokens:
                    result.add_warning(
                        category="PERSONA_PHRASING",
                        location=f"phrasings.boolean.{attr_name}",
                        message="true_phrase is raw token-like; use natural first-person text",
                    )
                if phr.false_phrase.strip().lower() in raw_tokens:
                    result.add_warning(
                        category="PERSONA_PHRASING",
                        location=f"phrasings.boolean.{attr_name}",
                        message="false_phrase is raw token-like; use natural first-person text",
                    )

        if attr.type == "categorical":
            phr = cat_map.get(attr_name)
            if phr is None:
                result.add_error(
                    category="PERSONA_PHRASING",
                    location="phrasings.categorical",
                    message=f"missing categorical phrasing for '{attr_name}'",
                )
            else:
                dist = attr.sampling.distribution
                if isinstance(dist, CategoricalDistribution):
                    expected = {_normalize_token(opt) for opt in dist.options}
                    actual = {_normalize_token(opt) for opt in phr.phrases.keys()}
                    if expected != actual:
                        missing = sorted(expected - actual)
                        extra = sorted(actual - expected)
                        detail = []
                        if missing:
                            detail.append(f"missing={missing}")
                        if extra:
                            detail.append(f"extra={extra}")
                        result.add_error(
                            category="PERSONA_PHRASING",
                            location=f"phrasings.categorical.{attr_name}",
                            message="categorical option coverage mismatch: "
                            + ", ".join(detail),
                        )

        treatment = treatment_lookup.get(attr_name)
        if treatment == "relative" and attr_name not in rel_map:
            result.add_error(
                category="PERSONA_PHRASING",
                location="phrasings.relative",
                message=f"missing relative phrasing for '{attr_name}'",
            )
        if (
            treatment == "concrete"
            and attr.type in {"int", "float"}
            and attr_name not in conc_map
        ):
            result.add_error(
                category="PERSONA_PHRASING",
                location="phrasings.concrete",
                message=f"missing concrete phrasing for '{attr_name}'",
            )

        # Float treatment should support mixed rendering quality.
        if (
            attr.type == "float"
            and treatment == "relative"
            and attr_name not in conc_map
        ):
            result.add_warning(
                category="PERSONA_PHRASING",
                location=f"phrasings.concrete.{attr_name}",
                message="relative float has no concrete fallback template",
            )

    return result
