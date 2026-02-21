"""Scenario-stage semantic role mapping for sampler/network runtime use.

This module runs a focused, low-context LLM call that maps semantic roles to
existing attribute names/options from the merged (base + extended) population.
The output is persisted in scenario.vN.yaml and consumed deterministically by
downstream stages.
"""

from __future__ import annotations

from typing import Any

from ..core.llm import reasoning_call
from ..core.models import PopulationSpec, SamplingSemanticRoles


_POLICIES = [
    "gaussian_offset",
    "same_group_rate",
    "same_country_rate",
    "same_value_probability",
]

_SAMPLING_SEMANTIC_ROLES_SCHEMA = {
    "type": "object",
    "properties": {
        "marital_roles": {
            "type": "object",
            "properties": {
                "attr": {"type": ["string", "null"]},
                "partnered_values": {"type": "array", "items": {"type": "string"}},
                "single_values": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["attr", "partnered_values", "single_values"],
            "additionalProperties": False,
        },
        "geo_roles": {
            "type": "object",
            "properties": {
                "country_attr": {"type": ["string", "null"]},
                "region_attr": {"type": ["string", "null"]},
                "urbanicity_attr": {"type": ["string", "null"]},
            },
            "required": ["country_attr", "region_attr", "urbanicity_attr"],
            "additionalProperties": False,
        },
        "partner_correlation_roles": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": _POLICIES,
            },
        },
        "school_parent_role": {
            "type": "object",
            "properties": {
                "dependents_attr": {"type": ["string", "null"]},
                "school_age_values": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["dependents_attr", "school_age_values"],
            "additionalProperties": False,
        },
        "religion_roles": {
            "type": "object",
            "properties": {
                "religion_attr": {"type": ["string", "null"]},
                "secular_values": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["religion_attr", "secular_values"],
            "additionalProperties": False,
        },
        "household_roles": {
            "type": "object",
            "properties": {
                "household_size_attr": {"type": ["string", "null"]},
            },
            "required": ["household_size_attr"],
            "additionalProperties": False,
        },
    },
    "required": [
        "marital_roles",
        "geo_roles",
        "partner_correlation_roles",
        "school_parent_role",
        "religion_roles",
        "household_roles",
    ],
    "additionalProperties": False,
}


def _categorical_options(attr: Any) -> list[str]:
    dist = getattr(getattr(attr, "sampling", None), "distribution", None)
    if dist is None or not hasattr(dist, "options"):
        return []
    options = getattr(dist, "options", None) or []
    return [str(opt) for opt in options]


def _build_prompt(population_spec: PopulationSpec) -> str:
    max_attrs = 60
    lines: list[str] = []
    for attr in population_spec.attributes[:max_attrs]:
        options = _categorical_options(attr)
        options_preview = ""
        if options:
            preview = ", ".join(options[:8])
            if len(options) > 8:
                preview += ", ..."
            options_preview = f" | options: [{preview}]"
        lines.append(
            f"- {attr.name} | type={attr.type} | scope={attr.scope}{options_preview}"
        )

    omitted_note = ""
    if len(population_spec.attributes) > max_attrs:
        omitted = len(population_spec.attributes) - max_attrs
        omitted_note = (
            f"\nNote: {omitted} additional attributes omitted to keep prompt compact."
        )

    attrs_block = "\n".join(lines) if lines else "- (no attributes)"

    return f"""Map sampler semantic roles to exact attribute names/options from this merged population.

Population: {population_spec.meta.description}
Geography/context: {population_spec.meta.geography or "unspecified"}

Available attributes:
{attrs_block}
{omitted_note}

Return ONLY valid mappings from these attributes.

Role definitions:
1) marital_roles:
- attr: marital/relationship status attribute (or null)
- partnered_values: values meaning partnered/married
- single_values: values meaning not partnered

2) geo_roles:
- country_attr: country/nationality/citizenship-style attribute for naming locale
- region_attr: region/state/province/city bucket
- urbanicity_attr: urban/rural/suburban-style bucket

3) partner_correlation_roles:
- map attribute_name -> policy
- policy enum:
  - gaussian_offset (age-like numeric correlation)
  - same_group_rate (identity group assortativity)
  - same_country_rate (country/citizenship assortativity)
  - same_value_probability (generic assortativity)

4) school_parent_role:
- dependents_attr: dependents/children list attribute
- school_age_values: labels that indicate school-age dependents

5) religion_roles:
- religion_attr: religion/faith affiliation attribute
- secular_values: values indicating no religion

6) household_roles:
- household_size_attr: attribute representing realized household size

Rules:
- Never invent attribute names.
- If a role is unavailable, use null and empty arrays/maps.
- partnered_values and single_values must be non-overlapping.
- Keep the mapping minimal and practical."""


def _sanitize_roles(
    data: dict[str, Any], population_spec: PopulationSpec
) -> SamplingSemanticRoles:
    known_attrs = {attr.name: attr for attr in population_spec.attributes}

    def keep_attr(name: Any) -> str | None:
        if isinstance(name, str) and name in known_attrs:
            return name
        return None

    def keep_values(attr_name: str | None, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        if attr_name is None:
            return []
        attr = known_attrs.get(attr_name)
        if attr is None:
            return []
        valid_options = set(_categorical_options(attr))
        if not valid_options:
            return []
        return [v for v in values if isinstance(v, str) and v in valid_options]

    marital_raw = data.get("marital_roles") or {}
    marital_attr = keep_attr(marital_raw.get("attr"))
    partnered_values = keep_values(marital_attr, marital_raw.get("partnered_values"))
    single_values = keep_values(marital_attr, marital_raw.get("single_values"))
    overlap = set(partnered_values) & set(single_values)
    if overlap:
        partnered_values = [v for v in partnered_values if v not in overlap]
        single_values = [v for v in single_values if v not in overlap]

    geo_raw = data.get("geo_roles") or {}
    geo_roles = {
        "country_attr": keep_attr(geo_raw.get("country_attr")),
        "region_attr": keep_attr(geo_raw.get("region_attr")),
        "urbanicity_attr": keep_attr(geo_raw.get("urbanicity_attr")),
    }

    policies_raw = data.get("partner_correlation_roles") or {}
    policies: dict[str, str] = {}
    if isinstance(policies_raw, dict):
        for attr_name, policy in policies_raw.items():
            if (
                attr_name in known_attrs
                and isinstance(policy, str)
                and policy in _POLICIES
            ):
                policies[attr_name] = policy

    school_raw = data.get("school_parent_role") or {}
    dependents_attr = keep_attr(school_raw.get("dependents_attr"))
    school_age_values = []
    if isinstance(school_raw.get("school_age_values"), list):
        school_age_values = [
            v
            for v in school_raw["school_age_values"]
            if isinstance(v, str) and v.strip()
        ]

    religion_raw = data.get("religion_roles") or {}
    religion_attr = keep_attr(religion_raw.get("religion_attr"))
    secular_values = []
    if isinstance(religion_raw.get("secular_values"), list):
        secular_values = [
            v
            for v in religion_raw["secular_values"]
            if isinstance(v, str) and v.strip()
        ]

    household_raw = data.get("household_roles") or {}
    household_size_attr = keep_attr(household_raw.get("household_size_attr"))

    return SamplingSemanticRoles.model_validate(
        {
            "marital_roles": {
                "attr": marital_attr,
                "partnered_values": partnered_values,
                "single_values": single_values,
            },
            "geo_roles": geo_roles,
            "partner_correlation_roles": policies,
            "school_parent_role": {
                "dependents_attr": dependents_attr,
                "school_age_values": school_age_values,
            },
            "religion_roles": {
                "religion_attr": religion_attr,
                "secular_values": secular_values,
            },
            "household_roles": {
                "household_size_attr": household_size_attr,
            },
        }
    )


def generate_sampling_semantic_roles(
    population_spec: PopulationSpec,
    model: str | None = None,
) -> SamplingSemanticRoles | None:
    """Generate scenario-stage semantic role mappings for sampler/network use.

    Returns None on provider/runtime failure so scenario compilation remains
    backward-compatible and non-blocking.
    """
    prompt = _build_prompt(population_spec)
    try:
        data = reasoning_call(
            prompt=prompt,
            response_schema=_SAMPLING_SEMANTIC_ROLES_SCHEMA,
            schema_name="sampling_semantic_roles",
            model=model,
            reasoning_effort="low",
            max_retries=1,
        )
    except Exception:
        return None

    try:
        return _sanitize_roles(data, population_spec)
    except Exception:
        return None
