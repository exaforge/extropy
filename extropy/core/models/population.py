"""Population Spec models and YAML I/O for Extropy.

A PopulationSpec is a complete blueprint for generating a population,
containing all attribute definitions, distributions, and sampling order.

This module contains all Phase 1 (Population Creation) models:
- Distributions: Normal, Uniform, Categorical, Boolean, etc.
- Modifiers: Numeric, Categorical, Boolean modifiers
- Sampling: SamplingConfig, Constraint
- Attributes: AttributeSpec, DiscoveredAttribute, HydratedAttribute
- Spec: PopulationSpec with YAML I/O
"""

from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# =============================================================================
# Household Models
# =============================================================================


class HouseholdType(str, Enum):
    SINGLE = "single"
    COUPLE = "couple"
    SINGLE_PARENT = "single_parent"
    COUPLE_WITH_KIDS = "couple_with_kids"
    MULTI_GENERATIONAL = "multi_generational"


class Dependent(BaseModel):
    """NPC dependent (child, elderly parent)."""

    name: str
    age: int
    gender: str
    relationship: str  # "son", "daughter", "mother", etc.
    school_status: str | None = None  # "home", "elementary", "middle_school", etc.


class LifeStageThreshold(BaseModel):
    """Age threshold for a dependent life stage."""

    max_age: int = Field(description="Up to (exclusive) this age")
    label: str = Field(description="Stage label, e.g. 'elementary', 'high_school'")


class HouseholdConfig(BaseModel):
    """Household composition and dependent generation parameters.

    Defaults are US Census-derived. At spec time, the LLM researches
    population-appropriate values. These defaults are the safety net.
    """

    # Age bracket boundaries: list of [upper_bound_exclusive, bracket_label]
    age_brackets: list[tuple[int, str]] = Field(
        default=[(30, "18-29"), (45, "30-44"), (65, "45-64"), (999, "65+")]
    )
    # Bracket label -> {household_type_value -> probability}
    household_type_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "18-29": {
                "single": 0.45,
                "couple": 0.25,
                "single_parent": 0.08,
                "couple_with_kids": 0.15,
                "multi_generational": 0.07,
            },
            "30-44": {
                "single": 0.20,
                "couple": 0.15,
                "single_parent": 0.12,
                "couple_with_kids": 0.40,
                "multi_generational": 0.13,
            },
            "45-64": {
                "single": 0.25,
                "couple": 0.35,
                "single_parent": 0.08,
                "couple_with_kids": 0.20,
                "multi_generational": 0.12,
            },
            "65+": {
                "single": 0.35,
                "couple": 0.40,
                "single_parent": 0.02,
                "couple_with_kids": 0.05,
                "multi_generational": 0.18,
            },
        }
    )
    # Partner correlation: same-group rates by race/ethnicity
    same_group_rates: dict[str, float] = Field(
        default_factory=lambda: {
            "white": 0.90,
            "black": 0.82,
            "hispanic": 0.78,
            "asian": 0.75,
            "other": 0.50,
        }
    )
    default_same_group_rate: float = 0.85
    assortative_mating: dict[str, float] = Field(
        default_factory=lambda: {
            "education_level": 0.6,
            "religious_affiliation": 0.7,
            "political_orientation": 0.5,
        }
    )
    partner_age_gap_mean: float = -2.0
    partner_age_gap_std: float = 3.0
    min_adult_age: int = 18

    # Dependent generation
    child_min_parent_offset: int = 20
    child_max_parent_offset: int = 40
    max_dependent_child_age: int = 17
    elderly_min_offset: int = 22
    elderly_max_offset: int = 35

    # Life stages (replaces hardcoded _school_status)
    life_stages: list[LifeStageThreshold] = Field(
        default_factory=lambda: [
            LifeStageThreshold(max_age=5, label="home"),
            LifeStageThreshold(max_age=11, label="elementary"),
            LifeStageThreshold(max_age=14, label="middle_school"),
            LifeStageThreshold(max_age=18, label="high_school"),
        ]
    )
    adult_stage_label: str = "adult"
    avg_household_size: float = 2.5


# =============================================================================
# Name Generation Models
# =============================================================================


class NameEntry(BaseModel):
    name: str
    weight: float = 1.0


class NameConfig(BaseModel):
    """Name frequency tables for culturally-appropriate name generation.

    None on SpecMeta = use bundled US CSV data.
    """

    male_first_names: list[NameEntry] = Field(default_factory=list)
    female_first_names: list[NameEntry] = Field(default_factory=list)
    last_names: list[NameEntry] = Field(default_factory=list)


# Standard personality attributes that spec builders should include.
# `conformity` (float, 0-1, correlated with agreeableness) is consumed by
# Phase C for threshold behavior in simulation.
STANDARD_PERSONALITY_ATTRIBUTES = [
    "neuroticism",
    "extraversion",
    "openness",
    "conscientiousness",
    "agreeableness",
    "conformity",
]


# =============================================================================
# Grounding Information
# =============================================================================


class GroundingInfo(BaseModel):
    """Information about how well an attribute is grounded in real data."""

    level: Literal["strong", "medium", "low"] = Field(
        description="How well grounded in real data: strong (direct data), medium (extrapolated), low (estimated)"
    )
    method: Literal["researched", "extrapolated", "estimated", "computed"] = Field(
        description="How the distribution was determined"
    )
    source: str | None = Field(default=None, description="Citation or URL if available")
    note: str | None = Field(default=None, description="Any caveats or notes")


# =============================================================================
# Distribution Configurations
# =============================================================================


class NormalDistribution(BaseModel):
    """Normal/Gaussian distribution parameters.

    For conditional attributes with continuous dependencies, use mean_formula
    instead of mean to express the relationship (e.g., "age - 28" for experience).

    For dynamic bounds that depend on other attributes, use min_formula/max_formula
    (e.g., "max(0, household_size - 1)" for children_count).
    """

    type: Literal["normal"] = "normal"
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    mean_formula: str | None = Field(
        default=None, description="Formula for mean, e.g., 'age - 28'"
    )
    std_formula: str | None = Field(
        default=None, description="Formula for std (rare, but supported)"
    )
    min_formula: str | None = Field(
        default=None,
        description="Formula for dynamic min bound, e.g., '0'. Evaluated with agent context.",
    )
    max_formula: str | None = Field(
        default=None,
        description="Formula for dynamic max bound, e.g., 'household_size - 1'. Evaluated with agent context.",
    )


class LognormalDistribution(BaseModel):
    """Lognormal distribution parameters.

    For dynamic bounds that depend on other attributes, use min_formula/max_formula.
    """

    type: Literal["lognormal"] = "lognormal"
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    mean_formula: str | None = None
    std_formula: str | None = None
    min_formula: str | None = Field(
        default=None,
        description="Formula for dynamic min bound, e.g., '0'. Evaluated with agent context.",
    )
    max_formula: str | None = Field(
        default=None,
        description="Formula for dynamic max bound, e.g., 'household_size - 1'. Evaluated with agent context.",
    )


class UniformDistribution(BaseModel):
    """Uniform distribution parameters."""

    type: Literal["uniform"] = "uniform"
    min: float
    max: float


class BetaDistribution(BaseModel):
    """Beta distribution parameters (useful for probabilities and proportions).

    For dynamic bounds that depend on other attributes, use min_formula/max_formula.
    """

    type: Literal["beta"] = "beta"
    alpha: float
    beta: float
    min: float | None = None
    max: float | None = None
    min_formula: str | None = Field(
        default=None,
        description="Formula for dynamic min bound. Evaluated with agent context.",
    )
    max_formula: str | None = Field(
        default=None,
        description="Formula for dynamic max bound. Evaluated with agent context.",
    )


class CategoricalDistribution(BaseModel):
    """Categorical distribution with options and weights."""

    type: Literal["categorical"] = "categorical"
    options: list[str]
    weights: list[float] = Field(description="Probabilities, should sum to ~1.0")


class BooleanDistribution(BaseModel):
    """Boolean distribution."""

    type: Literal["boolean"] = "boolean"
    probability_true: float = Field(ge=0, le=1)


Distribution = (
    NormalDistribution
    | LognormalDistribution
    | UniformDistribution
    | BetaDistribution
    | CategoricalDistribution
    | BooleanDistribution
)


# =============================================================================
# Sampling Configuration
# =============================================================================


class Modifier(BaseModel):
    """A conditional modifier for sampling.

    Modifies distributions based on conditions. The fields used depend on
    the distribution type:
    - Numeric (normal, lognormal, uniform, beta): use multiply/add
    - Categorical: use weight_overrides
    - Boolean: use probability_override

    Validation ensures type-appropriate fields are used.
    """

    when: str = Field(description="Python expression using other attribute names")
    multiply: float | None = None
    add: float | None = None
    weight_overrides: dict[str, float] | None = None
    probability_override: float | None = None


class SamplingConfig(BaseModel):
    """Configuration for how to sample an attribute."""

    strategy: Literal["independent", "derived", "conditional"] = Field(
        description="independent: sample directly; derived: compute from formula; conditional: sample then modify"
    )
    distribution: Distribution | None = Field(
        default=None,
        description="Distribution to sample from (for independent/conditional)",
    )
    formula: str | None = Field(
        default=None, description="Python expression for derived attributes"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="Attributes this depends on"
    )
    modifiers: list[Modifier] = Field(
        default_factory=list,
        description="Conditional modifiers (for conditional strategy)",
    )


# =============================================================================
# Constraint
# =============================================================================


class Constraint(BaseModel):
    """A constraint on an attribute value.

    Constraints should be set WIDER than observed data to preserve valid outliers.
    E.g., if research shows surgeons are typically 28-65, set hard_min: 26, hard_max: 78.

    Constraint types:
    - hard_min/hard_max: Static bounds for clamping sampled values
    - expression: Agent-level constraints validated after sampling (e.g., 'children_count <= household_size - 1')
    - spec_expression: Spec-level constraints that validate the YAML definition itself (e.g., 'sum(weights)==1'),
                       NOT evaluated against individual agents
    - min/max: Legacy aliases for hard_min/hard_max
    """

    type: Literal[
        "hard_min", "hard_max", "expression", "spec_expression", "min", "max"
    ] = Field(
        description="hard_min/hard_max for bounds, expression for agent-level constraints, spec_expression for spec-level validation. 'min'/'max' are legacy aliases."
    )
    value: float | None = Field(
        default=None, description="Value for hard_min/hard_max constraints"
    )
    expression: str | None = Field(
        default=None,
        description="Python expression for expression constraints, e.g., 'value <= age - 24'",
    )
    reason: str | None = Field(default=None, description="Why this constraint exists")


# =============================================================================
# Attribute Spec
# =============================================================================


class AttributeSpec(BaseModel):
    """Complete specification for a single attribute."""

    name: str = Field(description="Attribute name in snake_case")
    type: Literal["int", "float", "categorical", "boolean"] = Field(
        description="Data type of the attribute"
    )
    category: Literal[
        "universal", "population_specific", "context_specific", "personality"
    ] = Field(description="Category of attribute")
    description: str = Field(description="What this attribute represents")
    scope: Literal["individual", "household"] = Field(
        default="individual",
        description="Whether this attribute is sampled per-individual or shared across a household",
    )
    sampling: SamplingConfig
    grounding: GroundingInfo
    constraints: list[Constraint] = Field(default_factory=list)


# =============================================================================
# Spec Metadata
# =============================================================================


class SpecMeta(BaseModel):
    """Metadata about the population spec."""

    description: str = Field(description="Original population description")
    size: int = Field(description="Number of agents to generate")
    geography: str | None = Field(default=None, description="Geographic scope")
    agent_focus: str | None = Field(
        default=None,
        description="Who the study agents represent. Determines agent vs NPC partitioning in household sampling.",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0", description="Spec format version")
    persona_template: str | None = Field(
        default=None,
        description="Jinja2 template for generating persona text from agent attributes",
    )
    scenario_description: str | None = Field(
        default=None,
        description="Scenario description from extend command, used by scenario command",
    )
    household_config: HouseholdConfig = Field(default_factory=HouseholdConfig)
    name_config: NameConfig | None = Field(default=None)


# =============================================================================
# Overall Grounding Summary
# =============================================================================


class GroundingSummary(BaseModel):
    """Summary of grounding quality across all attributes."""

    overall: Literal["strong", "medium", "low"]
    sources_count: int
    strong_count: int = Field(description="Number of strongly grounded attributes")
    medium_count: int = Field(description="Number of medium grounded attributes")
    low_count: int = Field(description="Number of weakly grounded attributes")
    sources: list[str] = Field(default_factory=list, description="All sources used")


# =============================================================================
# Population Spec
# =============================================================================


class PopulationSpec(BaseModel):
    """Complete specification for generating a population."""

    meta: SpecMeta
    grounding: GroundingSummary
    attributes: list[AttributeSpec]
    sampling_order: list[str] = Field(
        description="Order in which attributes should be sampled (respects dependencies)"
    )

    def to_yaml(self, path: Path | str) -> None:
        """Save spec to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling datetime
        data = self.model_dump(mode="json")

        with open(path, "w") as f:
            yaml.dump(
                data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PopulationSpec":
        """Load spec from YAML file."""
        path = Path(path)

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def get_attribute(self, name: str) -> AttributeSpec | None:
        """Get an attribute by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None

    def summary(self) -> str:
        """Get a text summary of the spec."""
        lines = [
            f"Population: {self.meta.description}",
            f"Size: {self.meta.size}",
            f"Grounding: {self.grounding.overall} ({self.grounding.sources_count} sources)",
            f"Attributes: {len(self.attributes)}",
            "",
            "Sampling order:",
        ]
        for i, attr_name in enumerate(self.sampling_order, 1):
            attr = self.get_attribute(attr_name)
            if attr:
                lines.append(
                    f"  {i}. {attr_name} ({attr.type}) - {attr.grounding.level}"
                )

        return "\n".join(lines)

    def merge(self, extension: "PopulationSpec") -> "PopulationSpec":
        """
        Merge an extension spec into this base spec.

        The extension adds new attributes that can depend on base attributes.
        Base attributes are preserved; extension attributes are appended.
        Sampling order is recomputed to handle cross-layer dependencies.

        Args:
            extension: The extension spec to merge (scenario-specific attributes)

        Returns:
            New PopulationSpec with merged attributes
        """
        # Combine attributes (base first, then extension)
        base_names = {attr.name for attr in self.attributes}
        merged_attributes = list(self.attributes)

        for attr in extension.attributes:
            if attr.name not in base_names:
                merged_attributes.append(attr)

        # Merge sources (deduplicate)
        all_sources = list(set(self.grounding.sources + extension.grounding.sources))

        # Recompute grounding summary
        strong_count = sum(
            1 for a in merged_attributes if a.grounding.level == "strong"
        )
        medium_count = sum(
            1 for a in merged_attributes if a.grounding.level == "medium"
        )
        low_count = sum(1 for a in merged_attributes if a.grounding.level == "low")
        total = len(merged_attributes)

        if total == 0:
            overall = "low"
        elif strong_count / total >= 0.6:
            overall = "strong"
        elif (strong_count + medium_count) / total >= 0.5:
            overall = "medium"
        else:
            overall = "low"

        merged_grounding = GroundingSummary(
            overall=overall,
            sources_count=len(all_sources),
            strong_count=strong_count,
            medium_count=medium_count,
            low_count=low_count,
            sources=all_sources,
        )

        # Recompute sampling order via topological sort
        merged_order = self._compute_sampling_order(merged_attributes)

        # Create merged metadata
        # Note: persona_template is intentionally left as None here.
        # It will be regenerated by the CLI after merge to include all
        # attributes (base + extension).
        # Prefer extension's household_config if it differs from defaults,
        # otherwise keep base's researched config
        ext_hh = extension.meta.household_config
        base_hh = self.meta.household_config
        merged_hh = ext_hh if ext_hh != HouseholdConfig() else base_hh

        # Prefer extension's name_config if present, else keep base's
        merged_nc = extension.meta.name_config or self.meta.name_config

        merged_meta = SpecMeta(
            description=f"{self.meta.description} + {extension.meta.description}",
            size=self.meta.size,
            geography=self.meta.geography,
            agent_focus=self.meta.agent_focus,
            created_at=datetime.now(),
            version=self.meta.version,
            persona_template=None,
            household_config=merged_hh,
            name_config=merged_nc,
        )

        return PopulationSpec(
            meta=merged_meta,
            grounding=merged_grounding,
            attributes=merged_attributes,
            sampling_order=merged_order,
        )

    @staticmethod
    def _compute_sampling_order(attributes: list["AttributeSpec"]) -> list[str]:
        """Compute sampling order via topological sort (Kahn's algorithm).

        Note: This is a standalone implementation rather than using
        validation.graphs.topological_sort to keep core/models dependency-free.
        Cycles are handled gracefully by appending remaining nodes.
        """

        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = {a.name: 0 for a in attributes}
        attr_names = {a.name for a in attributes}

        for attr in attributes:
            for dep in attr.sampling.depends_on:
                if dep in attr_names:
                    graph[dep].append(attr.name)
                    in_degree[attr.name] += 1

        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            queue.sort()  # Deterministic ordering
            node = queue.pop(0)
            order.append(node)

            for dependent in graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # If cycle detected, append remaining (graceful degradation for merge)
        if len(order) != len(attributes):
            remaining = [a.name for a in attributes if a.name not in order]
            order.extend(remaining)

        return order


# =============================================================================
# Intermediate Types (used during spec building)
# =============================================================================


class DiscoveredAttribute(BaseModel):
    """An attribute discovered during attribute selection (Step 1).

    Includes strategy to indicate how the attribute should be sampled,
    which determines which hydration step processes it.
    """

    name: str
    type: Literal["int", "float", "categorical", "boolean"]
    category: Literal[
        "universal", "population_specific", "context_specific", "personality"
    ]
    description: str
    strategy: Literal["independent", "derived", "conditional"] = Field(
        default="independent",
        description="independent: sample directly; derived: zero-variance formula; conditional: probabilistic dependency",
    )
    scope: Literal["individual", "household"] = Field(
        default="individual",
        description="individual: varies per person; household: shared across household members",
    )
    depends_on: list[str] = Field(default_factory=list)


# hydrated attribute seems to be an extension of discovered attribute.


class HydratedAttribute(BaseModel):
    """An attribute with distribution data from research (Step 2).

    Contains the complete sampling configuration including distribution,
    modifiers (for conditional), formula (for derived), and constraints.
    """

    name: str
    type: Literal["int", "float", "categorical", "boolean"]
    category: Literal[
        "universal", "population_specific", "context_specific", "personality"
    ]
    description: str
    strategy: Literal["independent", "derived", "conditional"] = Field(
        default="independent", description="Sampling strategy determined in Step 1"
    )
    scope: Literal["individual", "household"] = Field(
        default="individual",
        description="individual: varies per person; household: shared across household members",
    )
    depends_on: list[str] = Field(default_factory=list)
    sampling: SamplingConfig
    grounding: GroundingInfo
    constraints: list[Constraint] = Field(default_factory=list)


class SufficiencyResult(BaseModel):
    """Result from context sufficiency check (Step 0)."""

    sufficient: bool
    size: int = Field(default=1000, description="Extracted or default population size")
    geography: str | None = None
    agent_focus: str | None = Field(
        default=None,
        description="Who this study is about, e.g. 'surgeons', 'high school students', 'retired couples', 'families'",
    )
    clarifications_needed: list[str] = Field(default_factory=list)
