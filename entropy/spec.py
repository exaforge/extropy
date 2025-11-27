"""Population Spec models and YAML I/O for Entropy.

A PopulationSpec is a complete blueprint for generating a population,
containing all attribute definitions, distributions, and sampling order.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


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
    """Normal/Gaussian distribution parameters."""

    type: Literal["normal"] = "normal"
    mean: float
    std: float
    min: float | None = None
    max: float | None = None


class UniformDistribution(BaseModel):
    """Uniform distribution parameters."""

    type: Literal["uniform"] = "uniform"
    min: float
    max: float


class CategoricalDistribution(BaseModel):
    """Categorical distribution with options and weights."""

    type: Literal["categorical"] = "categorical"
    options: list[str]
    weights: list[float] = Field(description="Probabilities, should sum to ~1.0")


class BooleanDistribution(BaseModel):
    """Boolean distribution."""

    type: Literal["boolean"] = "boolean"
    probability_true: float = Field(ge=0, le=1)


Distribution = NormalDistribution | UniformDistribution | CategoricalDistribution | BooleanDistribution


# =============================================================================
# Sampling Configuration
# =============================================================================


class Modifier(BaseModel):
    """A conditional modifier for sampling."""

    when: str = Field(description="Python expression using other attribute names")
    multiply: float | None = None
    add: float | None = None


class SamplingConfig(BaseModel):
    """Configuration for how to sample an attribute."""

    strategy: Literal["independent", "derived", "conditional"] = Field(
        description="independent: sample directly; derived: compute from formula; conditional: sample then modify"
    )
    distribution: Distribution | None = Field(
        default=None, description="Distribution to sample from (for independent/conditional)"
    )
    formula: str | None = Field(
        default=None, description="Python expression for derived attributes"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="Attributes this depends on"
    )
    modifiers: list[Modifier] = Field(
        default_factory=list, description="Conditional modifiers (for conditional strategy)"
    )


# =============================================================================
# Constraint
# =============================================================================


class Constraint(BaseModel):
    """A constraint on an attribute value."""

    type: Literal["min", "max", "expression"] = Field(
        description="min/max for bounds, expression for complex constraints"
    )
    value: float | None = Field(default=None, description="Value for min/max constraints")
    expression: str | None = Field(
        default=None, description="Python expression for expression constraints"
    )


# =============================================================================
# Attribute Spec
# =============================================================================


class AttributeSpec(BaseModel):
    """Complete specification for a single attribute."""

    name: str = Field(description="Attribute name in snake_case")
    type: Literal["int", "float", "categorical", "boolean"] = Field(
        description="Data type of the attribute"
    )
    category: Literal["universal", "population_specific", "context_specific", "personality"] = Field(
        description="Category of attribute"
    )
    description: str = Field(description="What this attribute represents")
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
    created_at: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0", description="Spec format version")


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
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

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
                lines.append(f"  {i}. {attr_name} ({attr.type}) - {attr.grounding.level}")

        return "\n".join(lines)


# =============================================================================
# Intermediate Types (used during spec building)
# =============================================================================


class DiscoveredAttribute(BaseModel):
    """An attribute discovered during attribute selection (Step 1)."""

    name: str
    type: Literal["int", "float", "categorical", "boolean"]
    category: Literal["universal", "population_specific", "context_specific", "personality"]
    description: str
    depends_on: list[str] = Field(default_factory=list)


class HydratedAttribute(BaseModel):
    """An attribute with distribution data from research (Step 2)."""

    name: str
    type: Literal["int", "float", "categorical", "boolean"]
    category: Literal["universal", "population_specific", "context_specific", "personality"]
    description: str
    depends_on: list[str] = Field(default_factory=list)
    sampling: SamplingConfig
    grounding: GroundingInfo
    constraints: list[Constraint] = Field(default_factory=list)


class SufficiencyResult(BaseModel):
    """Result from context sufficiency check (Step 0)."""

    sufficient: bool
    size: int = Field(default=1000, description="Extracted or default population size")
    geography: str | None = None
    clarifications_needed: list[str] = Field(default_factory=list)

