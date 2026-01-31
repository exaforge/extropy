"""Persona configuration models.

PersonaConfig defines how to render agent attributes into first-person personas.
Generated once per population via LLM, then applied to all agents via templates.
"""

from datetime import datetime
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TreatmentType(str, Enum):
    """How to treat an attribute when rendering."""
    
    CONCRETE = "concrete"  # Keep the actual number/value
    RELATIVE = "relative"  # Express relative to population mean


class RelativeLabels(BaseModel):
    """Labels for relative positioning (z-score based)."""
    
    much_below: str = Field(description="Label for z < -1.0")
    below: str = Field(description="Label for -1.0 <= z < -0.3")
    average: str = Field(description="Label for -0.3 <= z <= 0.3")
    above: str = Field(description="Label for 0.3 < z <= 1.0")
    much_above: str = Field(description="Label for z > 1.0")
    
    def get_label(self, z_score: float) -> str:
        """Get the appropriate label for a z-score."""
        if z_score < -1.0:
            return self.much_below
        elif z_score < -0.3:
            return self.below
        elif z_score <= 0.3:
            return self.average
        elif z_score <= 1.0:
            return self.above
        else:
            return self.much_above


class AttributeTreatment(BaseModel):
    """How to treat a specific attribute."""
    
    attribute: str = Field(description="Attribute name")
    treatment: TreatmentType = Field(description="Concrete or relative")
    group: str = Field(description="Which group this attribute belongs to")


class AttributeGroup(BaseModel):
    """A logical grouping of attributes."""
    
    name: str = Field(description="Internal group name (e.g., 'commute')")
    label: str = Field(description="Display label (e.g., 'My Commute')")
    attributes: list[str] = Field(description="Ordered list of attribute names in this group")


class BooleanPhrasing(BaseModel):
    """First-person phrasing for a boolean attribute."""
    
    attribute: str
    true_phrase: str = Field(description="Phrase when value is True")
    false_phrase: str = Field(description="Phrase when value is False")


class CategoricalPhrasing(BaseModel):
    """First-person phrasing for a categorical attribute."""
    
    attribute: str
    phrases: dict[str, str] = Field(
        description="Map from option value to first-person phrase"
    )
    fallback: str | None = Field(
        default=None,
        description="Fallback phrase if value not in phrases"
    )


class RelativePhrasing(BaseModel):
    """First-person phrasing for a relative (psychological) attribute."""
    
    attribute: str
    labels: RelativeLabels = Field(description="Labels for each z-score bucket")


class ConcretePhrasing(BaseModel):
    """First-person phrasing template for a concrete attribute."""
    
    attribute: str
    template: str = Field(
        description="Template with {value} placeholder, e.g., 'I drive {value} miles to work'"
    )
    format_spec: str | None = Field(
        default=None,
        description="Python format spec for the value, e.g., ',.0f' for '$95,000'"
    )
    prefix: str = Field(default="", description="Prefix before value, e.g., '$'")
    suffix: str = Field(default="", description="Suffix after value, e.g., ' miles'")


class AttributePhrasing(BaseModel):
    """Union type for all phrasing types."""
    
    boolean: list[BooleanPhrasing] = Field(default_factory=list)
    categorical: list[CategoricalPhrasing] = Field(default_factory=list)
    relative: list[RelativePhrasing] = Field(default_factory=list)
    concrete: list[ConcretePhrasing] = Field(default_factory=list)
    
    def get_phrasing(self, attr_name: str) -> BooleanPhrasing | CategoricalPhrasing | RelativePhrasing | ConcretePhrasing | None:
        """Get phrasing for a specific attribute."""
        for p in self.boolean:
            if p.attribute == attr_name:
                return p
        for p in self.categorical:
            if p.attribute == attr_name:
                return p
        for p in self.relative:
            if p.attribute == attr_name:
                return p
        for p in self.concrete:
            if p.attribute == attr_name:
                return p
        return None


class PopulationStats(BaseModel):
    """Statistics for each attribute, used for relative positioning."""
    
    stats: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Map from attribute name to {mean, std, min, max}"
    )
    
    def get_z_score(self, attr_name: str, value: float) -> float | None:
        """Calculate z-score for a value given population stats."""
        if attr_name not in self.stats:
            return None
        s = self.stats[attr_name]
        if s.get("std", 0) == 0:
            return 0.0
        return (value - s["mean"]) / s["std"]


class PersonaConfig(BaseModel):
    """Complete configuration for rendering agent personas.
    
    Generated once via LLM, then applied to all agents via templates.
    """
    
    # Metadata
    population_description: str = Field(description="Description of the population")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Intro template (narrative hook)
    intro_template: str = Field(
        description="First-person intro template with {attribute} placeholders"
    )
    
    # Attribute treatments
    treatments: list[AttributeTreatment] = Field(
        description="Treatment for each attribute"
    )
    
    # Groupings
    groups: list[AttributeGroup] = Field(
        description="Logical groupings of attributes"
    )
    
    # Phrasings
    phrasings: AttributePhrasing = Field(
        description="First-person phrasing templates"
    )
    
    # Population statistics (computed from sampled agents)
    population_stats: PopulationStats = Field(
        default_factory=PopulationStats,
        description="Mean/std for relative positioning"
    )
    
    def get_treatment(self, attr_name: str) -> AttributeTreatment | None:
        """Get treatment for a specific attribute."""
        for t in self.treatments:
            if t.attribute == attr_name:
                return t
        return None
    
    def get_group(self, group_name: str) -> AttributeGroup | None:
        """Get a group by name."""
        for g in self.groups:
            if g.name == group_name:
                return g
        return None
    
    def to_yaml(self) -> str:
        """Serialize to YAML."""
        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PersonaConfig":
        """Load from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)
    
    @classmethod
    def from_file(cls, path: str) -> "PersonaConfig":
        """Load from YAML file."""
        with open(path, "r") as f:
            return cls.from_yaml(f.read())
    
    def to_file(self, path: str) -> None:
        """Save to YAML file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())
