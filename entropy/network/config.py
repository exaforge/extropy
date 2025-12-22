"""Configuration for network generation.

This module defines default attribute weights for similarity calculations
and degree correction multipliers for network generation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttributeWeightConfig:
    """Configuration for how an attribute contributes to similarity.

    Attributes:
        weight: Base weight for this attribute (higher = more important)
        match_type: How to compute match score:
            - "exact": 1 if exact match, 0 otherwise
            - "numeric_range": 1 - |A - B| / range (normalized difference)
            - "within_n": 1 if within n levels, 0 otherwise
        range_value: For numeric_range, the normalization range; for within_n, the allowed difference
    """
    weight: float
    match_type: str = "exact"
    range_value: float | None = None


# Default attribute weights for network generation
# Based on the design document for German surgeons
DEFAULT_ATTRIBUTE_WEIGHTS: dict[str, AttributeWeightConfig] = {
    # Same hospital = daily interaction
    "employer_type": AttributeWeightConfig(weight=3.0, match_type="exact"),
    # Specialty societies, conferences, shared cases
    "surgical_specialty": AttributeWeightConfig(weight=2.5, match_type="exact"),
    # Geographic proximity, regional networks
    "federal_state": AttributeWeightConfig(weight=2.0, match_type="exact"),
    # Chiefs know senior physicians, not residents
    "role_seniority": AttributeWeightConfig(weight=1.5, match_type="within_n", range_value=1),
    # Tertiary centers form networks
    "care_level": AttributeWeightConfig(weight=1.0, match_type="exact"),
    # Generational cohorts, training cohorts
    "age": AttributeWeightConfig(weight=1.0, match_type="numeric_range", range_value=10),
    # Academic networks
    "participation_in_research": AttributeWeightConfig(weight=0.5, match_type="exact"),
    # Society meetings, committees
    "professional_society_membership": AttributeWeightConfig(weight=0.5, match_type="exact"),
}

# Total weight for normalization (sum of all weights)
DEFAULT_TOTAL_WEIGHT = sum(cfg.weight for cfg in DEFAULT_ATTRIBUTE_WEIGHTS.values())


# Seniority levels for comparison (used in influence calculation)
SENIORITY_LEVELS: dict[str, int] = {
    "resident": 1,
    "specialist_attending": 2,
    "senior_physician_Oberarzt": 3,
    "chief_physician_Chefarzt": 4,
}


@dataclass
class DegreeMultiplierConfig:
    """Configuration for degree correction multipliers.

    Certain agents are more connected based on their attributes.
    Multipliers stack multiplicatively.
    """
    attribute: str
    condition: Any  # Value to match (or callable for complex conditions)
    multiplier: float
    rationale: str


# Default degree correction multipliers
DEFAULT_DEGREE_MULTIPLIERS: list[DegreeMultiplierConfig] = [
    DegreeMultiplierConfig(
        attribute="role_seniority",
        condition="chief_physician_Chefarzt",
        multiplier=2.0,
        rationale="Department heads know everyone"
    ),
    DegreeMultiplierConfig(
        attribute="role_seniority",
        condition="senior_physician_Oberarzt",
        multiplier=1.3,
        rationale="Mid-level leadership"
    ),
    DegreeMultiplierConfig(
        attribute="teaching_responsibility",
        condition=True,
        multiplier=1.4,
        rationale="Mentors many residents"
    ),
    DegreeMultiplierConfig(
        attribute="participation_in_research",
        condition=True,
        multiplier=1.3,
        rationale="Collaborations, publications"
    ),
    DegreeMultiplierConfig(
        attribute="professional_society_membership",
        condition=True,
        multiplier=1.2,
        rationale="Committee work, conferences"
    ),
    DegreeMultiplierConfig(
        attribute="employer_type",
        condition="university_hospital",
        multiplier=1.2,
        rationale="Larger institutions, more colleagues"
    ),
]


@dataclass
class NetworkConfig:
    """Complete configuration for network generation.

    Attributes:
        avg_degree: Target average degree (connections per agent)
        rewire_prob: Watts-Strogatz rewiring probability
        similarity_threshold: Sigmoid threshold for edge probability
        similarity_steepness: Sigmoid steepness for edge probability
        attribute_weights: Weights for similarity calculation
        degree_multipliers: Multipliers for degree correction
        seed: Random seed for reproducibility
    """
    avg_degree: float = 20.0
    rewire_prob: float = 0.05
    similarity_threshold: float = 0.3
    similarity_steepness: float = 10.0
    attribute_weights: dict[str, AttributeWeightConfig] = field(
        default_factory=lambda: dict(DEFAULT_ATTRIBUTE_WEIGHTS)
    )
    degree_multipliers: list[DegreeMultiplierConfig] = field(
        default_factory=lambda: list(DEFAULT_DEGREE_MULTIPLIERS)
    )
    seed: int | None = None

    def get_total_weight(self) -> float:
        """Get total weight for normalization."""
        return sum(cfg.weight for cfg in self.attribute_weights.values())
