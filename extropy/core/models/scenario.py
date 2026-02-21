"""Scenario models for Extropy (Phase 2).

A ScenarioSpec defines how an event/information propagates through a population
and what outcomes to measure. It is the bridge between population creation (Phase 1)
and simulation execution (Phase 3).

This module contains:
- Event: EventType, Event
- Exposure: ExposureChannel, ExposureRule, SeedExposure
- Interaction: InteractionConfig, SpreadModifier, SpreadConfig
- Outcomes: OutcomeType, OutcomeDefinition, OutcomeConfig
- Config: TimestepUnit, ScenarioSimConfig
- Spec: ScenarioMeta, ScenarioSpec with YAML I/O
- Validation: ValidationError, ValidationWarning, ValidationResult
"""

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .population import AttributeSpec, HouseholdConfig

if TYPE_CHECKING:
    pass


# =============================================================================
# Event Definition
# =============================================================================


class EventType(str, Enum):
    """Type of event being introduced to the population."""

    ANNOUNCEMENT = "announcement"
    NEWS = "news"
    RUMOR = "rumor"
    POLICY_CHANGE = "policy_change"
    PRODUCT_LAUNCH = "product_launch"
    EMERGENCY = "emergency"
    OBSERVATION = "observation"


class Event(BaseModel):
    """Definition of the event/information being introduced."""

    type: EventType = Field(description="Type of event")
    content: str = Field(description="The actual information/announcement text")
    source: str = Field(
        description="Who/what originated this (e.g., 'Netflix', 'hospital administration')"
    )
    credibility: float = Field(
        ge=0,
        le=1,
        description="How credible is the source (0=not credible, 1=fully credible)",
    )
    ambiguity: float = Field(
        ge=0,
        le=1,
        description="How clear/unclear is the information (0=crystal clear, 1=very ambiguous)",
    )
    emotional_valence: float = Field(
        ge=-1,
        le=1,
        description="Emotional framing (-1=very negative, 0=neutral, 1=very positive)",
    )


# =============================================================================
# Seed Exposure
# =============================================================================


class ExposureChannel(BaseModel):
    """A channel through which agents can be exposed to the event."""

    name: str = Field(
        description="Channel identifier in snake_case (e.g., 'email_notification')"
    )
    description: str = Field(description="Human-readable description of the channel")
    credibility_modifier: float = Field(
        default=1.0,
        description="How the channel affects perceived credibility (1.0=no change)",
    )
    experience_template: str | None = Field(
        default=None,
        description="How the agent experiences this channel, e.g. 'I saw this on {channel_name}'",
    )


class ExposureRule(BaseModel):
    """A rule determining which agents are exposed through which channel."""

    channel: str = Field(description="References ExposureChannel.name")
    when: str = Field(
        description="Python expression using agent attributes (e.g., 'age < 45'). Use 'true' for all agents."
    )
    probability: float = Field(
        ge=0, le=1, description="Probability of exposure given the condition is met"
    )
    timestep: int = Field(ge=0, description="When this exposure occurs (0=immediately)")


class SeedExposure(BaseModel):
    """Configuration for initial event exposure."""

    channels: list[ExposureChannel] = Field(
        default_factory=list, description="Available exposure channels"
    )
    rules: list[ExposureRule] = Field(
        default_factory=list, description="Rules for exposing agents through channels"
    )


class TimelineEvent(BaseModel):
    """A development in the scenario timeline.

    Timeline events represent how a scenario evolves over time. For evolving
    scenarios (crises, campaigns), multiple events occur at different timesteps.
    Static scenarios (policy announcements) have no timeline events.
    """

    timestep: int = Field(ge=0, description="When this development occurs")
    event: Event = Field(description="The event content at this timestep")
    exposure_rules: list[ExposureRule] | None = Field(
        default=None,
        description="Custom exposure rules; if None, reuses seed_exposure.rules with updated content",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable context for this development",
    )
    re_reasoning_intensity: Literal["normal", "high", "extreme"] | None = Field(
        default=None,
        description=(
            "How broadly this event should trigger committed-agent re-reasoning. "
            "normal=direct only, high=direct+traced network, extreme=high+all aware."
        ),
    )


class InteractionConfig(BaseModel):
    """Configuration for how agents interact about the event."""

    description: str = Field(
        default="",
        description="Human-readable notes about social dynamics (informational only).",
    )


class SpreadModifier(BaseModel):
    """Modifier that adjusts spread probability based on conditions."""

    when: str = Field(
        description="Condition (can reference agent attrs or edge attrs like 'edge_type')"
    )
    multiply: float = Field(default=1.0, description="Multiplicative adjustment")
    add: float = Field(default=0.0, description="Additive adjustment")


class SpreadConfig(BaseModel):
    """Configuration for how information spreads through the network."""

    share_probability: float = Field(
        ge=0, le=1, description="Base probability that an agent shares with a neighbor"
    )
    share_modifiers: list[SpreadModifier] = Field(
        default_factory=list, description="Adjustments based on conditions"
    )
    decay_per_hop: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Information loses fidelity each hop (0=no decay)",
    )
    max_hops: int | None = Field(
        default=None, description="Limit propagation depth (None=unlimited)"
    )

    @field_validator("max_hops", mode="before")
    @classmethod
    def coerce_max_hops(cls, v):
        if v is None or v == "null" or v == "None":
            return None
        if isinstance(v, str):
            return int(v)
        return v


# =============================================================================
# Outcomes
# =============================================================================


class OutcomeType(str, Enum):
    """Type of outcome measurement."""

    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    FLOAT = "float"
    OPEN_ENDED = "open_ended"


class OutcomeDefinition(BaseModel):
    """Definition of a single outcome to measure."""

    name: str = Field(description="Outcome identifier in snake_case")
    type: OutcomeType = Field(description="Type of the outcome")
    description: str = Field(description="What this outcome measures")
    options: list[str] | None = Field(
        default=None, description="For categorical outcomes: the possible values"
    )
    range: tuple[float, float] | None = Field(
        default=None, description="For float outcomes: (min, max) range"
    )
    required: bool = Field(
        default=True, description="Whether this outcome must be extracted"
    )
    option_friction: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional per-option behavior friction scores in [0,1] where higher "
            "means harder to sustain in real behavior. Used by engine dynamics for "
            "categorical outcomes."
        ),
    )


class OutcomeConfig(BaseModel):
    """Configuration for outcome measurement."""

    suggested_outcomes: list[OutcomeDefinition] = Field(
        default_factory=list, description="Outcomes to measure"
    )
    capture_full_reasoning: bool = Field(
        default=True, description="Whether to capture agent's full reasoning"
    )
    extraction_instructions: str | None = Field(
        default=None, description="Hints for Phase 3 outcome extraction"
    )
    decision_relevant_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes most relevant to this scenario's decision (for trait salience in persona rendering)",
    )


# =============================================================================
# Simulation Parameters
# =============================================================================


class TimestepUnit(str, Enum):
    """Unit of time for simulation timesteps."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class ScenarioSimConfig(BaseModel):
    """Configuration for simulation execution."""

    max_timesteps: int = Field(ge=1, description="Maximum number of timesteps to run")
    timestep_unit: TimestepUnit = Field(
        default=TimestepUnit.HOUR, description="What each timestep represents"
    )
    stop_conditions: list[str] | None = Field(
        default=None,
        description="Conditions that trigger early stop (e.g., 'exposure_rate > 0.95')",
    )
    allow_early_convergence: bool | None = Field(
        default=None,
        description=(
            "Override auto convergence/quiescence stopping behavior: "
            "None=auto (disable when future timeline events exist), "
            "True=allow, False=disable."
        ),
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )


# =============================================================================
# Conversation Relationship Weights
# =============================================================================

# Default relationship weights used when scenario doesn't specify custom weights
DEFAULT_RELATIONSHIP_WEIGHTS: dict[str, float] = {
    "partner": 1.0,
    "household": 0.9,
    "close_friend": 0.7,
    "coworker": 0.6,
    "neighbor": 0.4,
    "congregation": 0.4,
    "school_parent": 0.35,
    "acquaintance": 0.2,
    "online_contact": 0.15,
}


# =============================================================================
# Complete Scenario Spec
# =============================================================================


class ScenarioMeta(BaseModel):
    """Metadata about the scenario spec."""

    name: str = Field(description="Short identifier for the scenario")
    description: str = Field(description="Full scenario description")
    # New flow: base_population is the versioned reference (e.g. "population.v2")
    base_population: str | None = Field(
        default=None,
        description="Base population version reference (e.g. population.v2)",
    )
    # Legacy fields for backwards compatibility
    population_spec: str | None = Field(
        default=None, description="Path to population YAML (legacy)"
    )
    study_db: str | None = Field(
        default=None, description="Path to canonical study DB (legacy)"
    )
    population_id: str = Field(
        default="default",
        description="[Deprecated] Population ID in study DB — use scenario name via scenario_id instead",
    )
    network_id: str = Field(
        default="default",
        description="[Deprecated] Network ID in study DB — use scenario name via scenario_id instead",
    )
    created_at: datetime = Field(default_factory=datetime.now)

    def get_population_ref(self) -> tuple[str, int | None]:
        """Parse the population reference from base_population or population_spec.

        Handles versioned references like 'population.v2' → ('population', 2)
        and plain names like 'population' → ('population', None).

        Returns:
            Tuple of (name, version) where version is None for unversioned refs.

        Raises:
            ValueError: If neither base_population nor population_spec is set.
        """
        ref = self.base_population or self.population_spec
        if ref is None:
            raise ValueError(
                "Scenario has no base_population or population_spec reference"
            )
        match = re.match(r"^(.+)\.v(\d+)$", ref)
        if match:
            return match.group(1), int(match.group(2))
        return ref, None


class IdentityDimension(BaseModel):
    """An identity dimension that may be threatened or activated by a scenario."""

    dimension: Literal[
        "political_orientation",
        "religious_affiliation",
        "race_ethnicity",
        "gender_identity",
        "sexual_orientation",
        "parental_status",
        "citizenship",
        "socioeconomic_class",
        "professional_identity",
        "generational_identity",
    ] = Field(description="The identity dimension being activated")
    relevance: str = Field(
        description="Why this dimension is relevant to the scenario (1-2 sentences)"
    )


PartnerCorrelationPolicy = Literal[
    "gaussian_offset",
    "same_group_rate",
    "same_country_rate",
    "same_value_probability",
]


class MaritalRoles(BaseModel):
    """Explicit marital semantics for deterministic household reconciliation."""

    attr: str | None = Field(
        default=None,
        description="Attribute name representing marital/relationship status",
    )
    partnered_values: list[str] = Field(
        default_factory=list,
        description="Categorical values indicating partnered state",
    )
    single_values: list[str] = Field(
        default_factory=list,
        description="Categorical values indicating single/non-partnered state",
    )


class GeoRoles(BaseModel):
    """Geographic attribute roles used for naming and contextual defaults."""

    country_attr: str | None = Field(
        default=None, description="Attribute name representing country"
    )
    region_attr: str | None = Field(
        default=None,
        description="Attribute name representing region/state/province/city scope",
    )
    urbanicity_attr: str | None = Field(
        default=None,
        description="Attribute name representing urbanicity (urban/rural/etc.)",
    )


class SchoolParentRole(BaseModel):
    """Role mapping for identifying school-parent contexts in structure generation."""

    dependents_attr: str | None = Field(
        default=None,
        description="Attribute name containing dependents/children list",
    )
    school_age_values: list[str] = Field(
        default_factory=list,
        description="Dependent school-status values treated as school-age",
    )


class ReligionRoles(BaseModel):
    """Role mapping for religion semantics in structural edge generation."""

    religion_attr: str | None = Field(
        default=None,
        description="Attribute name representing religion/faith affiliation",
    )
    secular_values: list[str] = Field(
        default_factory=list,
        description="Values treated as secular/no-religion",
    )


class HouseholdRoles(BaseModel):
    """Role mapping for household-coupled structural attributes."""

    household_size_attr: str | None = Field(
        default=None,
        description="Attribute name representing total realized household size",
    )


class SamplingSemanticRoles(BaseModel):
    """Scenario-owned semantic role mapping for deterministic sample/runtime logic."""

    marital_roles: MaritalRoles | None = Field(
        default=None,
        description="Marital/relationship status semantics",
    )
    geo_roles: GeoRoles | None = Field(
        default=None,
        description="Geographic role mapping",
    )
    partner_correlation_roles: dict[str, PartnerCorrelationPolicy] = Field(
        default_factory=dict,
        description="Per-attribute partner-correlation policy overrides",
    )
    school_parent_role: SchoolParentRole | None = Field(
        default=None,
        description="School-parent role mapping",
    )
    religion_roles: ReligionRoles | None = Field(
        default=None,
        description="Religion role mapping",
    )
    household_roles: HouseholdRoles | None = Field(
        default=None,
        description="Household structural role mapping",
    )


class ScenarioSpec(BaseModel):
    """Complete specification for a scenario simulation."""

    meta: ScenarioMeta
    event: Event
    timeline: list[TimelineEvent] | None = Field(
        default=None,
        description="Subsequent developments; None or empty = static scenario",
    )
    seed_exposure: SeedExposure
    interaction: InteractionConfig
    spread: SpreadConfig
    outcomes: OutcomeConfig
    simulation: ScenarioSimConfig
    background_context: str | None = Field(
        default=None,
        description="Optional background context injected into reasoning prompts",
    )
    relationship_weights: dict[str, float] | None = Field(
        default=None,
        description="Scenario-specific edge weights for conversation priority and peer ordering",
    )
    # Extended attributes from scenario (new CLI flow)
    extended_attributes: list[AttributeSpec] | None = Field(
        default=None,
        description="Scenario-specific attributes that extend the base population",
    )
    # Household configuration for sampling
    household_config: HouseholdConfig | None = Field(
        default=None,
        description="Household sampling config. Generated during scenario compilation.",
    )
    # Household agent scope — determines which household members become agents
    agent_focus_mode: Literal["primary_only", "couples", "all"] | None = Field(
        default=None,
        description="Household agent scope. primary_only: only primary adult is agent; "
        "couples: both partners; all: everyone including children. "
        "Inferred during scenario sufficiency check.",
    )
    # Identity dimensions activated by this scenario (for identity-threat framing)
    identity_dimensions: list[IdentityDimension] | None = Field(
        default=None,
        description="Identity dimensions that may feel threatened or activated by this scenario. Set by LLM during scenario creation.",
    )
    # Scenario-owned semantic role map for sample/network deterministic behavior
    sampling_semantic_roles: SamplingSemanticRoles | None = Field(
        default=None,
        description="Semantic role mappings generated during scenario compilation for deterministic sample/network execution.",
    )

    def to_yaml(self, path: Path | str) -> None:
        """Save scenario spec to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling datetime and enums
        data = self.model_dump(mode="json")

        with open(path, "w") as f:
            yaml.dump(
                data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ScenarioSpec":
        """Load scenario spec from YAML file."""
        path = Path(path)

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Scenario YAML must parse to an object")

        meta = data.get("meta", {})
        if isinstance(meta, dict) and ("agents_file" in meta or "network_file" in meta):
            raise ValueError(
                "Legacy scenario schema detected (meta.agents_file/meta.network_file). "
                "Migrate with: extropy migrate scenario --input "
                f"{path} --study-db study.db --population-id default --network-id default"
            )

        try:
            return cls.model_validate(data)
        except Exception as e:
            if isinstance(meta, dict) and (
                "study_db" not in meta
                or "population_id" not in meta
                or "network_id" not in meta
            ):
                raise ValueError(
                    "Scenario metadata must include meta.study_db, meta.population_id, "
                    "and meta.network_id. If this is an older scenario, run: "
                    "extropy migrate scenario --input "
                    f"{path} --study-db study.db --population-id default --network-id default"
                ) from e
            raise

    def summary(self) -> str:
        """Get a text summary of the scenario spec."""
        lines = [
            f"Scenario: {self.meta.name}",
            f"Event: {self.event.type.value} — {self.event.content[:50]}...",
            f"Source: {self.event.source} (credibility: {self.event.credibility:.2f})",
            "",
            f"Exposure channels: {len(self.seed_exposure.channels)}",
            f"Exposure rules: {len(self.seed_exposure.rules)}",
            "",
            f"Share probability: {self.spread.share_probability:.2f}",
            "",
            f"Outcomes: {len(self.outcomes.suggested_outcomes)}",
            f"Simulation: {self.simulation.max_timesteps} {self.simulation.timestep_unit.value}s",
        ]
        if self.interaction and self.interaction.description:
            lines.insert(
                8,
                f"Interaction notes: {self.interaction.description[:60]}...",
            )
        return "\n".join(lines)
