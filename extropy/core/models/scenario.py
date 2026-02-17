"""Scenario models for Extropy (Phase 2).

A ScenarioSpec defines how an event/information propagates through a population
and what outcomes to measure. It is the bridge between population creation (Phase 1)
and simulation execution (Phase 3).

This module contains:
- Event: EventType, Event
- Exposure: ExposureChannel, ExposureRule, SeedExposure
- Interaction: InteractionType, InteractionConfig, SpreadModifier, SpreadConfig
- Outcomes: OutcomeType, OutcomeDefinition, OutcomeConfig
- Config: TimestepUnit, SimulationConfig
- Spec: ScenarioMeta, ScenarioSpec with YAML I/O
- Validation: ValidationError, ValidationWarning, ValidationResult
"""

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

from .population import AttributeSpec

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
    reach: Literal["broadcast", "targeted", "organic"] = Field(
        description="broadcast=everyone, targeted=specific criteria, organic=through network"
    )
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


# =============================================================================
# Interaction Model
# =============================================================================


class InteractionType(str, Enum):
    """Type of agent interaction model."""

    PASSIVE_OBSERVATION = "passive_observation"  # Social media style
    DIRECT_CONVERSATION = "direct_conversation"  # One-on-one or small group
    BROADCAST_RESPONSE = "broadcast_response"  # Authority broadcasts, agents react
    DELIBERATIVE = "deliberative"  # Group deliberation with multiple rounds


class InteractionConfig(BaseModel):
    """Configuration for how agents interact about the event."""

    primary_model: InteractionType = Field(
        description="Primary interaction model for this scenario"
    )
    secondary_model: InteractionType | None = Field(
        default=None,
        description="Optional secondary interaction model (for blended scenarios)",
    )
    description: str = Field(
        description="Human-readable description of how interactions work"
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


class SimulationConfig(BaseModel):
    """Configuration for simulation execution."""

    max_timesteps: int = Field(ge=1, description="Maximum number of timesteps to run")
    timestep_unit: TimestepUnit = Field(
        default=TimestepUnit.HOUR, description="What each timestep represents"
    )
    stop_conditions: list[str] | None = Field(
        default=None,
        description="Conditions that trigger early stop (e.g., 'exposure_rate > 0.95')",
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
        default="default", description="[Deprecated] Population ID in study DB — use scenario name via scenario_id instead"
    )
    network_id: str = Field(default="default", description="[Deprecated] Network ID in study DB — use scenario name via scenario_id instead")
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
    simulation: SimulationConfig
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
            f"Interaction: {self.interaction.primary_model.value}",
            f"Share probability: {self.spread.share_probability:.2f}",
            "",
            f"Outcomes: {len(self.outcomes.suggested_outcomes)}",
            f"Simulation: {self.simulation.max_timesteps} {self.simulation.timestep_unit.value}s",
        ]
        return "\n".join(lines)
