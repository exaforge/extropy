"""Pydantic models for the Simulation Engine (Phase 3).

Defines all state and event models used during simulation execution.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Event Types
# =============================================================================


class SimulationEventType(str, Enum):
    """Type of simulation event."""

    SEED_EXPOSURE = "seed_exposure"  # Initial exposure from channel
    NETWORK_EXPOSURE = "network_exposure"  # Heard from another agent
    AGENT_REASONED = "agent_reasoned"  # Agent processed and formed opinion
    AGENT_SHARED = "agent_shared"  # Agent shared with network
    STATE_CHANGED = "state_changed"  # Agent's position/intent changed


# =============================================================================
# Exposure Records
# =============================================================================


class ExposureRecord(BaseModel):
    """Record of a single exposure event for an agent."""

    timestep: int = Field(description="When this exposure occurred")
    channel: str = Field(description="Which channel exposed them (or 'network')")
    source_agent_id: str | None = Field(
        default=None, description="If from network, who told them"
    )
    content: str = Field(description="What they heard")
    credibility: float = Field(
        ge=0, le=1, description="Perceived credibility of this exposure"
    )


# =============================================================================
# Agent State
# =============================================================================


class AgentState(BaseModel):
    """Complete state of an agent during simulation."""

    agent_id: str = Field(description="Unique agent identifier")
    aware: bool = Field(default=False, description="Has heard about event")
    exposure_count: int = Field(default=0, description="How many times exposed")
    exposures: list[ExposureRecord] = Field(
        default_factory=list, description="History of exposures"
    )
    last_reasoning_timestep: int = Field(
        default=-1, description="When they last reasoned"
    )
    position: str | None = Field(
        default=None, description="Current position (from outcomes)"
    )
    sentiment: float | None = Field(default=None, description="Current sentiment")
    action_intent: str | None = Field(
        default=None, description="What they intend to do"
    )
    will_share: bool = Field(default=False, description="Will they propagate")
    outcomes: dict[str, Any] = Field(
        default_factory=dict, description="All extracted outcomes"
    )
    raw_reasoning: str | None = Field(
        default=None, description="Full reasoning text"
    )
    updated_at: int = Field(default=0, description="Last state change timestep")


# =============================================================================
# Simulation Events (Timeline)
# =============================================================================


class SimulationEvent(BaseModel):
    """A single event in the simulation timeline."""

    timestep: int = Field(description="When this event occurred")
    event_type: SimulationEventType = Field(description="Type of event")
    agent_id: str = Field(description="Which agent was involved")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Event-specific data"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Wall clock time"
    )


# =============================================================================
# Peer Opinions (for reasoning context)
# =============================================================================


class PeerOpinion(BaseModel):
    """Opinion of a connected peer (for social influence)."""

    agent_id: str = Field(description="The peer's ID")
    relationship: str = Field(description="Edge type (colleague, mentor, etc.)")
    position: str | None = Field(default=None, description="Their current position")
    sentiment: float | None = Field(default=None, description="Their sentiment")


# =============================================================================
# Reasoning Context
# =============================================================================


class ReasoningContext(BaseModel):
    """Context provided to agent for reasoning."""

    agent_id: str = Field(description="Agent being reasoned")
    persona: str = Field(description="Generated persona text")
    event_content: str = Field(description="What the event is")
    exposure_history: list[ExposureRecord] = Field(
        description="How they learned"
    )
    peer_opinions: list[PeerOpinion] = Field(
        default_factory=list, description="What neighbors think (if known)"
    )
    current_state: AgentState | None = Field(
        default=None, description="Previous state if re-reasoning"
    )


# =============================================================================
# Reasoning Response
# =============================================================================


class ReasoningResponse(BaseModel):
    """Response from agent reasoning LLM call."""

    position: str | None = Field(
        default=None, description="Maps to outcome"
    )
    sentiment: float | None = Field(default=None, description="Sentiment value")
    action_intent: str | None = Field(
        default=None, description="Intended action"
    )
    will_share: bool = Field(default=False, description="Will they share")
    reasoning: str = Field(
        default="", description="1-3 sentence explanation"
    )
    outcomes: dict[str, Any] = Field(
        default_factory=dict, description="All structured outcomes"
    )


# =============================================================================
# Simulation Configuration
# =============================================================================


class SimulationRunConfig(BaseModel):
    """Configuration for a simulation run."""

    scenario_path: str = Field(description="Path to scenario YAML")
    output_dir: str = Field(description="Directory for results output")
    model: str = Field(
        default="gpt-5-mini", description="LLM model for agent reasoning"
    )
    reasoning_effort: str = Field(
        default="low", description="Reasoning effort level"
    )
    multi_touch_threshold: int = Field(
        default=3, description="Re-reason after N new exposures"
    )
    max_retries: int = Field(default=3, description="Max LLM retry attempts")
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )


# =============================================================================
# Timestep Summary
# =============================================================================


class TimestepSummary(BaseModel):
    """Summary statistics for a single timestep."""

    timestep: int = Field(description="Timestep number")
    new_exposures: int = Field(default=0, description="New exposures this step")
    agents_reasoned: int = Field(
        default=0, description="Agents who reasoned this step"
    )
    shares_occurred: int = Field(
        default=0, description="Share events this step"
    )
    state_changes: int = Field(
        default=0, description="Agents whose state changed"
    )
    exposure_rate: float = Field(
        default=0.0, description="Fraction of population aware"
    )
    position_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count per position"
    )
    average_sentiment: float | None = Field(
        default=None, description="Mean sentiment of aware agents"
    )
