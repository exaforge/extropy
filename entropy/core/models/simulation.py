"""Simulation Engine models for Entropy (Phase 3).

Defines all state and event models used during simulation execution:
- SimulationEventType: Enum of event types
- ExposureRecord: Record of a single exposure
- AgentState: Complete agent state during simulation
- SimulationEvent: Timeline event
- PeerOpinion: Opinion of connected peer
- ReasoningContext: Context for agent reasoning
- ReasoningResponse: Response from agent LLM call (Pass 1)
- ClassificationResponse: Response from classification pass (Pass 2)
- MemoryEntry: Agent reasoning memory trace
- SimulationRunConfig: Configuration for a run
- TimestepSummary: Summary statistics per timestep
- ConvictionLevel: Categorical conviction levels
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Conviction Levels
# =============================================================================


class ConvictionLevel(str, Enum):
    """Categorical conviction levels.

    The agent picks from these in Pass 1. Mapped to floats for storage/thresholds.
    The agent never sees the numeric value.
    """

    VERY_UNCERTAIN = "very_uncertain"
    LEANING = "leaning"
    MODERATE = "moderate"
    FIRM = "firm"
    ABSOLUTE = "absolute"


# Map conviction levels to float values for storage and threshold math
CONVICTION_MAP: dict[str, float] = {
    ConvictionLevel.VERY_UNCERTAIN: 0.1,
    ConvictionLevel.LEANING: 0.3,
    ConvictionLevel.MODERATE: 0.5,
    ConvictionLevel.FIRM: 0.7,
    ConvictionLevel.ABSOLUTE: 0.9,
}

# Reverse map: float -> level name (for display in re-reasoning prompts)
CONVICTION_REVERSE_MAP: dict[float, str] = {v: k for k, v in CONVICTION_MAP.items()}


def conviction_to_float(level: str | None) -> float | None:
    """Convert a conviction level string to its float value."""
    if level is None:
        return None
    return CONVICTION_MAP.get(level)


def float_to_conviction(value: float | None) -> str | None:
    """Convert a float to the nearest conviction level string."""
    if value is None:
        return None
    # Find nearest level
    closest = min(CONVICTION_MAP.items(), key=lambda x: abs(x[1] - value))
    return closest[0].value


def score_to_conviction_float(score: int | float | None) -> float | None:
    """Convert a 0-100 conviction score to conviction float.

    Buckets:
        0-15  → very_uncertain (0.1)
        16-35 → leaning (0.3)
        36-60 → moderate (0.5)
        61-85 → firm (0.7)
        86-100 → absolute (0.9)
    """
    if score is None:
        return None
    score = int(score)
    if score <= 15:
        return 0.1  # very_uncertain
    elif score <= 35:
        return 0.3  # leaning
    elif score <= 60:
        return 0.5  # moderate
    elif score <= 85:
        return 0.7  # firm
    else:
        return 0.9  # absolute


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
# Memory Entry
# =============================================================================


class MemoryEntry(BaseModel):
    """A single entry in an agent's reasoning memory trace.

    Stored after each reasoning pass — capped at 3 entries per agent.
    Fed back into the prompt so agents remember their own reasoning history.
    """

    timestep: int = Field(description="When this reasoning occurred")
    sentiment: float | None = Field(default=None, description="Sentiment at this time")
    conviction: float | None = Field(
        default=None, description="Conviction float value at this time"
    )
    summary: str = Field(description="1-sentence summary of reasoning at this time")


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
        default=None, description="Current position (from Pass 2 classification)"
    )
    sentiment: float | None = Field(
        default=None, description="Current sentiment (-1 to 1)"
    )
    conviction: float | None = Field(
        default=None, description="Conviction as float (from categorical level)"
    )
    public_statement: str | None = Field(
        default=None, description="1-2 sentence argument for peers"
    )
    action_intent: str | None = Field(
        default=None, description="What they intend to do"
    )
    will_share: bool = Field(default=False, description="Will they propagate")
    committed: bool = Field(
        default=False,
        description="Conviction >= firm; won't re-reason from repeated network exposure",
    )
    outcomes: dict[str, Any] = Field(
        default_factory=dict, description="All extracted outcomes (from Pass 2)"
    )
    raw_reasoning: str | None = Field(default=None, description="Full reasoning text")
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
    """Opinion of a connected peer (for social influence).

    In the redesigned simulation, peers influence via public_statement + sentiment,
    NOT position labels. Position is output-only (Pass 2).
    """

    agent_id: str = Field(description="The peer's ID")
    relationship: str = Field(description="Edge type (colleague, mentor, etc.)")
    position: str | None = Field(
        default=None, description="Their current position (for backwards compat only)"
    )
    sentiment: float | None = Field(default=None, description="Their sentiment")
    public_statement: str | None = Field(
        default=None, description="Their argument/statement"
    )
    credibility: float = Field(
        default=0.85, description="Dynamic credibility of this peer"
    )


# =============================================================================
# Reasoning Context
# =============================================================================


class ReasoningContext(BaseModel):
    """Context provided to agent for reasoning."""

    agent_id: str = Field(description="Agent being reasoned")
    persona: str = Field(description="Generated persona text")
    event_content: str = Field(description="What the event is")
    exposure_history: list[ExposureRecord] = Field(description="How they learned")
    peer_opinions: list[PeerOpinion] = Field(
        default_factory=list, description="What neighbors think (if known)"
    )
    current_state: AgentState | None = Field(
        default=None, description="Previous state if re-reasoning"
    )
    memory_trace: list[MemoryEntry] = Field(
        default_factory=list, description="Agent's previous reasoning summaries (max 3)"
    )


# =============================================================================
# Reasoning Response (Pass 1 — free-text role-play)
# =============================================================================


class ReasoningResponse(BaseModel):
    """Response from agent reasoning LLM call (Pass 1).

    This is the free-text role-play pass. No categorical enums here —
    the agent reasons naturally. Classification happens in Pass 2.
    """

    position: str | None = Field(
        default=None, description="Classified position (filled by Pass 2)"
    )
    sentiment: float | None = Field(
        default=None, description="Sentiment value (-1 to 1)"
    )
    conviction: float | None = Field(
        default=None, description="Conviction as float (mapped from categorical)"
    )
    public_statement: str | None = Field(
        default=None, description="1-2 sentence argument for peers"
    )
    reasoning_summary: str | None = Field(
        default=None, description="1-sentence summary for memory trace"
    )
    action_intent: str | None = Field(default=None, description="Intended action")
    will_share: bool = Field(default=False, description="Will they share")
    reasoning: str = Field(default="", description="Full internal monologue")
    outcomes: dict[str, Any] = Field(
        default_factory=dict, description="All structured outcomes (from Pass 2)"
    )


# =============================================================================
# Simulation Configuration
# =============================================================================


class SimulationRunConfig(BaseModel):
    """Configuration for a simulation run."""

    scenario_path: str = Field(description="Path to scenario YAML")
    output_dir: str = Field(description="Directory for results output")
    model: str = Field(
        default="",
        description="LLM model for agent reasoning (empty = use config default)",
    )
    pivotal_model: str = Field(
        default="",
        description="Model for pivotal reasoning (default: same as model)",
    )
    routine_model: str = Field(
        default="",
        description="Cheap model for routine reasoning + classification (default: provider cheap tier)",
    )
    reasoning_effort: str = Field(default="low", description="Reasoning effort level")
    multi_touch_threshold: int = Field(
        default=3, description="Re-reason after N new exposures"
    )
    max_retries: int = Field(default=3, description="Max LLM retry attempts")
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    chunk_size: int = Field(
        default=50, description="Agents per reasoning chunk for checkpointing"
    )


# =============================================================================
# Timestep Summary
# =============================================================================


class TimestepSummary(BaseModel):
    """Summary statistics for a single timestep."""

    timestep: int = Field(description="Timestep number")
    new_exposures: int = Field(default=0, description="New exposures this step")
    agents_reasoned: int = Field(default=0, description="Agents who reasoned this step")
    shares_occurred: int = Field(default=0, description="Share events this step")
    state_changes: int = Field(default=0, description="Agents whose state changed")
    exposure_rate: float = Field(
        default=0.0, description="Fraction of population aware"
    )
    position_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count per position"
    )
    average_sentiment: float | None = Field(
        default=None, description="Mean sentiment of aware agents"
    )
    average_conviction: float | None = Field(
        default=None, description="Mean conviction of aware agents"
    )
    sentiment_variance: float | None = Field(
        default=None, description="Variance of sentiment (for convergence detection)"
    )
