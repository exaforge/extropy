"""Pydantic models for simulation results.

Defines the structure of results data for loading, querying,
and displaying simulation outcomes.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SimulationSummary(BaseModel):
    """Summary of a completed simulation run."""

    scenario_name: str = Field(description="Name of the scenario")
    population_size: int = Field(description="Number of agents")
    total_timesteps: int = Field(description="Timesteps completed")
    stopped_reason: str | None = Field(
        default=None, description="Why simulation stopped"
    )
    total_reasoning_calls: int = Field(description="Total LLM calls made")
    total_exposures: int = Field(description="Total exposure events")
    final_exposure_rate: float = Field(description="Final awareness rate")
    outcome_distributions: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Distribution of outcomes"
    )
    runtime_seconds: float = Field(description="Total runtime")
    model_used: str = Field(description="LLM model used")
    completed_at: datetime = Field(description="When simulation completed")


class AgentFinalState(BaseModel):
    """Final state of a single agent."""

    agent_id: str = Field(description="Agent identifier")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Agent attributes from population"
    )
    aware: bool = Field(default=False, description="Was exposed to event")
    exposure_count: int = Field(default=0, description="Number of exposures")
    position: str | None = Field(default=None, description="Final position")
    sentiment: float | None = Field(default=None, description="Final sentiment")
    action_intent: str | None = Field(default=None, description="Intended action")
    will_share: bool = Field(default=False, description="Will share with others")
    raw_reasoning: str | None = Field(default=None, description="LLM reasoning text")
    outcomes: dict[str, Any] = Field(
        default_factory=dict, description="All extracted outcomes"
    )
    reasoning_count: int = Field(default=0, description="Times agent reasoned")


class SegmentAggregate(BaseModel):
    """Aggregate statistics for a population segment."""

    segment_attribute: str = Field(description="Attribute used for segmentation")
    segment_value: str = Field(description="Value of the segment attribute")
    agent_count: int = Field(description="Number of agents in segment")
    aware_count: int = Field(default=0, description="Number aware of event")
    position_distribution: dict[str, float] = Field(
        default_factory=dict, description="Position distribution (normalized)"
    )
    position_counts: dict[str, int] = Field(
        default_factory=dict, description="Position counts (raw)"
    )
    average_sentiment: float | None = Field(
        default=None, description="Mean sentiment in segment"
    )


class TimelinePoint(BaseModel):
    """A single point in the simulation timeline."""

    timestep: int = Field(description="Timestep number")
    exposure_rate: float = Field(description="Fraction of population aware")
    position_distribution: dict[str, float] = Field(
        default_factory=dict, description="Position distribution at this timestep"
    )
    average_sentiment: float | None = Field(
        default=None, description="Mean sentiment at this timestep"
    )
    cumulative_shares: int = Field(
        default=0, description="Total share events up to this point"
    )
    new_exposures: int = Field(default=0, description="New exposures this timestep")
    agents_reasoned: int = Field(
        default=0, description="Agents who reasoned this timestep"
    )


class RunMeta(BaseModel):
    """Metadata about a simulation run."""

    scenario_name: str = Field(description="Name of the scenario")
    scenario_path: str = Field(description="Path to scenario file")
    population_size: int = Field(description="Number of agents")
    model: str = Field(description="LLM model used")
    seed: int = Field(description="Random seed used")
    multi_touch_threshold: int = Field(description="Multi-touch threshold")
    completed_at: datetime = Field(description="When simulation completed")


class SimulationResults(BaseModel):
    """Complete simulation results."""

    meta: RunMeta = Field(description="Run metadata")
    summary: SimulationSummary | None = Field(
        default=None, description="Quick summary"
    )
    timeline: list[TimelinePoint] = Field(
        default_factory=list, description="Timeline data"
    )
    agent_states: list[AgentFinalState] = Field(
        default_factory=list, description="Final agent states"
    )
    segments: dict[str, list[SegmentAggregate]] = Field(
        default_factory=dict, description="Segment breakdowns by attribute"
    )
    outcome_distributions: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Final outcome distributions"
    )
