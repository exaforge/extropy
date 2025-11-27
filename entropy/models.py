"""Pydantic models for Entropy."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Phase 1: Population Creation Models
# =============================================================================


class ParsedContext(BaseModel):
    """Structured output from context parsing."""

    size: int = Field(description="Number of agents to create")
    base_population: str = Field(description="Base population type (e.g., 'US adults')")
    context_type: str = Field(description="Type of context (e.g., 'subscription', 'ownership')")
    context_entity: str | None = Field(default=None, description="Entity (e.g., 'Netflix', 'Tesla')")
    geography: str | None = Field(default=None, description="Geographic filter")
    filters: list[str] = Field(default_factory=list, description="Additional filters")


class Demographics(BaseModel):
    """Fixed demographic attributes."""

    age: int
    gender: str
    income: int
    education: str  # high_school, some_college, bachelors, masters, doctorate
    occupation: str
    location: dict = Field(default_factory=lambda: {"state": "", "urban_rural": ""})
    ethnicity: str
    marital_status: str
    household_size: int


class Psychographics(BaseModel):
    """Big Five personality traits and values."""

    openness: float = Field(ge=0, le=1)
    conscientiousness: float = Field(ge=0, le=1)
    extraversion: float = Field(ge=0, le=1)
    agreeableness: float = Field(ge=0, le=1)
    neuroticism: float = Field(ge=0, le=1)
    values: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)


class Cognitive(BaseModel):
    """Cognitive traits affecting information processing and influence."""

    information_processing: str  # analytical, intuitive, balanced
    openness_to_change: float = Field(ge=0, le=1)
    trust_in_institutions: float = Field(ge=0, le=1)
    confirmation_bias: float = Field(ge=0, le=1, description="Affects processing of contradicting info")
    persuadability: float = Field(ge=0, le=1, description="Affects social influence susceptibility")


class InformationEnvironment(BaseModel):
    """Media consumption and information exposure patterns."""

    news_sources: list[str] = Field(default_factory=list)
    social_media: list[str] = Field(default_factory=list)
    media_hours_daily: float = Field(ge=0)
    trust_in_media: float = Field(ge=0, le=1)
    exposure_rate: float = Field(ge=0, le=1, description="How likely to see new info")


class Connection(BaseModel):
    """A connection to another agent."""

    target_id: str
    strength: float = Field(ge=0, le=1)
    connection_type: str = Field(default="social")  # social, family, work, etc.


class Network(BaseModel):
    """Agent's network position and connections."""

    connections: list[Connection] = Field(default_factory=list)
    influence_score: float = Field(default=0.0, ge=0, description="Derived from network position")


class AgentState(BaseModel):
    """Mutable state updated during simulation (Phase 3)."""

    beliefs: dict[str, Any] = Field(default_factory=dict)
    exposures: list[str] = Field(default_factory=list, description="Events they've seen")
    emotional_state: str = Field(default="neutral")


class Agent(BaseModel):
    """A synthetic agent in a population."""

    id: str
    demographics: Demographics
    psychographics: Psychographics
    cognitive: Cognitive
    information_env: InformationEnvironment
    situation: dict[str, Any] = Field(
        default_factory=dict,
        description="Context-specific attributes, populated dynamically based on research",
    )
    network: Network = Field(default_factory=Network)
    persona: str = Field(default="", description="Natural language persona description")
    state: AgentState = Field(default_factory=AgentState)


class SituationSchema(BaseModel):
    """Schema for dynamically generated situation attributes."""

    name: str
    field_type: str  # int, float, str, list
    description: str
    min_value: float | None = None
    max_value: float | None = None
    options: list[str] | None = None


class ResearchData(BaseModel):
    """Research results from web search and extraction."""

    demographics: dict[str, Any] = Field(default_factory=dict)
    psychographics: dict[str, Any] = Field(default_factory=dict)
    situation_schema: list[SituationSchema] = Field(default_factory=list)
    situation_distributions: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    grounding_level: str = Field(default="low")  # low, medium, strong


class Population(BaseModel):
    """A population of agents."""

    name: str
    size: int
    context_raw: str = Field(description="Original natural language context")
    context_parsed: ParsedContext
    research: ResearchData = Field(default_factory=ResearchData)
    agents: list[Agent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Phase 2: Scenario Models
# =============================================================================


class Scenario(BaseModel):
    """A scenario to inject into a population."""

    name: str
    population_name: str
    description: str
    event_type: str
    channels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Phase 3: Simulation Models
# =============================================================================


class SimulationResult(BaseModel):
    """Results from a simulation run."""

    population_name: str
    scenario_name: str
    mode: str  # single, continuous
    results: dict[str, Any] = Field(default_factory=dict)
    timeline: list[dict[str, Any]] | None = None
    created_at: datetime = Field(default_factory=datetime.now)

