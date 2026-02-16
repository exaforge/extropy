"""All Pydantic models for Extropy, organized by domain.

This package centralizes all model definitions:
- population.py: Population specs, attributes, distributions, sampling configs
- scenario.py: Scenario specs, events, exposure rules, interaction models
- simulation.py: Agent state and runtime models
- results.py: Simulation results and aggregation models
"""

# Population models (Phase 1)
from .population import (
    # Grounding
    GroundingInfo,
    GroundingSummary,
    # Distributions
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    Distribution,
    # Modifiers
    Modifier,
    # Sampling
    SamplingConfig,
    Constraint,
    # Attributes
    AttributeSpec,
    DiscoveredAttribute,
    HydratedAttribute,
    # Spec
    SpecMeta,
    PopulationSpec,
    # Pipeline types
    SufficiencyResult,
    # Household models
    HouseholdType,
    Dependent,
    LifeStageThreshold,
    HouseholdConfig,
    # Name models
    NameEntry,
    NameConfig,
    STANDARD_PERSONALITY_ATTRIBUTES,
)

# Validation models (shared across population and scenario)
from .validation import (
    Severity,
    ValidationIssue,
    ValidationResult,
)

# Scenario models (Phase 2)
from .scenario import (
    # Event
    EventType,
    Event,
    # Exposure
    ExposureChannel,
    ExposureRule,
    SeedExposure,
    # Interaction
    InteractionType,
    InteractionConfig,
    SpreadModifier,
    SpreadConfig,
    # Outcomes
    OutcomeType,
    OutcomeDefinition,
    OutcomeConfig,
    # Simulation config
    TimestepUnit,
    SimulationConfig,
    # Scenario
    ScenarioMeta,
    ScenarioSpec,
)

# Simulation models (Phase 3)
from .simulation import (
    ConvictionLevel,
    CONVICTION_MAP,
    CONVICTION_REVERSE_MAP,
    conviction_to_float,
    float_to_conviction,
    score_to_conviction_float,
    SimulationEventType,
    ExposureRecord,
    MemoryEntry,
    AgentState,
    SimulationEvent,
    PeerOpinion,
    ReasoningContext,
    ReasoningResponse,
    SimulationRunConfig,
    TimestepSummary,
)

# Results models (Phase 4)
from .results import (
    SimulationSummary,
    AgentFinalState,
    SegmentAggregate,
    TimelinePoint,
    RunMeta,
    SimulationResults,
)

# Sampling models (runtime)
from .sampling import (
    SamplingStats,
    SamplingResult,
)

# Network models (runtime)
from .network import (
    Edge,
    NetworkResult,
    NetworkMetrics,
    NodeMetrics,
)

__all__ = [
    # Population - Grounding
    "GroundingInfo",
    "GroundingSummary",
    # Population - Distributions
    "NormalDistribution",
    "LognormalDistribution",
    "UniformDistribution",
    "BetaDistribution",
    "CategoricalDistribution",
    "BooleanDistribution",
    "Distribution",
    # Population - Modifiers
    "Modifier",
    # Population - Sampling
    "SamplingConfig",
    "Constraint",
    # Population - Attributes
    "AttributeSpec",
    "DiscoveredAttribute",
    "HydratedAttribute",
    # Population - Spec
    "SpecMeta",
    "PopulationSpec",
    # Population - Pipeline types
    "SufficiencyResult",
    # Population - Household models
    "HouseholdType",
    "Dependent",
    "LifeStageThreshold",
    "HouseholdConfig",
    # Population - Name models
    "NameEntry",
    "NameConfig",
    "STANDARD_PERSONALITY_ATTRIBUTES",
    # Scenario - Event
    "EventType",
    "Event",
    # Scenario - Exposure
    "ExposureChannel",
    "ExposureRule",
    "SeedExposure",
    # Scenario - Interaction
    "InteractionType",
    "InteractionConfig",
    "SpreadModifier",
    "SpreadConfig",
    # Scenario - Outcomes
    "OutcomeType",
    "OutcomeDefinition",
    "OutcomeConfig",
    # Scenario - Config
    "TimestepUnit",
    "SimulationConfig",
    # Scenario - Spec
    "ScenarioMeta",
    "ScenarioSpec",
    # Validation (shared)
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    # Simulation
    "ConvictionLevel",
    "CONVICTION_MAP",
    "CONVICTION_REVERSE_MAP",
    "conviction_to_float",
    "float_to_conviction",
    "score_to_conviction_float",
    "SimulationEventType",
    "ExposureRecord",
    "MemoryEntry",
    "AgentState",
    "SimulationEvent",
    "PeerOpinion",
    "ReasoningContext",
    "ReasoningResponse",
    "SimulationRunConfig",
    "TimestepSummary",
    # Results
    "SimulationSummary",
    "AgentFinalState",
    "SegmentAggregate",
    "TimelinePoint",
    "RunMeta",
    "SimulationResults",
    # Sampling
    "SamplingStats",
    "SamplingResult",
    # Network
    "Edge",
    "NetworkResult",
    "NetworkMetrics",
    "NodeMetrics",
]
