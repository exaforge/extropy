"""All Pydantic models for Entropy, organized by domain.

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
    NumericModifier,
    CategoricalModifier,
    BooleanModifier,
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
    # Validation
    ValidationError,
    ValidationWarning,
    ValidationResult,
)

# Simulation models (Phase 3)
from .simulation import (
    SimulationEventType,
    ExposureRecord,
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
    "NumericModifier",
    "CategoricalModifier",
    "BooleanModifier",
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
    # Scenario - Validation
    "ValidationError",
    "ValidationWarning",
    "ValidationResult",
    # Simulation
    "SimulationEventType",
    "ExposureRecord",
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
]
