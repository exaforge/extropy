"""Scenario Compiler for Extropy (Phase 2).

The scenario compiler transforms natural language scenario descriptions
into machine-readable scenario specs that Phase 3 executes.

Pipeline:
    Step 1: parse_scenario() - Parse description into Event
    Step 2: generate_seed_exposure() - Generate exposure channels and rules
    Step 3: determine_interaction_model() - Select interaction model and spread config
    Step 4: define_outcomes() - Define what outcomes to measure
    Step 5: create_scenario() - Orchestrate full pipeline, assemble spec
    Step 6: validate_scenario() - Validate spec against population/network

Usage:
    >>> from extropy.scenario import create_scenario, ScenarioSpec
    >>> spec, result = create_scenario(
    ...     "Netflix announces $3 price increase",
    ...     "population.yaml",
    ...     study_db_path="study.db",
    ...     population_id="default",
    ...     network_id="default",
    ...     "scenario.yaml"
    ... )
    >>> result.valid
    True
"""

from ..core.models import (
    # Event
    EventType,
    Event,
    # Timeline
    TimelineEvent,
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
    # Simulation
    TimestepUnit,
    SimulationConfig,
    # Scenario
    ScenarioMeta,
    ScenarioSpec,
    # Validation
    Severity,
    ValidationIssue,
    ValidationResult,
)

from .parser import parse_scenario
from .exposure import generate_seed_exposure
from .interaction import determine_interaction_model
from .outcomes import define_outcomes
from .timeline import generate_timeline
from .compiler import create_scenario, compile_scenario_from_files
from .validator import validate_scenario, load_and_validate_scenario


__all__ = [
    # Models - Event
    "EventType",
    "Event",
    # Models - Timeline
    "TimelineEvent",
    # Models - Exposure
    "ExposureChannel",
    "ExposureRule",
    "SeedExposure",
    # Models - Interaction
    "InteractionType",
    "InteractionConfig",
    "SpreadModifier",
    "SpreadConfig",
    # Models - Outcomes
    "OutcomeType",
    "OutcomeDefinition",
    "OutcomeConfig",
    # Models - Simulation
    "TimestepUnit",
    "SimulationConfig",
    # Models - Scenario
    "ScenarioMeta",
    "ScenarioSpec",
    # Models - Validation
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    # Pipeline functions
    "parse_scenario",
    "generate_seed_exposure",
    "determine_interaction_model",
    "define_outcomes",
    "generate_timeline",
    "create_scenario",
    "compile_scenario_from_files",
    "validate_scenario",
    "load_and_validate_scenario",
]
