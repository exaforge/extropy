"""Scenario Compiler for Extropy (Phase 2).

The scenario compiler transforms natural language scenario descriptions
into machine-readable scenario specs that Phase 3 executes.

Pipeline:
    Step 0: check_scenario_sufficiency() - Verify description completeness
    Step 1: parse_scenario() - Parse description into Event
    Step 2: generate_seed_exposure() - Generate exposure channels and rules
    Step 3: determine_interaction_model() - Select interaction model and spread config
    Step 4: generate_timeline_and_outcomes() - Timeline, outcomes, and background context
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
    InteractionConfig,
    SpreadModifier,
    SpreadConfig,
    # Outcomes
    OutcomeType,
    OutcomeDefinition,
    OutcomeConfig,
    # Simulation
    TimestepUnit,
    ScenarioSimConfig,
    # Scenario
    ScenarioMeta,
    ScenarioSpec,
    SamplingSemanticRoles,
    # Validation
    Severity,
    ValidationIssue,
    ValidationResult,
)

from .parser import parse_scenario
from .exposure import generate_seed_exposure
from .interaction import determine_interaction_model
from .timeline import generate_timeline_and_outcomes
from .compiler import create_scenario, create_scenario_spec, compile_scenario_from_files
from .validator import validate_scenario, load_and_validate_scenario
from .sufficiency import (
    check_scenario_sufficiency,
    check_scenario_sufficiency_with_answers,
    ScenarioSufficiencyResult,
)


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
    "InteractionConfig",
    "SpreadModifier",
    "SpreadConfig",
    # Models - Outcomes
    "OutcomeType",
    "OutcomeDefinition",
    "OutcomeConfig",
    # Models - Simulation
    "TimestepUnit",
    "ScenarioSimConfig",
    # Models - Scenario
    "ScenarioMeta",
    "ScenarioSpec",
    "SamplingSemanticRoles",
    # Models - Validation
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    # Pipeline functions
    "parse_scenario",
    "generate_seed_exposure",
    "determine_interaction_model",
    "generate_timeline_and_outcomes",
    "create_scenario",
    "create_scenario_spec",
    "compile_scenario_from_files",
    "validate_scenario",
    "load_and_validate_scenario",
    # Sufficiency check
    "check_scenario_sufficiency",
    "check_scenario_sufficiency_with_answers",
    "ScenarioSufficiencyResult",
]
