"""Simulation Engine module for Entropy (Phase 3).

Executes scenarios against populations, simulating opinion dynamics
with agent reasoning, network propagation, and state evolution.

Usage:
    from entropy.simulation import run_simulation, SimulationEngine

    # Simple API
    summary = run_simulation(
        scenario_path="scenario.yaml",
        output_dir="results/",
        model="gpt-5-mini",
    )

    # Or manual control
    engine = SimulationEngine(scenario, pop_spec, agents, network, config)
    summary = engine.run()

Key Concepts:
    - Seed Exposure: Agents exposed via channels (email, social media, etc.)
    - Network Propagation: Agents share with neighbors
    - Agent Reasoning: LLM-based opinion formation
    - Multi-touch: Re-reasoning after multiple exposures
    - Stopping Conditions: Exposure rate, convergence, max timesteps

Output:
    Results directory containing:
    - simulation.db: SQLite database with all state
    - timeline.jsonl: Streaming event log
    - agent_states.json: Final state per agent
    - by_timestep.json: Metrics over time
    - outcome_distributions.json: Final outcome distributions
    - meta.json: Run configuration
"""

from ..core.models import (
    # Event types
    SimulationEventType,
    # State models
    ExposureRecord,
    AgentState,
    SimulationEvent,
    PeerOpinion,
    ReasoningContext,
    ReasoningResponse,
    SimulationRunConfig,
    TimestepSummary,
)

from .engine import (
    SimulationEngine,
    SimulationSummary,
    run_simulation,
)

from .state import StateManager
from .persona import generate_persona, generate_persona_for_reasoning
from .reasoning import reason_agent, build_reasoning_prompt, build_response_schema
from .propagation import (
    apply_seed_exposures,
    propagate_through_network,
    evaluate_exposure_rule,
)
from .stopping import evaluate_stopping_conditions
from .timeline import TimelineManager, TimelineReader
from .aggregation import (
    compute_timestep_summary,
    compute_final_aggregates,
    compute_segment_aggregates,
    compute_outcome_distributions,
    compute_timeline_aggregates,
)

__all__ = [
    # Main API
    "run_simulation",
    "SimulationEngine",
    "SimulationSummary",
    # Models
    "SimulationEventType",
    "ExposureRecord",
    "AgentState",
    "SimulationEvent",
    "PeerOpinion",
    "ReasoningContext",
    "ReasoningResponse",
    "SimulationRunConfig",
    "TimestepSummary",
    # State management
    "StateManager",
    # Persona generation
    "generate_persona",
    "generate_persona_for_reasoning",
    # Reasoning
    "reason_agent",
    "build_reasoning_prompt",
    "build_response_schema",
    # Exposure
    "apply_seed_exposures",
    "propagate_through_network",
    "evaluate_exposure_rule",
    # Stopping
    "evaluate_stopping_conditions",
    # Timeline
    "TimelineManager",
    "TimelineReader",
    # Aggregation
    "compute_timestep_summary",
    "compute_final_aggregates",
    "compute_segment_aggregates",
    "compute_outcome_distributions",
    "compute_timeline_aggregates",
]
