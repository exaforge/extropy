"""Simulation Engine module for Extropy (Phase 3).

Executes scenarios against populations, simulating opinion dynamics
with agent reasoning, network propagation, and state evolution.

Usage:
    from extropy.simulation import run_simulation, SimulationEngine

    # Simple API (model resolved from extropy config)
    summary = run_simulation(
        scenario_path="scenario.yaml",
        output_dir="results/",
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
    - study.db: Canonical SQLite database with simulation state/checkpoints
    - by_timestep.json: Metrics over time
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
    # New Phase 0 models
    ConvictionLevel,
    CONVICTION_MAP,
    CONVICTION_REVERSE_MAP,
    conviction_to_float,
    float_to_conviction,
    MemoryEntry,
)

from .engine import (
    SimulationEngine,
    SimulationSummary,
    run_simulation,
)

from .progress import SimulationProgress

from .state import StateManager
from .persona import generate_persona, render_persona
from .reasoning import (
    reason_agent,
    build_pass1_prompt,
    build_pass1_schema,
    build_pass2_prompt,
    build_pass2_schema,
    batch_reason_agents,
    batch_reason_agents_async,
    create_reasoning_context,
)
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
    "SimulationProgress",
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
    # Conviction
    "ConvictionLevel",
    "CONVICTION_MAP",
    "CONVICTION_REVERSE_MAP",
    "conviction_to_float",
    "float_to_conviction",
    # Memory
    "MemoryEntry",
    # State management
    "StateManager",
    # Persona generation
    "generate_persona",
    "render_persona",
    # Reasoning (two-pass)
    "reason_agent",
    "build_pass1_prompt",
    "build_pass1_schema",
    "build_pass2_prompt",
    "build_pass2_schema",
    "batch_reason_agents",
    "batch_reason_agents_async",
    "create_reasoning_context",
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
