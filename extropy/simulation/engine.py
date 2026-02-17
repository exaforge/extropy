"""Main simulation engine orchestrator.

Coordinates the simulation loop: exposure, reasoning, propagation,
and aggregation across timesteps until stopping conditions are met.

Implements Phase 0 redesign:
- Two-pass reasoning (Pass 1: role-play, Pass 2: classification)
- Conviction-gated sharing (very_uncertain agents don't share)
- Conviction-based flip resistance
- Memory traces (sliding window of 3 per agent)
- Semantic peer influence (public_statement + sentiment, not position labels)
- Rate limiter integration (replaces hardcoded semaphore)
"""

import json
import logging
import queue
import random
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.models import (
    PopulationSpec,
    ScenarioSpec,
    AgentState,
    ConvictionLevel,
    CONVICTION_MAP,
    MemoryEntry,
    PeerOpinion,
    ReasoningContext,
    ReasoningResponse,
    SimulationEvent,
    SimulationEventType,
    SimulationRunConfig,
    TimestepSummary,
    float_to_conviction,
)
from ..core.rate_limiter import DualRateLimiter
from ..population.persona import PersonaConfig
from ..storage import open_study_db
from .progress import SimulationProgress
from .state import StateManager
from .persona import generate_persona
from .reasoning import (
    batch_reason_agents_async,
    create_reasoning_context,
)
from .conversation import (
    collect_conversation_requests,
    prioritize_and_resolve_conflicts,
    execute_conversation_batch_async,
    ConversationResult,
)
from .propagation import (
    apply_seed_exposures,
    apply_timeline_exposures,
    propagate_through_network,
)
from .stopping import evaluate_stopping_conditions
from ..utils.callbacks import TimestepProgressCallback
from ..utils.resource_governor import ResourceGovernor
from .aggregation import (
    compute_timestep_summary,
    compute_final_aggregates,
    compute_outcome_distributions,
    compute_timeline_aggregates,
    compute_conversation_stats,
    compute_most_impactful_conversations,
    export_elaborations_csv,
)

logger = logging.getLogger(__name__)

# Conviction thresholds derived from the canonical model
_FIRM_CONVICTION = CONVICTION_MAP[ConvictionLevel.FIRM]
_MODERATE_CONVICTION = CONVICTION_MAP[ConvictionLevel.MODERATE]
_SHARING_CONVICTION_THRESHOLD = CONVICTION_MAP[ConvictionLevel.VERY_UNCERTAIN]
_CONVICTION_DECAY_RATE = 0.05
_BOUNDED_CONFIDENCE_RHO = 0.35
_PRIVATE_ADJUSTMENT_RHO = 0.12
_PRIVATE_FLIP_CONVICTION = CONVICTION_MAP[ConvictionLevel.FIRM]


class _StateTimelineAdapter:
    """Timeline adapter that persists events into StateManager timeline table."""

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def log_event(self, event: SimulationEvent) -> None:
        self.state_manager.log_event(event)

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class SimulationSummary:
    """Summary of a completed simulation run."""

    def __init__(
        self,
        scenario_name: str,
        run_id: str | None,
        population_size: int,
        total_timesteps: int,
        stopped_reason: str | None,
        total_reasoning_calls: int,
        total_exposures: int,
        final_exposure_rate: float,
        outcome_distributions: dict[str, Any],
        runtime_seconds: float,
        model_used: str,
        completed_at: datetime,
    ):
        self.scenario_name = scenario_name
        self.run_id = run_id
        self.population_size = population_size
        self.total_timesteps = total_timesteps
        self.stopped_reason = stopped_reason
        self.total_reasoning_calls = total_reasoning_calls
        self.total_exposures = total_exposures
        self.final_exposure_rate = final_exposure_rate
        self.outcome_distributions = outcome_distributions
        self.runtime_seconds = runtime_seconds
        self.model_used = model_used
        self.completed_at = completed_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "run_id": self.run_id,
            "population_size": self.population_size,
            "total_timesteps": self.total_timesteps,
            "stopped_reason": self.stopped_reason,
            "total_reasoning_calls": self.total_reasoning_calls,
            "total_exposures": self.total_exposures,
            "final_exposure_rate": self.final_exposure_rate,
            "outcome_distributions": self.outcome_distributions,
            "runtime_seconds": self.runtime_seconds,
            "model_used": self.model_used,
            "completed_at": self.completed_at.isoformat(),
        }


class SimulationEngine:
    """Main orchestrator for simulation execution.

    Coordinates the simulation loop: exposure, reasoning, propagation,
    and aggregation across timesteps.
    """

    def __init__(
        self,
        scenario: ScenarioSpec,
        population_spec: PopulationSpec,
        agents: list[dict[str, Any]],
        network: dict[str, Any],
        config: SimulationRunConfig,
        persona_config: PersonaConfig | None = None,
        rate_limiter: DualRateLimiter | None = None,
        chunk_size: int = 50,
        state_db_path: Path | str | None = None,
        run_id: str | None = None,
        checkpoint_every_chunks: int = 1,
        retention_lite: bool = False,
        writer_queue_size: int = 256,
        db_write_batch_size: int = 100,
        resource_governor: ResourceGovernor | None = None,
    ):
        """Initialize simulation engine.

        Args:
            scenario: Scenario specification
            population_spec: Population specification
            agents: List of agent dictionaries
            network: Network data
            config: Simulation run configuration
            persona_config: Optional PersonaConfig for embodied persona rendering
            rate_limiter: Optional DualRateLimiter for API pacing (pivotal + routine)
            chunk_size: Number of agents per reasoning chunk for checkpointing
        """
        self.scenario = scenario
        self.population_spec = population_spec
        self.agents = agents
        self.network = network
        self.config = config
        self.persona_config = persona_config
        self.rate_limiter = rate_limiter
        self.chunk_size = chunk_size  # updated below after concurrency is resolved
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.checkpoint_every_chunks = max(1, checkpoint_every_chunks)
        self.retention_lite = retention_lite
        self.writer_queue_size = max(1, writer_queue_size)
        self.db_write_batch_size = max(1, db_write_batch_size)
        self.resource_governor = resource_governor
        # Auto-derive from rate limiter RPM, or use explicit override
        if config.max_concurrent is not None:
            self.reasoning_max_concurrency = config.max_concurrent
        elif rate_limiter:
            self.reasoning_max_concurrency = rate_limiter.pivotal.max_safe_concurrent
        else:
            self.reasoning_max_concurrency = 50
        # Auto-size chunk_size to match concurrency when using the default.
        # Small explicit chunk sizes (for fine-grained checkpointing) are respected.
        if chunk_size == 50 and self.chunk_size < self.reasoning_max_concurrency:
            self.chunk_size = self.reasoning_max_concurrency
            logger.info(
                f"[ENGINE] Auto-sized chunk_size to {self.chunk_size} "
                f"to match concurrency"
            )
        self._last_guardrail_timestep = -1

        # Build agent map for quick lookup
        self.agent_map = {a.get("_id", str(i)): a for i, a in enumerate(agents)}

        # Pre-build agent name lookup for peer reference resolution
        self._agent_names: dict[str, str] = {
            aid: a.get("first_name", "")
            for aid, a in self.agent_map.items()
            if a.get("first_name")
        }

        # Build adjacency list for O(1) neighbor lookups (both directions)
        self.adjacency: dict[str, list[tuple[str, dict]]] = {}
        for edge in network.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            if src is not None and tgt is not None:
                self.adjacency.setdefault(src, []).append((tgt, edge))
                self.adjacency.setdefault(tgt, []).append((src, edge))

        # Initialize RNG
        seed = config.random_seed
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.rng = random.Random(seed)
        self.seed = seed

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state manager
        state_db_file = (
            Path(state_db_path) if state_db_path else self.output_dir / "study.db"
        )
        self.state_manager = StateManager(
            state_db_file,
            agents,
            run_id=self.run_id,
        )
        self.study_db = open_study_db(state_db_file)

        # Initialize timeline manager
        self.timeline = _StateTimelineAdapter(self.state_manager)

        # Pre-generate personas for all agents
        # Extract decision-relevant attributes from outcome config (trait salience)
        decision_attrs = (
            scenario.outcomes.decision_relevant_attributes
            if hasattr(scenario.outcomes, "decision_relevant_attributes")
            else None
        )
        self._personas: dict[str, str] = {}
        for i, agent in enumerate(agents):
            agent_id = agent.get("_id", str(i))
            self._personas[agent_id] = generate_persona(
                agent,
                population_spec,
                persona_config=persona_config,
                decision_relevant_attributes=decision_attrs or None,
            )

        # Main categorical outcome used as the canonical behavioral position.
        categorical = [
            o
            for o in scenario.outcomes.suggested_outcomes
            if getattr(o.type, "value", str(o.type)) == "categorical"
        ]
        required = [o for o in categorical if o.required]
        primary = required[0] if required else (categorical[0] if categorical else None)
        self._primary_position_outcome = primary.name if primary else None
        self._primary_position_options = (
            primary.options if primary and primary.options else []
        )
        self._primary_option_friction: dict[str, float] = {}
        if primary and getattr(primary, "option_friction", None):
            for option, score in primary.option_friction.items():
                try:
                    self._primary_option_friction[str(option)] = max(
                        0.0, min(1.0, float(score))
                    )
                except (TypeError, ValueError):
                    continue
        self._private_anchor_position = self._infer_private_anchor_position(
            self._primary_position_options
        )

        # Tracking variables
        self.recent_summaries: list[TimestepSummary] = []
        self.total_reasoning_calls = 0
        self.total_exposures = 0

        # Timeline state (active event for current timestep, if any)
        self._active_timeline_event: Any = None

        # Token usage tracking
        self.pivotal_input_tokens = 0
        self.pivotal_output_tokens = 0
        self.routine_input_tokens = 0
        self.routine_output_tokens = 0

        # Progress callback
        self._on_progress: TimestepProgressCallback | None = None

        # Live progress state (thread-safe, for CLI display)
        self._progress: SimulationProgress | None = None
        self._summary_interval: int = 50

    def set_progress_callback(self, callback: TimestepProgressCallback) -> None:
        """Set progress callback.

        Args:
            callback: Function(timestep, max_timesteps, status)
        """
        self._on_progress = callback

    def set_progress_state(self, progress: SimulationProgress) -> None:
        """Set shared progress state for live display.

        Args:
            progress: Thread-safe SimulationProgress instance
        """
        self._progress = progress

    def _apply_runtime_guardrails(self, timestep: int) -> None:
        """Downshift runtime knobs when process memory nears configured budget."""
        if (
            self.resource_governor is None
            or self.resource_governor.resource_mode != "auto"
        ):
            return

        ratio = self.resource_governor.memory_pressure_ratio()
        if ratio < 0.85:
            return

        factor = 0.5 if ratio >= 0.98 else 0.75
        old_concurrency = self.reasoning_max_concurrency
        old_batch = self.db_write_batch_size
        old_queue = self.writer_queue_size

        self.reasoning_max_concurrency = self.resource_governor.downshift_int(
            self.reasoning_max_concurrency, factor=factor, minimum=1
        )
        self.db_write_batch_size = self.resource_governor.downshift_int(
            self.db_write_batch_size, factor=factor, minimum=1
        )
        self.writer_queue_size = self.resource_governor.downshift_int(
            self.writer_queue_size, factor=factor, minimum=4
        )

        changed = (
            old_concurrency != self.reasoning_max_concurrency
            or old_batch != self.db_write_batch_size
            or old_queue != self.writer_queue_size
        )
        if changed and timestep != self._last_guardrail_timestep:
            self._last_guardrail_timestep = timestep
            logger.warning(
                "[RESOURCE] Memory pressure %.2fx budget; "
                "reasoning_concurrency %d->%d, writer_batch %d->%d, writer_queue %d->%d",
                ratio,
                old_concurrency,
                self.reasoning_max_concurrency,
                old_batch,
                self.db_write_batch_size,
                old_queue,
                self.writer_queue_size,
            )

    def _report_progress(self, timestep: int, status: str) -> None:
        """Report progress to callback."""
        if self._on_progress:
            self._on_progress(
                timestep,
                self.scenario.simulation.max_timesteps,
                status,
            )

    def _log_verbose_summary(self, snap: dict) -> None:
        """Log a periodic summary block with distribution and averages.

        Args:
            snap: Snapshot dict from SimulationProgress.snapshot()
        """
        counts = snap.get("position_counts", {})
        total = sum(counts.values()) or 1
        avg_sent = snap.get("avg_sentiment")
        avg_conv = snap.get("avg_conviction")
        done = snap.get("agents_done", 0)
        agents_total = snap.get("agents_total", 0)

        lines = [
            f"[SUMMARY] Timestep {snap['timestep']} | {done}/{agents_total} agents"
        ]

        # Distribution sorted by count desc
        for position, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"  {position}: {pct:.0f}% ({count})")

        if avg_sent is not None:
            lines.append(f"  avg_sentiment: {avg_sent:.2f}")
        if avg_conv is not None:
            lines.append(f"  avg_conviction: {avg_conv:.2f}")

        logger.info("\n".join(lines))

    def _get_resume_timestep(self) -> int:
        """Determine which timestep to start/resume from.

        Returns:
            Timestep number to begin execution at.
        """
        checkpoint = self.state_manager.get_checkpoint_timestep()
        if checkpoint is not None:
            # Crashed mid-timestep — resume it
            logger.info(f"Resuming from checkpoint timestep {checkpoint}")
            return checkpoint

        last_completed = self.state_manager.get_last_completed_timestep()
        if last_completed >= 0:
            # Completed some timesteps — start from next one
            logger.info(f"Resuming from timestep {last_completed + 1}")
            return last_completed + 1

        return 0

    def run(self) -> SimulationSummary:
        """Execute the full simulation.

        Supports automatic resume: if the output directory contains a
        study.db with partial progress, the engine picks up where
        it left off.

        Returns:
            SimulationSummary with results
        """
        start_time = time.time()
        stopped_reason = None
        final_timestep = 0

        start_timestep = self._get_resume_timestep()

        # Reload recent_summaries from DB on resume
        if start_timestep > 0:
            existing_summaries = self.state_manager.get_timestep_summaries()
            self.recent_summaries = existing_summaries[-20:]

        try:
            for timestep in range(
                start_timestep, self.scenario.simulation.max_timesteps
            ):
                final_timestep = timestep

                # Report progress
                exposure_rate = self.state_manager.get_exposure_rate()
                self._report_progress(timestep, f"Exposure: {exposure_rate:.1%}")

                # Run timestep
                summary = self._run_timestep(timestep)
                self.recent_summaries.append(summary)

                # Keep only last 20 summaries
                if len(self.recent_summaries) > 20:
                    self.recent_summaries.pop(0)

                # Check stopping conditions
                should_stop, reason = evaluate_stopping_conditions(
                    timestep,
                    self.scenario.simulation,
                    self.state_manager,
                    self.recent_summaries,
                )

                if should_stop:
                    stopped_reason = reason
                    logger.info(f"Stopping at timestep {timestep}: {reason}")
                    break

        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
            stopped_reason = "interrupted"

        # Finalize and export
        runtime = time.time() - start_time
        try:
            summary = self._finalize(final_timestep, stopped_reason, runtime)
            self._export_results()
        finally:
            self.state_manager.close()
            self.study_db.close()

        return summary

    def _run_timestep(self, timestep: int) -> TimestepSummary:
        """Execute one timestep of simulation.

        Each phase (exposures, reasoning chunks, summary) has its own
        transaction. This enables per-chunk checkpointing so that a crash
        mid-timestep doesn't lose all progress.

        Args:
            timestep: Current timestep number

        Returns:
            TimestepSummary for this timestep
        """
        logger.info(f"[TIMESTEP {timestep}] ========== STARTING ==========")

        self.state_manager.mark_timestep_started(timestep)

        return self._execute_timestep(timestep)

    def _execute_timestep(self, timestep: int) -> TimestepSummary:
        """Execute the actual timestep logic — orchestrates sub-steps.

        Args:
            timestep: Current timestep number

        Returns:
            TimestepSummary for this timestep
        """
        # 1. Exposures (seed + network) — own transaction
        with self.state_manager.transaction():
            total_new_exposures = self._apply_exposures(timestep)
        self.total_exposures += total_new_exposures

        # 2. Chunked reasoning — each chunk has its own transaction
        agents_reasoned, state_changes, shares_occurred, reasoning_results = (
            self._reason_agents(timestep)
        )

        # 2c. Execute conversations (if fidelity > low)
        conversations_executed = 0
        conversation_state_changes = 0
        if self.config.fidelity != "low" and reasoning_results:
            conv_results = self._execute_conversations(timestep, reasoning_results)
            conversations_executed = len(conv_results)
            conversation_state_changes = self._apply_conversation_overrides(
                timestep, conv_results
            )
            if conversations_executed > 0:
                logger.info(
                    f"[TIMESTEP {timestep}] Conversations: {conversations_executed} executed, "
                    f"{conversation_state_changes} state changes"
                )

        # 2e. Record social posts from agents who shared
        posts_recorded = self._record_social_posts(timestep)
        if posts_recorded > 0:
            logger.info(
                f"[TIMESTEP {timestep}] Social posts recorded: {posts_recorded}"
            )

        # 2d. Decay conviction for agents that did not reason this timestep.
        # This adds attention-fade dynamics without forcing additional LLM calls.
        with self.state_manager.transaction():
            decayed = self.state_manager.apply_conviction_decay(
                timestep=timestep,
                decay_rate=_CONVICTION_DECAY_RATE,
                sharing_threshold=_SHARING_CONVICTION_THRESHOLD,
                firm_threshold=_FIRM_CONVICTION,
            )
        if decayed:
            logger.info(
                f"[TIMESTEP {timestep}] Conviction decay applied to {decayed} agents"
            )

        # 3. Compute and save timestep summary — own transaction
        with self.state_manager.transaction():
            summary = compute_timestep_summary(
                timestep,
                self.state_manager,
                self.recent_summaries[-1] if self.recent_summaries else None,
            )
            summary.new_exposures = total_new_exposures
            summary.agents_reasoned = agents_reasoned
            summary.state_changes = state_changes
            summary.shares_occurred = shares_occurred

            self.state_manager.save_timestep_summary(summary)

        self.state_manager.mark_timestep_completed(timestep)

        # Flush timeline periodically
        if timestep % 10 == 0:
            self.timeline.flush()

        return summary

    def _apply_exposures(self, timestep: int) -> int:
        """Apply seed, timeline, and network exposures for this timestep.

        Returns:
            Total new exposures this timestep.
        """
        new_seed = apply_seed_exposures(
            timestep,
            self.scenario,
            self.agents,
            self.state_manager,
            self.rng,
        )
        logger.info(f"[TIMESTEP {timestep}] Seed exposures: {new_seed}")

        # Apply timeline event exposures (if any timeline event fires this timestep)
        new_timeline, active_event = apply_timeline_exposures(
            timestep,
            self.scenario,
            self.agents,
            self.state_manager,
            self.rng,
        )
        if new_timeline > 0:
            logger.info(f"[TIMESTEP {timestep}] Timeline exposures: {new_timeline}")

        # Store active timeline event for prompt rendering
        self._active_timeline_event = active_event

        new_network = propagate_through_network(
            timestep,
            self.scenario,
            self.agents,
            self.network,
            self.state_manager,
            self.rng,
            adjacency=self.adjacency,
            agent_map=self.agent_map,
        )
        logger.info(f"[TIMESTEP {timestep}] Network exposures: {new_network}")

        return new_seed + new_timeline + new_network

    def _reason_agents(
        self, timestep: int
    ) -> tuple[int, int, int, list[tuple[str, ReasoningResponse | None]]]:
        """Identify agents needing reasoning, run in chunks, commit per-chunk.

        On resume, agents already processed this timestep are skipped.

        Returns:
            Tuple of (agents_reasoned, state_changes, shares_occurred, reasoning_results).
        """
        self._apply_runtime_guardrails(timestep)
        agents_to_reason = self.state_manager.get_agents_to_reason(
            timestep,
            self.config.multi_touch_threshold,
        )

        # Filter out agents already processed this timestep (resume support)
        already_done = self.state_manager.get_agents_already_reasoned_this_timestep(
            timestep
        )
        if already_done:
            agents_to_reason = [a for a in agents_to_reason if a not in already_done]
            logger.info(
                f"[TIMESTEP {timestep}] Skipping {len(already_done)} already-processed agents"
            )

        logger.info(f"[TIMESTEP {timestep}] Agents to reason: {len(agents_to_reason)}")

        # Update progress state for live display (even if 0 agents, so display updates)
        if self._progress:
            exposure_rate = self.state_manager.get_exposure_rate()
            self._progress.begin_timestep(
                timestep=timestep,
                max_timesteps=self.scenario.simulation.max_timesteps,
                agents_total=len(agents_to_reason),
                exposure_rate=exposure_rate,
            )

        if not agents_to_reason:
            return 0, 0, 0, []

        # Create on_agent_done closure for progress tracking
        def _on_agent_done(agent_id: str, result: Any) -> None:
            if result is None:
                return
            if self._progress:
                self._progress.record_agent_done(
                    position=result.position,
                    sentiment=result.sentiment,
                    conviction=result.conviction,
                )
                # Log verbose summary at intervals
                if (
                    self._progress.agents_done % self._summary_interval == 0
                    and self._progress.agents_done > 0
                ):
                    self._log_verbose_summary(self._progress.snapshot())

        # Build contexts and old states
        contexts = []
        old_states: dict[str, AgentState] = {}
        for agent_id in agents_to_reason:
            old_state = self.state_manager.get_agent_state(agent_id)
            old_states[agent_id] = old_state
            context = self._build_reasoning_context(agent_id, old_state, timestep)
            contexts.append(context)

        completed_chunks = self.study_db.get_completed_simulation_chunks(
            self.run_id, timestep
        )
        totals = {"reasoned": 0, "changes": 0, "shares": 0}
        all_reasoning_results: list[tuple[str, ReasoningResponse | None]] = []
        work_queue: queue.Queue[tuple[int, list[tuple[str, Any]], bool] | object] = (
            queue.Queue(maxsize=self.writer_queue_size)
        )
        sentinel = object()
        writer_error: list[Exception] = []

        def _writer_loop() -> None:
            chunks_since_checkpoint = 0
            pending_chunks: list[tuple[int, list[tuple[str, Any]], bool]] = []

            def _flush_pending() -> None:
                nonlocal chunks_since_checkpoint
                if not pending_chunks:
                    return

                with self.state_manager.transaction():
                    for chunk_index, chunk_results, _is_last_chunk in pending_chunks:
                        reasoned, changes, shares = self._process_reasoning_chunk(
                            timestep, chunk_results, old_states
                        )
                        totals["reasoned"] += reasoned
                        totals["changes"] += changes
                        totals["shares"] += shares

                for chunk_index, _chunk_results, is_last_chunk in pending_chunks:
                    self.study_db.save_simulation_checkpoint(
                        run_id=self.run_id,
                        timestep=timestep,
                        chunk_index=chunk_index,
                        status="done",
                    )
                    chunks_since_checkpoint += 1
                    if (
                        chunks_since_checkpoint >= self.checkpoint_every_chunks
                        or is_last_chunk
                    ):
                        self.study_db.set_run_metadata(
                            self.run_id,
                            "last_checkpoint",
                            f"{timestep}:{chunk_index}",
                        )
                        chunks_since_checkpoint = 0

                pending_chunks.clear()

            try:
                while True:
                    item = work_queue.get()
                    try:
                        if item is sentinel:
                            _flush_pending()
                            break

                        chunk_index, chunk_results, is_last_chunk = item
                        if chunk_index in completed_chunks:
                            continue
                        pending_chunks.append(
                            (chunk_index, chunk_results, is_last_chunk)
                        )
                        if (
                            len(pending_chunks) >= self.db_write_batch_size
                            or is_last_chunk
                        ):
                            _flush_pending()
                    finally:
                        work_queue.task_done()
            except Exception as e:  # pragma: no cover - surfaced to caller
                writer_error.append(e)

        writer_thread = threading.Thread(
            target=_writer_loop,
            name=f"sim-writer-{self.run_id}-{timestep}",
            daemon=True,
        )
        writer_thread.start()

        import asyncio

        from ..core.providers import close_simulation_provider

        async def _run_all_chunks():
            try:
                for chunk_start in range(0, len(contexts), self.chunk_size):
                    if writer_error:
                        break
                    self._apply_runtime_guardrails(timestep)
                    chunk_index = chunk_start // self.chunk_size
                    if chunk_index in completed_chunks:
                        logger.info(
                            f"[TIMESTEP {timestep}] Skipping completed chunk {chunk_index}"
                        )
                        continue
                    chunk_contexts = contexts[
                        chunk_start : chunk_start + self.chunk_size
                    ]

                    reasoning_start = time.time()
                    chunk_results, chunk_usage = await batch_reason_agents_async(
                        chunk_contexts,
                        self.scenario,
                        self.config,
                        max_concurrency=self.reasoning_max_concurrency,
                        rate_limiter=self.rate_limiter,
                        on_agent_done=_on_agent_done,
                    )
                    reasoning_elapsed = time.time() - reasoning_start
                    self.total_reasoning_calls += len(chunk_results)

                    self.pivotal_input_tokens += chunk_usage.pivotal_input_tokens
                    self.pivotal_output_tokens += chunk_usage.pivotal_output_tokens
                    self.routine_input_tokens += chunk_usage.routine_input_tokens
                    self.routine_output_tokens += chunk_usage.routine_output_tokens

                    logger.info(
                        f"[TIMESTEP {timestep}] Chunk {chunk_start // self.chunk_size + 1}: "
                        f"{len(chunk_results)} agents in {reasoning_elapsed:.2f}s"
                        if chunk_results
                        else f"[TIMESTEP {timestep}] Chunk empty"
                    )
                    is_last_chunk = chunk_start + self.chunk_size >= len(contexts)
                    work_queue.put((chunk_index, chunk_results, is_last_chunk))
                    # Collect results for conversation phase
                    all_reasoning_results.extend(chunk_results)
            finally:
                await close_simulation_provider()

        asyncio.run(_run_all_chunks())

        work_queue.put(sentinel)
        while work_queue.unfinished_tasks > 0:
            if writer_error:
                while True:
                    try:
                        work_queue.get_nowait()
                        work_queue.task_done()
                    except queue.Empty:
                        break
                break
            time.sleep(0.01)

        work_queue.join()
        writer_thread.join(timeout=1)
        if writer_error:
            raise writer_error[0]

        return (
            totals["reasoned"],
            totals["changes"],
            totals["shares"],
            all_reasoning_results,
        )

    def _process_reasoning_chunk(
        self,
        timestep: int,
        results: list[tuple[str, Any]],
        old_states: dict[str, AgentState],
    ) -> tuple[int, int, int]:
        """Process a chunk of reasoning results and update agent states.

        Returns:
            Tuple of (agents_reasoned, state_changes, shares_occurred).
        """
        agents_reasoned = 0
        state_changes = 0
        shares_occurred = 0
        state_updates: list[tuple[str, AgentState]] = []

        for agent_id, response in results:
            if response is None:
                continue

            old_state = old_states[agent_id]

            old_public_sentiment = (
                old_state.public_sentiment
                if old_state.public_sentiment is not None
                else old_state.sentiment
            )
            old_public_conviction = (
                old_state.public_conviction
                if old_state.public_conviction is not None
                else old_state.conviction
            )
            old_public_position = old_state.public_position or old_state.position

            # Public state: what the agent says and propagates.
            public_sentiment = response.sentiment
            if old_public_sentiment is not None and response.sentiment is not None:
                public_sentiment = old_public_sentiment + _BOUNDED_CONFIDENCE_RHO * (
                    response.sentiment - old_public_sentiment
                )
                public_sentiment = max(-1.0, min(1.0, public_sentiment))

            public_conviction = response.conviction
            if old_public_conviction is not None and response.conviction is not None:
                public_conviction = old_public_conviction + _BOUNDED_CONFIDENCE_RHO * (
                    response.conviction - old_public_conviction
                )
                public_conviction = max(0.0, min(1.0, public_conviction))

            public_will_share = response.will_share
            candidate_public_position = response.public_position or response.position
            public_position = candidate_public_position

            if (
                old_public_conviction is not None
                and old_public_conviction >= _FIRM_CONVICTION
            ):
                if (
                    old_public_position is not None
                    and candidate_public_position is not None
                    and old_public_position != candidate_public_position
                ):
                    new_conviction = (
                        public_conviction if public_conviction is not None else 0.0
                    )
                    if new_conviction < _MODERATE_CONVICTION:
                        logger.info(
                            f"[CONVICTION] Agent {agent_id}: public flip from {old_public_position} "
                            f"to {candidate_public_position} rejected (old conviction={float_to_conviction(old_public_conviction)}, "
                            f"new conviction={float_to_conviction(public_conviction)})"
                        )
                        public_position = old_public_position

            if (
                response.conviction is not None
                and response.conviction <= _SHARING_CONVICTION_THRESHOLD
            ) or (
                public_conviction is not None
                and public_conviction <= _SHARING_CONVICTION_THRESHOLD
            ):
                public_will_share = False

            # Private state: what the agent is likely to actually do.
            old_private_position = old_state.private_position or old_state.position
            old_private_sentiment = (
                old_state.private_sentiment
                if old_state.private_sentiment is not None
                else old_state.sentiment
            )
            old_private_conviction = (
                old_state.private_conviction
                if old_state.private_conviction is not None
                else old_state.conviction
            )

            private_sentiment = public_sentiment
            if old_private_sentiment is not None and public_sentiment is not None:
                private_sentiment = old_private_sentiment + _PRIVATE_ADJUSTMENT_RHO * (
                    public_sentiment - old_private_sentiment
                )
                private_sentiment = max(-1.0, min(1.0, private_sentiment))

            private_conviction = public_conviction
            if old_private_conviction is not None and public_conviction is not None:
                private_conviction = (
                    old_private_conviction
                    + _PRIVATE_ADJUSTMENT_RHO
                    * (public_conviction - old_private_conviction)
                )
                private_conviction = max(0.0, min(1.0, private_conviction))

            recent_sources = {
                exp.source_agent_id
                for exp in old_state.exposures
                if (
                    exp.source_agent_id
                    and exp.timestep > old_state.last_reasoning_timestep
                )
            }
            recent_source_count = len(recent_sources)

            private_position = old_private_position
            if private_position is None:
                public_friction = self._position_action_friction(public_position)
                if (
                    self._private_anchor_position is not None
                    and public_position is not None
                    and public_friction >= 0.65
                    and (public_conviction if public_conviction is not None else 0.0)
                    < 0.90
                ):
                    private_position = self._private_anchor_position
                else:
                    private_position = public_position
            elif (
                public_position is not None
                and public_position != private_position
                and (
                    old_private_conviction is None
                    or old_private_conviction < _FIRM_CONVICTION
                )
            ):
                public_friction = self._position_action_friction(public_position)
                required_conviction = (
                    0.90 if public_friction >= 0.65 else _PRIVATE_FLIP_CONVICTION
                )
                required_sources = 2 if public_friction >= 0.65 else 1
                if (
                    public_conviction if public_conviction is not None else 0.0
                ) >= required_conviction and recent_source_count >= required_sources:
                    private_position = public_position

            private_outcomes = (
                dict(response.outcomes)
                if response.outcomes
                else dict(old_state.private_outcomes or old_state.outcomes or {})
            )
            if self._primary_position_outcome and private_position is not None:
                private_outcomes[self._primary_position_outcome] = private_position
            if private_sentiment is not None:
                private_outcomes["sentiment"] = private_sentiment

            is_committed = (
                private_conviction is not None
                and private_conviction >= _FIRM_CONVICTION
            )

            new_state = AgentState(
                agent_id=agent_id,
                aware=True,
                exposure_count=old_state.exposure_count,
                exposures=old_state.exposures,
                last_reasoning_timestep=timestep,
                position=private_position,
                sentiment=private_sentiment,
                conviction=private_conviction,
                public_statement=response.public_statement,
                action_intent=response.action_intent,
                will_share=public_will_share,
                public_position=public_position,
                public_sentiment=public_sentiment,
                public_conviction=public_conviction,
                private_position=private_position,
                private_sentiment=private_sentiment,
                private_conviction=private_conviction,
                private_outcomes=private_outcomes,
                committed=is_committed,
                outcomes=private_outcomes,
                raw_reasoning=None if self.retention_lite else response.reasoning,
                updated_at=timestep,
            )

            if self._state_changed(old_state, new_state):
                state_changes += 1

            if new_state.will_share and not old_state.will_share:
                shares_occurred += 1

            state_updates.append((agent_id, new_state))
            agents_reasoned += 1

            if response.reasoning_summary:
                memory_entry = MemoryEntry(
                    timestep=timestep,
                    sentiment=private_sentiment,
                    conviction=private_conviction,
                    summary=response.reasoning_summary,
                    raw_reasoning=None if self.retention_lite else response.reasoning,
                    action_intent=response.action_intent,
                )
                self.state_manager.save_memory_entry(agent_id, memory_entry)

            self.timeline.log_event(
                SimulationEvent(
                    timestep=timestep,
                    event_type=SimulationEventType.AGENT_REASONED,
                    agent_id=agent_id,
                    details={
                        "public_position": new_state.public_position,
                        "private_position": new_state.private_position,
                        "public_sentiment": new_state.public_sentiment,
                        "private_sentiment": new_state.private_sentiment,
                        "public_conviction": new_state.public_conviction,
                        "private_conviction": new_state.private_conviction,
                        "will_share": new_state.will_share,
                        "raw_reasoning": None
                        if self.retention_lite
                        else response.reasoning,
                    },
                )
            )

        if state_updates:
            self.state_manager.batch_update_states(state_updates, timestep)

        return agents_reasoned, state_changes, shares_occurred

    def _execute_conversations(
        self,
        timestep: int,
        reasoning_results: list[tuple[str, ReasoningResponse | None]],
    ) -> list[ConversationResult]:
        """Execute conversations based on talk_to actions from reasoning.

        Args:
            timestep: Current simulation timestep
            reasoning_results: List of (agent_id, response) pairs

        Returns:
            List of conversation results
        """
        # Get households for NPC resolution
        households = self.study_db.get_households(self.scenario.meta.name)

        # Collect conversation requests from actions
        requests = collect_conversation_requests(
            reasoning_results=reasoning_results,
            adjacency=self.adjacency,
            agent_map=self.agent_map,
            relationship_weights=self.scenario.relationship_weights,
            households=households,
        )

        if not requests:
            return []

        # Prioritize and resolve conflicts
        batches, _deferred = prioritize_and_resolve_conflicts(
            requests, fidelity=self.config.fidelity
        )

        if not batches:
            return []

        # Build contexts for all agents involved in conversations
        contexts: dict[str, ReasoningContext] = {}
        for batch in batches:
            for req in batch:
                if req.initiator_id not in contexts:
                    state = self.state_manager.get_agent_state(req.initiator_id)
                    contexts[req.initiator_id] = self._build_reasoning_context(
                        req.initiator_id, state, timestep
                    )
                if not req.target_is_npc and req.target_id not in contexts:
                    state = self.state_manager.get_agent_state(req.target_id)
                    contexts[req.target_id] = self._build_reasoning_context(
                        req.target_id, state, timestep
                    )

        # Execute all batches (in practice, usually just one batch)
        all_results: list[ConversationResult] = []

        import asyncio
        from ..core.providers import close_simulation_provider

        async def _run_conversations():
            try:
                for batch in batches:
                    batch_results = await execute_conversation_batch_async(
                        requests=batch,
                        contexts=contexts,
                        agent_map=self.agent_map,
                        scenario=self.scenario,
                        config=self.config,
                        rate_limiter=self.rate_limiter,
                        timestep=timestep,
                    )
                    all_results.extend(batch_results)
            finally:
                await close_simulation_provider()

        asyncio.run(_run_conversations())

        # Save conversations to DB
        for result in all_results:
            self.study_db.save_conversation(
                run_id=self.run_id,
                conversation_data={
                    "id": result.id,
                    "timestep": timestep,
                    "initiator_id": result.initiator_id,
                    "target_id": result.target_id,
                    "target_is_npc": result.target_is_npc,
                    "target_npc_profile": result.target_npc_profile,
                    "messages": [m.model_dump() for m in result.messages],
                    "initiator_state_change": (
                        result.initiator_state_change.model_dump()
                        if result.initiator_state_change
                        else None
                    ),
                    "target_state_change": (
                        result.target_state_change.model_dump()
                        if result.target_state_change
                        else None
                    ),
                    "priority_score": result.priority_score,
                },
            )

        logger.info(f"[TIMESTEP {timestep}] Executed {len(all_results)} conversations")
        return all_results

    def _apply_conversation_overrides(
        self,
        timestep: int,
        conversation_results: list[ConversationResult],
    ) -> int:
        """Apply state changes from conversations.

        Conversation outcomes OVERRIDE the provisional reasoning state.

        Args:
            timestep: Current simulation timestep
            conversation_results: List of conversation results

        Returns:
            Number of agents whose state was changed
        """
        state_changes = 0
        state_updates: list[tuple[str, AgentState]] = []

        for result in conversation_results:
            # Skip empty conversations
            if not result.messages:
                continue

            # Apply initiator state change
            if result.initiator_state_change:
                change = result.initiator_state_change
                state = self.state_manager.get_agent_state(result.initiator_id)

                updated = False
                if change.sentiment is not None:
                    state.sentiment = change.sentiment
                    state.private_sentiment = change.sentiment
                    state.public_sentiment = change.sentiment
                    updated = True
                if change.conviction is not None:
                    state.conviction = change.conviction
                    state.private_conviction = change.conviction
                    state.public_conviction = change.conviction
                    updated = True
                if change.position is not None:
                    state.position = change.position
                    state.private_position = change.position
                    state.public_position = change.position
                    updated = True

                if updated:
                    state.updated_at = timestep
                    state_updates.append((result.initiator_id, state))
                    state_changes += 1

            # Apply target state change (only for non-NPC targets)
            if not result.target_is_npc and result.target_state_change:
                change = result.target_state_change
                state = self.state_manager.get_agent_state(result.target_id)

                updated = False
                if change.sentiment is not None:
                    state.sentiment = change.sentiment
                    state.private_sentiment = change.sentiment
                    state.public_sentiment = change.sentiment
                    updated = True
                if change.conviction is not None:
                    state.conviction = change.conviction
                    state.private_conviction = change.conviction
                    state.public_conviction = change.conviction
                    updated = True
                if change.position is not None:
                    state.position = change.position
                    state.private_position = change.position
                    state.public_position = change.position
                    updated = True

                if updated:
                    state.updated_at = timestep
                    state_updates.append((result.target_id, state))
                    state_changes += 1

        # Batch update states
        if state_updates:
            with self.state_manager.transaction():
                self.state_manager.batch_update_states(state_updates, timestep)

        return state_changes

    def _record_social_posts(self, timestep: int) -> int:
        """Record social posts from agents who are sharing.

        Captures public_statement from agents with will_share=True as social posts.

        Args:
            timestep: Current simulation timestep

        Returns:
            Number of posts recorded
        """
        # Get all agents who are sharing this timestep
        sharers = self.state_manager.get_sharers()

        if not sharers:
            return 0

        posts = []
        for agent_id in sharers:
            state = self.state_manager.get_agent_state(agent_id)

            # Only record if they have a public statement
            if not state.public_statement:
                continue

            # Get agent name
            agent = self.agent_map.get(agent_id, {})
            agent_name = agent.get("first_name", agent_id)

            posts.append(
                {
                    "timestep": timestep,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "statement": state.public_statement,
                    "position": state.public_position or state.position,
                    "sentiment": state.public_sentiment or state.sentiment,
                    "conviction": state.public_conviction or state.conviction,
                }
            )

        if posts:
            self.study_db.save_social_posts_batch(self.run_id, posts)

        return len(posts)

    def _position_action_friction(self, position: str | None) -> float:
        """Estimate behavior-change friction from a position label."""
        if not position:
            return 0.5
        if position in self._primary_option_friction:
            return self._primary_option_friction[position]

        token = position.lower()
        low = (
            "maintain",
            "continue",
            "keep",
            "stay",
            "deny",
            "remove_shared_access",
            "no_change",
        )
        medium = ("reduce", "abandon", "pause")
        high = (
            "switch",
            "migrate",
            "cancel",
            "subscribe",
            "purchase",
            "buy",
            "upgrade",
        )

        if any(k in token for k in low):
            return 0.2
        if any(k in token for k in medium):
            return 0.4
        if any(k in token for k in high):
            return 0.75
        return 0.5

    def _infer_private_anchor_position(self, options: list[str]) -> str | None:
        """Pick the lowest-friction option as the private default anchor."""
        if not options:
            return None
        return min(options, key=self._position_action_friction)

    def _build_reasoning_context(
        self, agent_id: str, state: AgentState, timestep: int = 0
    ) -> ReasoningContext:
        """Build reasoning context for an agent.

        Args:
            agent_id: Agent ID
            state: Current agent state
            timestep: Current simulation timestep

        Returns:
            ReasoningContext for LLM call
        """
        agent = self.agent_map.get(agent_id, {})
        persona = self._personas.get(agent_id, "")

        # Get peer opinions from neighbors (with public statements, not position labels)
        peer_opinions = self._get_peer_opinions(agent_id)

        # Get memory trace
        memory_trace = self.state_manager.get_memory_traces(agent_id)

        # Derive prior action intent from most recent memory entry
        prior_action_intent = None
        if memory_trace:
            prior_action_intent = memory_trace[-1].action_intent

        # Build macro and local mood summaries (zero API calls)
        macro_summary = None
        if self.recent_summaries:
            prev_summary = (
                self.recent_summaries[-2] if len(self.recent_summaries) >= 2 else None
            )
            macro_summary = self._render_macro_summary(
                self.recent_summaries[-1],
                prev_summary=prev_summary,
            )
        local_mood_summary = self._render_local_mood(agent_id)

        ctx = create_reasoning_context(
            agent_id=agent_id,
            agent=agent,
            persona=persona,
            exposures=state.exposures,
            scenario=self.scenario,
            peer_opinions=peer_opinions,
            current_state=state if state.last_reasoning_timestep >= 0 else None,
            memory_trace=memory_trace,
        )
        # Populate Phase A fields
        ctx.timestep = timestep
        ctx.timestep_unit = self.scenario.simulation.timestep_unit.value
        ctx.agent_name = agent.get("first_name")
        ctx.prior_action_intent = prior_action_intent
        ctx.macro_summary = macro_summary
        ctx.local_mood_summary = local_mood_summary
        ctx.background_context = self.scenario.background_context
        ctx.agent_names = self._agent_names

        # Populate Phase C fields
        ctx.observable_peer_actions = self._compute_observable_adoption(agent_id)
        ctx.conformity = agent.get("conformity")

        # Populate Phase D fields (available contacts for conversation)
        if self.config.fidelity != "low":
            ctx.available_contacts = self._build_available_contacts(agent_id, agent)

        # Build social feed from recent posts (beyond direct network)
        ctx.social_feed = self._build_social_feed(agent_id, timestep)

        # Build timeline recap (accumulated events up to current timestep)
        if self.scenario.timeline:
            recap = []
            current_dev = None
            unit = self.scenario.simulation.timestep_unit.value
            for te in self.scenario.timeline:
                if te.timestep < timestep:
                    desc = te.description or te.event.content[:80]
                    recap.append(f"{unit} {te.timestep + 1}: {desc}")
                elif te.timestep == timestep:
                    current_dev = te.event.content
            ctx.timeline_recap = recap if recap else None
            ctx.current_development = current_dev

        # Build identity threat summary from scenario's identity dimensions
        ctx.identity_threat_summary = self._render_identity_threat_summary(agent)

        return ctx

    def _render_identity_threat_summary(
        self, agent: dict[str, Any]
    ) -> str | None:
        """Render identity threat framing from scenario's identity_dimensions.

        Uses the identity_dimensions field from ScenarioSpec (set by LLM during
        scenario creation) to determine which aspects of the agent's identity
        might feel threatened.
        """
        if not self.scenario.identity_dimensions:
            return None

        # Map dimension names to agent attribute keys
        dimension_attr_keys = {
            "political_orientation": ("political_orientation", "political_ideology", "party_affiliation"),
            "religious_affiliation": ("religious_affiliation", "religion", "faith_tradition"),
            "race_ethnicity": ("race_ethnicity", "race", "ethnicity"),
            "gender_identity": ("gender_identity", "gender"),
            "sexual_orientation": ("sexual_orientation",),
            "parental_status": ("parental_status", "household_role", "has_children"),
            "citizenship": ("citizenship_status", "nationality", "country_of_origin"),
            "socioeconomic_class": ("socioeconomic_class", "income_bracket", "social_class"),
            "professional_identity": ("occupation", "profession", "job_title"),
            "generational_identity": ("generation", "age_group"),
        }

        relevant_dimensions = []
        for dim in self.scenario.identity_dimensions:
            # Check if agent has a value for this dimension
            attr_keys = dimension_attr_keys.get(dim.dimension, ())
            agent_value = None
            for key in attr_keys:
                val = agent.get(key)
                if val and str(val).lower() not in ("unknown", "none", "n/a", ""):
                    agent_value = val
                    break

            # Special handling for parental_status - check dependents
            if dim.dimension == "parental_status" and not agent_value:
                if agent.get("dependents") or agent.get("has_children"):
                    agent_value = "parent"

            if agent_value:
                relevant_dimensions.append(
                    f"{dim.dimension.replace('_', ' ')} ({agent_value}): {dim.relevance}"
                )

        if not relevant_dimensions:
            return None

        return (
            "This development can feel identity-relevant, not just practical. "
            "Parts of who I am that may feel implicated:\n- " +
            "\n- ".join(relevant_dimensions) +
            "\n\nIf it feels personal, acknowledge that in both your internal reaction "
            "and what you choose to say publicly."
        )

    def _get_peer_opinions(self, agent_id: str) -> list[PeerOpinion]:
        """Get opinions of connected peers who have visibly shared.

        Only includes peers who have will_share=True — this models real-world
        observability where agents can only perceive what peers have explicitly
        shared or posted. Silent position changes are invisible.

        Peer limit is fidelity-gated: low=5, medium=5, high=10.

        Args:
            agent_id: Agent ID

        Returns:
            List of peer opinions (only from peers who shared)
        """
        neighbors = self.adjacency.get(agent_id, [])
        opinions = []

        # Fidelity-gated peer limit: low/medium=5, high=10
        max_peers = 10 if self.config.fidelity == "high" else 5

        for neighbor_id, edge_data in neighbors[:max_peers]:
            neighbor_state = self.state_manager.get_agent_state(neighbor_id)

            # Only include peer if they actively shared (observable behavior)
            if not neighbor_state.will_share:
                continue

            peer_sentiment = (
                neighbor_state.public_sentiment
                if neighbor_state.public_sentiment is not None
                else neighbor_state.sentiment
            )
            peer_position = neighbor_state.public_position or neighbor_state.position

            # Include peer if they have formed any public opinion AND shared
            if peer_sentiment is not None or neighbor_state.public_statement:
                peer_data = self.agent_map.get(neighbor_id, {})
                opinions.append(
                    PeerOpinion(
                        agent_id=neighbor_id,
                        peer_name=peer_data.get("first_name"),
                        relationship=edge_data.get("type", "contact"),
                        position=peer_position,
                        sentiment=peer_sentiment,
                        public_statement=neighbor_state.public_statement,
                        credibility=0.85,
                    )
                )

        return opinions

    def _compute_observable_adoption(self, agent_id: str) -> int | None:
        """Count neighbors who have visibly acted (will_share=True).

        Real-world model: agents can only perceive what peers have
        explicitly shared or posted. Silent position changes are invisible.

        Args:
            agent_id: Agent ID

        Returns:
            Count of neighbors who shared, or None if no neighbors
        """
        neighbors = self.adjacency.get(agent_id, [])
        if not neighbors:
            return None

        visible_actors = 0
        for neighbor_id, _ in neighbors:
            ns = self.state_manager.get_agent_state(neighbor_id)
            # Only count peers who shared/posted — observable behavior
            if ns.will_share:
                visible_actors += 1

        return visible_actors

    def _build_available_contacts(
        self, agent_id: str, agent: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Build list of contacts the agent can talk to.

        Includes network neighbors and household NPCs (partner, dependents).
        Fidelity controls how many contacts are shown.

        Args:
            agent_id: Agent ID
            agent: Agent attributes dict

        Returns:
            List of contact dicts with name, relationship, last_opinion
        """
        contacts: list[dict[str, Any]] = []

        # Determine limits based on fidelity
        max_peers = 10 if self.config.fidelity == "high" else 5

        # Add partner (agent or NPC)
        partner_npc = agent.get("partner_npc")
        if partner_npc:
            contacts.append(
                {
                    "name": partner_npc.get("name", "Partner"),
                    "relationship": "partner",
                    "last_opinion": None,
                    "is_npc": True,
                }
            )
        elif agent.get("partner_agent_id"):
            partner_id = agent["partner_agent_id"]
            partner = self.agent_map.get(partner_id, {})
            partner_state = self.state_manager.get_agent_state(partner_id)
            contacts.append(
                {
                    "name": partner.get("first_name", "Partner"),
                    "relationship": "partner",
                    "last_opinion": partner_state.public_statement,
                    "is_npc": False,
                }
            )

        # Add network neighbors (sorted by relationship weight)
        weights = self.scenario.relationship_weights or {}
        from ..core.models.scenario import DEFAULT_RELATIONSHIP_WEIGHTS

        for k, v in DEFAULT_RELATIONSHIP_WEIGHTS.items():
            weights.setdefault(k, v)

        neighbors = self.adjacency.get(agent_id, [])
        scored_neighbors = []
        for neighbor_id, edge_data in neighbors:
            rel = edge_data.get("type", "contact")
            edge_weight = edge_data.get("weight", 0.5)
            rel_weight = weights.get(rel, 0.3)
            score = edge_weight * rel_weight
            scored_neighbors.append((neighbor_id, edge_data, score))

        # Sort by score descending
        scored_neighbors.sort(key=lambda x: x[2], reverse=True)

        for neighbor_id, edge_data, _score in scored_neighbors[:max_peers]:
            neighbor = self.agent_map.get(neighbor_id, {})
            neighbor_state = self.state_manager.get_agent_state(neighbor_id)
            name = neighbor.get("first_name")
            if not name:
                continue

            contacts.append(
                {
                    "name": name,
                    "relationship": edge_data.get("type", "contact"),
                    "last_opinion": neighbor_state.public_statement
                    if neighbor_state.will_share
                    else None,
                    "is_npc": False,
                }
            )

        # Add household dependents (kids, elderly) as NPCs
        # These would come from household data if available
        dependents = agent.get("dependents", [])
        for dep in dependents:
            contacts.append(
                {
                    "name": dep.get("name", "Family member"),
                    "relationship": dep.get("relationship", "household"),
                    "last_opinion": None,
                    "is_npc": True,
                }
            )

        return contacts

    def _build_social_feed(self, agent_id: str, timestep: int) -> list[dict[str, Any]]:
        """Build social feed from recent posts beyond agent's direct network.

        The social feed represents public discourse the agent can perceive
        from the broader population, like trending posts on social media.

        Args:
            agent_id: Agent ID
            timestep: Current simulation timestep

        Returns:
            List of post dicts with agent_name, statement, position
        """
        if timestep == 0:
            return []

        # Get recent posts (not including current timestep)
        recent_posts = self.study_db.get_recent_social_posts(
            self.run_id,
            current_timestep=timestep,
            lookback=3,
            limit=15,
        )

        if not recent_posts:
            return []

        # Get agent's direct neighbors to exclude
        neighbors = self.adjacency.get(agent_id, [])
        neighbor_ids = {n[0] for n in neighbors}
        neighbor_ids.add(agent_id)  # Exclude self

        # Filter to posts from non-neighbors (the "broader public")
        feed = []
        for post in recent_posts:
            if post["agent_id"] in neighbor_ids:
                continue

            feed.append(
                {
                    "agent_name": post["agent_name"],
                    "statement": post["statement"],
                    "position": post["position"],
                }
            )

            # Limit to 5 posts from beyond direct network
            if len(feed) >= 5:
                break

        return feed

    def _render_macro_summary(
        self,
        summary: TimestepSummary,
        prev_summary: TimestepSummary | None = None,
    ) -> str:
        """Convert a TimestepSummary into a human-readable vibes sentence.

        Pure string formatting from numeric aggregates — zero API calls.
        """
        parts = []

        # Position distribution summary
        dist = summary.position_distribution
        if dist:
            total = sum(dist.values()) or 1
            top = sorted(dist.items(), key=lambda x: -x[1])
            leader, leader_count = top[0]
            leader_pct = leader_count / total * 100
            runner_up = top[1][0] if len(top) > 1 else None
            runner_up_pct = ((top[1][1] / total) * 100) if len(top) > 1 else 0.0
            if leader_pct > 60:
                msg = f"Most people are choosing '{leader}'."
            elif leader_pct > 40:
                msg = (
                    f"Opinion is split, with '{leader}' slightly ahead"
                    f"{f' of {runner_up!r}' if runner_up else ''}."
                )
            else:
                msg = "People are still quite divided on this."

            if runner_up and runner_up_pct >= 20 and leader_pct >= 45:
                msg += f" A sizable minority is backing '{runner_up}'."

            if prev_summary and prev_summary.position_distribution:
                prev_total = sum(prev_summary.position_distribution.values()) or 1
                prev_leader_count = prev_summary.position_distribution.get(leader, 0)
                prev_leader_pct = prev_leader_count / prev_total
                delta = (leader_count / total) - prev_leader_pct
                if delta >= 0.05:
                    msg += f" Support for '{leader}' is growing."
                elif delta <= -0.05:
                    msg += f" Support for '{leader}' is slipping."

            parts.append(msg)

        # Sentiment summary
        avg_sent = summary.average_sentiment
        if avg_sent is not None:
            if prev_summary and prev_summary.average_sentiment is not None:
                delta = avg_sent - prev_summary.average_sentiment
            else:
                delta = 0.0

            if avg_sent > 0.3:
                tone = "cautiously positive"
            elif avg_sent < -0.3:
                tone = "cautiously negative"
            else:
                tone = "mixed"

            if delta >= 0.08:
                trend = " and getting more positive"
            elif delta <= -0.08:
                trend = " and getting more negative"
            else:
                trend = ""

            parts.append(f"The general mood is {tone}{trend}.")

        # Exposure rate
        if summary.exposure_rate is not None:
            pct = summary.exposure_rate * 100
            if summary.exposure_rate >= 0.95:
                parts.append("Almost everyone has heard about this now.")
            elif summary.exposure_rate >= 0.75:
                parts.append("Most people have heard about this by now.")
            else:
                parts.append(f"About {pct:.0f}% have heard about this.")

        # Action momentum
        if prev_summary:
            if summary.shares_occurred > prev_summary.shares_occurred:
                parts.append("More people are actively taking action this round.")
            elif (
                summary.shares_occurred == 0
                and prev_summary.shares_occurred == 0
                and summary.state_changes == 0
            ):
                parts.append("Most people are still watching and waiting.")

        return " ".join(parts) if parts else ""

    def _render_local_mood(self, agent_id: str) -> str | None:
        """Compute a mood summary from neighbor agent states.

        Pure aggregation from stored states — zero API calls.
        Returns None if no neighbors have reasoned yet.
        """
        neighbors = self.adjacency.get(agent_id, [])
        if not neighbors:
            return None

        sentiments: list[float] = []
        for neighbor_id, _ in neighbors[:10]:
            ns = self.state_manager.get_agent_state(neighbor_id)
            s = ns.public_sentiment if ns.public_sentiment is not None else ns.sentiment
            if s is not None:
                sentiments.append(s)

        if not sentiments:
            return None

        avg = sum(sentiments) / len(sentiments)
        if avg > 0.3:
            mood = "optimistic"
        elif avg > 0.0:
            mood = "cautiously positive"
        elif avg > -0.3:
            mood = "uneasy"
        else:
            mood = "anxious"

        return f"Most people I know are {mood} about this."

    def _state_changed(self, old: AgentState, new: AgentState) -> bool:
        """Check if agent state changed meaningfully.

        Uses sentiment + conviction changes as primary signals,
        not position (which is output-only from Pass 2).

        Args:
            old: Previous state
            new: New state

        Returns:
            True if state changed
        """
        old_private_sentiment = (
            old.private_sentiment
            if old.private_sentiment is not None
            else old.sentiment
        )
        new_private_sentiment = (
            new.private_sentiment
            if new.private_sentiment is not None
            else new.sentiment
        )
        old_public_sentiment = (
            old.public_sentiment if old.public_sentiment is not None else old.sentiment
        )
        new_public_sentiment = (
            new.public_sentiment if new.public_sentiment is not None else new.sentiment
        )

        # Sentiment shift (private or public).
        if old_private_sentiment is not None and new_private_sentiment is not None:
            if abs(old_private_sentiment - new_private_sentiment) > 0.1:
                return True
        if old_public_sentiment is not None and new_public_sentiment is not None:
            if abs(old_public_sentiment - new_public_sentiment) > 0.1:
                return True

        old_private_conviction = (
            old.private_conviction
            if old.private_conviction is not None
            else old.conviction
        )
        new_private_conviction = (
            new.private_conviction
            if new.private_conviction is not None
            else new.conviction
        )
        old_public_conviction = (
            old.public_conviction
            if old.public_conviction is not None
            else old.conviction
        )
        new_public_conviction = (
            new.public_conviction
            if new.public_conviction is not None
            else new.conviction
        )

        # Conviction shift (private or public).
        if old_private_conviction is not None and new_private_conviction is not None:
            if abs(old_private_conviction - new_private_conviction) > 0.15:
                return True
        if old_public_conviction is not None and new_public_conviction is not None:
            if abs(old_public_conviction - new_public_conviction) > 0.15:
                return True

        # Position change on either track.
        if (old.private_position or old.position) != (
            new.private_position or new.position
        ):
            return True
        if (old.public_position or old.position) != (
            new.public_position or new.position
        ):
            return True

        # Sharing intent change
        if old.will_share != new.will_share:
            return True

        return False

    def _finalize(
        self,
        final_timestep: int,
        stopped_reason: str | None,
        runtime: float,
    ) -> SimulationSummary:
        """Finalize simulation and create summary.

        Args:
            final_timestep: Last completed timestep
            stopped_reason: Why simulation stopped
            runtime: Total runtime in seconds

        Returns:
            SimulationSummary
        """
        # Close timeline
        self.timeline.flush()
        self.timeline.close()

        # Compute final aggregates
        compute_final_aggregates(
            self.state_manager,
            self.agents,
            self.population_spec,
        )

        # Compute outcome distributions
        outcome_dists = compute_outcome_distributions(
            self.state_manager,
            self.scenario.outcomes.suggested_outcomes,
        )

        return SimulationSummary(
            scenario_name=self.scenario.meta.name,
            run_id=self.run_id,
            population_size=len(self.agents),
            total_timesteps=final_timestep + 1,
            stopped_reason=stopped_reason,
            total_reasoning_calls=self.total_reasoning_calls,
            total_exposures=self.total_exposures,
            final_exposure_rate=self.state_manager.get_exposure_rate(),
            outcome_distributions=outcome_dists,
            runtime_seconds=runtime,
            model_used=self.config.strong,
            completed_at=datetime.now(),
        )

    def _compute_cost(self) -> dict[str, Any]:
        """Compute cost from actual token usage using pricing data.

        Returns:
            Cost dictionary with token counts and estimated USD.
        """
        from ..core.cost.pricing import get_pricing
        from ..config import get_config

        cost: dict[str, Any] = {
            "pivotal_input_tokens": self.pivotal_input_tokens,
            "pivotal_output_tokens": self.pivotal_output_tokens,
            "routine_input_tokens": self.routine_input_tokens,
            "routine_output_tokens": self.routine_output_tokens,
            "total_input_tokens": self.pivotal_input_tokens + self.routine_input_tokens,
            "total_output_tokens": self.pivotal_output_tokens
            + self.routine_output_tokens,
        }

        # Resolve effective model names for pricing
        config = get_config()
        from ..config import parse_model_string

        strong_model_str = self.config.strong or config.resolve_sim_strong()
        fast_model_str = self.config.fast or config.resolve_sim_fast()

        # Strip provider prefix for pricing lookup (pricing is keyed by bare model name)
        _, pivotal_model = parse_model_string(strong_model_str)
        _, routine_model = parse_model_string(fast_model_str)

        cost["pivotal_model"] = pivotal_model
        cost["routine_model"] = routine_model

        # Compute USD cost per pass
        estimated_usd = 0.0
        has_pricing = False

        pivotal_pricing = get_pricing(pivotal_model)
        if pivotal_pricing:
            pivotal_cost = (
                self.pivotal_input_tokens / 1_000_000
            ) * pivotal_pricing.input_per_mtok + (
                self.pivotal_output_tokens / 1_000_000
            ) * pivotal_pricing.output_per_mtok
            estimated_usd += pivotal_cost
            has_pricing = True
        else:
            logger.warning(
                f"[COST] Unknown pricing for pivotal model '{pivotal_model}'"
            )

        routine_pricing = get_pricing(routine_model)
        if routine_pricing:
            routine_cost = (
                self.routine_input_tokens / 1_000_000
            ) * routine_pricing.input_per_mtok + (
                self.routine_output_tokens / 1_000_000
            ) * routine_pricing.output_per_mtok
            estimated_usd += routine_cost
            has_pricing = True
        else:
            logger.warning(
                f"[COST] Unknown pricing for routine model '{routine_model}'"
            )

        cost["estimated_usd"] = round(estimated_usd, 4) if has_pricing else None

        return cost

    def _export_results(self) -> None:
        """Export compact default artifacts to output directory."""
        # Export summary
        summaries = self.state_manager.get_timestep_summaries()
        timeline_agg = compute_timeline_aggregates(summaries)

        with open(self.output_dir / "by_timestep.json", "w") as f:
            json.dump(timeline_agg, f, indent=2)

        # Export meta information
        meta = {
            "scenario_name": self.scenario.meta.name,
            "scenario_path": self.config.scenario_path,
            "population_size": len(self.agents),
            "strong_model": self.config.strong,
            "fast_model": self.config.fast,
            "fidelity": self.config.fidelity,
            "seed": self.seed,
            "multi_touch_threshold": self.config.multi_touch_threshold,
            "completed_at": datetime.now().isoformat(),
        }

        if self.rate_limiter:
            meta["rate_limiter_stats"] = self.rate_limiter.stats()

        # Compute cost from actual token usage
        meta["cost"] = self._compute_cost()

        # Compute and export conversation statistics (Phase D)
        if self.config.fidelity != "low":
            conv_stats = compute_conversation_stats(
                study_db=self.study_db,
                run_id=self.run_id,
                max_timesteps=self.scenario.simulation.max_timesteps,
            )
            meta["conversation_stats"] = conv_stats

            # Export conversations detail if any occurred
            if conv_stats["total_conversations"] > 0:
                all_conversations = []
                for ts in range(self.scenario.simulation.max_timesteps):
                    convs = self.study_db.get_conversations_for_timestep(
                        self.run_id, ts
                    )
                    all_conversations.extend(convs)

                with open(self.output_dir / "conversations.json", "w") as f:
                    json.dump(all_conversations, f, indent=2)

            # Export social posts
            all_posts = self.study_db.get_all_social_posts(self.run_id)
            if all_posts:
                meta["social_posts_count"] = len(all_posts)
                with open(self.output_dir / "social_posts.json", "w") as f:
                    json.dump(all_posts, f, indent=2)

            # Export most impactful conversations
            if conv_stats["total_conversations"] > 0:
                impactful = compute_most_impactful_conversations(
                    study_db=self.study_db,
                    run_id=self.run_id,
                    max_timesteps=self.scenario.simulation.max_timesteps,
                    top_n=10,
                )
                if impactful:
                    meta["most_impactful_conversations"] = impactful

        # Export flattened elaborations CSV for DS workflows
        csv_path = self.output_dir / "elaborations.csv"
        rows_exported = export_elaborations_csv(
            state_manager=self.state_manager,
            agent_map=self.agent_map,
            output_path=str(csv_path),
        )
        if rows_exported > 0:
            meta["elaborations_csv_rows"] = rows_exported

        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def run_simulation(
    scenario_path: str | Path,
    output_dir: str | Path,
    study_db_path: str | Path | None = None,
    strong: str = "",
    fast: str = "",
    multi_touch_threshold: int = 3,
    random_seed: int | None = None,
    on_progress: TimestepProgressCallback | None = None,
    persona_config_path: str | Path | None = None,
    rate_tier: int | None = None,
    rpm_override: int | None = None,
    tpm_override: int | None = None,
    chunk_size: int = 50,
    progress: SimulationProgress | None = None,
    run_id: str | None = None,
    resume: bool = False,
    checkpoint_every_chunks: int = 1,
    retention_lite: bool = False,
    writer_queue_size: int = 256,
    db_write_batch_size: int = 100,
    resource_governor: ResourceGovernor | None = None,
    merged_pass: bool = False,
    fidelity: str = "medium",
) -> SimulationSummary:
    """Run a simulation from a scenario file.

    This is the main entry point for running simulations.

    Args:
        scenario_path: Path to scenario YAML file
        output_dir: Directory for results output
        strong: Strong model for Pass 1 reasoning (provider/model format)
        fast: Fast model for Pass 2 classification (provider/model format)
        multi_touch_threshold: Re-reason after N new exposures
        random_seed: Random seed for reproducibility
        on_progress: Progress callback(timestep, max, status)
        persona_config_path: Optional path to PersonaConfig YAML for embodied personas
        rate_tier: Rate limit tier (1-4, None = Tier 1)
        rpm_override: Override RPM limit
        tpm_override: Override TPM limit
        chunk_size: Agents per reasoning chunk for checkpointing
        progress: Optional SimulationProgress for live display tracking
        run_id: Optional run identifier for resume and bookkeeping
        resume: Resume a prior run from DB checkpoints
        checkpoint_every_chunks: Mark simulation checkpoint every N chunks
        retention_lite: Reduce payload volume by dropping full raw reasoning text
        writer_queue_size: Maximum buffered chunks waiting for DB writer
        db_write_batch_size: Number of chunks applied per DB writer transaction
        resource_governor: Optional governor for runtime downshift guardrails
        merged_pass: Use single merged reasoning pass instead of two-pass (experimental)
        fidelity: Conversation fidelity level (low, medium, high)

    Returns:
        SimulationSummary with results
    """
    scenario_path = Path(scenario_path)
    output_dir = Path(output_dir)
    if resume and not run_id:
        raise ValueError("--resume requires --run-id")

    def _reset_runtime_tables(path: Path, run_key: str) -> None:
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            statements = [
                "DELETE FROM agent_states WHERE run_id = ?",
                "DELETE FROM exposures WHERE run_id = ?",
                "DELETE FROM memory_traces WHERE run_id = ?",
                "DELETE FROM timeline WHERE run_id = ?",
                "DELETE FROM timestep_summaries WHERE run_id = ?",
                "DELETE FROM shared_to WHERE run_id = ?",
                "DELETE FROM simulation_metadata WHERE run_id = ?",
            ]
            for sql in statements:
                try:
                    cur.execute(sql, (run_key,))
                except sqlite3.OperationalError:
                    # Legacy table shape fallback.
                    table = sql.split()[2]
                    cur.execute(f"DELETE FROM {table}")
            conn.commit()
        except sqlite3.OperationalError:
            # First run on this DB may not have simulation tables yet.
            pass
        finally:
            conn.close()

    # Load scenario
    scenario = ScenarioSpec.from_yaml(scenario_path)

    # Load population spec (resolve from base_population or legacy population_spec)
    pop_name, pop_version = scenario.meta.get_population_ref()

    if pop_version is not None:
        # Versioned ref (e.g. "population.v1") — resolve relative to study root
        # Study structure: study_root/scenario/<name>/scenario.v1.yaml
        study_root = scenario_path.parent.parent.parent
        pop_path = study_root / f"{pop_name}.v{pop_version}.yaml"
    else:
        pop_path = Path(pop_name)
        if not pop_path.is_absolute():
            pop_path = scenario_path.parent / pop_path
    population_spec = PopulationSpec.from_yaml(pop_path)

    # Resolve canonical study DB
    if study_db_path is None:
        if not getattr(scenario.meta, "study_db", None):
            raise ValueError(
                "Legacy scenario format detected. Rebuild scenario with --study-db."
            )
        study_db_resolved = Path(scenario.meta.study_db)
        if not study_db_resolved.is_absolute():
            study_db_resolved = scenario_path.parent / study_db_resolved
    else:
        study_db_resolved = Path(study_db_path)

    if not study_db_resolved.exists():
        raise FileNotFoundError(f"Study DB not found: {study_db_resolved}")

    resolved_run_id = (
        run_id
        or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    with open_study_db(study_db_resolved) as db:
        # Try scenario-based lookup first (new CLI flow), fall back to legacy IDs
        agents = db.get_agents_by_scenario(scenario.meta.name)
        if not agents:
            agents = db.get_agents(scenario.meta.population_id)
        if not agents:
            raise ValueError(
                f"No agents for scenario '{scenario.meta.name}' in {study_db_resolved}"
            )
        network = db.get_network(scenario.meta.name)
        if not network.get("edges"):
            network = db.get_network(scenario.meta.network_id)
        if not network.get("edges"):
            raise ValueError(
                f"No network edges for scenario '{scenario.meta.name}' in {study_db_resolved}"
            )
        db.create_simulation_run(
            run_id=resolved_run_id,
            scenario_name=scenario.meta.name,
            population_id=scenario.meta.name,
            network_id=scenario.meta.name,
            config={
                "scenario_path": str(scenario_path),
                "output_dir": str(output_dir),
                "strong": strong,
                "fast": fast,
                "multi_touch_threshold": multi_touch_threshold,
                "chunk_size": chunk_size,
                "checkpoint_every_chunks": checkpoint_every_chunks,
                "retention_lite": retention_lite,
                "writer_queue_size": writer_queue_size,
                "db_write_batch_size": db_write_batch_size,
                "resume": resume,
            },
            seed=random_seed,
            status="running",
        )
        db.set_run_metadata(resolved_run_id, "output_dir", str(output_dir))
        db.set_run_metadata(resolved_run_id, "study_db", str(study_db_resolved))

    if not resume:
        _reset_runtime_tables(study_db_resolved, resolved_run_id)

    # Load persona config if provided
    persona_config = None
    if persona_config_path:
        persona_config_path = Path(persona_config_path)
        if not persona_config_path.is_absolute():
            persona_config_path = scenario_path.parent / persona_config_path
        if persona_config_path.exists():
            persona_config = PersonaConfig.from_file(str(persona_config_path))
    else:
        # Try to find persona config automatically
        auto_config_path = pop_path.with_suffix(".persona.yaml")
        if auto_config_path.exists():
            persona_config = PersonaConfig.from_file(str(auto_config_path))

    # Create config
    # Resolve effective model strings for rate limiting
    from ..config import get_config

    entropy_config = get_config()

    # Validate fidelity
    if fidelity not in ("low", "medium", "high"):
        raise ValueError(f"Invalid fidelity '{fidelity}', must be low/medium/high")

    config = SimulationRunConfig(
        scenario_path=str(scenario_path),
        output_dir=str(output_dir),
        strong=strong,
        fast=fast,
        multi_touch_threshold=multi_touch_threshold,
        random_seed=random_seed,
        max_concurrent=entropy_config.simulation.max_concurrent,
        merged_pass=merged_pass,
        fidelity=fidelity,  # type: ignore[arg-type]
    )
    effective_strong = strong or entropy_config.resolve_sim_strong()
    effective_fast = fast or entropy_config.resolve_sim_fast()

    rate_limiter = DualRateLimiter.create(
        strong_model_string=effective_strong,
        fast_model_string=effective_fast,
        tier=rate_tier,
        rpm_override=rpm_override,
        tpm_override=tpm_override,
    )

    # Create and run engine
    engine = SimulationEngine(
        scenario=scenario,
        population_spec=population_spec,
        agents=agents,
        network=network,
        config=config,
        persona_config=persona_config,
        rate_limiter=rate_limiter,
        chunk_size=chunk_size,
        state_db_path=study_db_resolved,
        run_id=resolved_run_id,
        checkpoint_every_chunks=checkpoint_every_chunks,
        retention_lite=retention_lite,
        writer_queue_size=writer_queue_size,
        db_write_batch_size=db_write_batch_size,
        resource_governor=resource_governor,
    )

    if on_progress:
        engine.set_progress_callback(on_progress)

    if progress:
        engine.set_progress_state(progress)

    try:
        summary = engine.run()
    except Exception as e:
        with open_study_db(study_db_resolved) as db:
            db.update_simulation_run(
                run_id=resolved_run_id,
                status="failed",
                stopped_reason=str(e),
            )
        raise

    final_status = "stopped" if summary.stopped_reason else "completed"
    with open_study_db(study_db_resolved) as db:
        db.update_simulation_run(
            run_id=resolved_run_id,
            status=final_status,
            stopped_reason=summary.stopped_reason,
        )

    return summary
