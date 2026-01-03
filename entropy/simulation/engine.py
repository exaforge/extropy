"""Main simulation engine orchestrator.

Coordinates the simulation loop: exposure, reasoning, propagation,
and aggregation across timesteps until stopping conditions are met.
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..core.models import (
    PopulationSpec,
    ScenarioSpec,
    AgentState,
    PeerOpinion,
    ReasoningContext,
    SimulationEvent,
    SimulationEventType,
    SimulationRunConfig,
    TimestepSummary,
)
from ..population.network import load_agents_json
from .state import StateManager
from .persona import generate_persona
from .reasoning import reason_agent, create_reasoning_context
from .propagation import apply_seed_exposures, propagate_through_network, get_neighbors
from .stopping import evaluate_stopping_conditions
from .timeline import TimelineManager
from .aggregation import (
    compute_timestep_summary,
    compute_final_aggregates,
    compute_outcome_distributions,
    compute_timeline_aggregates,
)

logger = logging.getLogger(__name__)


class SimulationSummary:
    """Summary of a completed simulation run."""

    def __init__(
        self,
        scenario_name: str,
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
    ):
        """Initialize simulation engine.

        Args:
            scenario: Scenario specification
            population_spec: Population specification
            agents: List of agent dictionaries
            network: Network data
            config: Simulation run configuration
        """
        self.scenario = scenario
        self.population_spec = population_spec
        self.agents = agents
        self.network = network
        self.config = config

        # Build agent map for quick lookup
        self.agent_map = {a.get("_id", str(i)): a for i, a in enumerate(agents)}

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
        self.state_manager = StateManager(
            self.output_dir / "simulation.db",
            agents,
        )

        # Initialize timeline manager
        self.timeline = TimelineManager(self.output_dir / "timeline.jsonl")

        # Pre-generate personas for all agents
        self._personas: dict[str, str] = {}
        for agent in agents:
            agent_id = agent.get("_id", str(agents.index(agent)))
            self._personas[agent_id] = generate_persona(agent, population_spec)

        # Tracking variables
        self.recent_summaries: list[TimestepSummary] = []
        self.total_reasoning_calls = 0
        self.total_exposures = 0

        # Progress callback
        self._on_progress: Callable[[int, int, str], None] | None = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Set progress callback.

        Args:
            callback: Function(timestep, max_timesteps, status)
        """
        self._on_progress = callback

    def _report_progress(self, timestep: int, status: str) -> None:
        """Report progress to callback."""
        if self._on_progress:
            self._on_progress(
                timestep,
                self.scenario.simulation.max_timesteps,
                status,
            )

    def run(self) -> SimulationSummary:
        """Execute the full simulation.

        Returns:
            SimulationSummary with results
        """
        start_time = time.time()
        stopped_reason = None
        final_timestep = 0

        try:
            for timestep in range(self.scenario.simulation.max_timesteps):
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
        summary = self._finalize(final_timestep, stopped_reason, runtime)
        self._export_results()

        return summary

    def _run_timestep(self, timestep: int) -> TimestepSummary:
        """Execute one timestep of simulation.

        Args:
            timestep: Current timestep number

        Returns:
            TimestepSummary for this timestep
        """
        # 1. Apply seed exposures
        new_seed_exposures = apply_seed_exposures(
            timestep,
            self.scenario,
            self.agents,
            self.state_manager,
            self.rng,
        )

        # 2. Network propagation from previous sharers
        new_network_exposures = propagate_through_network(
            timestep,
            self.scenario,
            self.agents,
            self.network,
            self.state_manager,
            self.rng,
        )

        total_new_exposures = new_seed_exposures + new_network_exposures
        self.total_exposures += total_new_exposures

        # 3. Identify agents who need to reason
        agents_to_reason = self.state_manager.get_agents_to_reason(
            timestep,
            self.config.multi_touch_threshold,
        )

        # 4. Run reasoning for each agent
        agents_reasoned = 0
        state_changes = 0
        shares_occurred = 0

        for agent_id in agents_to_reason:
            old_state = self.state_manager.get_agent_state(agent_id)

            # Build reasoning context
            context = self._build_reasoning_context(agent_id, old_state)

            # Call LLM for reasoning
            response = reason_agent(context, self.scenario, self.config)
            self.total_reasoning_calls += 1

            if response is None:
                continue

            # Create new state from response
            new_state = AgentState(
                agent_id=agent_id,
                aware=True,
                exposure_count=old_state.exposure_count,
                exposures=old_state.exposures,
                last_reasoning_timestep=timestep,
                position=response.position,
                sentiment=response.sentiment,
                action_intent=response.action_intent,
                will_share=response.will_share,
                outcomes=response.outcomes,
                raw_reasoning=response.reasoning,
                updated_at=timestep,
            )

            # Check for state change
            if self._state_changed(old_state, new_state):
                state_changes += 1

            # Track shares
            if new_state.will_share and not old_state.will_share:
                shares_occurred += 1

            # Update state
            self.state_manager.update_agent_state(agent_id, new_state, timestep)
            agents_reasoned += 1

            # Log event
            self.timeline.log_event(
                SimulationEvent(
                    timestep=timestep,
                    event_type=SimulationEventType.AGENT_REASONED,
                    agent_id=agent_id,
                    details={
                        "position": new_state.position,
                        "sentiment": new_state.sentiment,
                        "will_share": new_state.will_share,
                    },
                )
            )

        # 5. Compute and save timestep summary
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

        # Flush timeline periodically
        if timestep % 10 == 0:
            self.timeline.flush()

        return summary

    def _build_reasoning_context(
        self, agent_id: str, state: AgentState
    ) -> ReasoningContext:
        """Build reasoning context for an agent.

        Args:
            agent_id: Agent ID
            state: Current agent state

        Returns:
            ReasoningContext for LLM call
        """
        agent = self.agent_map.get(agent_id, {})
        persona = self._personas.get(agent_id, "")

        # Get peer opinions from neighbors
        peer_opinions = self._get_peer_opinions(agent_id)

        return create_reasoning_context(
            agent_id=agent_id,
            agent=agent,
            persona=persona,
            exposures=state.exposures,
            scenario=self.scenario,
            peer_opinions=peer_opinions,
            current_state=state if state.last_reasoning_timestep >= 0 else None,
        )

    def _get_peer_opinions(self, agent_id: str) -> list[PeerOpinion]:
        """Get opinions of connected peers.

        Args:
            agent_id: Agent ID

        Returns:
            List of peer opinions
        """
        neighbors = get_neighbors(self.network, agent_id)
        opinions = []

        for neighbor_id, edge_data in neighbors[:5]:  # Limit to 5 peers
            neighbor_state = self.state_manager.get_agent_state(neighbor_id)

            if neighbor_state.position:  # Only include if they have an opinion
                opinions.append(
                    PeerOpinion(
                        agent_id=neighbor_id,
                        relationship=edge_data.get("type", "contact"),
                        position=neighbor_state.position,
                        sentiment=neighbor_state.sentiment,
                    )
                )

        return opinions

    def _state_changed(self, old: AgentState, new: AgentState) -> bool:
        """Check if agent state changed meaningfully.

        Args:
            old: Previous state
            new: New state

        Returns:
            True if state changed
        """
        if old.position != new.position:
            return True
        if old.will_share != new.will_share:
            return True
        if old.sentiment is not None and new.sentiment is not None:
            if abs(old.sentiment - new.sentiment) > 0.1:
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
            population_size=len(self.agents),
            total_timesteps=final_timestep + 1,
            stopped_reason=stopped_reason,
            total_reasoning_calls=self.total_reasoning_calls,
            total_exposures=self.total_exposures,
            final_exposure_rate=self.state_manager.get_exposure_rate(),
            outcome_distributions=outcome_dists,
            runtime_seconds=runtime,
            model_used=self.config.model,
            completed_at=datetime.now(),
        )

    def _export_results(self) -> None:
        """Export all results to output directory."""
        # Export summary
        summaries = self.state_manager.get_timestep_summaries()
        timeline_agg = compute_timeline_aggregates(summaries)

        with open(self.output_dir / "by_timestep.json", "w") as f:
            json.dump(timeline_agg, f, indent=2)

        # Export final agent states
        final_states = self.state_manager.export_final_states()

        # Merge with agent attributes
        agent_results = []
        for state in final_states:
            agent_id = state["agent_id"]
            agent = self.agent_map.get(agent_id, {})

            agent_results.append(
                {
                    "agent_id": agent_id,
                    "attributes": {
                        k: v for k, v in agent.items() if not k.startswith("_")
                    },
                    "final_state": state,
                    "reasoning_count": (
                        1 if state["last_reasoning_timestep"] >= 0 else 0
                    ),
                }
            )

        with open(self.output_dir / "agent_states.json", "w") as f:
            json.dump(agent_results, f, indent=2)

        # Export outcome distributions
        outcome_dists = compute_outcome_distributions(
            self.state_manager,
            self.scenario.outcomes.suggested_outcomes,
        )

        with open(self.output_dir / "outcome_distributions.json", "w") as f:
            json.dump(outcome_dists, f, indent=2)

        # Export meta information
        meta = {
            "scenario_name": self.scenario.meta.name,
            "scenario_path": self.config.scenario_path,
            "population_size": len(self.agents),
            "model": self.config.model,
            "seed": self.seed,
            "multi_touch_threshold": self.config.multi_touch_threshold,
            "completed_at": datetime.now().isoformat(),
        }

        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)


def run_simulation(
    scenario_path: str | Path,
    output_dir: str | Path,
    model: str = "gpt-5-mini",
    multi_touch_threshold: int = 3,
    random_seed: int | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> SimulationSummary:
    """Run a simulation from a scenario file.

    This is the main entry point for running simulations.

    Args:
        scenario_path: Path to scenario YAML file
        output_dir: Directory for results output
        model: LLM model for agent reasoning
        multi_touch_threshold: Re-reason after N new exposures
        random_seed: Random seed for reproducibility
        on_progress: Progress callback(timestep, max, status)

    Returns:
        SimulationSummary with results
    """
    scenario_path = Path(scenario_path)
    output_dir = Path(output_dir)

    # Load scenario
    scenario = ScenarioSpec.from_yaml(scenario_path)

    # Load population spec
    pop_path = Path(scenario.meta.population_spec)
    if not pop_path.is_absolute():
        pop_path = scenario_path.parent / pop_path
    population_spec = PopulationSpec.from_yaml(pop_path)

    # Load agents
    agents_path = Path(scenario.meta.agents_file)
    if not agents_path.is_absolute():
        agents_path = scenario_path.parent / agents_path
    agents = load_agents_json(agents_path)

    # Load network
    network_path = Path(scenario.meta.network_file)
    if not network_path.is_absolute():
        network_path = scenario_path.parent / network_path
    with open(network_path) as f:
        network = json.load(f)

    # Create config
    config = SimulationRunConfig(
        scenario_path=str(scenario_path),
        output_dir=str(output_dir),
        model=model,
        multi_touch_threshold=multi_touch_threshold,
        random_seed=random_seed,
    )

    # Create and run engine
    engine = SimulationEngine(
        scenario=scenario,
        population_spec=population_spec,
        agents=agents,
        network=network,
        config=config,
    )

    if on_progress:
        engine.set_progress_callback(on_progress)

    return engine.run()
