"""Results reader for loading and querying simulation results.

Provides a clean API for loading results from a results directory
and computing on-demand aggregations.
"""

import json
from pathlib import Path
from typing import Any
from datetime import datetime

from .models import (
    SimulationSummary,
    AgentFinalState,
    SegmentAggregate,
    TimelinePoint,
    RunMeta,
    SimulationResults,
)


class ResultsReader:
    """Reader for simulation results directory.

    Loads results lazily and provides querying capabilities.
    """

    def __init__(self, results_dir: Path | str):
        """Initialize results reader.

        Args:
            results_dir: Path to results directory
        """
        self.results_dir = Path(results_dir)

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # Cache for loaded data
        self._meta: RunMeta | None = None
        self._timeline: list[TimelinePoint] | None = None
        self._agent_states: list[AgentFinalState] | None = None
        self._outcome_distributions: dict[str, dict[str, float]] | None = None

    def get_meta(self) -> RunMeta:
        """Get run metadata.

        Returns:
            RunMeta with configuration info
        """
        if self._meta is None:
            meta_path = self.results_dir / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"meta.json not found in {self.results_dir}")

            with open(meta_path) as f:
                data = json.load(f)

            self._meta = RunMeta(
                scenario_name=data.get("scenario_name", "unknown"),
                scenario_path=data.get("scenario_path", ""),
                population_size=data.get("population_size", 0),
                model=data.get("model", "unknown"),
                seed=data.get("seed", 0),
                multi_touch_threshold=data.get("multi_touch_threshold", 3),
                completed_at=datetime.fromisoformat(data.get("completed_at", datetime.now().isoformat())),
            )

        return self._meta

    def get_timeline(self) -> list[TimelinePoint]:
        """Get timeline data.

        Returns:
            List of TimelinePoint objects
        """
        if self._timeline is None:
            timeline_path = self.results_dir / "by_timestep.json"
            if not timeline_path.exists():
                return []

            with open(timeline_path) as f:
                data = json.load(f)

            self._timeline = [
                TimelinePoint(
                    timestep=point.get("timestep", 0),
                    exposure_rate=point.get("exposure_rate", 0),
                    position_distribution=point.get("position_distribution", {}),
                    average_sentiment=point.get("average_sentiment"),
                    cumulative_shares=point.get("cumulative_shares", 0),
                    new_exposures=point.get("new_exposures", 0),
                    agents_reasoned=point.get("agents_reasoned", 0),
                )
                for point in data
            ]

        return self._timeline

    def get_agent_states(self) -> list[AgentFinalState]:
        """Get final agent states.

        Returns:
            List of AgentFinalState objects
        """
        if self._agent_states is None:
            states_path = self.results_dir / "agent_states.json"
            if not states_path.exists():
                return []

            with open(states_path) as f:
                data = json.load(f)

            self._agent_states = [
                AgentFinalState(
                    agent_id=agent.get("agent_id", ""),
                    attributes=agent.get("attributes", {}),
                    aware=agent.get("final_state", {}).get("aware", False),
                    exposure_count=agent.get("final_state", {}).get("exposure_count", 0),
                    position=agent.get("final_state", {}).get("position"),
                    sentiment=agent.get("final_state", {}).get("sentiment"),
                    action_intent=agent.get("final_state", {}).get("action_intent"),
                    will_share=agent.get("final_state", {}).get("will_share", False),
                    raw_reasoning=agent.get("final_state", {}).get("raw_reasoning"),
                    outcomes=agent.get("final_state", {}).get("outcomes", {}),
                    reasoning_count=agent.get("reasoning_count", 0),
                )
                for agent in data
            ]

        return self._agent_states

    def get_agent_state(self, agent_id: str) -> AgentFinalState | None:
        """Get state for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            AgentFinalState or None if not found
        """
        states = self.get_agent_states()
        for state in states:
            if state.agent_id == agent_id:
                return state
        return None

    def get_outcome_distributions(self) -> dict[str, dict[str, float]]:
        """Get outcome distributions.

        Returns:
            Dict mapping outcome name to distribution
        """
        if self._outcome_distributions is None:
            dist_path = self.results_dir / "outcome_distributions.json"
            if not dist_path.exists():
                return {}

            with open(dist_path) as f:
                self._outcome_distributions = json.load(f)

        return self._outcome_distributions

    def compute_segment(self, attribute: str) -> list[SegmentAggregate]:
        """Compute segment breakdown for an attribute.

        Args:
            attribute: Attribute to segment by

        Returns:
            List of SegmentAggregate objects
        """
        states = self.get_agent_states()

        # Group by attribute value
        segments: dict[str, list[AgentFinalState]] = {}
        for state in states:
            value = state.attributes.get(attribute, "unknown")
            value_str = str(value)
            if value_str not in segments:
                segments[value_str] = []
            segments[value_str].append(state)

        # Compute aggregates
        results = []
        for value, segment_states in segments.items():
            position_counts: dict[str, int] = {}
            sentiments: list[float] = []
            aware_count = 0

            for state in segment_states:
                if state.aware:
                    aware_count += 1
                if state.position:
                    position_counts[state.position] = position_counts.get(state.position, 0) + 1
                if state.sentiment is not None:
                    sentiments.append(state.sentiment)

            total = len(segment_states)
            position_dist = {
                k: v / total for k, v in position_counts.items()
            } if total > 0 else {}

            results.append(SegmentAggregate(
                segment_attribute=attribute,
                segment_value=value,
                agent_count=len(segment_states),
                aware_count=aware_count,
                position_distribution=position_dist,
                position_counts=position_counts,
                average_sentiment=sum(sentiments) / len(sentiments) if sentiments else None,
            ))

        # Sort by count
        results.sort(key=lambda x: x.agent_count, reverse=True)
        return results

    def get_summary(self) -> SimulationSummary | None:
        """Compute or get simulation summary.

        Returns:
            SimulationSummary or None
        """
        meta = self.get_meta()
        timeline = self.get_timeline()
        states = self.get_agent_states()
        outcomes = self.get_outcome_distributions()

        if not states:
            return None

        # Compute summary stats
        total = len(states)
        aware = sum(1 for s in states if s.aware)
        reasoned = sum(1 for s in states if s.reasoning_count > 0)

        final_timestep = timeline[-1].timestep if timeline else 0

        # Estimate total exposures and reasoning calls
        total_exposures = sum(s.exposure_count for s in states)
        total_reasoning = sum(s.reasoning_count for s in states)

        return SimulationSummary(
            scenario_name=meta.scenario_name,
            population_size=total,
            total_timesteps=final_timestep + 1,
            stopped_reason=None,  # Not stored in current format
            total_reasoning_calls=total_reasoning,
            total_exposures=total_exposures,
            final_exposure_rate=aware / total if total > 0 else 0,
            outcome_distributions=outcomes,
            runtime_seconds=0,  # Not stored in current format
            model_used=meta.model,
            completed_at=meta.completed_at,
        )

    def load_all(self) -> SimulationResults:
        """Load all results into a single object.

        Returns:
            SimulationResults with all data
        """
        return SimulationResults(
            meta=self.get_meta(),
            summary=self.get_summary(),
            timeline=self.get_timeline(),
            agent_states=self.get_agent_states(),
            segments={},  # Computed on demand
            outcome_distributions=self.get_outcome_distributions(),
        )

    def get_position_summary(self) -> dict[str, Any]:
        """Get quick summary of position distribution.

        Returns:
            Dict with position counts and percentages
        """
        states = self.get_agent_states()
        total = len(states)

        if total == 0:
            return {"total": 0, "positions": {}}

        position_counts: dict[str, int] = {}
        for state in states:
            if state.position:
                position_counts[state.position] = position_counts.get(state.position, 0) + 1

        return {
            "total": total,
            "aware": sum(1 for s in states if s.aware),
            "positions": position_counts,
            "position_pct": {
                k: v / total for k, v in position_counts.items()
            },
        }


def load_results(results_dir: Path | str) -> ResultsReader:
    """Load simulation results from a directory.

    Args:
        results_dir: Path to results directory

    Returns:
        ResultsReader instance
    """
    return ResultsReader(results_dir)
