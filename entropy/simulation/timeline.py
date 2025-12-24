"""Timeline management for simulation events.

Provides streaming event logging to JSONL files and post-simulation
querying capabilities.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..core.models import SimulationEvent, SimulationEventType


class TimelineManager:
    """Manages timeline events with streaming file output.

    Events are written to a JSONL file as they occur for crash safety
    and memory efficiency with large simulations.
    """

    def __init__(self, output_path: Path | str):
        """Initialize timeline manager.

        Args:
            output_path: Path to JSONL output file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode for streaming
        self._file = open(self.output_path, "w")
        self._event_count = 0

    def log_event(self, event: SimulationEvent) -> None:
        """Log an event to the timeline.

        Args:
            event: Event to log
        """
        event_dict = {
            "timestep": event.timestep,
            "event_type": event.event_type.value,
            "agent_id": event.agent_id,
            "details": event.details,
            "timestamp": event.timestamp.isoformat(),
        }

        self._file.write(json.dumps(event_dict) + "\n")
        self._event_count += 1

    def flush(self) -> None:
        """Flush buffer to disk."""
        self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        self._file.close()

    def get_event_count(self) -> int:
        """Get total number of logged events."""
        return self._event_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class TimelineReader:
    """Reads and queries timeline events from a JSONL file."""

    def __init__(self, timeline_path: Path | str):
        """Initialize timeline reader.

        Args:
            timeline_path: Path to timeline JSONL file
        """
        self.timeline_path = Path(timeline_path)

    def iter_events(self) -> Iterator[dict[str, Any]]:
        """Iterate over all events in the timeline.

        Yields:
            Event dictionaries
        """
        with open(self.timeline_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def get_all_events(self) -> list[dict[str, Any]]:
        """Load all events into memory.

        Returns:
            List of event dictionaries
        """
        return list(self.iter_events())

    def get_events_for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        """Get all events involving a specific agent.

        Args:
            agent_id: Agent ID to filter by

        Returns:
            List of events for this agent
        """
        return [e for e in self.iter_events() if e["agent_id"] == agent_id]

    def get_events_for_timestep(self, timestep: int) -> list[dict[str, Any]]:
        """Get all events in a specific timestep.

        Args:
            timestep: Timestep to filter by

        Returns:
            List of events in this timestep
        """
        return [e for e in self.iter_events() if e["timestep"] == timestep]

    def get_events_by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Get all events of a specific type.

        Args:
            event_type: Event type to filter by

        Returns:
            List of events of this type
        """
        return [e for e in self.iter_events() if e["event_type"] == event_type]

    def get_cascade_tree(self, initial_agent_id: str) -> dict[str, Any]:
        """Build cascade tree showing propagation from an initial agent.

        Args:
            initial_agent_id: The agent who started a cascade

        Returns:
            Tree structure showing who spread to whom
        """
        # Build mapping of exposures
        network_exposures = self.get_events_by_type("network_exposure")

        # Map: target -> source
        exposure_sources: dict[str, list[str]] = defaultdict(list)
        for event in network_exposures:
            target = event["agent_id"]
            source = event["details"].get("source")
            if source:
                exposure_sources[target].append(source)

        # Build tree from initial agent
        def build_subtree(agent_id: str, visited: set) -> dict[str, Any]:
            if agent_id in visited:
                return {"agent_id": agent_id, "children": [], "cycle": True}

            visited.add(agent_id)

            # Find who this agent exposed
            children = []
            for target, sources in exposure_sources.items():
                if agent_id in sources:
                    child_tree = build_subtree(target, visited.copy())
                    children.append(child_tree)

            return {
                "agent_id": agent_id,
                "children": children,
            }

        return build_subtree(initial_agent_id, set())

    def get_exposure_flow(self) -> list[dict[str, Any]]:
        """Get exposure flow over time.

        Returns:
            List of dicts with timestep, channel/source counts
        """
        events = self.get_all_events()

        timestep_data: dict[int, dict[str, int]] = defaultdict(
            lambda: {"seed": 0, "network": 0}
        )

        for event in events:
            t = event["timestep"]
            if event["event_type"] == "seed_exposure":
                timestep_data[t]["seed"] += 1
            elif event["event_type"] == "network_exposure":
                timestep_data[t]["network"] += 1

        return [
            {"timestep": t, **counts}
            for t, counts in sorted(timestep_data.items())
        ]

    def get_reasoning_summary(self) -> dict[str, Any]:
        """Get summary of reasoning events.

        Returns:
            Summary dict with counts and distributions
        """
        reasoning_events = self.get_events_by_type("agent_reasoned")

        positions: dict[str, int] = defaultdict(int)
        sentiments: list[float] = []
        share_count = 0

        for event in reasoning_events:
            details = event["details"]
            if "position" in details and details["position"]:
                positions[details["position"]] += 1
            if "sentiment" in details and details["sentiment"] is not None:
                sentiments.append(details["sentiment"])
            if details.get("will_share"):
                share_count += 1

        return {
            "total_reasoning_events": len(reasoning_events),
            "position_counts": dict(positions),
            "sentiment_mean": sum(sentiments) / len(sentiments) if sentiments else None,
            "sentiment_std": (
                (sum((s - sum(sentiments) / len(sentiments)) ** 2 for s in sentiments) / len(sentiments)) ** 0.5
                if sentiments else None
            ),
            "share_rate": share_count / len(reasoning_events) if reasoning_events else 0,
        }

    def get_unique_agents(self) -> set[str]:
        """Get set of all agent IDs in the timeline.

        Returns:
            Set of agent IDs
        """
        return {e["agent_id"] for e in self.iter_events()}

    def count_events(self) -> int:
        """Count total events in timeline.

        Returns:
            Event count
        """
        count = 0
        with open(self.timeline_path) as f:
            for _ in f:
                count += 1
        return count
