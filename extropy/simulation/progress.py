"""Thread-safe simulation progress tracking.

Provides a shared state object that the simulation thread updates
per-agent, and the CLI display thread reads via snapshot().
"""

import threading
from dataclasses import dataclass, field


@dataclass
class SimulationProgress:
    """Thread-safe progress state shared between simulation and display threads.

    The simulation thread calls record_agent_done() per agent.
    The CLI display thread calls snapshot() to get a safe copy for rendering.
    Position counts are cumulative across all timesteps (shows overall distribution).
    """

    timestep: int = 0
    max_timesteps: int = 0
    agents_total: int = 0
    agents_done: int = 0
    exposure_rate: float = 0.0
    position_counts: dict[str, int] = field(default_factory=dict)
    _sentiment_sum: float = 0.0
    _sentiment_count: int = 0
    _conviction_sum: float = 0.0
    _conviction_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def begin_timestep(
        self,
        timestep: int,
        max_timesteps: int,
        agents_total: int,
        exposure_rate: float,
    ) -> None:
        """Reset per-timestep counters for a new timestep.

        Args:
            timestep: Current timestep number
            max_timesteps: Total timesteps configured
            agents_total: Number of agents reasoning this timestep
            exposure_rate: Current exposure rate
        """
        with self._lock:
            self.timestep = timestep
            self.max_timesteps = max_timesteps
            self.agents_total = agents_total
            self.agents_done = 0
            self.exposure_rate = exposure_rate

    def record_agent_done(
        self,
        position: str | None,
        sentiment: float | None,
        conviction: float | None,
    ) -> None:
        """Record that an agent has completed reasoning.

        Args:
            position: Agent's position (None if no categorical outcome)
            sentiment: Agent's sentiment (-1 to 1)
            conviction: Agent's conviction float (0.1 to 0.9)
        """
        with self._lock:
            self.agents_done += 1
            if position is not None:
                self.position_counts[position] = (
                    self.position_counts.get(position, 0) + 1
                )
            if sentiment is not None:
                self._sentiment_sum += sentiment
                self._sentiment_count += 1
            if conviction is not None:
                self._conviction_sum += conviction
                self._conviction_count += 1

    def snapshot(self) -> dict:
        """Return a thread-safe copy of all display-relevant fields.

        Returns:
            Dictionary with: timestep, max_timesteps, agents_total, agents_done,
            exposure_rate, position_counts, avg_sentiment, avg_conviction.
        """
        with self._lock:
            return {
                "timestep": self.timestep,
                "max_timesteps": self.max_timesteps,
                "agents_total": self.agents_total,
                "agents_done": self.agents_done,
                "exposure_rate": self.exposure_rate,
                "position_counts": dict(self.position_counts),
                "avg_sentiment": (
                    self._sentiment_sum / self._sentiment_count
                    if self._sentiment_count > 0
                    else None
                ),
                "avg_conviction": (
                    self._conviction_sum / self._conviction_count
                    if self._conviction_count > 0
                    else None
                ),
            }
