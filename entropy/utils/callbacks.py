"""Typed callback protocols for progress reporting across the pipeline.

These Protocol classes provide type-safe callback signatures without
requiring runtime changes — existing callables continue to work via duck typing.
"""

from typing import Any, Protocol


class StepProgressCallback(Protocol):
    """Callback for step-based progress (scenario compiler, persona generator).

    Args:
        step: Step identifier (e.g. "1/5", "structure")
        status: Human-readable status message
    """

    def __call__(self, step: str, status: str) -> None: ...


class TimestepProgressCallback(Protocol):
    """Callback for simulation timestep progress.

    Args:
        timestep: Current timestep number
        max_timesteps: Total timesteps configured
        status: Human-readable status message
    """

    def __call__(self, timestep: int, max_timesteps: int, status: str) -> None: ...


class ItemProgressCallback(Protocol):
    """Callback for item-based progress (sampler, network generator).

    Args:
        current: Current item index
        total: Total items to process
    """

    def __call__(self, current: int, total: int) -> None: ...


class HydrationProgressCallback(Protocol):
    """Callback for hydration step progress.

    Args:
        step: Step identifier (e.g. "independent", "derived")
        status: Human-readable status message
        count: Optional item count for sub-progress
    """

    def __call__(self, step: str, status: str, count: int | None = None) -> None: ...


class NetworkProgressCallback(Protocol):
    """Callback for network generation progress.

    Args:
        stage: Stage name (e.g. "edges", "rewiring")
        current: Current progress count
        total: Total items in this stage
    """

    def __call__(self, stage: str, current: int, total: int) -> None: ...


class AgentDoneCallback(Protocol):
    """Callback invoked after each agent completes reasoning.

    Args:
        agent_id: The agent's identifier
        response: The ReasoningResponse (or None if failed)
    """

    def __call__(self, agent_id: str, response: Any) -> None: ...
