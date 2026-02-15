"""Stopping condition evaluation for simulation.

Determines when a simulation should stop based on various conditions
like max timesteps, exposure rate thresholds, or convergence.
"""

import logging
import re

from ..core.models import SimulationConfig, TimestepSummary
from .state import StateManager

logger = logging.getLogger(__name__)


def parse_comparison(condition: str) -> tuple[str, str, float] | None:
    """Parse a simple comparison condition.

    Supports: >, <, >=, <=, ==, !=

    Args:
        condition: Condition string like "exposure_rate > 0.95"

    Returns:
        Tuple of (variable, operator, value) or None if parsing fails
    """
    # Match patterns like "variable > 0.95" or "variable >= 0.95"
    pattern = r"(\w+)\s*(>=|<=|>|<|==|!=)\s*([\d.]+)"
    match = re.match(pattern, condition.strip())

    if not match:
        return None

    variable = match.group(1)
    operator = match.group(2)
    try:
        value = float(match.group(3))
    except ValueError:
        return None

    return (variable, operator, value)


def evaluate_comparison(
    variable: str,
    operator: str,
    threshold: float,
    state_manager: StateManager,
    recent_summaries: list[TimestepSummary],
) -> bool:
    """Evaluate a comparison condition.

    Args:
        variable: Variable name (e.g., "exposure_rate")
        operator: Comparison operator
        threshold: Threshold value
        state_manager: State manager for current values
        recent_summaries: Recent timestep summaries

    Returns:
        True if condition is met
    """
    # Get current value based on variable
    if variable == "exposure_rate":
        current_value = state_manager.get_exposure_rate()
    elif variable == "average_sentiment":
        current_value = state_manager.get_average_sentiment()
        if current_value is None:
            return False
    else:
        # Unknown variable
        logger.warning(f"Unknown stopping condition variable: {variable}")
        return False

    # Evaluate comparison
    if operator == ">":
        return current_value > threshold
    elif operator == "<":
        return current_value < threshold
    elif operator == ">=":
        return current_value >= threshold
    elif operator == "<=":
        return current_value <= threshold
    elif operator == "==":
        return abs(current_value - threshold) < 0.001
    elif operator == "!=":
        return abs(current_value - threshold) >= 0.001

    return False


def evaluate_no_state_changes(
    condition: str,
    recent_summaries: list[TimestepSummary],
) -> bool:
    """Evaluate a no_state_changes_for condition.

    Args:
        condition: Condition like "no_state_changes_for > 10"
        recent_summaries: Recent timestep summaries

    Returns:
        True if no state changes for the specified number of timesteps
    """
    # Extract threshold
    pattern = r"no_state_changes_for\s*>\s*(\d+)"
    match = re.match(pattern, condition.strip())

    if not match:
        return False

    threshold = int(match.group(1))

    if len(recent_summaries) < threshold:
        return False

    # Check if all recent summaries have zero state changes
    return all(s.state_changes == 0 for s in recent_summaries[-threshold:])


def evaluate_convergence(
    recent_summaries: list[TimestepSummary],
    window: int = 5,
    tolerance: float = 0.01,
) -> bool:
    """Check if position distribution has converged.

    Convergence is detected when the position distribution remains
    stable within tolerance for the specified window.

    Args:
        recent_summaries: Recent timestep summaries
        window: Number of timesteps to check
        tolerance: Maximum variance allowed

    Returns:
        True if converged
    """
    if len(recent_summaries) < window:
        return False

    recent = recent_summaries[-window:]

    # Get all positions across recent summaries
    all_positions = set()
    for summary in recent:
        all_positions.update(summary.position_distribution.keys())

    if not all_positions:
        return False

    # Check variance for each position
    for position in all_positions:
        values = []
        for summary in recent:
            total = sum(summary.position_distribution.values())
            if total > 0:
                count = summary.position_distribution.get(position, 0)
                values.append(count / total)
            else:
                values.append(0)

        if values:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            if variance > tolerance:
                return False

    return True


def evaluate_condition(
    condition: str,
    timestep: int,
    state_manager: StateManager,
    recent_summaries: list[TimestepSummary],
) -> bool:
    """Evaluate a single stopping condition.

    Supported conditions:
    - "exposure_rate > 0.95"
    - "exposure_rate >= 0.9"
    - "no_state_changes_for > 10"
    - "convergence"

    Args:
        condition: Condition string
        timestep: Current timestep
        state_manager: State manager
        recent_summaries: Recent timestep summaries

    Returns:
        True if condition is met
    """
    condition = condition.strip().lower()

    # Check for no_state_changes_for pattern
    if "no_state_changes_for" in condition:
        return evaluate_no_state_changes(condition, recent_summaries)

    # Check for convergence
    if condition == "convergence":
        return evaluate_convergence(recent_summaries)

    # Try to parse as simple comparison
    parsed = parse_comparison(condition)
    if parsed:
        variable, operator, threshold = parsed
        return evaluate_comparison(
            variable, operator, threshold, state_manager, recent_summaries
        )

    logger.warning(f"Could not parse stopping condition: {condition}")
    return False


def evaluate_stopping_conditions(
    timestep: int,
    config: SimulationConfig,
    state_manager: StateManager,
    recent_summaries: list[TimestepSummary],
) -> tuple[bool, str | None]:
    """Evaluate all stopping conditions.

    Args:
        timestep: Current timestep
        config: Simulation configuration
        state_manager: State manager
        recent_summaries: Recent timestep summaries

    Returns:
        Tuple of (should_stop, reason) where reason is the condition
        that triggered the stop, or None if no stop.
    """
    # Always check max timesteps
    if timestep >= config.max_timesteps - 1:
        return True, "max_timesteps_reached"

    # Evaluate custom conditions
    if config.stop_conditions:
        for condition in config.stop_conditions:
            if evaluate_condition(condition, timestep, state_manager, recent_summaries):
                return True, condition

    # Convergence auto-stop: position distribution stable for 3 timesteps
    if evaluate_convergence(recent_summaries, window=3, tolerance=0.005):
        return True, "converged"

    # Quiescence auto-stop: no agents reasoned for last 3 timesteps
    if len(recent_summaries) >= 3:
        last_3 = recent_summaries[-3:]
        if all(s.agents_reasoned == 0 for s in last_3):
            return True, "simulation_quiescent"

    return False, None


def estimate_remaining_timesteps(
    current_timestep: int,
    config: SimulationConfig,
    state_manager: StateManager,
    recent_summaries: list[TimestepSummary],
) -> int | None:
    """Estimate remaining timesteps until completion.

    Based on current trends, estimates when stopping conditions
    might be met.

    Args:
        current_timestep: Current timestep
        config: Simulation configuration
        state_manager: State manager
        recent_summaries: Recent timestep summaries

    Returns:
        Estimated remaining timesteps, or None if cannot estimate
    """
    remaining = config.max_timesteps - current_timestep - 1

    # If we have exposure rate conditions, try to estimate
    if config.stop_conditions and len(recent_summaries) >= 3:
        for condition in config.stop_conditions:
            if "exposure_rate" in condition:
                parsed = parse_comparison(condition)
                if parsed:
                    _, operator, threshold = parsed
                    if operator in (">", ">="):
                        current_rate = state_manager.get_exposure_rate()

                        # Estimate rate of change
                        recent_rates = [s.exposure_rate for s in recent_summaries[-5:]]
                        if len(recent_rates) >= 2:
                            rate_change = (recent_rates[-1] - recent_rates[0]) / len(
                                recent_rates
                            )
                            if rate_change > 0:
                                needed = threshold - current_rate
                                estimated = int(needed / rate_change)
                                remaining = min(remaining, estimated)

    return max(0, remaining)
