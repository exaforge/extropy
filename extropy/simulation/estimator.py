"""Simulation cost estimator.

Predicts API costs before running a simulation. No LLM calls, no API keys.
Loads the same data as the simulation engine, runs a simplified propagation
model, and returns estimated LLM calls, tokens, and USD cost.
"""

from dataclasses import dataclass, field
from typing import Any

from ..core.models import ScenarioSpec, PopulationSpec
from ..core.pricing import ModelPricing, get_pricing, resolve_default_model
from ..utils.eval_safe import eval_condition, ConditionError


@dataclass
class CostEstimate:
    """Result of a cost estimation run."""

    # Population info
    population_size: int
    avg_degree: float
    max_timesteps: int
    effective_timesteps: int

    # Model info
    pivotal_model: str
    routine_model: str
    pivotal_pricing: ModelPricing | None
    routine_pricing: ModelPricing | None

    # Call counts
    pass1_calls: int
    pass2_calls: int

    # Token estimates
    pass1_input_tokens: int
    pass1_output_tokens: int
    pass2_input_tokens: int
    pass2_output_tokens: int

    # USD cost
    pass1_cost: float | None
    pass2_cost: float | None
    total_cost: float | None

    # Per-timestep breakdown (timestep -> reasoning_calls)
    per_timestep: list[dict[str, Any]] = field(default_factory=list)


def _compute_avg_degree(network: dict[str, Any]) -> float:
    """Compute average node degree from network data."""
    edges = network.get("edges", [])
    if not edges:
        return 0.0

    # Count degree per node
    degree: dict[str, int] = {}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1

    if not degree:
        return 0.0
    return sum(degree.values()) / len(degree)


def _evaluate_rule_reach(
    rule_when: str,
    rule_probability: float,
    agents: list[dict[str, Any]],
) -> float:
    """Evaluate what fraction of agents match a seed exposure rule.

    Returns the expected number of agents exposed (matching fraction * probability).
    """
    if rule_when.lower() == "true" or rule_when == "1":
        return len(agents) * rule_probability

    matching = 0
    for agent in agents:
        try:
            if eval_condition(rule_when, agent, raise_on_error=True):
                matching += 1
        except ConditionError:
            pass

    return matching * rule_probability


# Token estimation constants for simulation cost prediction.
# These are calibrated estimates based on typical prompt sizes.
PERSONA_BASE_TOKENS = 80  # Base tokens for persona structure
PERSONA_PER_ATTRIBUTE_TOKENS = 15  # Additional tokens per agent attribute
CHARS_PER_TOKEN = 4  # Approximate characters per LLM token

# Pass 1 (role-play reasoning) token budget
PASS1_SYSTEM_OVERHEAD = 250  # System prompt + schema instructions
PASS1_EXPOSURE_OVERHEAD = 115  # Exposure history + peer opinions
PASS1_OUTPUT_TOKENS = 200  # Reasoning + public statement + conviction

# Pass 2 (classification) token budget
PASS2_INPUT_TOKENS = 300  # Reasoning text + classification schema
PASS2_OUTPUT_TOKENS = 70  # Position + outcomes JSON


def _estimate_token_counts(
    num_attributes: int,
    event_content_len: int,
) -> dict[str, int]:
    """Estimate per-call token counts for Pass 1 and Pass 2.

    Returns dict with keys: pass1_input, pass1_output, pass2_input, pass2_output.
    """
    persona_tokens = PERSONA_BASE_TOKENS + PERSONA_PER_ATTRIBUTE_TOKENS * num_attributes
    event_tokens = event_content_len // CHARS_PER_TOKEN

    pass1_input = (
        PASS1_SYSTEM_OVERHEAD + persona_tokens + event_tokens + PASS1_EXPOSURE_OVERHEAD
    )
    pass1_output = PASS1_OUTPUT_TOKENS

    pass2_input = PASS2_INPUT_TOKENS
    pass2_output = PASS2_OUTPUT_TOKENS

    return {
        "pass1_input": pass1_input,
        "pass1_output": pass1_output,
        "pass2_input": pass2_input,
        "pass2_output": pass2_output,
    }


def estimate_simulation_cost(
    scenario: ScenarioSpec,
    population_spec: PopulationSpec,
    agents: list[dict[str, Any]],
    network: dict[str, Any],
    strong_model: str = "",
    fast_model: str = "",
    multi_touch_threshold: int = 3,
) -> CostEstimate:
    """Estimate the cost of running a simulation.

    Uses a simplified SIR-like propagation model to predict how many
    agents reason at each timestep, then estimates tokens and cost.

    Args:
        scenario: Scenario specification
        population_spec: Population specification
        agents: List of agent dictionaries
        network: Network data dict
        strong_model: Model for Pass 1 (provider/model format, empty = config default)
        fast_model: Model for Pass 2 (provider/model format, empty = config default)
        multi_touch_threshold: Re-reasoning threshold

    Returns:
        CostEstimate with all predictions
    """
    n = len(agents)
    avg_degree = _compute_avg_degree(network)
    max_timesteps = scenario.simulation.max_timesteps
    share_prob = scenario.spread.share_probability
    will_share_rate = 0.55  # accounts for conviction-gated sharing

    # Resolve models — strip provider prefix for pricing lookup
    from ..config import get_config, parse_model_string

    config = get_config()
    eff_strong_str = strong_model or config.resolve_sim_strong()
    eff_fast_str = fast_model or config.resolve_sim_fast()
    _, eff_pivotal = parse_model_string(eff_strong_str)
    _, eff_routine = parse_model_string(eff_fast_str)

    # Pre-compute seed exposure schedule: timestep -> expected new seed exposures
    seed_schedule: dict[int, float] = {}
    for rule in scenario.seed_exposure.rules:
        reach = _evaluate_rule_reach(rule.when, rule.probability, agents)
        ts = rule.timestep
        seed_schedule[ts] = seed_schedule.get(ts, 0) + reach

    # Simplified propagation model
    exposed = 0.0  # number of exposed agents (float for fractional tracking)
    reasoned_ever = 0.0  # number that have reasoned at least once
    total_pass1 = 0
    total_pass2 = 0
    per_timestep: list[dict[str, Any]] = []
    effective_timesteps = max_timesteps

    for t in range(max_timesteps):
        # Seed exposures this timestep
        new_seed = seed_schedule.get(t, 0.0)

        # Network propagation: sharers * avg_degree * share_prob * (1 - exposure_rate)
        exposure_rate = min(exposed / n, 1.0) if n > 0 else 0.0
        sharers = reasoned_ever * will_share_rate
        new_network = sharers * avg_degree * share_prob * (1.0 - exposure_rate)

        total_new = new_seed + new_network

        # Cap at remaining unexposed
        remaining = n - exposed
        total_new = min(total_new, remaining)
        total_new = max(total_new, 0.0)

        # First-time reasoning: all newly exposed agents reason
        first_time_reasoning = total_new

        # Re-reasoning: ~2% of previously-reasoned agents per timestep
        # (committed agents with conviction >= firm skip re-reasoning;
        # one-shot sharing + unique-source multi-touch reduce triggers)
        re_reasoning = reasoned_ever * 0.02 if reasoned_ever > 0 and t > 0 else 0.0

        reasoning_this_step = int(round(first_time_reasoning + re_reasoning))

        # Each reasoning event = 1 Pass 1 + 1 Pass 2
        total_pass1 += reasoning_this_step
        total_pass2 += reasoning_this_step

        # Update tracking
        exposed += total_new
        reasoned_ever += first_time_reasoning

        per_timestep.append(
            {
                "timestep": t,
                "exposure_rate": min(exposed / n, 1.0) if n > 0 else 0.0,
                "new_exposures": int(round(total_new)),
                "reasoning_calls": reasoning_this_step,
            }
        )

        # Early stop check
        if exposure_rate > 0.95 and total_new < 1:
            effective_timesteps = t + 1
            break

    # Token estimation
    num_attributes = len(population_spec.attributes)
    event_content_len = len(scenario.event.content)
    tok = _estimate_token_counts(num_attributes, event_content_len)

    pass1_input_total = total_pass1 * tok["pass1_input"]
    pass1_output_total = total_pass1 * tok["pass1_output"]
    pass2_input_total = total_pass2 * tok["pass2_input"]
    pass2_output_total = total_pass2 * tok["pass2_output"]

    # Cost calculation
    pivotal_pricing = get_pricing(eff_pivotal)
    routine_pricing = get_pricing(eff_routine)

    pass1_cost = None
    pass2_cost = None
    total_cost = None

    if pivotal_pricing:
        pass1_cost = (
            pass1_input_total * pivotal_pricing.input_per_mtok
            + pass1_output_total * pivotal_pricing.output_per_mtok
        ) / 1_000_000

    if routine_pricing:
        pass2_cost = (
            pass2_input_total * routine_pricing.input_per_mtok
            + pass2_output_total * routine_pricing.output_per_mtok
        ) / 1_000_000

    if pass1_cost is not None and pass2_cost is not None:
        total_cost = pass1_cost + pass2_cost

    return CostEstimate(
        population_size=n,
        avg_degree=avg_degree,
        max_timesteps=max_timesteps,
        effective_timesteps=effective_timesteps,
        pivotal_model=eff_pivotal,
        routine_model=eff_routine,
        pivotal_pricing=pivotal_pricing,
        routine_pricing=routine_pricing,
        pass1_calls=total_pass1,
        pass2_calls=total_pass2,
        pass1_input_tokens=pass1_input_total,
        pass1_output_tokens=pass1_output_total,
        pass2_input_tokens=pass2_input_total,
        pass2_output_tokens=pass2_output_total,
        pass1_cost=pass1_cost,
        pass2_cost=pass2_cost,
        total_cost=total_cost,
        per_timestep=per_timestep,
    )
