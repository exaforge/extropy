"""Similarity calculation for network generation.

Computes weighted similarity between agent pairs for edge probability.
Similarity is normalized to [0, 1] based on available attribute weights.
"""

from typing import Any

from .config import (
    AttributeWeightConfig,
    DEFAULT_ATTRIBUTE_WEIGHTS,
    SENIORITY_LEVELS,
    NetworkConfig,
)


def compute_match_score(
    value_a: Any,
    value_b: Any,
    config: AttributeWeightConfig,
) -> float:
    """Compute match score between two attribute values.

    Args:
        value_a: First agent's attribute value
        value_b: Second agent's attribute value
        config: Configuration for how to compare

    Returns:
        Match score in [0, 1]
    """
    if value_a is None or value_b is None:
        return 0.0

    if config.match_type == "exact":
        return 1.0 if value_a == value_b else 0.0

    elif config.match_type == "numeric_range":
        # 1 - |A - B| / range (normalized difference)
        try:
            diff = abs(float(value_a) - float(value_b))
            range_val = config.range_value or 1.0
            return max(0.0, 1.0 - diff / range_val)
        except (TypeError, ValueError):
            return 0.0

    elif config.match_type == "within_n":
        # 1 if within n levels, 0 otherwise
        # Used for ordinal attributes like seniority
        try:
            # Check if these are seniority values that need level lookup
            if isinstance(value_a, str) and value_a in SENIORITY_LEVELS:
                level_a = SENIORITY_LEVELS[value_a]
                level_b = SENIORITY_LEVELS.get(value_b, 0)
            else:
                level_a = float(value_a)
                level_b = float(value_b)

            n = config.range_value or 1
            return 1.0 if abs(level_a - level_b) <= n else 0.0
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def compute_similarity(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    attribute_weights: dict[str, AttributeWeightConfig] | None = None,
) -> float:
    """Compute normalized similarity between two agents.

    Similarity is computed as:
        raw_sim(A, B) = sum(weight_i * match_i(A, B))
        sim(A, B) = raw_sim(A, B) / sum(weight_i)    # Normalized to [0, 1]

    If an attribute is not present in the spec (e.g., care_level), it's skipped
    and the total weight is reduced accordingly.

    Args:
        agent_a: First agent's attributes
        agent_b: Second agent's attributes
        attribute_weights: Weights for each attribute (uses defaults if None)

    Returns:
        Normalized similarity score in [0, 1]
    """
    if attribute_weights is None:
        attribute_weights = DEFAULT_ATTRIBUTE_WEIGHTS

    raw_similarity = 0.0
    total_weight = 0.0

    for attr_name, config in attribute_weights.items():
        value_a = agent_a.get(attr_name)
        value_b = agent_b.get(attr_name)

        # Skip if either agent doesn't have this attribute
        if value_a is None or value_b is None:
            continue

        match_score = compute_match_score(value_a, value_b, config)
        raw_similarity += config.weight * match_score
        total_weight += config.weight

    # Normalize against available weights
    if total_weight == 0:
        return 0.0

    return raw_similarity / total_weight


def compute_similarity_matrix_sparse(
    agents: list[dict[str, Any]],
    config: NetworkConfig,
    threshold: float = 0.1,
) -> dict[tuple[int, int], float]:
    """Compute sparse similarity matrix for all agent pairs.

    Only stores pairs with similarity above threshold to save memory.
    For N agents, the full matrix would be N*(N-1)/2 entries.

    Args:
        agents: List of agent dictionaries
        config: Network configuration with attribute weights
        threshold: Minimum similarity to store (default 0.1)

    Returns:
        Dictionary mapping (i, j) pairs to similarity scores
        where i < j (only upper triangle stored)
    """
    n = len(agents)
    similarities: dict[tuple[int, int], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(agents[i], agents[j], config.attribute_weights)
            if sim >= threshold:
                similarities[(i, j)] = sim

    return similarities


def compute_degree_factor(
    agent: dict[str, Any],
    config: NetworkConfig,
) -> float:
    """Compute degree correction factor for an agent.

    Certain agents have more connections based on their attributes.
    Multipliers stack multiplicatively.

    Example: A teaching research chief at a university hospital gets
    2.0 × 1.4 × 1.3 × 1.2 × 1.2 = 5.2× baseline connections.

    Args:
        agent: Agent's attributes
        config: Network configuration with degree multipliers

    Returns:
        Degree correction factor (>= 1.0)
    """
    factor = 1.0

    for multiplier in config.degree_multipliers:
        value = agent.get(multiplier.attribute)
        if value is None:
            continue

        # Check if condition matches
        if callable(multiplier.condition):
            if multiplier.condition(value):
                factor *= multiplier.multiplier
        else:
            if value == multiplier.condition:
                factor *= multiplier.multiplier

    return factor


def sigmoid(x: float, threshold: float = 0.3, steepness: float = 10.0) -> float:
    """Compute sigmoid function for edge probability.

    Makes probability increase sharply around the threshold.

    Args:
        x: Input value (similarity)
        threshold: Center point of sigmoid
        steepness: How sharp the transition is

    Returns:
        Sigmoid output in (0, 1)
    """
    import math

    z = steepness * (x - threshold)
    # Clamp to avoid overflow
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def compute_edge_probability(
    similarity: float,
    degree_factor_a: float,
    degree_factor_b: float,
    base_rate: float,
    config: NetworkConfig,
) -> float:
    """Compute probability of edge between two agents.

    P(edge A<->B) = base_rate × sigmoid(similarity, threshold, steepness)
                    × degree_factor(A) × degree_factor(B)

    Clamped to [0, 1].

    Args:
        similarity: Similarity score between agents
        degree_factor_a: Degree correction for first agent
        degree_factor_b: Degree correction for second agent
        base_rate: Base edge probability (calibrated for target avg_degree)
        config: Network configuration

    Returns:
        Edge probability in [0, 1]
    """
    sig = sigmoid(similarity, config.similarity_threshold, config.similarity_steepness)
    prob = base_rate * sig * degree_factor_a * degree_factor_b
    return min(1.0, prob)
