"""Network generation module for Entropy.

Creates realistic social graphs between agents sampled from a PopulationSpec.
The goal is to model how professionals are connected through work relationships,
professional societies, and informal ties.

Usage:
    from entropy.network import generate_network, NetworkConfig, NetworkResult

    # Load agents from JSON
    agents = load_agents_json("agents.json")

    # Generate network with default config
    result = generate_network(agents)
    result.save_json("network.json")

    # Generate with custom config and compute metrics
    config = NetworkConfig(avg_degree=25, rewire_prob=0.1)
    result = generate_network_with_metrics(agents, config)

Key Concepts:
    - Homophily: People preferentially connect with similar others
    - Small-World: High clustering + short path lengths
    - Weak Ties: Sparse connections between clusters
    - Degree Distribution: Some nodes are hubs (opinion leaders)

Algorithm:
    1. Compute weighted similarity between agent pairs
    2. Sample edges proportional to similarity
    3. Apply degree correction for high-influence agents
    4. Add Watts-Strogatz rewiring for small-world properties

Edge Types:
    - colleague: Same employer + similar experience
    - mentor_mentee: Same employer + large experience gap
    - society: Same specialty + both society members
    - conference: Same specialty + met at conferences
    - regional: Same region + different specialty
    - weak_tie: Random rewiring (incidental connection)
"""

from .config import (
    NetworkConfig,
    AttributeWeightConfig,
    DegreeMultiplierConfig,
    DEFAULT_ATTRIBUTE_WEIGHTS,
    DEFAULT_DEGREE_MULTIPLIERS,
    SENIORITY_LEVELS,
)
from .similarity import (
    compute_similarity,
    compute_degree_factor,
    compute_edge_probability,
    compute_match_score,
    sigmoid,
)
from .generator import (
    generate_network,
    generate_network_with_metrics,
    load_agents_json,
    Edge,
    NetworkResult,
)
from .metrics import (
    NetworkMetrics,
    NodeMetrics,
    compute_network_metrics,
    compute_node_metrics,
    validate_network,
)

__all__ = [
    # Main functions
    "generate_network",
    "generate_network_with_metrics",
    "load_agents_json",
    # Configuration
    "NetworkConfig",
    "AttributeWeightConfig",
    "DegreeMultiplierConfig",
    "DEFAULT_ATTRIBUTE_WEIGHTS",
    "DEFAULT_DEGREE_MULTIPLIERS",
    "SENIORITY_LEVELS",
    # Result types
    "Edge",
    "NetworkResult",
    "NetworkMetrics",
    "NodeMetrics",
    # Similarity functions
    "compute_similarity",
    "compute_degree_factor",
    "compute_edge_probability",
    "compute_match_score",
    "sigmoid",
    # Metrics functions
    "compute_network_metrics",
    "compute_node_metrics",
    "validate_network",
]
