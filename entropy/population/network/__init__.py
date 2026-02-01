"""Network generation module for Entropy.

Creates realistic social graphs between agents sampled from a PopulationSpec.
The goal is to model how professionals are connected through work relationships,
professional societies, and informal ties.

Usage:
    from entropy.network import generate_network, NetworkConfig, NetworkResult

    # Load agents from JSON
    agents = load_agents_json("agents.json")

    # Generate network with default config (flat — no similarity structure)
    result = generate_network(agents)
    result.save_json("network.json")

    # Generate with a population-aware config (LLM-generated or manual)
    config = NetworkConfig.from_yaml("network-config.yaml")
    result = generate_network_with_metrics(agents, config)

    # Use the German surgeon reference config for backward compat
    from entropy.population.network.config import GERMAN_SURGEON_CONFIG
    result = generate_network(agents, GERMAN_SURGEON_CONFIG)

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

Edge Types (data-driven via EdgeTypeRule):
    Configured per-network via edge_type_rules in NetworkConfig.
    Rules are evaluated in priority order against agent pairs.
    The default_edge_type is used when no rule matches.
"""

from .config import (
    NetworkConfig,
    AttributeWeightConfig,
    DegreeMultiplierConfig,
    EdgeTypeRule,
    InfluenceFactorConfig,
    GERMAN_SURGEON_CONFIG,
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
)
from .config_generator import generate_network_config
from .metrics import (
    compute_network_metrics,
    compute_node_metrics,
    validate_network,
)
from ...core.models import Edge, NetworkResult, NetworkMetrics, NodeMetrics

__all__ = [
    # Main functions
    "generate_network",
    "generate_network_with_metrics",
    "generate_network_config",
    "load_agents_json",
    # Configuration
    "NetworkConfig",
    "AttributeWeightConfig",
    "DegreeMultiplierConfig",
    "EdgeTypeRule",
    "InfluenceFactorConfig",
    "GERMAN_SURGEON_CONFIG",
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
