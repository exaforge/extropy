"""Network metrics and validation for generated social graphs.

Computes validation metrics and per-agent derived metrics including:
- Network validation: avg degree, clustering, path length, modularity
- Node metrics: PageRank, betweenness, cluster ID, echo chamber score
"""

import logging
from collections import defaultdict
from typing import Any

from ...core.models import NetworkMetrics, NodeMetrics

logger = logging.getLogger(__name__)

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def _build_networkx_graph(
    edges: list[dict[str, Any]],
    agent_ids: list[str],
) -> "nx.Graph":
    """Build NetworkX graph from edge list.

    Args:
        edges: List of edge dictionaries with source, target, weight
        agent_ids: List of agent IDs for nodes

    Returns:
        NetworkX Graph object
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for network metrics. Install with: pip install networkx"
        )

    G = nx.Graph()
    G.add_nodes_from(agent_ids)

    for edge in edges:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge.get("weight", 1.0),
            edge_type=edge.get("type", "unknown"),
        )

    return G


def compute_network_metrics(
    edges: list[dict[str, Any]],
    agent_ids: list[str],
) -> NetworkMetrics:
    """Compute validation metrics for the network.

    Args:
        edges: List of edge dictionaries
        agent_ids: List of agent IDs

    Returns:
        NetworkMetrics with validation metrics
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for network metrics. Install with: pip install networkx"
        )

    G = _build_networkx_graph(edges, agent_ids)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Average degree
    avg_degree = 2 * m / n if n > 0 else 0.0

    # Clustering coefficient
    clustering = nx.average_clustering(G)

    # Average path length (only for largest connected component)
    avg_path_length = None
    components = list(nx.connected_components(G))
    if components:
        largest_cc = max(components, key=len)
        largest_cc_ratio = len(largest_cc) / n if n > 0 else 0.0

        if len(largest_cc) > 1:
            subgraph = G.subgraph(largest_cc)
            try:
                avg_path_length = nx.average_shortest_path_length(subgraph)
            except nx.NetworkXError:
                avg_path_length = None
    else:
        largest_cc_ratio = 0.0

    # Modularity via Louvain community detection
    try:
        communities = nx.community.louvain_communities(G, seed=42)
        modularity = nx.community.modularity(G, communities)
    except Exception:
        communities = []
        modularity = 0.0

    # Degree assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortativity = 0.0

    # Degree distribution
    degree_dist = defaultdict(int)
    for _, degree in G.degree():
        degree_dist[degree] += 1

    return NetworkMetrics(
        node_count=n,
        edge_count=m,
        avg_degree=avg_degree,
        clustering_coefficient=clustering,
        avg_path_length=avg_path_length,
        modularity=modularity,
        largest_component_ratio=largest_cc_ratio,
        degree_assortativity=assortativity,
        degree_distribution=dict(degree_dist),
    )


def compute_node_metrics(
    edges: list[dict[str, Any]],
    agent_ids: list[str],
) -> dict[str, NodeMetrics]:
    """Compute per-agent metrics for simulation.

    Args:
        edges: List of edge dictionaries
        agent_ids: List of agent IDs

    Returns:
        Dictionary mapping agent_id to NodeMetrics
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for node metrics. Install with: pip install networkx"
        )

    G = _build_networkx_graph(edges, agent_ids)

    # Compute metrics
    degrees = dict(G.degree())
    pagerank = nx.pagerank(G, alpha=0.85)
    betweenness = nx.betweenness_centrality(G)
    local_clustering = nx.clustering(G)

    # Community detection
    try:
        communities = nx.community.louvain_communities(G, seed=42)
        # Build node -> cluster_id mapping
        node_to_cluster = {}
        for cluster_id, community in enumerate(communities):
            for node in community:
                node_to_cluster[node] = cluster_id
    except Exception:
        node_to_cluster = {agent_id: 0 for agent_id in agent_ids}

    # Build adjacency for echo chamber calculation
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])

    # Compute echo chamber score (% of edges within same cluster)
    def compute_echo_chamber(agent_id: str) -> float:
        neighbors = adjacency[agent_id]
        if not neighbors:
            return 0.0
        my_cluster = node_to_cluster.get(agent_id, -1)
        same_cluster = sum(1 for n in neighbors if node_to_cluster.get(n) == my_cluster)
        return same_cluster / len(neighbors)

    # Build result
    result = {}
    for agent_id in agent_ids:
        result[agent_id] = NodeMetrics(
            degree=degrees.get(agent_id, 0),
            influence_score=pagerank.get(agent_id, 0.0),
            betweenness=betweenness.get(agent_id, 0.0),
            cluster_id=node_to_cluster.get(agent_id, 0),
            echo_chamber_score=compute_echo_chamber(agent_id),
            local_clustering=local_clustering.get(agent_id, 0.0),
        )

    return result


def validate_network(
    edges: list[dict[str, Any]],
    agent_ids: list[str],
    verbose: bool = False,
) -> tuple[bool, NetworkMetrics, list[str]]:
    """Validate network against expected metric ranges.

    Args:
        edges: List of edge dictionaries
        agent_ids: List of agent IDs
        verbose: If True, log detailed metrics

    Returns:
        Tuple of (is_valid, metrics, warnings)
    """
    metrics = compute_network_metrics(edges, agent_ids)
    is_valid, warnings = metrics.is_valid()

    if verbose:
        logger.info("Network Validation Report:")
        logger.info("  Nodes: %s", metrics.node_count)
        logger.info("  Edges: %s", metrics.edge_count)
        logger.info("  Avg Degree: %.2f", metrics.avg_degree)
        logger.info("  Clustering: %.3f", metrics.clustering_coefficient)
        logger.info(
            f"  Avg Path Length: {metrics.avg_path_length:.2f}"
            if metrics.avg_path_length
            else "  Avg Path Length: N/A (disconnected)"
        )
        logger.info("  Modularity: %.3f", metrics.modularity)
        logger.info("  Largest Component: %.1f%%", metrics.largest_component_ratio * 100)
        logger.info("  Degree Assortativity: %.3f", metrics.degree_assortativity)

        if warnings:
            logger.info("Warnings:")
            for w in warnings:
                logger.info("  - %s", w)
        else:
            logger.info("All metrics within expected ranges.")

    return is_valid, metrics, warnings
