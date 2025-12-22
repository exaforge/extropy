"""Network metrics and validation for generated social graphs.

Computes validation metrics and per-agent derived metrics including:
- Network validation: avg degree, clustering, path length, modularity
- Node metrics: PageRank, betweenness, cluster ID, echo chamber score
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class NetworkMetrics:
    """Validation metrics for the generated network.

    Expected ranges from design doc:
        - avg_degree: 15-25
        - clustering_coefficient: 0.3-0.5
        - avg_path_length: 3-5
        - modularity: 0.4-0.7
        - largest_component_ratio: >0.95
        - degree_assortativity: 0.1-0.3
    """
    node_count: int
    edge_count: int
    avg_degree: float
    clustering_coefficient: float
    avg_path_length: float | None  # None if graph is disconnected
    modularity: float
    largest_component_ratio: float
    degree_assortativity: float
    degree_distribution: dict[int, int] = field(default_factory=dict)

    def is_valid(self) -> tuple[bool, list[str]]:
        """Check if metrics are within expected ranges.

        Returns:
            Tuple of (is_valid, list of warning messages)
        """
        warnings = []

        if self.avg_degree < 15:
            warnings.append(f"avg_degree {self.avg_degree:.1f} below expected range (15-25)")
        elif self.avg_degree > 25:
            warnings.append(f"avg_degree {self.avg_degree:.1f} above expected range (15-25)")

        if self.clustering_coefficient < 0.3:
            warnings.append(f"clustering_coefficient {self.clustering_coefficient:.2f} below expected range (0.3-0.5)")
        elif self.clustering_coefficient > 0.5:
            warnings.append(f"clustering_coefficient {self.clustering_coefficient:.2f} above expected range (0.3-0.5)")

        if self.avg_path_length is not None:
            if self.avg_path_length < 3:
                warnings.append(f"avg_path_length {self.avg_path_length:.2f} below expected range (3-5)")
            elif self.avg_path_length > 5:
                warnings.append(f"avg_path_length {self.avg_path_length:.2f} above expected range (3-5)")

        if self.modularity < 0.4:
            warnings.append(f"modularity {self.modularity:.2f} below expected range (0.4-0.7)")
        elif self.modularity > 0.7:
            warnings.append(f"modularity {self.modularity:.2f} above expected range (0.4-0.7)")

        if self.largest_component_ratio < 0.95:
            warnings.append(f"largest_component_ratio {self.largest_component_ratio:.2f} below expected (>0.95)")

        return len(warnings) == 0, warnings


@dataclass
class NodeMetrics:
    """Per-agent derived metrics for simulation.

    Attributes:
        degree: Number of connections
        influence_score: PageRank score (whose opinions spread further)
        betweenness: Betweenness centrality (brokers between communities)
        cluster_id: Community/cluster ID from Louvain detection
        echo_chamber_score: % of edges within same cluster
        local_clustering: Node clustering coefficient
    """
    degree: int
    influence_score: float
    betweenness: float
    cluster_id: int
    echo_chamber_score: float
    local_clustering: float


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
        raise ImportError("networkx is required for network metrics. Install with: pip install networkx")

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
        raise ImportError("networkx is required for network metrics. Install with: pip install networkx")

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
        raise ImportError("networkx is required for node metrics. Install with: pip install networkx")

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
        verbose: If True, print detailed metrics

    Returns:
        Tuple of (is_valid, metrics, warnings)
    """
    metrics = compute_network_metrics(edges, agent_ids)
    is_valid, warnings = metrics.is_valid()

    if verbose:
        print(f"Network Validation Report:")
        print(f"  Nodes: {metrics.node_count}")
        print(f"  Edges: {metrics.edge_count}")
        print(f"  Avg Degree: {metrics.avg_degree:.2f}")
        print(f"  Clustering: {metrics.clustering_coefficient:.3f}")
        print(f"  Avg Path Length: {metrics.avg_path_length:.2f}" if metrics.avg_path_length else "  Avg Path Length: N/A (disconnected)")
        print(f"  Modularity: {metrics.modularity:.3f}")
        print(f"  Largest Component: {metrics.largest_component_ratio:.1%}")
        print(f"  Degree Assortativity: {metrics.degree_assortativity:.3f}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("\nAll metrics within expected ranges.")

    return is_valid, metrics, warnings
