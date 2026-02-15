"""Network-related models.

Contains models for network generation results and metrics.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Edge(BaseModel):
    """A single edge in the network.

    Attributes:
        source: Source agent ID
        target: Target agent ID
        weight: Edge weight (similarity-derived, 0-1)
        edge_type: Type of connection (colleague, mentor_mentee, etc.)
        bidirectional: Whether communication flows both ways (always True)
        influence_weight: Asymmetric influence weights
    """

    source: str
    target: str
    weight: float
    edge_type: str
    bidirectional: bool = True
    influence_weight: dict[str, float] = Field(default_factory=dict)
    structural: bool = False
    context: str | None = None  # "household", "workplace", "neighborhood", etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "type": self.edge_type,
            "bidirectional": self.bidirectional,
            "influence_weight": {
                "source_to_target": round(
                    self.influence_weight.get("source_to_target", self.weight), 4
                ),
                "target_to_source": round(
                    self.influence_weight.get("target_to_source", self.weight), 4
                ),
            },
        }
        if self.structural:
            d["structural"] = True
        if self.context:
            d["context"] = self.context
        return d


class NodeMetrics(BaseModel):
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


class NetworkMetrics(BaseModel):
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
    degree_distribution: dict[int, int] = Field(default_factory=dict)

    def is_valid(self) -> tuple[bool, list[str]]:
        """Check if metrics are within expected ranges.

        Returns:
            Tuple of (is_valid, list of warning messages)
        """
        warnings = []

        if self.avg_degree < 15:
            warnings.append(
                f"avg_degree {self.avg_degree:.1f} below expected range (15-25)"
            )
        elif self.avg_degree > 25:
            warnings.append(
                f"avg_degree {self.avg_degree:.1f} above expected range (15-25)"
            )

        if self.clustering_coefficient < 0.3:
            warnings.append(
                f"clustering_coefficient {self.clustering_coefficient:.2f} below expected range (0.3-0.5)"
            )
        elif self.clustering_coefficient > 0.5:
            warnings.append(
                f"clustering_coefficient {self.clustering_coefficient:.2f} above expected range (0.3-0.5)"
            )

        if self.avg_path_length is not None:
            if self.avg_path_length < 3:
                warnings.append(
                    f"avg_path_length {self.avg_path_length:.2f} below expected range (3-5)"
                )
            elif self.avg_path_length > 5:
                warnings.append(
                    f"avg_path_length {self.avg_path_length:.2f} above expected range (3-5)"
                )

        if self.modularity < 0.4:
            warnings.append(
                f"modularity {self.modularity:.2f} below expected range (0.4-0.7)"
            )
        elif self.modularity > 0.7:
            warnings.append(
                f"modularity {self.modularity:.2f} above expected range (0.4-0.7)"
            )

        if self.largest_component_ratio < 0.95:
            warnings.append(
                f"largest_component_ratio {self.largest_component_ratio:.2f} below expected (>0.95)"
            )

        return len(warnings) == 0, warnings


class NetworkResult(BaseModel):
    """Result of network generation.

    Attributes:
        meta: Metadata about the generation
        edges: List of edges
        node_metrics: Per-agent metrics (if computed)
        network_metrics: Network-level metrics (if computed)
    """

    meta: dict[str, Any]
    edges: list[Edge]
    node_metrics: dict[str, NodeMetrics] | None = None
    network_metrics: NetworkMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "meta": self.meta,
            "edges": [e.to_dict() for e in self.edges],
        }

        if self.node_metrics:
            result["node_metrics"] = {
                agent_id: {
                    "degree": m.degree,
                    "influence_score": round(m.influence_score, 6),
                    "betweenness": round(m.betweenness, 6),
                    "cluster_id": m.cluster_id,
                    "echo_chamber_score": round(m.echo_chamber_score, 4),
                }
                for agent_id, m in self.node_metrics.items()
            }

        return result

    def save_json(self, path: Path | str) -> None:
        """Save network to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
