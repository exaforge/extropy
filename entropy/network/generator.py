"""Network generation algorithms for creating social graphs between agents.

Implements the hybrid approach: attribute similarity + Watts-Strogatz rewiring.
"""

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import NetworkConfig, SENIORITY_LEVELS
from .similarity import (
    compute_similarity,
    compute_degree_factor,
    compute_edge_probability,
)
from .metrics import NetworkMetrics, NodeMetrics, compute_network_metrics, compute_node_metrics


@dataclass
class Edge:
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
    influence_weight: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "type": self.edge_type,
            "bidirectional": self.bidirectional,
            "influence_weight": {
                "source_to_target": round(self.influence_weight.get("source_to_target", self.weight), 4),
                "target_to_source": round(self.influence_weight.get("target_to_source", self.weight), 4),
            },
        }


@dataclass
class NetworkResult:
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


def _get_seniority_level(agent: dict[str, Any]) -> int:
    """Get numeric seniority level for an agent."""
    role = agent.get("role_seniority", "")
    return SENIORITY_LEVELS.get(role, 1)


def _infer_edge_type(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    is_rewired: bool = False,
) -> str:
    """Infer edge type based on agent attributes.

    Edge types from design doc:
        - mentor_mentee: Same employer + years_experience diff > 10
        - colleague: Same employer + years_experience diff <= 10
        - society: Same specialty + different employer + both society members
        - conference: Same specialty + different employer + not both society members
        - regional: Same federal_state + different specialty + different employer
        - weak_tie: Random rewiring (incidental connection)
    """
    if is_rewired:
        return "weak_tie"

    same_employer = agent_a.get("employer_type") == agent_b.get("employer_type")
    same_specialty = agent_a.get("surgical_specialty") == agent_b.get("surgical_specialty")
    same_state = agent_a.get("federal_state") == agent_b.get("federal_state")

    # Get experience difference
    exp_a = agent_a.get("years_experience", 0) or 0
    exp_b = agent_b.get("years_experience", 0) or 0
    exp_diff = abs(exp_a - exp_b)

    # Both have society membership?
    society_a = agent_a.get("professional_society_membership", False)
    society_b = agent_b.get("professional_society_membership", False)
    both_society = society_a and society_b

    # Infer type based on conditions
    if same_employer:
        if exp_diff > 10:
            return "mentor_mentee"
        return "colleague"

    if same_specialty:
        if both_society:
            return "society"
        return "conference"

    if same_state:
        return "regional"

    return "weak_tie"


def _compute_influence_weights(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    edge_weight: float,
) -> dict[str, float]:
    """Compute asymmetric influence weights for an edge.

    influence(A -> B) = base_influence × seniority_ratio × expertise_ratio

    seniority_ratio = seniority_level(A) / seniority_level(B)

    expertise_ratio = 1.0 + 0.2 * (int(research_A) - int(research_B))
                          + 0.1 * (int(teaching_A) - int(teaching_B))
    """
    level_a = _get_seniority_level(agent_a)
    level_b = _get_seniority_level(agent_b)

    research_a = 1 if agent_a.get("participation_in_research", False) else 0
    research_b = 1 if agent_b.get("participation_in_research", False) else 0
    teaching_a = 1 if agent_a.get("teaching_responsibility", False) else 0
    teaching_b = 1 if agent_b.get("teaching_responsibility", False) else 0

    # A -> B influence
    seniority_ratio_a_to_b = level_a / level_b if level_b > 0 else 1.0
    expertise_ratio_a_to_b = 1.0 + 0.2 * (research_a - research_b) + 0.1 * (teaching_a - teaching_b)
    influence_a_to_b = edge_weight * seniority_ratio_a_to_b * max(0.1, expertise_ratio_a_to_b)

    # B -> A influence
    seniority_ratio_b_to_a = level_b / level_a if level_a > 0 else 1.0
    expertise_ratio_b_to_a = 1.0 + 0.2 * (research_b - research_a) + 0.1 * (teaching_b - teaching_a)
    influence_b_to_a = edge_weight * seniority_ratio_b_to_a * max(0.1, expertise_ratio_b_to_a)

    return {
        "source_to_target": influence_a_to_b,
        "target_to_source": influence_b_to_a,
    }


def _calibrate_base_rate(
    n_agents: int,
    target_avg_degree: float,
    degree_factors: list[float],
    similarities: dict[tuple[int, int], float],
    config: NetworkConfig,
) -> float:
    """Calibrate base rate to achieve target average degree.

    Uses binary search to find the base rate that produces
    approximately the target number of edges.
    """
    target_edges = int(n_agents * target_avg_degree / 2)

    def estimate_edges(base_rate: float) -> float:
        total = 0.0
        for (i, j), sim in similarities.items():
            prob = compute_edge_probability(
                sim,
                degree_factors[i],
                degree_factors[j],
                base_rate,
                config,
            )
            total += prob
        return total

    # Binary search for base rate
    low, high = 0.0001, 1.0
    for _ in range(50):
        mid = (low + high) / 2
        estimated = estimate_edges(mid)
        if estimated < target_edges:
            low = mid
        else:
            high = mid

    return mid


def generate_network(
    agents: list[dict[str, Any]],
    config: NetworkConfig | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> NetworkResult:
    """Generate a social network from sampled agents.

    Uses the hybrid approach:
    1. Compute similarity matrix (sparse)
    2. Sample edges proportional to similarity
    3. Apply degree correction for high-influence agents
    4. Add random rewiring (5-10% of edges)

    Args:
        agents: List of agent dictionaries (must have _id field)
        config: Network configuration (uses defaults if None)
        on_progress: Optional callback(stage, current, total) for progress

    Returns:
        NetworkResult with edges and optionally metrics
    """
    if config is None:
        config = NetworkConfig()

    # Initialize RNG
    seed = config.seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = random.Random(seed)

    n = len(agents)
    agent_ids = [a.get("_id", f"agent_{i}") for i, a in enumerate(agents)]

    if on_progress:
        on_progress("Computing similarities", 0, n)

    # Step 1: Compute degree factors for all agents
    degree_factors = [compute_degree_factor(a, config) for a in agents]

    # Step 2: Compute similarity matrix (sparse, only above threshold)
    similarities: dict[tuple[int, int], float] = {}
    threshold = 0.05  # Store pairs with very low similarity too for completeness

    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_similarity(agents[i], agents[j], config.attribute_weights)
            if sim >= threshold:
                similarities[(i, j)] = sim

        if on_progress and i % 50 == 0:
            on_progress("Computing similarities", i, n)

    if on_progress:
        on_progress("Computing similarities", n, n)

    # Step 3: Calibrate base rate for target avg degree
    if on_progress:
        on_progress("Calibrating edge probability", 0, 1)

    base_rate = _calibrate_base_rate(
        n, config.avg_degree, degree_factors, similarities, config
    )

    if on_progress:
        on_progress("Calibrating edge probability", 1, 1)

    # Step 4: Sample edges
    if on_progress:
        on_progress("Sampling edges", 0, len(similarities))

    edges: list[Edge] = []
    edge_set: set[tuple[str, str]] = set()

    for idx, ((i, j), sim) in enumerate(similarities.items()):
        prob = compute_edge_probability(
            sim,
            degree_factors[i],
            degree_factors[j],
            base_rate,
            config,
        )

        if rng.random() < prob:
            agent_a = agents[i]
            agent_b = agents[j]
            id_a = agent_ids[i]
            id_b = agent_ids[j]

            edge_type = _infer_edge_type(agent_a, agent_b, is_rewired=False)
            influence_weights = _compute_influence_weights(agent_a, agent_b, sim)

            edge = Edge(
                source=id_a,
                target=id_b,
                weight=sim,
                edge_type=edge_type,
                influence_weight=influence_weights,
            )
            edges.append(edge)
            edge_set.add((id_a, id_b))
            edge_set.add((id_b, id_a))

        if on_progress and idx % 1000 == 0:
            on_progress("Sampling edges", idx, len(similarities))

    if on_progress:
        on_progress("Sampling edges", len(similarities), len(similarities))

    # Step 5: Watts-Strogatz rewiring
    if on_progress:
        on_progress("Rewiring edges", 0, len(edges))

    n_rewire = int(len(edges) * config.rewire_prob)
    rewired_count = 0

    for _ in range(n_rewire):
        if not edges:
            break

        # Pick random edge to rewire
        edge_idx = rng.randint(0, len(edges) - 1)
        old_edge = edges[edge_idx]

        # Pick random new target (different from source and existing neighbors)
        source_idx = agent_ids.index(old_edge.source)
        attempts = 0
        while attempts < 10:
            new_target_idx = rng.randint(0, n - 1)
            new_target_id = agent_ids[new_target_idx]

            # Check not self-loop and not existing edge
            if new_target_idx != source_idx:
                if (old_edge.source, new_target_id) not in edge_set:
                    # Remove old edge from set
                    edge_set.discard((old_edge.source, old_edge.target))
                    edge_set.discard((old_edge.target, old_edge.source))

                    # Create new edge
                    agent_a = agents[source_idx]
                    agent_b = agents[new_target_idx]

                    # Compute new similarity for weight
                    new_sim = compute_similarity(agent_a, agent_b, config.attribute_weights)
                    influence_weights = _compute_influence_weights(agent_a, agent_b, new_sim)

                    new_edge = Edge(
                        source=old_edge.source,
                        target=new_target_id,
                        weight=new_sim,
                        edge_type="weak_tie",  # Rewired edges are weak ties
                        influence_weight=influence_weights,
                    )
                    edges[edge_idx] = new_edge

                    edge_set.add((new_edge.source, new_edge.target))
                    edge_set.add((new_edge.target, new_edge.source))

                    rewired_count += 1
                    break

            attempts += 1

    if on_progress:
        on_progress("Rewiring edges", n_rewire, n_rewire)

    # Build metadata
    meta = {
        "agent_count": n,
        "edge_count": len(edges),
        "avg_degree": 2 * len(edges) / n if n > 0 else 0.0,
        "rewired_count": rewired_count,
        "algorithm": "hybrid",
        "seed": seed,
        "config": {
            "avg_degree_target": config.avg_degree,
            "rewire_prob": config.rewire_prob,
            "similarity_threshold": config.similarity_threshold,
        },
        "generated_at": datetime.now().isoformat(),
    }

    return NetworkResult(meta=meta, edges=edges)


def generate_network_with_metrics(
    agents: list[dict[str, Any]],
    config: NetworkConfig | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> NetworkResult:
    """Generate network and compute all metrics.

    Same as generate_network but also computes:
    - Network-level validation metrics
    - Per-agent node metrics (PageRank, betweenness, etc.)

    Args:
        agents: List of agent dictionaries
        config: Network configuration
        on_progress: Progress callback

    Returns:
        NetworkResult with edges and metrics
    """
    result = generate_network(agents, config, on_progress)

    # Get agent IDs
    agent_ids = [a.get("_id", f"agent_{i}") for i, a in enumerate(agents)]

    # Compute metrics
    if on_progress:
        on_progress("Computing metrics", 0, 2)

    edge_dicts = [e.to_dict() for e in result.edges]
    result.network_metrics = compute_network_metrics(edge_dicts, agent_ids)

    if on_progress:
        on_progress("Computing metrics", 1, 2)

    result.node_metrics = compute_node_metrics(edge_dicts, agent_ids)

    if on_progress:
        on_progress("Computing metrics", 2, 2)

    # Update meta with computed metrics
    if result.network_metrics:
        result.meta["clustering_coefficient"] = round(result.network_metrics.clustering_coefficient, 4)
        result.meta["avg_path_length"] = round(result.network_metrics.avg_path_length, 2) if result.network_metrics.avg_path_length else None
        result.meta["modularity"] = round(result.network_metrics.modularity, 4)

    return result


def load_agents_json(path: Path | str) -> list[dict[str, Any]]:
    """Load agents from JSON file.

    Expected format:
    {
        "meta": {...},
        "agents": [...]
    }

    Args:
        path: Path to agents JSON file

    Returns:
        List of agent dictionaries
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "agents" in data:
        return data["agents"]
    else:
        raise ValueError(f"Unexpected JSON format in {path}")
