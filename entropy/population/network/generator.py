"""Network generation algorithms for creating social graphs between agents.

Implements the hybrid approach: attribute similarity + Watts-Strogatz rewiring.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.models import Edge, NetworkResult
from ...utils.callbacks import NetworkProgressCallback
from .config import NetworkConfig, InfluenceFactorConfig
from .similarity import (
    compute_similarity,
    compute_degree_factor,
    compute_edge_probability,
)
from .metrics import (
    compute_network_metrics,
    compute_node_metrics,
)


def _eval_edge_condition(
    condition: str,
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
) -> bool:
    """Evaluate an edge type rule condition against two agents.

    Builds a context with a_{attr} and b_{attr} variables for each
    agent attribute, then evaluates the condition string.

    Only supports comparisons (==, !=, <, >, <=, >=) and boolean operators
    (and, or, not). No function calls or assignments.
    """
    context: dict[str, Any] = {}
    for key, val in agent_a.items():
        context[f"a_{key}"] = val
    for key, val in agent_b.items():
        context[f"b_{key}"] = val

    try:
        return bool(eval(condition, {"__builtins__": {}}, context))  # noqa: S307
    except Exception:
        return False


def _infer_edge_type(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    config: NetworkConfig | None = None,
    is_rewired: bool = False,
) -> str:
    """Infer edge type based on agent attributes and config rules.

    If config has edge_type_rules, evaluates them in priority order
    (highest first). Otherwise falls back to hardcoded German surgeon logic
    for backward compatibility.

    Args:
        agent_a: First agent's attributes
        agent_b: Second agent's attributes
        config: Network config with edge_type_rules
        is_rewired: Whether this edge was created by rewiring

    Returns:
        Edge type string
    """
    if config is None:
        config = NetworkConfig()

    if is_rewired:
        return config.default_edge_type

    # Data-driven: evaluate rules in priority order
    if config.edge_type_rules:
        sorted_rules = sorted(
            config.edge_type_rules, key=lambda r: r.priority, reverse=True
        )
        for rule in sorted_rules:
            if _eval_edge_condition(rule.condition, agent_a, agent_b):
                return rule.name
        return config.default_edge_type

    # No rules configured — return default edge type
    return config.default_edge_type


def _compute_influence_factor(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    factor: InfluenceFactorConfig,
) -> float:
    """Compute a single influence factor's contribution (A influencing B).

    Returns a multiplier >= 0.1 representing how much this factor
    amplifies or dampens A's influence on B.
    """
    val_a = agent_a.get(factor.attribute)
    val_b = agent_b.get(factor.attribute)

    if val_a is None or val_b is None:
        return 1.0

    if factor.type == "ordinal":
        if factor.levels is None:
            return 1.0
        level_a = factor.levels.get(str(val_a), 1)
        level_b = factor.levels.get(str(val_b), 1)
        ratio = level_a / level_b if level_b > 0 else 1.0
        # Scale by weight: closer to 1.0 means less impact
        return 1.0 + factor.weight * (ratio - 1.0)

    elif factor.type == "boolean":
        bool_a = 1 if val_a else 0
        bool_b = 1 if val_b else 0
        return 1.0 + factor.weight * (bool_a - bool_b)

    elif factor.type == "numeric":
        try:
            num_a = float(val_a)
            num_b = float(val_b)
            if num_b > 0:
                ratio = num_a / num_b
            else:
                ratio = 1.0
            # Dampen: sqrt to reduce extreme ratios
            import math

            dampened = (
                math.sqrt(ratio)
                if ratio >= 1.0
                else 1.0 / math.sqrt(1.0 / ratio)
                if ratio > 0
                else 1.0
            )
            return 1.0 + factor.weight * (dampened - 1.0)
        except (TypeError, ValueError):
            return 1.0

    return 1.0


def _compute_influence_weights(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    edge_weight: float,
    config: NetworkConfig | None = None,
) -> dict[str, float]:
    """Compute asymmetric influence weights for an edge.

    If config has influence_factors, uses them to compute data-driven
    asymmetric influence. Otherwise returns symmetric influence based
    on edge_weight alone.

    Args:
        agent_a: Source agent attributes
        agent_b: Target agent attributes
        edge_weight: Base edge weight (similarity)
        config: Network config with influence_factors

    Returns:
        Dict with "source_to_target" and "target_to_source" influence weights
    """
    if config is None:
        config = NetworkConfig()

    if not config.influence_factors:
        return {
            "source_to_target": edge_weight,
            "target_to_source": edge_weight,
        }

    # Compute A -> B influence
    influence_a_to_b = edge_weight
    for factor in config.influence_factors:
        influence_a_to_b *= max(
            0.1, _compute_influence_factor(agent_a, agent_b, factor)
        )

    # Compute B -> A influence
    influence_b_to_a = edge_weight
    for factor in config.influence_factors:
        influence_b_to_a *= max(
            0.1, _compute_influence_factor(agent_b, agent_a, factor)
        )

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
    on_progress: NetworkProgressCallback | None = None,
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
            sim = compute_similarity(
                agents[i], agents[j], config.attribute_weights, config.ordinal_levels
            )
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

            edge_type = _infer_edge_type(agent_a, agent_b, config, is_rewired=False)
            influence_weights = _compute_influence_weights(
                agent_a, agent_b, sim, config
            )

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
                    new_sim = compute_similarity(
                        agent_a,
                        agent_b,
                        config.attribute_weights,
                        config.ordinal_levels,
                    )
                    influence_weights = _compute_influence_weights(
                        agent_a, agent_b, new_sim, config
                    )

                    new_edge = Edge(
                        source=old_edge.source,
                        target=new_target_id,
                        weight=new_sim,
                        edge_type=config.default_edge_type,  # Rewired edges get default type
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
    on_progress: NetworkProgressCallback | None = None,
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
        result.meta["clustering_coefficient"] = round(
            result.network_metrics.clustering_coefficient, 4
        )
        result.meta["avg_path_length"] = (
            round(result.network_metrics.avg_path_length, 2)
            if result.network_metrics.avg_path_length
            else None
        )
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
