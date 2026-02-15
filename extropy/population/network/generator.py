"""Network generation algorithms for creating social graphs between agents.

Implements adaptive calibration to hit target metrics (avg_degree, clustering, modularity).
"""

import json
import logging
import hashlib
import multiprocessing as mp
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.models import Edge, NetworkResult
from ...storage import open_study_db
from ...utils.callbacks import NetworkProgressCallback
from ...utils.eval_safe import ConditionError, eval_condition
from .config import NetworkConfig, InfluenceFactorConfig
from .similarity import (
    compute_similarity,
    compute_degree_factor,
)

logger = logging.getLogger(__name__)

_SIM_WORKER_AGENTS: list[dict[str, Any]] | None = None
_SIM_WORKER_ATTRIBUTE_WEIGHTS = None
_SIM_WORKER_ORDINAL_LEVELS: dict[str, dict[str, int]] | None = None
_SIM_WORKER_THRESHOLD: float = 0.05
_SIM_WORKER_CANDIDATE_MAP: list[list[int]] | None = None


def _is_db_checkpoint(path: Path | None) -> bool:
    return path is not None and path.suffix.lower() == ".db"


def _choose_blocking_attributes(config: NetworkConfig) -> list[str]:
    """Choose blocking attributes for candidate pruning."""
    if config.blocking_attributes:
        return list(config.blocking_attributes)

    weighted = sorted(
        config.attribute_weights.items(),
        key=lambda x: x[1].weight,
        reverse=True,
    )
    preferred = [
        attr for attr, cfg in weighted if cfg.match_type in {"exact", "within_n"}
    ]

    if preferred:
        return preferred[:3]

    return [attr for attr, _ in weighted[:2]]


def _build_blocked_candidate_map(
    agents: list[dict[str, Any]],
    config: NetworkConfig,
    seed: int,
) -> tuple[list[list[int]] | None, list[str]]:
    """Build per-agent candidate lists for blocked similarity mode."""
    attrs = _choose_blocking_attributes(config)
    n = len(agents)

    if not attrs or n <= 1:
        return None, attrs

    blocks: dict[str, dict[Any, list[int]]] = {attr: {} for attr in attrs}

    for idx, agent in enumerate(agents):
        for attr in attrs:
            val = agent.get(attr)
            if val is None:
                continue
            blocks[attr].setdefault(val, []).append(idx)

    target_pool = max(config.min_candidate_pool, int(config.avg_degree * config.candidate_pool_multiplier))
    target_pool = max(1, min(n - 1, target_pool))

    candidate_map: list[list[int]] = [[] for _ in range(n)]

    for i, agent in enumerate(agents):
        scores: dict[int, int] = {}

        for attr in attrs:
            val = agent.get(attr)
            if val is None:
                continue
            for j in blocks[attr].get(val, []):
                if j == i:
                    continue
                scores[j] = scores.get(j, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        chosen = [j for j, _ in ranked[:target_pool]]

        if len(chosen) < target_pool:
            rng = random.Random(seed + (i + 1) * 7919)
            seen = set(chosen)
            seen.add(i)
            while len(chosen) < target_pool and len(seen) < n:
                j = rng.randrange(n)
                if j in seen:
                    continue
                seen.add(j)
                chosen.append(j)

        candidate_map[i] = sorted(chosen)

    return candidate_map, attrs


def _similarity_checkpoint_signature(
    n: int,
    seed: int,
    config: NetworkConfig,
    blocking_attrs: list[str],
) -> dict[str, Any]:
    """Build a minimal signature to validate checkpoint compatibility."""
    return {
        "n": n,
        "seed": seed,
        "candidate_mode": config.candidate_mode,
        "threshold": config.similarity_store_threshold,
        "candidate_pool_multiplier": config.candidate_pool_multiplier,
        "min_candidate_pool": config.min_candidate_pool,
        "blocking_attributes": blocking_attrs,
    }


def _similarity_checkpoint_job_id(signature: dict[str, Any]) -> str:
    raw = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _save_similarity_checkpoint(
    path: Path,
    similarities: dict[tuple[int, int], float],
    completed_rows: int,
    signature: dict[str, Any],
    completed_chunks: list[tuple[int, int]] | None = None,
) -> None:
    """Persist sparse similarities so generation can resume after interruption."""
    payload = {
        "version": 1,
        "completed_rows": completed_rows,
        "completed_chunks": completed_chunks or [],
        "signature": signature,
        "similarities": similarities,
        "saved_at": datetime.now().isoformat(),
    }
    if _is_db_checkpoint(path):
        job_id = _similarity_checkpoint_job_id(signature)
        with open_study_db(path) as db:
            db.init_network_similarity_job(
                network_run_id=f"checkpoint:{job_id}",
                signature=signature,
                job_id=job_id,
            )
            db.save_similarity_snapshot(job_id=job_id, payload=pickle.dumps(payload))
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def _load_similarity_checkpoint(
    path: Path,
    expected_signature: dict[str, Any],
) -> tuple[dict[tuple[int, int], float], int, set[int]]:
    """Load checkpoint and validate compatibility with current run settings."""
    if _is_db_checkpoint(path):
        job_id = _similarity_checkpoint_job_id(expected_signature)
        with open_study_db(path) as db:
            signature = db.get_network_similarity_job_signature(job_id)
            if signature is not None:
                if signature != expected_signature:
                    raise ValueError(
                        "Checkpoint settings do not match current run. "
                        "Delete checkpoint or run with matching config."
                    )

                done_chunks = db.list_completed_similarity_chunks(job_id)
                done_starts = {start for start, _ in done_chunks}
                similarities = db.load_similarity_pairs(job_id)

                # Resume serial fallback only from contiguous completed prefix.
                contiguous_rows = 0
                for start, end in done_chunks:
                    if start != contiguous_rows:
                        break
                    contiguous_rows = end

                return similarities, max(0, contiguous_rows), done_starts

            payload_bytes = db.get_similarity_snapshot(job_id)
            if payload_bytes is None:
                raise ValueError(f"Checkpoint not found in study DB: job_id={job_id}")
            payload = pickle.loads(payload_bytes)
    else:
        with open(path, "rb") as f:
            payload = pickle.load(f)

    signature = payload.get("signature", {})
    if signature != expected_signature:
        raise ValueError(
            "Checkpoint settings do not match current run. "
            "Delete checkpoint or run with matching config."
        )

    similarities = payload.get("similarities", {})
    completed_rows = int(payload.get("completed_rows", 0))
    completed_chunk_starts: set[int] = set()
    for item in payload.get("completed_chunks", []):
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
        ):
            completed_chunk_starts.add(item[0])

    if not isinstance(similarities, dict):
        raise ValueError("Invalid checkpoint similarities payload")

    return similarities, max(0, completed_rows), completed_chunk_starts


def _init_similarity_worker(
    agents: list[dict[str, Any]],
    attribute_weights,
    ordinal_levels: dict[str, dict[str, int]] | None,
    threshold: float,
    candidate_map: list[list[int]] | None,
) -> None:
    """Initialize process-local globals for similarity workers."""
    global _SIM_WORKER_AGENTS
    global _SIM_WORKER_ATTRIBUTE_WEIGHTS
    global _SIM_WORKER_ORDINAL_LEVELS
    global _SIM_WORKER_THRESHOLD
    global _SIM_WORKER_CANDIDATE_MAP

    _SIM_WORKER_AGENTS = agents
    _SIM_WORKER_ATTRIBUTE_WEIGHTS = attribute_weights
    _SIM_WORKER_ORDINAL_LEVELS = ordinal_levels
    _SIM_WORKER_THRESHOLD = threshold
    _SIM_WORKER_CANDIDATE_MAP = candidate_map


def _compute_similarity_chunk(task: tuple[int, int]) -> tuple[int, list[tuple[int, int, float]]]:
    """Compute similarities for a chunk of row indices in a worker process."""
    start, end = task
    if _SIM_WORKER_AGENTS is None:
        raise RuntimeError("Similarity worker not initialized")

    n = len(_SIM_WORKER_AGENTS)
    rows: list[tuple[int, int, float]] = []

    for i in range(start, min(end, n)):
        if _SIM_WORKER_CANDIDATE_MAP is None:
            candidates = range(i + 1, n)
        else:
            candidates = _SIM_WORKER_CANDIDATE_MAP[i]

        for j in candidates:
            if j <= i:
                continue
            sim = compute_similarity(
                _SIM_WORKER_AGENTS[i],
                _SIM_WORKER_AGENTS[j],
                _SIM_WORKER_ATTRIBUTE_WEIGHTS,
                _SIM_WORKER_ORDINAL_LEVELS,
            )
            if sim >= _SIM_WORKER_THRESHOLD:
                rows.append((i, j, sim))

    return end, rows


def _compute_similarities_parallel(
    agents: list[dict[str, Any]],
    config: NetworkConfig,
    candidate_map: list[list[int]] | None,
    on_progress: NetworkProgressCallback | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_signature: dict[str, Any] | None = None,
    initial_similarities: dict[tuple[int, int], float] | None = None,
    completed_rows: int = 0,
    completed_chunk_starts: set[int] | None = None,
    checkpoint_job_id: str | None = None,
) -> dict[tuple[int, int], float]:
    """Compute sparse similarities with process parallelism."""
    n = len(agents)
    similarities: dict[tuple[int, int], float] = dict(initial_similarities or {})

    chunk_size = max(8, config.similarity_chunk_size)
    tasks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    task_ends = {start: end for start, end in tasks}
    completed_starts: set[int] = set(completed_chunk_starts or set())
    for start, _ in tasks:
        if start < completed_rows:
            completed_starts.add(start)
    pending_tasks = [(s, e) for s, e in tasks if s not in completed_starts]
    workers = max(1, config.similarity_workers)

    completed_row_count = sum((e - s) for s, e in tasks if s in completed_starts)
    if on_progress and completed_row_count > 0:
        on_progress("Computing similarities", min(completed_row_count, n), n)

    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_similarity_worker,
            initargs=(
                agents,
                config.attribute_weights,
                config.ordinal_levels,
                config.similarity_store_threshold,
                candidate_map,
            ),
        ) as ex:
            futures = {
                ex.submit(_compute_similarity_chunk, task): task for task in pending_tasks
            }
            pending_results: dict[int, list[tuple[int, int, float]]] = {}
            sorted_starts = [start for start, _ in tasks]
            next_commit_idx = 0

            for fut in as_completed(futures):
                task_start, _task_end = futures[fut]
                _row_end, local_rows = fut.result()
                pending_results[task_start] = local_rows

                # Deterministic merge: commit completed chunks in chunk_start order.
                while next_commit_idx < len(sorted_starts):
                    current_start = sorted_starts[next_commit_idx]
                    current_end = task_ends[current_start]
                    if current_start in completed_starts:
                        next_commit_idx += 1
                        continue
                    if current_start not in pending_results:
                        break

                    chunk_rows = pending_results.pop(current_start)
                    for i, j, sim in chunk_rows:
                        similarities[(i, j)] = sim
                    completed_starts.add(current_start)
                    completed_row_count += current_end - current_start
                    completed_rows = max(completed_rows, current_end)

                    if (
                        checkpoint_path is not None
                        and checkpoint_signature is not None
                    ):
                        if _is_db_checkpoint(checkpoint_path) and checkpoint_job_id:
                            with open_study_db(checkpoint_path) as db:
                                db.save_similarity_chunk_rows(
                                    job_id=checkpoint_job_id,
                                    chunk_start=current_start,
                                    chunk_end=current_end,
                                    rows=chunk_rows,
                                )
                        else:
                            completed_chunks = [
                                (s, e) for s, e in tasks if s in completed_starts
                            ]
                            _save_similarity_checkpoint(
                                path=checkpoint_path,
                                similarities=similarities,
                                completed_rows=min(completed_row_count, n),
                                signature=checkpoint_signature,
                                completed_chunks=completed_chunks,
                            )

                    if on_progress:
                        on_progress(
                            "Computing similarities", min(completed_row_count, n), n
                        )
                    next_commit_idx += 1

    except Exception as e:
        logger.warning(
            "Parallel similarity failed (%s). Falling back to serial mode.", e
        )
        return _compute_similarities_serial(
            agents=agents,
            config=config,
            candidate_map=candidate_map,
            on_progress=on_progress,
            checkpoint_path=checkpoint_path,
            initial_similarities=similarities,
            start_row=completed_rows,
            checkpoint_signature=checkpoint_signature,
            completed_chunk_starts=completed_starts,
            checkpoint_job_id=checkpoint_job_id,
        )

    return similarities


def _compute_similarities_serial(
    agents: list[dict[str, Any]],
    config: NetworkConfig,
    candidate_map: list[list[int]] | None = None,
    on_progress: NetworkProgressCallback | None = None,
    checkpoint_path: Path | None = None,
    initial_similarities: dict[tuple[int, int], float] | None = None,
    start_row: int = 0,
    checkpoint_signature: dict[str, Any] | None = None,
    completed_chunk_starts: set[int] | None = None,
    checkpoint_job_id: str | None = None,
) -> dict[tuple[int, int], float]:
    """Compute sparse similarities serially, with optional checkpointing."""
    n = len(agents)
    threshold = config.similarity_store_threshold
    similarities = dict(initial_similarities or {})
    checkpoint_every = max(1, config.checkpoint_every_rows)
    chunk_size = max(8, config.similarity_chunk_size)
    tasks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    completed_starts: set[int] = set(completed_chunk_starts or set())
    for start, _ in tasks:
        if start < start_row:
            completed_starts.add(start)
    completed_row_count = sum((e - s) for s, e in tasks if s in completed_starts)

    for chunk_idx, (start, end) in enumerate(tasks):
        if start in completed_starts:
            continue

        local_rows: list[tuple[int, int, float]] = []
        for i in range(start, end):
            if candidate_map is None:
                candidates = range(i + 1, n)
            else:
                candidates = candidate_map[i]

            for j in candidates:
                if j <= i:
                    continue
                sim = compute_similarity(
                    agents[i], agents[j], config.attribute_weights, config.ordinal_levels
                )
                if sim >= threshold:
                    similarities[(i, j)] = sim
                    local_rows.append((i, j, sim))

        completed_starts.add(start)
        completed_row_count += end - start

        if checkpoint_path is not None and checkpoint_signature is not None:
            if _is_db_checkpoint(checkpoint_path) and checkpoint_job_id:
                with open_study_db(checkpoint_path) as db:
                    db.save_similarity_chunk_rows(
                        job_id=checkpoint_job_id,
                        chunk_start=start,
                        chunk_end=end,
                        rows=local_rows,
                    )
            elif (
                completed_row_count % checkpoint_every == 0
                or chunk_idx == len(tasks) - 1
            ):
                completed_chunks = [(s, e) for s, e in tasks if s in completed_starts]
                _save_similarity_checkpoint(
                    path=checkpoint_path,
                    similarities=similarities,
                    completed_rows=min(completed_row_count, n),
                    signature=checkpoint_signature,
                    completed_chunks=completed_chunks,
                )

        if on_progress:
            on_progress("Computing similarities", min(completed_row_count, n), n)

    return similarities


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
        return eval_condition(condition, context, raise_on_error=True)
    except ConditionError:
        return False


def _infer_edge_type(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    config: NetworkConfig | None = None,
    is_rewired: bool = False,
) -> str:
    """Infer edge type based on agent attributes and config rules.

    If config has edge_type_rules, evaluates them in priority order
    (highest first). Otherwise falls back to default edge type.
    """
    if config is None:
        config = NetworkConfig()

    if is_rewired:
        return config.default_edge_type

    if config.edge_type_rules:
        sorted_rules = sorted(
            config.edge_type_rules, key=lambda r: r.priority, reverse=True
        )
        for rule in sorted_rules:
            if _eval_edge_condition(rule.condition, agent_a, agent_b):
                return rule.name
        return config.default_edge_type

    return config.default_edge_type


def _compute_influence_factor(
    agent_a: dict[str, Any],
    agent_b: dict[str, Any],
    factor: InfluenceFactorConfig,
) -> float:
    """Compute a single influence factor's contribution (A influencing B)."""
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
    """Compute asymmetric influence weights for an edge."""
    if config is None:
        config = NetworkConfig()

    if not config.influence_factors:
        return {
            "source_to_target": edge_weight,
            "target_to_source": edge_weight,
        }

    influence_a_to_b = edge_weight
    for factor in config.influence_factors:
        influence_a_to_b *= max(
            0.1, _compute_influence_factor(agent_a, agent_b, factor)
        )

    influence_b_to_a = edge_weight
    for factor in config.influence_factors:
        influence_b_to_a *= max(
            0.1, _compute_influence_factor(agent_b, agent_a, factor)
        )

    return {
        "source_to_target": influence_a_to_b,
        "target_to_source": influence_b_to_a,
    }


def _assign_communities(
    agents: list[dict],
    similarities: dict[tuple[int, int], float],
    n_communities: int,
    rng: random.Random,
) -> list[int]:
    """Assign agents to communities using fast k-medoids-style clustering."""
    n = len(agents)
    k = min(n_communities, n)

    if k <= 1:
        return [0] * n

    def get_sim(i: int, j: int) -> float:
        if i == j:
            return 1.0
        key = (min(i, j), max(i, j))
        return similarities.get(key, 0.0)

    # Select k diverse initial centers using k-means++ style
    centers = [rng.randint(0, n - 1)]

    for _ in range(k - 1):
        best_agent = -1
        best_min_sim = float("inf")

        for i in range(n):
            if i in centers:
                continue
            max_sim_to_centers = max(get_sim(i, c) for c in centers)
            if max_sim_to_centers < best_min_sim:
                best_min_sim = max_sim_to_centers
                best_agent = i

        if best_agent >= 0:
            centers.append(best_agent)

    # Iterative assignment and center update
    assignments = [0] * n
    max_iterations = 10

    for _ in range(max_iterations):
        changed = False
        for i in range(n):
            best_center = 0
            best_sim = get_sim(i, centers[0])
            for c_idx, c in enumerate(centers[1:], 1):
                sim = get_sim(i, c)
                if sim > best_sim:
                    best_sim = sim
                    best_center = c_idx
            if assignments[i] != best_center:
                changed = True
                assignments[i] = best_center

        if not changed:
            break

        for c_idx in range(k):
            cluster_members = [i for i in range(n) if assignments[i] == c_idx]
            if not cluster_members:
                continue

            best_medoid = cluster_members[0]
            best_avg_sim = 0.0

            for candidate in cluster_members:
                avg_sim = sum(get_sim(candidate, m) for m in cluster_members) / len(
                    cluster_members
                )
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_medoid = candidate

            centers[c_idx] = best_medoid

    return assignments


def _compute_local_clustering(adjacency: dict[int, set[int]], node: int) -> float:
    """Compute local clustering coefficient for a single node."""
    neighbors = adjacency.get(node, set())
    k = len(neighbors)
    if k < 2:
        return 0.0

    edges_between = 0
    neighbor_list = list(neighbors)
    for i, n1 in enumerate(neighbor_list):
        for n2 in neighbor_list[i + 1 :]:
            if n2 in adjacency.get(n1, set()):
                edges_between += 1

    max_edges = k * (k - 1) / 2
    return edges_between / max_edges if max_edges > 0 else 0.0


def _compute_avg_clustering(adjacency: dict[int, set[int]], n: int) -> float:
    """Compute average clustering coefficient over all nodes."""
    if n == 0:
        return 0.0
    total = sum(_compute_local_clustering(adjacency, i) for i in range(n))
    return total / n


def _sample_edges(
    agents: list[dict],
    agent_ids: list[str],
    similarities: dict[tuple[int, int], float],
    communities: list[int],
    degree_factors: list[float],
    config: NetworkConfig,
    intra_scale: float,
    inter_scale: float,
    rng: random.Random,
) -> tuple[list[Edge], set[tuple[str, str]]]:
    """Sample edges with local attachment for high clustering.

    Uses a two-phase approach:
    1. Sample initial edges based on similarity
    2. Preferentially connect to friends-of-friends (local attachment)

    This creates networks with naturally high clustering.
    """
    n = len(agents)
    edges: list[Edge] = []
    edge_set: set[tuple[str, str]] = set()
    added_pairs: set[tuple[int, int]] = set()

    # Build index-based adjacency (updated as edges are added)
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}

    def add_edge(i: int, j: int, sim: float) -> None:
        """Add an edge between agents i and j."""
        pair = (min(i, j), max(i, j))
        if pair in added_pairs:
            return
        added_pairs.add(pair)

        agent_a = agents[i]
        agent_b = agents[j]
        id_a = agent_ids[i]
        id_b = agent_ids[j]

        edge_type = _infer_edge_type(agent_a, agent_b, config, is_rewired=False)
        influence_weights = _compute_influence_weights(agent_a, agent_b, sim, config)

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
        adjacency[i].add(j)
        adjacency[j].add(i)

    # Separate intra and inter community pairs
    intra_pairs = []
    inter_pairs = []
    for (i, j), sim in similarities.items():
        if communities[i] == communities[j]:
            intra_pairs.append((i, j, sim))
        else:
            inter_pairs.append((i, j, sim))

    n_intra = len(intra_pairs)
    n_inter = len(inter_pairs)

    # Calculate target edges (before triadic closure adds more)
    # Aim for ~45% of final target since triadic closure will add ~55% more
    target_edges = int(n * config.avg_degree / 2 * 0.45)

    # Base probabilities calibrated for target edges
    avg_sim_intra = sum(s for _, _, s in intra_pairs) / n_intra if n_intra > 0 else 0.5
    avg_sim_inter = sum(s for _, _, s in inter_pairs) / n_inter if n_inter > 0 else 0.3

    total_weighted_pairs = (
        n_intra * avg_sim_intra * intra_scale + n_inter * avg_sim_inter * inter_scale
    )
    if total_weighted_pairs > 0:
        base_prob = target_edges / total_weighted_pairs
    else:
        base_prob = 0.1
    base_prob = min(0.8, max(0.01, base_prob))

    # Phase 1: Sample initial edges based on similarity
    # Sort by similarity (highest first) and sample top candidates
    all_pairs = [
        (i, j, sim, communities[i] == communities[j])
        for (i, j), sim in similarities.items()
    ]
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, sim, same_community in all_pairs:
        if len(edges) >= target_edges:
            break

        scale = intra_scale if same_community else inter_scale
        prob = base_prob * sim * scale * degree_factors[i] * degree_factors[j]

        # Boost probability for friends-of-friends (local attachment)
        common_neighbors = len(adjacency[i] & adjacency[j])
        if common_neighbors > 0:
            # Significantly boost prob for pairs with common neighbors
            prob = min(0.95, prob * (1 + common_neighbors * 0.5))

        prob = min(0.95, prob)

        if rng.random() < prob:
            add_edge(i, j, sim)

    # Phase 2: Local attachment - preferentially add edges to friends-of-friends
    # This is key for high clustering
    local_attachment_budget = int(
        target_edges * 0.5
    )  # 50% of edges via local attachment
    local_edges_added = 0

    # Build list of potential local attachments (pairs with common neighbors but no edge)
    fof_candidates = []
    for i in range(n):
        neighbors = adjacency[i]
        for neighbor in neighbors:
            for fof in adjacency[neighbor]:
                if fof != i and fof not in neighbors:
                    pair = (min(i, fof), max(i, fof))
                    if pair not in added_pairs:
                        sim = similarities.get(pair, 0.0)
                        if sim > 0:
                            # Score by similarity and number of common neighbors
                            common = len(adjacency[i] & adjacency[fof])
                            score = sim * (1 + common * 0.3)
                            fof_candidates.append((score, i, fof, sim))

    # Deduplicate and sort
    seen_fof = set()
    unique_fof = []
    for score, i, j, sim in fof_candidates:
        pair = (min(i, j), max(i, j))
        if pair not in seen_fof:
            seen_fof.add(pair)
            unique_fof.append((score, i, j, sim))
    unique_fof.sort(reverse=True, key=lambda x: x[0])

    # Add top local attachment edges
    for score, i, j, sim in unique_fof[: local_attachment_budget * 2]:
        if local_edges_added >= local_attachment_budget:
            break
        if (min(i, j), max(i, j)) in added_pairs:
            continue

        # High probability for local attachment (friends-of-friends)
        if rng.random() < 0.7:
            add_edge(i, j, sim)
            local_edges_added += 1

    return edges, edge_set


def _triadic_closure(
    agents: list[dict],
    agent_ids: list[str],
    edges: list[Edge],
    edge_set: set[tuple[str, str]],
    config: NetworkConfig,
    rng: random.Random,
    similarities: dict[tuple[int, int], float] | None = None,
    communities: list[int] | None = None,
    target_clustering: float = 0.35,
    max_edge_increase: float = 1.5,
) -> tuple[list[Edge], set[tuple[str, str]]]:
    """Apply triadic closure to increase clustering coefficient.

    Runs until target_clustering is reached or edge budget exhausted.
    """
    n = len(agents)
    if n < 3 or config.triadic_closure_prob <= 0:
        return edges, edge_set

    # Build adjacency list
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for edge in edges:
        i = id_to_idx[edge.source]
        j = id_to_idx[edge.target]
        adjacency[i].add(j)
        adjacency[j].add(i)

    current_clustering = _compute_avg_clustering(adjacency, n)
    if current_clustering >= target_clustering:
        return edges, edge_set

    # Edge budget: don't exceed max_edge_increase factor
    max_edges = int(len(edges) * max_edge_increase)

    max_iterations = 25  # More iterations for better convergence
    max_closures_per_iter = int(n * 0.25)  # More closures per iteration

    for iteration in range(max_iterations):
        # Find open triads
        open_triads: list[tuple[int, int, int]] = []

        for b in range(n):
            neighbors_of_b = list(adjacency[b])
            for i, a in enumerate(neighbors_of_b):
                for c in neighbors_of_b[i + 1 :]:
                    if c not in adjacency[a]:
                        open_triads.append((a, c, b))

        if not open_triads:
            break

        if len(edges) >= max_edges:
            break

        # Score triads by similarity and community membership
        triad_with_score = []
        for a, c, b in open_triads:
            pair = (min(a, c), max(a, c))
            sim = similarities.get(pair) if similarities is not None else None
            if sim is None:
                sim = compute_similarity(
                    agents[a],
                    agents[c],
                    config.attribute_weights,
                    config.ordinal_levels,
                )
                if similarities is not None:
                    similarities[pair] = sim
            same_community = (
                communities is not None and communities[a] == communities[c]
            )
            score = (5.0 if same_community else 0.0) + sim
            triad_with_score.append((score, sim, a, c, b))
        triad_with_score.sort(reverse=True, key=lambda x: x[0])

        top_candidates = triad_with_score[: max_closures_per_iter * 2]
        rng.shuffle(top_candidates)

        closures = 0

        for score, sim, a, c, b in top_candidates:
            if closures >= max_closures_per_iter:
                break
            if len(edges) >= max_edges:
                break

            if c in adjacency[a]:
                continue

            agent_a = agents[a]
            agent_c = agents[c]

            # Higher base probability for more aggressive triadic closure
            effective_prob = min(0.95, config.triadic_closure_prob * (0.7 + sim * 0.5))

            if rng.random() < effective_prob:
                id_a = agent_ids[a]
                id_c = agent_ids[c]

                edge_type = _infer_edge_type(agent_a, agent_c, config, is_rewired=False)
                influence_weights = _compute_influence_weights(
                    agent_a, agent_c, sim, config
                )

                new_edge = Edge(
                    source=id_a,
                    target=id_c,
                    weight=sim,
                    edge_type=edge_type,
                    influence_weight=influence_weights,
                )
                edges.append(new_edge)
                edge_set.add((id_a, id_c))
                edge_set.add((id_c, id_a))
                adjacency[a].add(c)
                adjacency[c].add(a)
                closures += 1

        current_clustering = _compute_avg_clustering(adjacency, n)
        if current_clustering >= target_clustering:
            break

    return edges, edge_set


def _apply_rewiring(
    agents: list[dict],
    agent_ids: list[str],
    edges: list[Edge],
    edge_set: set[tuple[str, str]],
    config: NetworkConfig,
    rng: random.Random,
) -> tuple[list[Edge], set[tuple[str, str]], int]:
    """Apply Watts-Strogatz rewiring for small-world properties.

    Returns (edges, edge_set, rewired_count).
    """
    n = len(agents)
    n_rewire = int(len(edges) * config.rewire_prob)
    rewired_count = 0
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    for _ in range(n_rewire):
        if not edges:
            break

        edge_idx = rng.randint(0, len(edges) - 1)
        old_edge = edges[edge_idx]

        source_idx = id_to_idx.get(old_edge.source)
        if source_idx is None:
            continue
        attempts = 0
        while attempts < 10:
            new_target_idx = rng.randint(0, n - 1)
            new_target_id = agent_ids[new_target_idx]

            if new_target_idx != source_idx:
                if (old_edge.source, new_target_id) not in edge_set:
                    edge_set.discard((old_edge.source, old_edge.target))
                    edge_set.discard((old_edge.target, old_edge.source))

                    agent_a = agents[source_idx]
                    agent_b = agents[new_target_idx]

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
                        edge_type=config.default_edge_type,
                        influence_weight=influence_weights,
                    )
                    edges[edge_idx] = new_edge

                    edge_set.add((new_edge.source, new_edge.target))
                    edge_set.add((new_edge.target, new_edge.source))

                    rewired_count += 1
                    break

            attempts += 1

    return edges, edge_set, rewired_count


def _compute_modularity_fast(
    edges: list[Edge],
    agent_ids: list[str],
    communities: list[int],
) -> float:
    """Compute modularity using the provided community assignments.

    This is faster than running Louvain since we already have communities.
    Modularity Q = (1/2m) * sum_ij[(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]
    """
    n = len(agent_ids)
    m = len(edges)
    if m == 0:
        return 0.0

    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    # Compute degrees
    degrees = [0] * n
    for edge in edges:
        i = id_to_idx[edge.source]
        j = id_to_idx[edge.target]
        degrees[i] += 1
        degrees[j] += 1

    # Compute modularity
    q = 0.0
    two_m = 2 * m

    # For edges within same community
    for edge in edges:
        i = id_to_idx[edge.source]
        j = id_to_idx[edge.target]
        if communities[i] == communities[j]:
            # A_ij = 1, and we count each edge once
            q += 2 * (1 - degrees[i] * degrees[j] / two_m)

    # Normalize
    q /= two_m

    return q


def _generate_network_single_pass(
    agents: list[dict],
    agent_ids: list[str],
    similarities: dict[tuple[int, int], float],
    communities: list[int],
    degree_factors: list[float],
    config: NetworkConfig,
    intra_scale: float,
    inter_scale: float,
    rng: random.Random,
) -> tuple[list[Edge], float, float, float]:
    """Generate network with given parameters and return metrics.

    Returns (edges, avg_degree, clustering, modularity).
    """
    n = len(agents)

    # Sample edges
    edges, edge_set = _sample_edges(
        agents,
        agent_ids,
        similarities,
        communities,
        degree_factors,
        config,
        intra_scale,
        inter_scale,
        rng,
    )

    # Apply triadic closure for clustering (allow more edges to be added)
    edges, edge_set = _triadic_closure(
        agents,
        agent_ids,
        edges,
        edge_set,
        config,
        rng,
        similarities=similarities,
        communities=communities,
        target_clustering=config.target_clustering,
        max_edge_increase=2.5,  # Allow up to 2.5x edges for better clustering
    )

    # Compute metrics
    avg_degree = 2 * len(edges) / n if n > 0 else 0.0

    # Build adjacency for clustering
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for edge in edges:
        i = id_to_idx[edge.source]
        j = id_to_idx[edge.target]
        adjacency[i].add(j)
        adjacency[j].add(i)

    clustering = _compute_avg_clustering(adjacency, n)
    modularity = _compute_modularity_fast(edges, agent_ids, communities)

    return edges, avg_degree, clustering, modularity


def generate_network(
    agents: list[dict[str, Any]],
    config: NetworkConfig | None = None,
    on_progress: NetworkProgressCallback | None = None,
    checkpoint_path: Path | str | None = None,
    resume_from_checkpoint: bool = False,
) -> NetworkResult:
    """Generate a social network from sampled agents.

    Uses adaptive calibration to hit target metrics:
    1. Compute similarity matrix and detect communities (once)
    2. Iteratively adjust intra/inter scales until metrics are in range
    3. Apply triadic closure for clustering
    4. Apply Watts-Strogatz rewiring for small-world properties

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
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else None

    # Step 1: Compute degree factors
    degree_factors = [compute_degree_factor(a, config) for a in agents]

    # Step 2: Build similarity candidates (exact/blocked)
    candidate_map: list[list[int]] | None = None
    blocking_attrs: list[str] = []
    candidate_mode = config.candidate_mode

    if config.candidate_mode == "blocked":
        if on_progress:
            on_progress("Preparing candidate blocks", 0, n)
        candidate_map, blocking_attrs = _build_blocked_candidate_map(agents, config, seed)
        if on_progress:
            on_progress("Preparing candidate blocks", n, n)
        if candidate_map is None:
            logger.warning(
                "Blocked candidate mode could not be initialized. Falling back to exact mode."
            )
            candidate_mode = "exact"

    if on_progress:
        on_progress("Computing similarities", 0, n)

    checkpoint_signature = _similarity_checkpoint_signature(
        n=n,
        seed=seed,
        config=config,
        blocking_attrs=blocking_attrs,
    )
    checkpoint_job_id: str | None = None
    if _is_db_checkpoint(checkpoint_file):
        checkpoint_job_id = _similarity_checkpoint_job_id(checkpoint_signature)
        if not resume_from_checkpoint:
            with open_study_db(checkpoint_file) as db:
                db.init_network_similarity_job(
                    network_run_id=f"checkpoint:{checkpoint_job_id}",
                    signature=checkpoint_signature,
                    job_id=checkpoint_job_id,
                )
                db.mark_similarity_job_running(checkpoint_job_id)

    similarities: dict[tuple[int, int], float]
    start_row = 0
    completed_chunk_starts: set[int] = set()

    if resume_from_checkpoint and checkpoint_file is None:
        raise ValueError("--resume-checkpoint requires --checkpoint path")

    if resume_from_checkpoint:
        if checkpoint_file is None or not checkpoint_file.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_file}")
        similarities, start_row, completed_chunk_starts = _load_similarity_checkpoint(
            checkpoint_file, checkpoint_signature
        )
        if checkpoint_job_id and checkpoint_file is not None:
            with open_study_db(checkpoint_file) as db:
                db.mark_similarity_job_running(checkpoint_job_id)
        if on_progress:
            on_progress("Computing similarities", min(start_row, n), n)
    else:
        similarities = {}

    use_parallel_similarity = config.similarity_workers > 1

    if use_parallel_similarity:
        similarities = _compute_similarities_parallel(
            agents=agents,
            config=config,
            candidate_map=candidate_map if candidate_mode == "blocked" else None,
            on_progress=on_progress,
            checkpoint_path=checkpoint_file,
            checkpoint_signature=checkpoint_signature,
            initial_similarities=similarities,
            completed_rows=start_row,
            completed_chunk_starts=completed_chunk_starts,
            checkpoint_job_id=checkpoint_job_id,
        )
    else:
        similarities = _compute_similarities_serial(
            agents=agents,
            config=config,
            candidate_map=candidate_map if candidate_mode == "blocked" else None,
            on_progress=on_progress,
            checkpoint_path=checkpoint_file,
            initial_similarities=similarities,
            start_row=start_row,
            checkpoint_signature=checkpoint_signature,
            completed_chunk_starts=completed_chunk_starts,
            checkpoint_job_id=checkpoint_job_id,
        )

    if checkpoint_job_id and checkpoint_file is not None:
        with open_study_db(checkpoint_file) as db:
            db.mark_similarity_job_complete(checkpoint_job_id)

    if on_progress:
        on_progress("Computing similarities", n, n)

    # Step 3: Assign communities
    if on_progress:
        on_progress("Detecting communities", 0, 1)

    n_communities = config.community_count
    if n_communities is None:
        # Target ~40 agents per community for balanced structure
        n_communities = max(5, int(n / 40))

    communities = _assign_communities(agents, similarities, n_communities, rng)

    if on_progress:
        on_progress("Detecting communities", 1, 1)

    # Step 4: Adaptive calibration loop
    if on_progress:
        on_progress("Calibrating network", 0, config.max_calibration_iterations)

    # Initial scales: favor inter-community since triadic closure adds intra
    intra_scale = 1.0
    inter_scale = 1.2

    # Target ranges
    target_degree_min, target_degree_max = 15.0, 25.0
    # Aim for LOWER modularity in sampling because triadic closure will increase it
    target_mod_min, target_mod_max = 0.3, 0.55
    target_cluster_min = 0.3

    best_edges = None
    best_score = float("inf")

    for iteration in range(config.max_calibration_iterations):
        if on_progress:
            on_progress(
                "Calibrating network", iteration, config.max_calibration_iterations
            )

        # Generate with current parameters
        # Use a derived seed for reproducibility within calibration
        iter_rng = random.Random(seed + iteration * 1000)

        edges, avg_degree, clustering, modularity = _generate_network_single_pass(
            agents,
            agent_ids,
            similarities,
            communities,
            degree_factors,
            config,
            intra_scale,
            inter_scale,
            iter_rng,
        )

        # Score: sum of squared deviations from target ranges
        score = 0.0

        # Degree deviation
        if avg_degree < target_degree_min:
            score += (target_degree_min - avg_degree) ** 2
        elif avg_degree > target_degree_max:
            score += (avg_degree - target_degree_max) ** 2

        # Modularity deviation
        if modularity < target_mod_min:
            score += ((target_mod_min - modularity) * 10) ** 2  # Weight modularity
        elif modularity > target_mod_max:
            score += ((modularity - target_mod_max) * 10) ** 2

        # Clustering deviation (heavily penalize if too low)
        if clustering < target_cluster_min:
            score += ((target_cluster_min - clustering) * 20) ** 2  # Strong weight

        logger.debug(
            f"Calibration iter {iteration}: degree={avg_degree:.1f}, "
            f"clustering={clustering:.3f}, modularity={modularity:.3f}, score={score:.2f}"
        )

        # Track best result
        if score < best_score:
            best_score = score
            best_edges = edges

        # Check if we're in range
        in_range = (
            target_degree_min <= avg_degree <= target_degree_max
            and target_mod_min <= modularity <= target_mod_max
            and clustering >= target_cluster_min
        )

        if in_range:
            logger.info(f"Calibration converged at iteration {iteration}")
            best_edges = edges
            break

        # Adjust parameters for next iteration
        # Degree adjustment: scale both proportionally
        if avg_degree < target_degree_min * 0.9:
            scale_adj = (
                min(1.5, target_degree_min / avg_degree) if avg_degree > 0 else 1.5
            )
            intra_scale *= scale_adj
            inter_scale *= scale_adj
        elif avg_degree > target_degree_max * 1.1:
            scale_adj = max(0.6, target_degree_max / avg_degree)
            intra_scale *= scale_adj
            inter_scale *= scale_adj

        # Modularity adjustment: change ratio of intra to inter
        if modularity > target_mod_max + 0.05:
            # Too modular: boost inter-community edges
            inter_scale *= 1.3
            intra_scale *= 0.9
        elif modularity < target_mod_min - 0.05:
            # Not modular enough: boost intra-community
            intra_scale *= 1.2
            inter_scale *= 0.85

        # Clustering adjustment: more intra-community = more triads = higher clustering
        if clustering < target_cluster_min - 0.05:
            # Clustering too low: boost intra-community for more triadic closure potential
            intra_scale *= 1.15
            # Don't reduce inter too much or modularity will rise too high
            if modularity < target_mod_max - 0.1:
                inter_scale *= 0.95

        # Clamp scales to reasonable range
        intra_scale = max(0.3, min(6.0, intra_scale))
        inter_scale = max(0.3, min(6.0, inter_scale))

    if on_progress:
        on_progress(
            "Calibrating network",
            config.max_calibration_iterations,
            config.max_calibration_iterations,
        )

    # Use best result found
    edges = best_edges if best_edges else edges

    # Rebuild edge_set from best edges
    edge_set: set[tuple[str, str]] = set()
    for edge in edges:
        edge_set.add((edge.source, edge.target))
        edge_set.add((edge.target, edge.source))

    # Step 5: Watts-Strogatz rewiring
    if on_progress:
        on_progress("Rewiring edges", 0, len(edges))

    edges, edge_set, rewired_count = _apply_rewiring(
        agents, agent_ids, edges, edge_set, config, rng
    )

    if on_progress:
        on_progress("Rewiring edges", len(edges), len(edges))

    # Build metadata
    meta = {
        "agent_count": n,
        "edge_count": len(edges),
        "avg_degree": 2 * len(edges) / n if n > 0 else 0.0,
        "rewired_count": rewired_count,
        "algorithm": "adaptive_calibration",
        "seed": seed,
        "candidate_mode": candidate_mode,
        "similarity_pairs": len(similarities),
        "blocking_attributes": blocking_attrs if candidate_mode == "blocked" else [],
        "resumed_from_checkpoint": resume_from_checkpoint,
        "config": {
            "avg_degree_target": config.avg_degree,
            "rewire_prob": config.rewire_prob,
            "target_clustering": config.target_clustering,
            "target_modularity": config.target_modularity,
        },
        "generated_at": datetime.now().isoformat(),
    }

    return NetworkResult(meta=meta, edges=edges)


def generate_network_with_metrics(
    agents: list[dict[str, Any]],
    config: NetworkConfig | None = None,
    on_progress: NetworkProgressCallback | None = None,
    checkpoint_path: Path | str | None = None,
    resume_from_checkpoint: bool = False,
) -> NetworkResult:
    """Generate network and compute all metrics.

    Same as generate_network but also computes:
    - Network-level validation metrics
    - Per-agent node metrics (PageRank, betweenness, etc.)
    """
    from .metrics import compute_network_metrics, compute_node_metrics

    result = generate_network(
        agents,
        config,
        on_progress,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    agent_ids = [a.get("_id", f"agent_{i}") for i, a in enumerate(agents)]

    if on_progress:
        on_progress("Computing metrics", 0, 2)

    edge_dicts = [e.to_dict() for e in result.edges]
    result.network_metrics = compute_network_metrics(edge_dicts, agent_ids)

    if on_progress:
        on_progress("Computing metrics", 1, 2)

    result.node_metrics = compute_node_metrics(edge_dicts, agent_ids)

    if on_progress:
        on_progress("Computing metrics", 2, 2)

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
    """Load agents from JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "agents" in data:
        return data["agents"]
    else:
        raise ValueError(f"Unexpected JSON format in {path}")
