"""Network generation algorithms for creating social graphs between agents.

Implements adaptive calibration to hit target metrics (avg_degree, clustering, modularity).
"""

import json
import logging
import hashlib
import multiprocessing as mp
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
_STATUS_HEARTBEAT_SECONDS = 5.0

_SIM_WORKER_AGENTS: list[dict[str, Any]] | None = None
_SIM_WORKER_ATTRIBUTE_WEIGHTS = None
_SIM_WORKER_ORDINAL_LEVELS: dict[str, dict[str, int]] | None = None
_SIM_WORKER_THRESHOLD: float = 0.05
_SIM_WORKER_CANDIDATE_MAP: list[list[int]] | None = None


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


def _filter_over_fragmented_attrs(
    attrs: list[str],
    agents: list[dict[str, Any]],
    max_cardinality_ratio: float = 0.35,
) -> list[str]:
    """Drop block attrs that fragment almost every node into tiny groups."""
    if not attrs:
        return []
    n = max(1, len(agents))
    kept: list[str] = []
    for attr in attrs:
        values = {a.get(attr) for a in agents if a.get(attr) is not None}
        if not values:
            continue
        ratio = len(values) / n
        if ratio <= max_cardinality_ratio:
            kept.append(attr)
    return kept


def _build_blocked_candidate_map(
    agents: list[dict[str, Any]],
    config: NetworkConfig,
    seed: int,
) -> tuple[list[list[int]] | None, list[str]]:
    """Build per-agent candidate lists for blocked similarity mode."""
    attrs = _choose_blocking_attributes(config)
    if not config.blocking_attributes:
        attrs = _filter_over_fragmented_attrs(attrs, agents)
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

    target_pool = max(
        config.min_candidate_pool,
        int(config.avg_degree * config.candidate_pool_multiplier),
    )
    target_pool = max(1, min(n - 1, target_pool))

    candidate_map: list[list[int]] = [[] for _ in range(n)]
    global_quota = max(2, int(target_pool * 0.1))

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

        rng = random.Random(seed + (i + 1) * 7919)
        seen = set(chosen)
        seen.add(i)

        # Deterministic global bridge quota to avoid complete siloing in blocked mode.
        bridge_added = 0
        while bridge_added < global_quota and len(seen) < n:
            j = rng.randrange(n)
            if j in seen:
                continue
            seen.add(j)
            chosen.append(j)
            bridge_added += 1

        if len(chosen) < target_pool:
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


def _load_similarity_checkpoint(
    checkpoint_db: Path,
    expected_signature: dict[str, Any],
) -> tuple[dict[tuple[int, int], float], int, set[int]]:
    """Load checkpoint and validate compatibility with current run settings."""
    job_id = _similarity_checkpoint_job_id(expected_signature)
    with open_study_db(checkpoint_db) as db:
        signature = db.get_network_similarity_job_signature(job_id)
        if signature is None:
            raise ValueError(f"Checkpoint not found in study DB: job_id={job_id}")
        if signature != expected_signature:
            raise ValueError(
                "Checkpoint settings do not match current run. "
                "Delete checkpoint or run with matching config."
            )

        done_chunks = db.list_completed_similarity_chunks(job_id)
        done_starts = {start for start, _ in done_chunks}
        similarities = db.load_similarity_pairs(job_id)

    contiguous_rows = 0
    for start, end in done_chunks:
        if start != contiguous_rows:
            break
        contiguous_rows = end

    return similarities, max(0, contiguous_rows), done_starts


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


def _compute_similarity_chunk(
    task: tuple[int, int],
) -> tuple[int, list[tuple[int, int, float]]]:
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
    for start, end in tasks:
        if end <= completed_rows:
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
                ex.submit(_compute_similarity_chunk, task): task
                for task in pending_tasks
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
                        and checkpoint_job_id is not None
                    ):
                        with open_study_db(checkpoint_path) as db:
                            db.save_similarity_chunk_rows(
                                job_id=checkpoint_job_id,
                                chunk_start=current_start,
                                chunk_end=current_end,
                                rows=chunk_rows,
                            )

                    if on_progress:
                        on_progress(
                            "Computing similarities", min(completed_row_count, n), n
                        )
                    next_commit_idx += 1

    except Exception as e:
        downgraded_config = config.model_copy(
            update={
                "similarity_workers": 1,
                "similarity_chunk_size": max(8, config.similarity_chunk_size // 2),
            }
        )
        logger.warning(
            "Parallel similarity failed (%s). Falling back to serial mode "
            "(chunk_size %d -> %d).",
            e,
            config.similarity_chunk_size,
            downgraded_config.similarity_chunk_size,
        )
        return _compute_similarities_serial(
            agents=agents,
            config=downgraded_config,
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
    for start, end in tasks:
        if end <= start_row:
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
                    agents[i],
                    agents[j],
                    config.attribute_weights,
                    config.ordinal_levels,
                )
                if sim >= threshold:
                    similarities[(i, j)] = sim
                    local_rows.append((i, j, sim))

        completed_starts.add(start)
        completed_row_count += end - start

        if (
            checkpoint_path is not None
            and checkpoint_signature is not None
            and checkpoint_job_id is not None
        ):
            if (
                completed_row_count % checkpoint_every == 0
                or chunk_idx == len(tasks) - 1
            ):
                with open_study_db(checkpoint_path) as db:
                    db.save_similarity_chunk_rows(
                        job_id=checkpoint_job_id,
                        chunk_start=start,
                        chunk_end=end,
                        rows=local_rows,
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
    """Backwards-compatible wrapper for deterministic graph clustering."""
    communities, _ = _assign_communities_with_diagnostics(
        agents=agents,
        similarities=similarities,
        n_communities=n_communities,
        rng=rng,
    )
    return communities


def _compute_similarity_coverage(
    n: int,
    similarities: dict[tuple[int, int], float],
) -> dict[str, float]:
    """Compute sparsity diagnostics to decide whether escalation is required."""
    if n <= 1:
        return {
            "pair_count": float(len(similarities)),
            "pairs_per_node": 0.0,
            "node_coverage_ratio": 1.0,
            "density": 0.0,
            "fragmentation": 0.0,
        }
    degree_counts = [0] * n
    for i, j in similarities.keys():
        degree_counts[i] += 1
        degree_counts[j] += 1
    covered = sum(1 for d in degree_counts if d > 0)
    max_pairs = n * (n - 1) / 2.0
    pair_count = float(len(similarities))
    pairs_per_node = (2.0 * pair_count) / n
    density = pair_count / max_pairs if max_pairs > 0 else 0.0
    node_coverage_ratio = covered / n
    return {
        "pair_count": pair_count,
        "pairs_per_node": pairs_per_node,
        "node_coverage_ratio": node_coverage_ratio,
        "density": density,
        "fragmentation": 1.0 - node_coverage_ratio,
    }


def _coverage_needs_escalation(
    diagnostics: dict[str, float],
    config: NetworkConfig,
    n: int,
) -> bool:
    min_pairs_per_node = max(6.0, config.avg_degree * 0.85)
    if n >= 5000:
        min_pairs_per_node = max(min_pairs_per_node, 10.0)
    return (
        diagnostics["pairs_per_node"] < min_pairs_per_node
        or diagnostics["node_coverage_ratio"] < 0.7
    )


def _expand_candidate_map_undercovered(
    candidate_map: list[list[int]],
    n: int,
    min_pool: int,
    seed: int,
) -> list[list[int]]:
    """Deterministically expand sparse candidate rows for under-covered nodes."""
    expanded = [list(row) for row in candidate_map]
    capped_pool = max(1, min(n - 1, min_pool))
    for i in range(n):
        current = expanded[i]
        if len(current) >= capped_pool:
            continue
        seen = set(current)
        seen.add(i)
        rng = random.Random(seed + (i + 1) * 13007)
        while len(current) < capped_pool and len(seen) < n:
            j = rng.randrange(n)
            if j in seen:
                continue
            seen.add(j)
            current.append(j)
        expanded[i] = sorted(current)
    return expanded


def _choose_partition_attributes(
    agents: list[dict[str, Any]],
    n: int,
    preferred_attrs: list[str] | None = None,
) -> list[str]:
    """Choose stable categorical attributes for attribute-grounded communities."""
    if not agents:
        return []
    chosen: list[str] = []
    if preferred_attrs:
        for attr in preferred_attrs:
            if attr not in agents[0]:
                continue
            values = {a.get(attr) for a in agents if a.get(attr) is not None}
            card = len(values)
            if 2 <= card <= max(8, int(n * 0.2)):
                chosen.append(attr)
            if len(chosen) >= 3:
                break
    if len(chosen) >= 2:
        return chosen[:3]

    candidate_attrs: list[tuple[str, int]] = []
    for key in agents[0].keys():
        if key == "_id" or key.startswith("_"):
            continue
        first_val = agents[0].get(key)
        if not isinstance(first_val, (str, int, float, bool)):
            continue
        values = {a.get(key) for a in agents if a.get(key) is not None}
        card = len(values)
        if 2 <= card <= max(8, int(n * 0.2)):
            candidate_attrs.append((key, card))
    candidate_attrs.sort(key=lambda x: (-x[1], x[0]))
    fallback = [name for name, _ in candidate_attrs[:3]]
    return (chosen + fallback)[:3]


def _build_attribute_partition(
    agents: list[dict[str, Any]],
    n_communities: int,
    attributes: list[str],
) -> list[int] | None:
    """Build deterministic, load-balanced partition from selected attributes."""
    n = len(agents)
    if n == 0 or n_communities <= 1 or not attributes:
        return None
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, agent in enumerate(agents):
        key = tuple(agent.get(attr) for attr in attributes)
        buckets[key].append(idx)
    if len(buckets) < 2:
        return None

    k = min(n_communities, n)
    target_size = max(1, (n + k - 1) // k)
    capacity = max(1, int(target_size * 1.25))
    bucket_items = sorted(
        ((bucket_key, sorted(members)) for bucket_key, members in buckets.items()),
        key=lambda item: (-len(item[1]), str(item[0])),
    )

    reassigned = [0] * n
    loads = [0] * k

    for _bucket_key, members in bucket_items:
        remaining = list(members)

        # If this bucket can fit in one reasonably empty community, keep it intact.
        if len(remaining) <= capacity:
            cid = min(range(k), key=lambda idx: (loads[idx], idx))
            for node in remaining:
                reassigned[node] = cid
            loads[cid] += len(remaining)
            continue

        # Large buckets are split deterministically across least-loaded communities.
        while remaining:
            cid = min(range(k), key=lambda idx: (loads[idx], idx))
            room = max(1, capacity - loads[cid])
            take = min(len(remaining), room)
            chunk = remaining[:take]
            remaining = remaining[take:]
            for node in chunk:
                reassigned[node] = cid
            loads[cid] += len(chunk)

    return reassigned


def _community_signal(
    similarities: dict[tuple[int, int], float],
    assignments: list[int],
) -> float:
    """Higher is better: within-community similarity minus cross-community similarity."""
    if not similarities:
        return 0.0
    within_sum = 0.0
    cross_sum = 0.0
    within_n = 0
    cross_n = 0
    for (i, j), sim in similarities.items():
        if assignments[i] == assignments[j]:
            within_sum += sim
            within_n += 1
        else:
            cross_sum += sim
            cross_n += 1
    within_mean = within_sum / within_n if within_n else 0.0
    cross_mean = cross_sum / cross_n if cross_n else 0.0
    return within_mean - cross_mean


def _assign_communities_with_diagnostics(
    agents: list[dict],
    similarities: dict[tuple[int, int], float],
    n_communities: int,
    rng: random.Random,
    preferred_attrs: list[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[int], dict[str, float]]:
    """Assign communities using deterministic weighted label propagation."""
    n = len(agents)
    k = min(max(1, n_communities), max(1, n))
    if n <= 1 or k <= 1:
        return [0] * n, {"low_signal": 0.0, "label_iterations": 0.0}

    coverage = _compute_similarity_coverage(n, similarities)
    # Low-signal short-circuit: avoid expensive refinement when the graph is too sparse.
    if coverage["pairs_per_node"] < 1.0 or coverage["node_coverage_ratio"] < 0.2:
        assignments = [i % k for i in range(n)]
        return assignments, {
            "low_signal": 1.0,
            "label_iterations": 0.0,
            "initial_labels": float(k),
            "final_labels": float(k),
        }

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), w in similarities.items():
        adjacency[i].append((j, w))
        adjacency[j].append((i, w))

    labels = list(range(n))
    max_iters = 8 if n >= 5000 else 12
    iters_used = 0

    for iteration in range(max_iters):
        iters_used = iteration + 1
        order = list(range(n))
        iter_rng = random.Random(rng.randint(0, 2**31 - 1) + iteration * 1009)
        iter_rng.shuffle(order)
        changed = 0
        for node in order:
            neighbors = adjacency[node]
            if not neighbors:
                continue
            scores: dict[int, float] = defaultdict(float)
            for nei, weight in neighbors:
                scores[labels[nei]] += weight
            best_label = labels[node]
            best_score = scores.get(best_label, -1.0)
            for label, score in scores.items():
                if score > best_score or (score == best_score and label < best_label):
                    best_label = label
                    best_score = score
            if best_label != labels[node]:
                labels[node] = best_label
                changed += 1
        if progress:
            progress("Detecting communities", iters_used, max_iters)
        if changed == 0:
            break

    # Compress to top-k labels by size; reassign tiny groups to nearest dominant label.
    label_members: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_members[label].append(idx)
    top_labels = [
        label
        for label, _members in sorted(
            label_members.items(),
            key=lambda x: (-len(x[1]), x[0]),
        )[:k]
    ]
    top_label_set = set(top_labels)

    major_index = {label: i for i, label in enumerate(top_labels)}
    assignments = [0] * n
    for node in range(n):
        own_label = labels[node]
        if own_label in top_label_set:
            assignments[node] = major_index[own_label]
            continue
        scores: dict[int, float] = defaultdict(float)
        for nei, weight in adjacency[node]:
            nei_label = labels[nei]
            if nei_label in top_label_set:
                scores[nei_label] += weight
        if scores:
            chosen = max(scores.items(), key=lambda x: (x[1], -x[0]))[0]
            assignments[node] = major_index[chosen]
        else:
            assignments[node] = node % k

    unique_labels = len(set(assignments))
    label_signal = _community_signal(similarities, assignments)

    selected_attrs = _choose_partition_attributes(agents, n, preferred_attrs)
    attr_assignments = _build_attribute_partition(agents, k, selected_attrs)
    attr_signal = -1.0
    chose_attr_partition = 0.0
    if attr_assignments is not None:
        attr_signal = _community_signal(similarities, attr_assignments)
        # Prefer attribute partition when it yields clearer assortative structure.
        if attr_signal >= label_signal + 0.01:
            assignments = attr_assignments
            unique_labels = len(set(assignments))
            label_signal = attr_signal
            chose_attr_partition = 1.0

    # Guard against community collapse (e.g., all nodes in one label).
    # If collapsed, enforce a deterministic attribute-grounded partition first.
    # Fall back to seeded partition only if no useful attributes are available.
    forced_partition = 0.0
    attribute_partition = 0.0
    min_required = max(2, min(k, n // 120))
    if unique_labels < min_required:
        if attr_assignments is not None:
            assignments = attr_assignments
            unique_labels = len(set(assignments))
            forced_partition = 1.0
            attribute_partition = 1.0

        if unique_labels < min_required:
            ordered_nodes = sorted(range(n), key=lambda i: (-len(adjacency[i]), i))
            seeds: list[int] = []
            if ordered_nodes:
                stride = max(1, len(ordered_nodes) // k)
                for idx in range(0, len(ordered_nodes), stride):
                    seeds.append(ordered_nodes[idx])
                    if len(seeds) >= k:
                        break
            seen = set(seeds)
            if len(seeds) < k:
                for idx in ordered_nodes:
                    if idx in seen:
                        continue
                    seeds.append(idx)
                    seen.add(idx)
                    if len(seeds) >= k:
                        break
            if not seeds:
                seeds = list(range(min(k, n)))

            def get_sim(i: int, j: int) -> float:
                if i == j:
                    return 1.0
                pair = (i, j) if i < j else (j, i)
                return similarities.get(pair, 0.0)

            reassigned = [0] * n
            for node in range(n):
                best_idx = node % len(seeds)
                best_sim = get_sim(node, seeds[best_idx])
                any_signal = best_sim > 0.0
                for seed_idx in range(1, len(seeds)):
                    sim = get_sim(node, seeds[seed_idx])
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = seed_idx
                        any_signal = True
                if any_signal:
                    reassigned[node] = best_idx
                else:
                    reassigned[node] = node % len(seeds)

            assignments = reassigned
            unique_labels = len(set(assignments))
            forced_partition = 1.0
        label_signal = _community_signal(similarities, assignments)

    return assignments, {
        "low_signal": 0.0,
        "label_iterations": float(iters_used),
        "initial_labels": float(len(label_members)),
        "final_labels": float(unique_labels),
        "community_signal": float(label_signal),
        "attr_signal": float(attr_signal),
        "chosen_attribute_partition": chose_attr_partition,
        "forced_partition": forced_partition,
        "attribute_partition": attribute_partition,
    }


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


def _compute_largest_component_ratio(adjacency: dict[int, set[int]], n: int) -> float:
    """Compute largest connected component size / n."""
    if n <= 0:
        return 0.0
    visited = [False] * n
    largest = 0
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nei in adjacency[node]:
                if not visited[nei]:
                    visited[nei] = True
                    stack.append(nei)
        if size > largest:
            largest = size
    return largest / n


def _build_adjacency_from_edges(
    edges: list[Edge], agent_ids: list[str], n: int
) -> dict[int, set[int]]:
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for edge in edges:
        i = id_to_idx[edge.source]
        j = id_to_idx[edge.target]
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def _acceptance_bounds(config: NetworkConfig) -> tuple[float, float, float, float, float]:
    degree_delta = max(1.0, config.avg_degree * config.target_degree_tolerance_pct)
    deg_min = max(1.0, config.avg_degree - degree_delta)
    deg_max = config.avg_degree + degree_delta
    clust_min = max(0.0, config.target_clustering - config.target_clustering_tolerance)
    mod_min = max(0.0, config.target_modularity - config.target_modularity_tolerance)
    mod_max = min(1.0, config.target_modularity + config.target_modularity_tolerance)
    return deg_min, deg_max, clust_min, mod_min, mod_max


def _apply_scale_target_caps(
    config: NetworkConfig,
    n: int,
) -> tuple[NetworkConfig, dict[str, float]]:
    """Cap strict topology targets to scale-feasible envelopes.

    This prevents large runs from chasing unrealistic clustering/modularity
    values produced by first-pass generated configs.
    """
    if n < 2000:
        return config, {}

    if n >= 5000:
        cluster_cap = 0.25 if config.quality_profile == "strict" else 0.22
        modularity_cap = 0.45 if config.quality_profile == "strict" else 0.40
    else:
        cluster_cap = 0.33 if config.quality_profile == "strict" else 0.30
        modularity_cap = 0.55 if config.quality_profile == "strict" else 0.50

    updates: dict[str, float] = {}
    caps: dict[str, float] = {}
    if config.target_clustering > cluster_cap:
        updates["target_clustering"] = cluster_cap
        caps["target_clustering"] = cluster_cap
    if config.target_modularity > modularity_cap:
        updates["target_modularity"] = modularity_cap
        caps["target_modularity"] = modularity_cap

    if not updates:
        return config, {}
    return config.model_copy(update=updates), caps


def _min_edge_floor(n: int, config: NetworkConfig) -> int:
    expected = n * config.avg_degree / 2.0
    if config.quality_profile == "strict":
        ratio = 0.75
    elif config.quality_profile == "fast":
        ratio = 0.55
    else:
        ratio = 0.65
    return max(1, int(expected * ratio))


def _apply_bridge_swaps(
    edges: list[Edge],
    agent_ids: list[str],
    communities: list[int],
    config: NetworkConfig,
    rng: random.Random,
    max_swaps_override: int | None = None,
) -> list[Edge]:
    """Degree-preserving swaps that increase cross-community bridges."""
    if config.swap_passes <= 0 or len(edges) < 4:
        return edges
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    edge_pairs = [(id_to_idx[e.source], id_to_idx[e.target]) for e in edges]
    edge_set = {tuple(sorted(pair)) for pair in edge_pairs}
    max_swaps = max(4, int(len(edges) * config.bridge_budget_fraction))
    if max_swaps_override is not None:
        max_swaps = max(1, min(max_swaps, max_swaps_override))

    for _ in range(config.swap_passes):
        swaps = 0
        for _attempt in range(max_swaps * 4):
            if swaps >= max_swaps:
                break
            a_idx = rng.randrange(len(edges))
            b_idx = rng.randrange(len(edges))
            if a_idx == b_idx:
                continue
            e1 = edges[a_idx]
            e2 = edges[b_idx]
            u, v = id_to_idx[e1.source], id_to_idx[e1.target]
            x, y = id_to_idx[e2.source], id_to_idx[e2.target]
            if len({u, v, x, y}) < 4:
                continue
            old_cross = int(communities[u] != communities[v]) + int(
                communities[x] != communities[y]
            )
            new_cross = int(communities[u] != communities[y]) + int(
                communities[x] != communities[v]
            )
            if new_cross <= old_cross:
                continue
            p1 = tuple(sorted((u, y)))
            p2 = tuple(sorted((x, v)))
            if p1 in edge_set or p2 in edge_set:
                continue
            edge_set.discard(tuple(sorted((u, v))))
            edge_set.discard(tuple(sorted((x, y))))
            edge_set.add(p1)
            edge_set.add(p2)
            edges[a_idx] = Edge(
                source=agent_ids[u],
                target=agent_ids[y],
                weight=e1.weight,
                edge_type=e1.edge_type,
                influence_weight=e1.influence_weight,
            )
            edges[b_idx] = Edge(
                source=agent_ids[x],
                target=agent_ids[v],
                weight=e2.weight,
                edge_type=e2.edge_type,
                influence_weight=e2.influence_weight,
            )
            swaps += 1
    return edges


def _target_intra_edge_fraction(
    communities: list[int],
    config: NetworkConfig,
    intra_scale: float,
    inter_scale: float,
) -> float:
    """Estimate intra-community edge fraction needed to hit target modularity.

    Approximation:
      Q ~= e_intra - sum_c (a_c^2)
    where e_intra is the fraction of within-community edges and a_c is the
    degree share for community c (approximated by size share).
    """
    n = len(communities)
    if n <= 0:
        return 0.5
    counts: dict[int, int] = defaultdict(int)
    for c in communities:
        counts[c] += 1
    sum_size_sq = sum((count / n) ** 2 for count in counts.values())

    # Base fraction anchored to target modularity.
    base = config.target_modularity + sum_size_sq + 0.02

    # Let calibration scales nudge assortativity.
    scale_delta = max(-2.0, min(2.0, intra_scale - inter_scale))
    adjusted = base + (0.06 * scale_delta)

    if config.quality_profile == "strict":
        lo, hi = 0.22, 0.95
    elif config.quality_profile == "fast":
        lo, hi = 0.18, 0.88
    else:
        lo, hi = 0.20, 0.92
    return max(lo, min(hi, adjusted))


def _pair_seed_jitter(pair_seed: int, i: int, j: int) -> float:
    """Small deterministic jitter so different seeds can produce different graphs."""
    h = ((i + 1) * 73856093) ^ ((j + 1) * 19349663) ^ pair_seed
    h &= 0xFFFFFFFF
    return (h / 0xFFFFFFFF) * 1e-6


def _ensure_intra_similarity_coverage(
    agents: list[dict],
    similarities: dict[tuple[int, int], float],
    communities: list[int],
    config: NetworkConfig,
    rng_seed: int,
    min_intra_candidates_per_node: int,
) -> int:
    """Backfill sparse within-community similarities for under-covered nodes.

    This prevents blocked candidate mode from starving calibration when
    community-local candidate coverage is too low.
    """
    n = len(agents)
    if n <= 1 or min_intra_candidates_per_node <= 0:
        return 0

    members_by_community: dict[int, list[int]] = defaultdict(list)
    for idx, c in enumerate(communities):
        members_by_community[c].append(idx)

    intra_counts = [0] * n
    for (i, j) in similarities.keys():
        if communities[i] == communities[j]:
            intra_counts[i] += 1
            intra_counts[j] += 1

    added = 0
    for i in range(n):
        need = min_intra_candidates_per_node - intra_counts[i]
        if need <= 0:
            continue
        peers = members_by_community.get(communities[i], [])
        if len(peers) <= 1:
            continue

        # Keep augmentation bounded for runtime stability.
        max_to_add = min(need, max(4, min_intra_candidates_per_node // 2))
        node_rng = random.Random(rng_seed + (i + 1) * 911)
        attempts = 0
        added_for_node = 0
        max_attempts = max_to_add * 10
        while added_for_node < max_to_add and attempts < max_attempts:
            attempts += 1
            j = peers[node_rng.randrange(len(peers))]
            if j == i:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in similarities:
                continue
            sim = compute_similarity(
                agents[i],
                agents[j],
                config.attribute_weights,
                config.ordinal_levels,
            )
            # Keep weak ties only if they are not near-zero noise.
            if sim <= 0.01:
                continue
            similarities[pair] = sim
            intra_counts[i] += 1
            intra_counts[j] += 1
            added_for_node += 1
            added += 1

    return added


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
    """Sample edges with deterministic community quotas and local attachment."""
    n = len(agents)
    edges: list[Edge] = []
    edge_set: set[tuple[str, str]] = set()
    added_pairs: set[tuple[int, int]] = set()
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    node_degree = [0] * n
    community_external: dict[int, int] = defaultdict(int)

    def add_edge(i: int, j: int, sim: float) -> bool:
        pair = (i, j) if i < j else (j, i)
        if i == j or pair in added_pairs:
            return False
        added_pairs.add(pair)

        agent_a = agents[i]
        agent_b = agents[j]
        id_a = agent_ids[i]
        id_b = agent_ids[j]

        edge_type = _infer_edge_type(agent_a, agent_b, config, is_rewired=False)
        influence_weights = _compute_influence_weights(agent_a, agent_b, sim, config)
        edges.append(
            Edge(
                source=id_a,
                target=id_b,
                weight=sim,
                edge_type=edge_type,
                influence_weight=influence_weights,
            )
        )
        edge_set.add((id_a, id_b))
        edge_set.add((id_b, id_a))
        adjacency[i].add(j)
        adjacency[j].add(i)
        node_degree[i] += 1
        node_degree[j] += 1
        if communities[i] != communities[j]:
            ci = communities[i]
            cj = communities[j]
            community_external[ci] += 1
            community_external[cj] += 1
        return True

    if not similarities:
        return edges, edge_set

    # Dynamic degree target so calibration scales can move degree.
    expected_edges = max(1, int(n * config.avg_degree / 2))
    baseline_scale = 0.65
    scale_mean = max(0.1, (intra_scale + inter_scale) / 2.0)
    target_multiplier = max(0.4, min(1.35, scale_mean / baseline_scale))
    target_edges = max(1, int(expected_edges * target_multiplier))

    # Ensure within-community coverage is not starved by blocked candidates.
    min_local_candidates = max(4, int(config.avg_degree * 1.1))
    _ensure_intra_similarity_coverage(
        agents=agents,
        similarities=similarities,
        communities=communities,
        config=config,
        rng_seed=rng.randrange(2**31 - 1),
        min_intra_candidates_per_node=min_local_candidates,
    )

    pair_seed = rng.randrange(2**31 - 1)
    intra_pairs: list[tuple[float, int, int, float]] = []
    inter_pairs: list[tuple[float, int, int, float]] = []
    for (i, j), sim in similarities.items():
        base = sim * degree_factors[i] * degree_factors[j]
        if communities[i] == communities[j]:
            score = (base * intra_scale) + _pair_seed_jitter(pair_seed, i, j)
            intra_pairs.append((score, i, j, sim))
        else:
            score = (base * inter_scale) + _pair_seed_jitter(pair_seed, i, j)
            inter_pairs.append((score, i, j, sim))

    intra_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    inter_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    intra_fraction = _target_intra_edge_fraction(
        communities=communities,
        config=config,
        intra_scale=intra_scale,
        inter_scale=inter_scale,
    )
    intra_target = int(round(target_edges * intra_fraction))
    intra_target = max(0, min(target_edges, intra_target))
    inter_target = target_edges - intra_target

    intra_available = len(intra_pairs)
    inter_available = len(inter_pairs)
    if intra_target > intra_available:
        intra_target = intra_available
        inter_target = min(inter_available, target_edges - intra_target)
    if inter_target > inter_available:
        inter_target = inter_available
        intra_target = min(intra_available, target_edges - inter_target)

    degree_soft_cap = max(2, int(config.avg_degree * 1.8 * max(0.8, target_multiplier)))
    seed_k = max(1, min(12, int(config.avg_degree * 0.35 * max(0.8, intra_scale))))

    # Seed local neighborhoods (kNN-like) for clustering.
    intra_by_node: list[list[tuple[float, int, float]]] = [[] for _ in range(n)]
    for score, i, j, sim in intra_pairs:
        intra_by_node[i].append((score, j, sim))
        intra_by_node[j].append((score, i, sim))
    for node in range(n):
        intra_by_node[node].sort(key=lambda x: (-x[0], x[1]))

    intra_added = 0
    for i in range(n):
        if intra_added >= intra_target:
            break
        picked = 0
        for _score, j, sim in intra_by_node[i]:
            if picked >= seed_k or intra_added >= intra_target:
                break
            if node_degree[i] >= degree_soft_cap and node_degree[j] >= degree_soft_cap:
                continue
            if add_edge(i, j, sim):
                intra_added += 1
                picked += 1

    # Fill remaining intra quota by highest-score pairs.
    for _score, i, j, sim in intra_pairs:
        if intra_added >= intra_target:
            break
        if node_degree[i] >= degree_soft_cap and node_degree[j] >= degree_soft_cap:
            continue
        if add_edge(i, j, sim):
            intra_added += 1

    # Guarantee at least one external bridge per community where possible.
    unique_communities = sorted(set(communities))
    min_bridges_per_community = 1 if len(unique_communities) > 1 else 0
    best_bridge_for_community: dict[int, tuple[int, int, float]] = {}
    for _score, i, j, sim in inter_pairs:
        ci = communities[i]
        cj = communities[j]
        if ci not in best_bridge_for_community:
            best_bridge_for_community[ci] = (i, j, sim)
        if cj not in best_bridge_for_community:
            best_bridge_for_community[cj] = (i, j, sim)
        if len(best_bridge_for_community) >= len(unique_communities):
            break

    inter_added = 0
    for community in unique_communities:
        if inter_added >= inter_target:
            break
        if community_external[community] >= min_bridges_per_community:
            continue
        bridge = best_bridge_for_community.get(community)
        if bridge is None:
            continue
        i, j, sim = bridge
        if node_degree[i] >= degree_soft_cap and node_degree[j] >= degree_soft_cap:
            continue
        if add_edge(i, j, sim):
            inter_added += 1

    # Fill remaining inter quota.
    for _score, i, j, sim in inter_pairs:
        if inter_added >= inter_target:
            break
        if node_degree[i] >= degree_soft_cap and node_degree[j] >= degree_soft_cap:
            continue
        if add_edge(i, j, sim):
            inter_added += 1

    # Local attachment: close open triads first, then backfill to target.
    if len(edges) < target_edges:
        fof_candidates: list[tuple[float, int, int, float]] = []
        seen_fof: set[tuple[int, int]] = set()
        for i in range(n):
            neighbors = adjacency[i]
            if len(neighbors) < 2:
                continue
            for neighbor in neighbors:
                for fof in adjacency[neighbor]:
                    if fof == i or fof in neighbors:
                        continue
                    pair = (i, fof) if i < fof else (fof, i)
                    if pair in added_pairs or pair in seen_fof:
                        continue
                    seen_fof.add(pair)
                    sim = similarities.get(pair)
                    if sim is None or sim <= 0.01:
                        continue
                    common = len(adjacency[i] & adjacency[fof])
                    same_community = communities[i] == communities[fof]
                    score = sim * (1.0 + (0.25 * common))
                    if same_community:
                        score *= 1.25
                    else:
                        score *= 0.55
                    score += _pair_seed_jitter(pair_seed, pair[0], pair[1])
                    fof_candidates.append((score, pair[0], pair[1], sim))

        fof_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        for _score, i, j, sim in fof_candidates:
            if len(edges) >= target_edges:
                break
            if node_degree[i] >= degree_soft_cap and node_degree[j] >= degree_soft_cap:
                continue
            add_edge(i, j, sim)

    # Final backfill if still short.
    if len(edges) < target_edges:
        all_pairs = intra_pairs + inter_pairs
        all_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
        for _score, i, j, sim in all_pairs:
            if len(edges) >= target_edges:
                break
            if node_degree[i] >= degree_soft_cap and node_degree[j] >= degree_soft_cap:
                continue
            add_edge(i, j, sim)

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
    """Apply bounded triadic closure with sampled open-triad proposals.

    This avoids global open-triad enumeration (which is expensive at 10k+ scale)
    while still aggressively adding triangle-closing edges.
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

    # Edge budget: do not exceed expansion factor and keep degree near target envelope.
    max_edges = int(len(edges) * max_edge_increase)
    degree_cap_edges = int(n * config.avg_degree / 2 * 1.2)
    if degree_cap_edges > 0:
        max_edges = min(max_edges, degree_cap_edges)
    available = max(0, max_edges - len(edges))
    if available <= 0:
        return edges, edge_set

    hubs = [i for i in range(n) if len(adjacency[i]) >= 2]
    if not hubs:
        return edges, edge_set

    # Calibrate closure budget to the actual clustering gap.
    # Small gaps should not trigger large structural shifts.
    gap = max(0.0, target_clustering - current_clustering)
    gap_ratio = gap / max(0.05, target_clustering)
    budget_ratio = min(0.40, max(0.10, gap_ratio * 0.50))
    closure_budget = min(available, max(200, int(available * budget_ratio)))
    attempts = max(closure_budget * 6, n)
    closures = 0

    for _ in range(attempts):
        if closures >= closure_budget or len(edges) >= max_edges:
            break

        b = hubs[rng.randrange(len(hubs))]
        neighbors = list(adjacency[b])
        if len(neighbors) < 2:
            continue
        a, c = rng.sample(neighbors, 2)
        if a == c or c in adjacency[a]:
            continue

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

        same_community = communities is not None and communities[a] == communities[c]
        # Enforce assortative closure: cross-community triad closure tends to destroy
        # modularity at scale and is better handled by explicit bridge logic.
        if not same_community:
            continue

        effective_prob = config.triadic_closure_prob * (0.85 + sim * 0.55)
        if same_community:
            effective_prob += 0.25
        effective_prob = min(0.98, max(0.05, effective_prob))

        if rng.random() >= effective_prob:
            continue

        agent_a = agents[a]
        agent_c = agents[c]
        id_a = agent_ids[a]
        id_c = agent_ids[c]
        edge_type = _infer_edge_type(agent_a, agent_c, config, is_rewired=False)
        influence_weights = _compute_influence_weights(agent_a, agent_c, sim, config)

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
    """Compute modularity from fixed community assignments.

    Uses the equivalent community-aggregate form:
      Q = sum_c [ (l_c / m) - (d_c / (2m))^2 ]
    where l_c is internal edge count in community c and d_c is total degree sum.
    """
    n = len(agent_ids)
    m = len(edges)
    if m == 0:
        return 0.0

    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    # Node degrees.
    degrees = [0] * n
    for edge in edges:
        i = id_to_idx[edge.source]
        j = id_to_idx[edge.target]
        degrees[i] += 1
        degrees[j] += 1

    # Internal edge counts per community.
    internal_edges: dict[int, int] = defaultdict(int)
    for edge in edges:
        i = id_to_idx[edge.source]
        j = id_to_idx[edge.target]
        ci = communities[i]
        if ci == communities[j]:
            internal_edges[ci] += 1

    # Degree sums per community.
    degree_sums: dict[int, int] = defaultdict(int)
    for idx, degree in enumerate(degrees):
        degree_sums[communities[idx]] += degree

    two_m = float(2 * m)
    q = 0.0
    for community, d_sum in degree_sums.items():
        l_c = internal_edges.get(community, 0)
        q += (l_c / m) - (d_sum / two_m) ** 2
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
) -> tuple[list[Edge], float, float, float, float]:
    """Generate network with given parameters and return metrics.

    Returns (edges, avg_degree, clustering, modularity, largest_component_ratio).
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

    adjacency = _build_adjacency_from_edges(edges, agent_ids, n)

    clustering = _compute_avg_clustering(adjacency, n)
    modularity = _compute_modularity_fast(edges, agent_ids, communities)
    largest_component_ratio = _compute_largest_component_ratio(adjacency, n)

    return edges, avg_degree, clustering, modularity, largest_component_ratio


def generate_network(
    agents: list[dict[str, Any]],
    config: NetworkConfig | None = None,
    on_progress: NetworkProgressCallback | None = None,
    checkpoint_path: Path | str | None = None,
    resume_from_checkpoint: bool = False,
    study_db_path: Path | str | None = None,
    network_run_id: str | None = None,
    resume_calibration: bool = False,
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
    config = config.apply_quality_profile_defaults()

    # Initialize RNG
    seed = config.seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = random.Random(seed)

    n = len(agents)
    config, target_caps = _apply_scale_target_caps(config, n)
    agent_ids = [a.get("_id", f"agent_{i}") for i, a in enumerate(agents)]
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
    study_db_file = Path(study_db_path) if study_db_path else None
    if checkpoint_file is not None and checkpoint_file.suffix.lower() != ".db":
        raise ValueError(
            "Network checkpoints are DB-only now. Use --study-db (or --checkpoint <study.db>)."
        )
    if resume_from_checkpoint and checkpoint_file is None:
        raise ValueError("--resume-checkpoint requires a checkpoint DB path")

    started_at = time.time()
    last_heartbeat = [0.0]

    def emit_progress(
        stage: str,
        current: int,
        total: int,
        message: str | dict[str, Any] | None = None,
        force_db: bool = False,
    ) -> None:
        if on_progress:
            on_progress(stage, current, total)
        should_write = force_db
        now = time.time()
        if now - last_heartbeat[0] >= _STATUS_HEARTBEAT_SECONDS:
            should_write = True
            last_heartbeat[0] = now
        if not should_write:
            return
        if study_db_file is not None and network_run_id:
            if isinstance(message, dict):
                message_text = json.dumps(message, sort_keys=True, separators=(",", ":"))
            else:
                message_text = message or stage
            with open_study_db(study_db_file) as db:
                db.upsert_network_generation_status(
                    network_run_id=network_run_id,
                    phase=stage,
                    current=current,
                    total=total,
                    message=message_text,
                )

    def _stage_plan() -> list[dict[str, Any]]:
        if config.candidate_mode == "exact":
            return [{"name": "exact", "candidate_mode": "exact", "pool_multiplier": 0.0}]
        base_mult = max(6.0, config.candidate_pool_multiplier)
        if config.quality_profile == "fast":
            return [
                {
                    "name": "blocked",
                    "candidate_mode": "blocked",
                    "pool_multiplier": base_mult,
                    "coverage_multiplier": 1.0,
                },
                {
                    "name": "blocked-expanded",
                    "candidate_mode": "blocked",
                    "pool_multiplier": max(base_mult * 1.5, base_mult + 5.0),
                    "coverage_multiplier": 1.5,
                },
            ]
        return [
            {
                "name": "blocked",
                "candidate_mode": "blocked",
                "pool_multiplier": base_mult,
                "coverage_multiplier": 1.0,
            },
            {
                "name": "blocked-expanded",
                "candidate_mode": "blocked",
                "pool_multiplier": max(base_mult * 1.8, base_mult + 8.0),
                "coverage_multiplier": 2.0,
            },
            {
                "name": "hybrid-dense",
                "candidate_mode": "blocked",
                "pool_multiplier": max(base_mult * 2.6, base_mult + 14.0),
                "coverage_multiplier": 2.8,
                "hybrid_expand": True,
            },
        ]

    # Step 1: Compute degree factors once.
    degree_factors = [compute_degree_factor(a, config) for a in agents]

    # Global gate bounds.
    deg_min, deg_max, cluster_min, mod_min, mod_max = _acceptance_bounds(config)
    min_edge_floor = _min_edge_floor(n, config)

    stage_plan = _stage_plan()
    elapsed_budget = max(1, int(config.max_calibration_minutes * 60))
    stage_budget = max(30, elapsed_budget // max(1, len(stage_plan)))

    accepted = False
    best_score = float("inf")
    best_edges: list[Edge] | None = None
    best_metrics: dict[str, float] | None = None
    best_stage = ""
    final_candidate_mode = config.candidate_mode
    final_blocking_attrs: list[str] = []
    final_similarity_pairs = 0
    stage_summaries: list[dict[str, Any]] = []
    calibration_step = 0
    calibration_total = max(
        1, len(stage_plan) * config.calibration_restarts * config.max_calibration_iterations
    )

    for stage_idx, stage in enumerate(stage_plan):
        if time.time() - started_at >= elapsed_budget:
            break
        stage_name = stage["name"]
        stage_started = time.time()
        stage_cfg = config.model_copy(
            update={
                "candidate_mode": stage["candidate_mode"],
                "candidate_pool_multiplier": stage.get(
                    "pool_multiplier", config.candidate_pool_multiplier
                ),
                "min_candidate_pool": max(
                    config.min_candidate_pool,
                    int(config.avg_degree * stage.get("coverage_multiplier", 1.0)),
                ),
            }
        )

        emit_progress(
            "Preparing candidates",
            stage_idx + 1,
            len(stage_plan),
            message={"stage": stage_name, "elapsed_s": int(time.time() - started_at)},
            force_db=True,
        )
        candidate_mode = stage_cfg.candidate_mode
        candidate_map: list[list[int]] | None = None
        blocking_attrs: list[str] = []
        if stage_cfg.candidate_mode == "blocked":
            candidate_map, blocking_attrs = _build_blocked_candidate_map(
                agents, stage_cfg, seed + stage_idx * 104729
            )
            if candidate_map is None:
                candidate_mode = "exact"
            elif stage.get("hybrid_expand"):
                expanded_pool = int(max(config.avg_degree * 20, stage_cfg.min_candidate_pool))
                candidate_map = _expand_candidate_map_undercovered(
                    candidate_map=candidate_map,
                    n=n,
                    min_pool=min(n - 1, expanded_pool),
                    seed=seed + stage_idx * 193,
                )

        emit_progress(
            "Computing similarities",
            0,
            n,
            message={"stage": stage_name, "mode": candidate_mode},
            force_db=True,
        )
        checkpoint_signature = _similarity_checkpoint_signature(
            n=n,
            seed=seed + stage_idx * 31,
            config=stage_cfg,
            blocking_attrs=blocking_attrs,
        )
        checkpoint_job_id: str | None = None
        similarities: dict[tuple[int, int], float] = {}
        start_row = 0
        completed_chunk_starts: set[int] = set()
        should_resume_similarity = resume_from_checkpoint and stage_idx == 0

        if checkpoint_file is not None:
            checkpoint_job_id = _similarity_checkpoint_job_id(checkpoint_signature)
            if not should_resume_similarity:
                with open_study_db(checkpoint_file) as db:
                    db.init_network_similarity_job(
                        network_run_id=f"checkpoint:{checkpoint_job_id}",
                        signature=checkpoint_signature,
                        job_id=checkpoint_job_id,
                    )
                    db.mark_similarity_job_running(checkpoint_job_id)

        if should_resume_similarity:
            if checkpoint_file is None or not checkpoint_file.exists():
                raise ValueError(f"Checkpoint not found: {checkpoint_file}")
            similarities, start_row, completed_chunk_starts = _load_similarity_checkpoint(
                checkpoint_file, checkpoint_signature
            )
            if checkpoint_job_id and checkpoint_file is not None:
                with open_study_db(checkpoint_file) as db:
                    db.mark_similarity_job_running(checkpoint_job_id)

        use_parallel_similarity = stage_cfg.similarity_workers > 1
        def similarity_progress(_stage: str, current: int, total: int) -> None:
            emit_progress(
                "Computing similarities",
                current,
                total,
                message={"stage": stage_name},
                force_db=current == total,
            )

        if use_parallel_similarity:
            similarities = _compute_similarities_parallel(
                agents=agents,
                config=stage_cfg,
                candidate_map=candidate_map if candidate_mode == "blocked" else None,
                on_progress=similarity_progress,
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
                config=stage_cfg,
                candidate_map=candidate_map if candidate_mode == "blocked" else None,
                on_progress=similarity_progress,
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

        coverage = _compute_similarity_coverage(n, similarities)
        needs_escalation = _coverage_needs_escalation(coverage, stage_cfg, n)
        emit_progress(
            "Computing similarities",
            n,
            n,
            message={
                "stage": stage_name,
                "pairs_per_node": round(coverage["pairs_per_node"], 3),
                "coverage": round(coverage["node_coverage_ratio"], 3),
                "escalate": needs_escalation,
            },
            force_db=True,
        )
        final_candidate_mode = candidate_mode
        final_blocking_attrs = blocking_attrs if candidate_mode == "blocked" else []
        final_similarity_pairs = len(similarities)

        if needs_escalation and stage_idx < len(stage_plan) - 1:
            stage_summaries.append(
                {
                    "stage": stage_name,
                    "candidate_mode": candidate_mode,
                    "coverage": coverage,
                    "escalated_before_calibration": True,
                    "elapsed_s": round(time.time() - stage_started, 2),
                }
            )
            continue

        # Step 3: Assign communities with deterministic, scalable algorithm.
        n_communities = config.community_count
        if n_communities is None:
            # Keep communities coarse enough for stable modularity/clustering at scale.
            n_communities = max(5, int(n / 200))
        emit_progress(
            "Detecting communities",
            0,
            12,
            message={"stage": stage_name},
            force_db=True,
        )
        communities, community_diag = _assign_communities_with_diagnostics(
            agents=agents,
            similarities=similarities,
            n_communities=n_communities,
            rng=rng,
            preferred_attrs=blocking_attrs,
            progress=lambda _s, c, t: emit_progress(
                "Detecting communities",
                c,
                t,
                message={"stage": stage_name},
            ),
        )
        emit_progress(
            "Detecting communities",
            12,
            12,
            message={"stage": stage_name, **community_diag},
            force_db=True,
        )

        if community_diag.get("low_signal", 0.0) >= 1.0 and stage_idx < len(stage_plan) - 1:
            stage_summaries.append(
                {
                    "stage": stage_name,
                    "candidate_mode": candidate_mode,
                    "coverage": coverage,
                    "community_diagnostics": community_diag,
                    "escalated_before_calibration": True,
                    "elapsed_s": round(time.time() - stage_started, 2),
                }
            )
            continue

        # Step 4: Calibrate for this stage.
        stage_deadline = min(started_at + elapsed_budget, stage_started + stage_budget)
        stage_accepted = False
        stage_best_score = float("inf")
        stage_best_metrics: dict[str, float] | None = None

        for restart in range(config.calibration_restarts):
            if time.time() >= stage_deadline:
                break
            restart_seed = seed + stage_idx * 100003 + restart * 1000003
            intra_scale = 1.0
            inter_scale = max(0.1, float(stage_cfg.inter_community_scale))
            calibration_run_id: str | None = None
            if study_db_file is not None and network_run_id:
                with open_study_db(study_db_file) as db:
                    calibration_run_id = db.create_network_calibration_run(
                        network_run_id=network_run_id,
                        restart_index=(stage_idx * config.calibration_restarts) + restart,
                        seed=restart_seed,
                    )

            for iteration in range(config.max_calibration_iterations):
                if time.time() >= stage_deadline:
                    break
                calibration_step += 1
                emit_progress(
                    f"Calibration restart {restart + 1}/{config.calibration_restarts}",
                    calibration_step,
                    calibration_total,
                    message={
                        "stage": stage_name,
                        "restart": restart + 1,
                        "iteration": iteration + 1,
                        "best_score": round(best_score, 4)
                        if best_score != float("inf")
                        else None,
                    },
                )
                iter_rng = random.Random(restart_seed + iteration * 1000)
                edges, avg_degree, clustering, modularity, largest_component_ratio = (
                    _generate_network_single_pass(
                        agents,
                        agent_ids,
                        similarities,
                        communities,
                        degree_factors,
                        stage_cfg,
                        intra_scale,
                        inter_scale,
                        iter_rng,
                    )
                )
                lcc_deficit = max(
                    0.0, config.target_largest_component_ratio - largest_component_ratio
                )
                should_bridge_swap = modularity > (mod_max + 0.02) or (
                    lcc_deficit >= 0.03 and modularity > mod_min
                )
                if should_bridge_swap:
                    if modularity > (mod_max + 0.02):
                        swap_budget = max(4, int(len(edges) * config.bridge_budget_fraction))
                    else:
                        # Connectivity repair should be conservative so we do not
                        # collapse modular structure while bridging components.
                        ratio = min(config.bridge_budget_fraction * 0.25, lcc_deficit * 0.20)
                        swap_budget = max(1, int(len(edges) * ratio))
                    edges = _apply_bridge_swaps(
                        edges,
                        agent_ids,
                        communities,
                        stage_cfg,
                        iter_rng,
                        max_swaps_override=swap_budget,
                    )
                emit_progress(
                    "Repair pass",
                    iteration + 1,
                    config.max_calibration_iterations,
                    message={"stage": stage_name, "restart": restart + 1},
                )
                adjacency = _build_adjacency_from_edges(edges, agent_ids, n)
                clustering = _compute_avg_clustering(adjacency, n)
                modularity = _compute_modularity_fast(edges, agent_ids, communities)
                largest_component_ratio = _compute_largest_component_ratio(adjacency, n)
                edge_count = len(edges)

                score = 0.0
                if avg_degree < deg_min:
                    score += (deg_min - avg_degree) ** 2
                elif avg_degree > deg_max:
                    score += (avg_degree - deg_max) ** 2
                if clustering < cluster_min:
                    score += ((cluster_min - clustering) * 20.0) ** 2
                if modularity < mod_min:
                    score += ((mod_min - modularity) * 10.0) ** 2
                elif modularity > mod_max:
                    score += ((modularity - mod_max) * 10.0) ** 2
                if largest_component_ratio < config.target_largest_component_ratio:
                    score += (
                        (
                            config.target_largest_component_ratio
                            - largest_component_ratio
                        )
                        * 20.0
                    ) ** 2
                if edge_count < min_edge_floor:
                    score += (
                        ((min_edge_floor - edge_count) / max(1.0, min_edge_floor)) ** 2
                    ) * 400

                in_range = (
                    deg_min <= avg_degree <= deg_max
                    and clustering >= cluster_min
                    and mod_min <= modularity <= mod_max
                    and largest_component_ratio >= config.target_largest_component_ratio
                    and edge_count >= min_edge_floor
                )
                metrics_payload = {
                    "edge_count": edge_count,
                    "avg_degree": avg_degree,
                    "clustering": clustering,
                    "modularity": modularity,
                    "largest_component_ratio": largest_component_ratio,
                }

                if score < stage_best_score:
                    stage_best_score = score
                    stage_best_metrics = metrics_payload
                if score < best_score:
                    best_score = score
                    best_edges = edges
                    best_metrics = metrics_payload
                    best_stage = stage_name

                if calibration_run_id and study_db_file is not None:
                    with open_study_db(study_db_file) as db:
                        db.append_network_calibration_iteration(
                            calibration_run_id=calibration_run_id,
                            iteration=iteration,
                            phase=f"{stage_name}:calibration",
                            intra_scale=intra_scale,
                            inter_scale=inter_scale,
                            edge_count=edge_count,
                            avg_degree=avg_degree,
                            clustering=clustering,
                            modularity=modularity,
                            largest_component_ratio=largest_component_ratio,
                            score=score,
                            accepted=in_range,
                        )

                emit_progress(
                    "Gate validation",
                    iteration + 1,
                    config.max_calibration_iterations,
                    message={
                        "stage": stage_name,
                        "accepted": in_range,
                        "score": round(score, 4),
                        "avg_degree": round(avg_degree, 4),
                        "clustering": round(clustering, 4),
                        "modularity": round(modularity, 4),
                        "lcc": round(largest_component_ratio, 4),
                    },
                )

                if in_range:
                    stage_accepted = True
                    accepted = True
                    if calibration_run_id and study_db_file is not None:
                        with open_study_db(study_db_file) as db:
                            db.complete_network_calibration_run(
                                calibration_run_id=calibration_run_id,
                                status="accepted",
                                best_score=score,
                                best_metrics=metrics_payload,
                            )
                    break

                if avg_degree < deg_min:
                    adj = min(1.5, deg_min / max(0.1, avg_degree))
                    intra_scale *= adj
                    inter_scale *= adj
                elif avg_degree > deg_max:
                    adj = max(0.6, deg_max / max(0.1, avg_degree))
                    intra_scale *= adj
                    inter_scale *= adj

                # Only apply structure-shaping pushes when degree is not already above cap.
                # Otherwise we can get trapped in a high-degree attractor.
                degree_allows_growth = avg_degree <= deg_max
                if modularity > mod_max:
                    inter_scale *= 1.2
                    intra_scale *= 0.92
                elif modularity < mod_min:
                    if degree_allows_growth:
                        intra_scale *= 1.35
                        inter_scale *= 0.55
                    else:
                        # If degree is high, increase assortativity mostly by suppressing inter edges.
                        inter_scale *= 0.55
                        intra_scale *= 1.02
                if clustering < cluster_min and degree_allows_growth:
                    intra_scale *= 1.15
                intra_scale = max(0.25, min(6.5, intra_scale))
                inter_scale = max(0.25, min(6.5, inter_scale))

            if calibration_run_id and study_db_file is not None and not stage_accepted:
                with open_study_db(study_db_file) as db:
                    db.complete_network_calibration_run(
                        calibration_run_id=calibration_run_id,
                        status="completed",
                        best_score=stage_best_score
                        if stage_best_score != float("inf")
                        else None,
                        best_metrics=stage_best_metrics,
                    )
            if stage_accepted:
                break

        stage_summaries.append(
            {
                "stage": stage_name,
                "candidate_mode": candidate_mode,
                "coverage": coverage,
                "community_diagnostics": community_diag,
                "accepted": stage_accepted,
                "best_score": stage_best_score,
                "best_metrics": stage_best_metrics or {},
                "elapsed_s": round(time.time() - stage_started, 2),
                "budget_s": stage_budget,
            }
        )
        if stage_accepted:
            break

    emit_progress("Calibrating network", calibration_total, calibration_total, force_db=True)
    edges = best_edges or []

    # Rebuild edge_set from best edges
    edge_set: set[tuple[str, str]] = set()
    for edge in edges:
        edge_set.add((edge.source, edge.target))
        edge_set.add((edge.target, edge.source))

    # Step 5: Watts-Strogatz rewiring
    emit_progress("Rewiring edges", 0, len(edges), force_db=True)
    edges, edge_set, rewired_count = _apply_rewiring(
        agents, agent_ids, edges, edge_set, config, rng
    )
    emit_progress("Rewiring edges", len(edges), len(edges), force_db=True)

    quality_deltas = {
        "degree_to_min": (best_metrics or {}).get("avg_degree", 0.0) - deg_min,
        "degree_to_max": deg_max - (best_metrics or {}).get("avg_degree", 0.0),
        "clustering_to_min": (best_metrics or {}).get("clustering", 0.0) - cluster_min,
        "modularity_to_min": (best_metrics or {}).get("modularity", 0.0) - mod_min,
        "modularity_to_max": mod_max - (best_metrics or {}).get("modularity", 0.0),
        "lcc_to_min": (best_metrics or {}).get("largest_component_ratio", 0.0)
        - config.target_largest_component_ratio,
    }

    meta = {
        "agent_count": n,
        "edge_count": len(edges),
        "avg_degree": 2 * len(edges) / n if n > 0 else 0.0,
        "rewired_count": rewired_count,
        "algorithm": "adaptive_calibration",
        "seed": seed,
        "candidate_mode": final_candidate_mode,
        "similarity_pairs": final_similarity_pairs,
        "blocking_attributes": final_blocking_attrs,
        "resumed_from_checkpoint": resume_from_checkpoint,
        "config": {
            "avg_degree_target": config.avg_degree,
            "rewire_prob": config.rewire_prob,
            "target_clustering": config.target_clustering,
            "target_modularity": config.target_modularity,
            "quality_profile": config.quality_profile,
        },
        "quality": {
            "topology_gate": config.topology_gate,
            "accepted": accepted,
            "best_score": best_score,
            "best_metrics": best_metrics or {},
            "best_stage": best_stage,
            "gate_deltas": quality_deltas,
            "target_caps_applied": target_caps,
            "bounds": {
                "degree_min": deg_min,
                "degree_max": deg_max,
                "clustering_min": cluster_min,
                "modularity_min": mod_min,
                "modularity_max": mod_max,
                "largest_component_min": config.target_largest_component_ratio,
                "min_edge_floor": min_edge_floor,
            },
            "ladder_stages": stage_summaries,
        },
        "resume_calibration_requested": resume_calibration,
        "generated_at": datetime.now().isoformat(),
    }

    return NetworkResult(meta=meta, edges=edges)


def generate_network_with_metrics(
    agents: list[dict[str, Any]],
    config: NetworkConfig | None = None,
    on_progress: NetworkProgressCallback | None = None,
    checkpoint_path: Path | str | None = None,
    resume_from_checkpoint: bool = False,
    study_db_path: Path | str | None = None,
    network_run_id: str | None = None,
    resume_calibration: bool = False,
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
        study_db_path=study_db_path,
        network_run_id=network_run_id,
        resume_calibration=resume_calibration,
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
