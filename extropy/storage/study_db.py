"""Canonical study database storage for Extropy.

This module provides the schema and helper operations for ``study.db``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import AgentDBRecord, NetworkEdgeDBRecord, ChatMessagePayload


def _now_iso() -> str:
    return datetime.now().isoformat()


def _dumps(data: Any) -> str:
    return json.dumps(data, default=str)


class StudyDB:
    """SQLite-backed canonical study store."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._set_pragmas()
        self.init_schema()

    def _set_pragmas(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA temp_store = MEMORY")
        self.conn.commit()

    def init_schema(self) -> None:
        """Create canonical schema and indexes."""
        cursor = self.conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS study_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS population_specs (
                population_id TEXT PRIMARY KEY,
                spec_yaml TEXT NOT NULL,
                source_path TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sample_runs (
                sample_run_id TEXT PRIMARY KEY,
                population_id TEXT NOT NULL,
                seed INTEGER,
                count INTEGER,
                created_at TEXT NOT NULL,
                meta_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agents (
                population_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                attrs_json TEXT NOT NULL,
                sample_run_id TEXT NOT NULL,
                PRIMARY KEY (population_id, agent_id)
            );

            CREATE TABLE IF NOT EXISTS network_runs (
                network_run_id TEXT PRIMARY KEY,
                population_id TEXT NOT NULL,
                network_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                seed INTEGER,
                candidate_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                meta_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS network_edges (
                network_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL,
                edge_type TEXT NOT NULL,
                influence_st REAL,
                influence_ts REAL,
                PRIMARY KEY (network_id, source_id, target_id)
            );

            CREATE TABLE IF NOT EXISTS network_metrics (
                network_id TEXT PRIMARY KEY,
                metrics_json TEXT NOT NULL,
                computed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS network_similarity_jobs (
                job_id TEXT PRIMARY KEY,
                network_run_id TEXT NOT NULL,
                signature_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS network_similarity_chunks (
                job_id TEXT NOT NULL,
                chunk_start INTEGER NOT NULL,
                chunk_end INTEGER NOT NULL,
                status TEXT NOT NULL,
                pair_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (job_id, chunk_start)
            );

            CREATE TABLE IF NOT EXISTS network_similarity_pairs (
                job_id TEXT NOT NULL,
                i INTEGER NOT NULL,
                j INTEGER NOT NULL,
                sim REAL NOT NULL,
                PRIMARY KEY (job_id, i, j)
            ) WITHOUT ROWID;

            CREATE TABLE IF NOT EXISTS simulation_runs (
                run_id TEXT PRIMARY KEY,
                scenario_name TEXT,
                population_id TEXT NOT NULL,
                network_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                seed INTEGER,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                stopped_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_states (
                run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                aware INTEGER DEFAULT 0,
                exposure_count INTEGER DEFAULT 0,
                last_reasoning_timestep INTEGER DEFAULT -1,
                position TEXT,
                sentiment REAL,
                conviction REAL,
                public_statement TEXT,
                action_intent TEXT,
                will_share INTEGER DEFAULT 0,
                outcomes_json TEXT,
                public_position TEXT,
                public_sentiment REAL,
                public_conviction REAL,
                private_position TEXT,
                private_sentiment REAL,
                private_conviction REAL,
                private_outcomes_json TEXT,
                raw_reasoning TEXT,
                committed INTEGER DEFAULT 0,
                network_hop_depth INTEGER,
                updated_at INTEGER DEFAULT 0,
                PRIMARY KEY (run_id, agent_id)
            );

            CREATE TABLE IF NOT EXISTS exposures (
                run_id TEXT NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestep INTEGER,
                channel TEXT,
                source_agent_id TEXT,
                content TEXT,
                credibility REAL,
                FOREIGN KEY (run_id, agent_id) REFERENCES agent_states(run_id, agent_id)
            );

            CREATE TABLE IF NOT EXISTS memory_traces (
                run_id TEXT NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestep INTEGER,
                sentiment REAL,
                conviction REAL,
                summary TEXT,
                FOREIGN KEY (run_id, agent_id) REFERENCES agent_states(run_id, agent_id)
            );

            CREATE TABLE IF NOT EXISTS timeline (
                run_id TEXT NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestep INTEGER,
                event_type TEXT,
                agent_id TEXT,
                details_json TEXT,
                wall_timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS timestep_summaries (
                run_id TEXT NOT NULL,
                timestep INTEGER NOT NULL,
                new_exposures INTEGER,
                agents_reasoned INTEGER,
                shares_occurred INTEGER,
                state_changes INTEGER,
                exposure_rate REAL,
                position_distribution_json TEXT,
                average_sentiment REAL,
                average_conviction REAL,
                sentiment_variance REAL,
                PRIMARY KEY (run_id, timestep)
            );

            CREATE TABLE IF NOT EXISTS shared_to (
                run_id TEXT NOT NULL,
                source_agent_id TEXT,
                target_agent_id TEXT,
                timestep INTEGER,
                position TEXT,
                PRIMARY KEY (run_id, source_agent_id, target_agent_id)
            );

            CREATE TABLE IF NOT EXISTS simulation_metadata (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT
                ,
                PRIMARY KEY (run_id, key)
            );

            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (run_id, key)
            );

            CREATE TABLE IF NOT EXISTS simulation_checkpoints (
                run_id TEXT NOT NULL,
                timestep INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, timestep, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                closed_at TEXT,
                meta_json TEXT
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations_json TEXT,
                token_usage_json TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, turn_index)
            );

            CREATE TABLE IF NOT EXISTS chat_artifacts (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY (session_id, key)
            );

            CREATE INDEX IF NOT EXISTS idx_agents_population ON agents(population_id);
            CREATE INDEX IF NOT EXISTS idx_network_edges_src ON network_edges(network_id, source_id);
            CREATE INDEX IF NOT EXISTS idx_network_edges_tgt ON network_edges(network_id, target_id);
            CREATE INDEX IF NOT EXISTS idx_net_sim_chunks_status ON network_similarity_chunks(job_id, status);
            CREATE INDEX IF NOT EXISTS idx_sim_ckpt ON simulation_checkpoints(run_id, timestep, chunk_index);
            CREATE INDEX IF NOT EXISTS idx_chat_session_agent ON chat_sessions(run_id, agent_id);
            CREATE INDEX IF NOT EXISTS idx_agent_states_aware ON agent_states(run_id, aware);
            CREATE INDEX IF NOT EXISTS idx_agent_states_will_share ON agent_states(run_id, will_share);
            CREATE INDEX IF NOT EXISTS idx_agent_states_last_reasoning ON agent_states(run_id, last_reasoning_timestep);
            CREATE INDEX IF NOT EXISTS idx_agent_states_run_awws
            ON agent_states(run_id, aware, will_share, last_reasoning_timestep);
            CREATE INDEX IF NOT EXISTS idx_exposures_agent_timestep ON exposures(run_id, agent_id, timestep);
            CREATE INDEX IF NOT EXISTS idx_timeline_timestep ON timeline(run_id, timestep);
            CREATE INDEX IF NOT EXISTS idx_shared_to_source ON shared_to(run_id, source_agent_id);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "StudyDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def save_population_spec(
        self,
        population_id: str,
        spec_yaml: str,
        source_path: str | None,
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO population_specs (population_id, spec_yaml, source_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (population_id, spec_yaml, source_path, _now_iso()),
        )
        self.conn.commit()

    def get_population_spec_yaml(self, population_id: str) -> str | None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT spec_yaml FROM population_specs WHERE population_id = ?",
            (population_id,),
        )
        row = cursor.fetchone()
        return str(row["spec_yaml"]) if row else None

    def save_sample_result(
        self,
        population_id: str,
        agents: list[dict[str, Any]],
        meta: dict[str, Any],
        seed: int | None = None,
        sample_run_id: str | None = None,
    ) -> str:
        run_id = sample_run_id or str(uuid.uuid4())
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO sample_runs
            (sample_run_id, population_id, seed, count, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, population_id, seed, len(agents), _now_iso(), _dumps(meta)),
        )

        cursor.execute("DELETE FROM agents WHERE population_id = ?", (population_id,))

        rows = []
        for i, agent in enumerate(agents):
            agent_id = str(agent.get("_id", f"agent_{i}"))
            row_agent = dict(agent)
            row_agent["_id"] = agent_id
            rec = AgentDBRecord(
                population_id=population_id,
                agent_id=agent_id,
                attrs_json=row_agent,
                sample_run_id=run_id,
            )
            rows.append(
                (
                    rec.population_id,
                    rec.agent_id,
                    _dumps(rec.attrs_json),
                    rec.sample_run_id,
                )
            )

        cursor.executemany(
            """
            INSERT INTO agents (population_id, agent_id, attrs_json, sample_run_id)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return run_id

    def get_agents(self, population_id: str) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT attrs_json
            FROM agents
            WHERE population_id = ?
            ORDER BY agent_id
            """,
            (population_id,),
        )
        agents = []
        for row in cursor.fetchall():
            try:
                agents.append(json.loads(row["attrs_json"]))
            except json.JSONDecodeError:
                continue
        return agents

    def get_agent_count(self, population_id: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM agents WHERE population_id = ?",
            (population_id,),
        )
        row = cursor.fetchone()
        return int(row["cnt"]) if row else 0

    def save_network_result(
        self,
        population_id: str,
        network_id: str,
        config: dict[str, Any],
        result_meta: dict[str, Any],
        edges: list[dict[str, Any]],
        seed: int | None,
        candidate_mode: str,
        network_metrics: dict[str, Any] | None = None,
        network_run_id: str | None = None,
    ) -> str:
        run_id = network_run_id or str(uuid.uuid4())
        cursor = self.conn.cursor()
        now = _now_iso()

        cursor.execute(
            """
            INSERT OR REPLACE INTO network_runs
            (network_run_id, population_id, network_id, config_json, seed, candidate_mode,
             status, created_at, completed_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                population_id,
                network_id,
                _dumps(config),
                seed,
                candidate_mode,
                "completed",
                now,
                now,
                _dumps(result_meta),
            ),
        )

        cursor.execute("DELETE FROM network_edges WHERE network_id = ?", (network_id,))

        rows = []
        for edge in edges:
            infl = edge.get("influence_weight") or {}
            rec = NetworkEdgeDBRecord(
                network_id=network_id,
                source_id=str(edge.get("source", "")),
                target_id=str(edge.get("target", "")),
                weight=float(edge.get("weight", 0.0)),
                edge_type=str(edge.get("type", edge.get("edge_type", "unknown"))),
                influence_st=float(infl.get("source_to_target", edge.get("weight", 0.0))),
                influence_ts=float(infl.get("target_to_source", edge.get("weight", 0.0))),
            )
            rows.append(
                (
                    rec.network_id,
                    rec.source_id,
                    rec.target_id,
                    rec.weight,
                    rec.edge_type,
                    rec.influence_st,
                    rec.influence_ts,
                )
            )

        cursor.executemany(
            """
            INSERT INTO network_edges
            (network_id, source_id, target_id, weight, edge_type, influence_st, influence_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        if network_metrics is not None:
            cursor.execute(
                """
                INSERT OR REPLACE INTO network_metrics (network_id, metrics_json, computed_at)
                VALUES (?, ?, ?)
                """,
                (network_id, _dumps(network_metrics), now),
            )

        self.conn.commit()
        return run_id

    def get_network(self, network_id: str) -> dict[str, Any]:
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT meta_json FROM network_runs WHERE network_id = ? ORDER BY completed_at DESC LIMIT 1",
            (network_id,),
        )
        run_row = cursor.fetchone()
        meta = {}
        if run_row:
            try:
                meta = json.loads(run_row["meta_json"])
            except json.JSONDecodeError:
                meta = {}

        cursor.execute(
            """
            SELECT source_id, target_id, weight, edge_type, influence_st, influence_ts
            FROM network_edges
            WHERE network_id = ?
            ORDER BY source_id, target_id
            """,
            (network_id,),
        )
        edges = []
        for row in cursor.fetchall():
            edges.append(
                {
                    "source": row["source_id"],
                    "target": row["target_id"],
                    "weight": row["weight"],
                    "type": row["edge_type"],
                    "bidirectional": True,
                    "influence_weight": {
                        "source_to_target": row["influence_st"],
                        "target_to_source": row["influence_ts"],
                    },
                }
            )

        return {"meta": meta, "edges": edges}

    def get_network_edge_count(self, network_id: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM network_edges WHERE network_id = ?",
            (network_id,),
        )
        row = cursor.fetchone()
        return int(row["cnt"]) if row else 0

    def init_network_similarity_job(
        self,
        network_run_id: str,
        signature: dict[str, Any],
        job_id: str | None = None,
    ) -> str:
        job = job_id or str(uuid.uuid4())
        cursor = self.conn.cursor()
        now = _now_iso()
        cursor.execute(
            """
            INSERT OR REPLACE INTO network_similarity_jobs
            (job_id, network_run_id, signature_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job, network_run_id, _dumps(signature), "running", now, now),
        )
        self.conn.commit()
        return job

    def get_network_similarity_job_signature(self, job_id: str) -> dict[str, Any] | None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT signature_json FROM network_similarity_jobs WHERE job_id = ?",
            (job_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row["signature_json"])
        except json.JSONDecodeError:
            return None

    def get_completed_similarity_chunks(self, job_id: str) -> set[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT chunk_start
            FROM network_similarity_chunks
            WHERE job_id = ? AND status = 'done'
            """,
            (job_id,),
        )
        return {int(row["chunk_start"]) for row in cursor.fetchall()}

    def list_completed_similarity_chunks(self, job_id: str) -> list[tuple[int, int]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT chunk_start, chunk_end
            FROM network_similarity_chunks
            WHERE job_id = ? AND status = 'done'
            ORDER BY chunk_start
            """,
            (job_id,),
        )
        return [
            (int(row["chunk_start"]), int(row["chunk_end"])) for row in cursor.fetchall()
        ]

    def save_similarity_chunk_rows(
        self,
        job_id: str,
        chunk_start: int,
        chunk_end: int,
        rows: list[tuple[int, int, float]],
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO network_similarity_chunks
            (job_id, chunk_start, chunk_end, status, pair_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, chunk_start, chunk_end, "running", len(rows), _now_iso()),
        )
        if rows:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO network_similarity_pairs (job_id, i, j, sim)
                VALUES (?, ?, ?, ?)
                """,
                [(job_id, i, j, sim) for i, j, sim in rows],
            )
        cursor.execute(
            """
            UPDATE network_similarity_chunks
            SET status = 'done', updated_at = ?
            WHERE job_id = ? AND chunk_start = ?
            """,
            (_now_iso(), job_id, chunk_start),
        )
        cursor.execute(
            """
            UPDATE network_similarity_jobs
            SET updated_at = ?
            WHERE job_id = ?
            """,
            (_now_iso(), job_id),
        )
        self.conn.commit()

    def mark_similarity_job_running(self, job_id: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE network_similarity_jobs
            SET status = 'running', updated_at = ?
            WHERE job_id = ?
            """,
            (_now_iso(), job_id),
        )
        self.conn.commit()

    def load_similarity_pairs(self, job_id: str) -> dict[tuple[int, int], float]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT i, j, sim FROM network_similarity_pairs WHERE job_id = ?",
            (job_id,),
        )
        return {(int(row["i"]), int(row["j"])): float(row["sim"]) for row in cursor.fetchall()}

    def mark_similarity_job_complete(self, job_id: str, drop_pairs: bool = False) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE network_similarity_jobs
            SET status = 'completed', updated_at = ?
            WHERE job_id = ?
            """,
            (_now_iso(), job_id),
        )
        if drop_pairs:
            cursor.execute("DELETE FROM network_similarity_pairs WHERE job_id = ?", (job_id,))
        self.conn.commit()

    def create_simulation_run(
        self,
        run_id: str,
        scenario_name: str,
        population_id: str,
        network_id: str,
        config: dict[str, Any],
        seed: int | None,
        status: str = "running",
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO simulation_runs
            (run_id, scenario_name, population_id, network_id, config_json, seed, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, scenario_name, population_id, network_id, _dumps(config), seed, status, _now_iso()),
        )
        self.conn.commit()

    def update_simulation_run(
        self,
        run_id: str,
        status: str,
        stopped_reason: str | None = None,
    ) -> None:
        cursor = self.conn.cursor()
        completed_at = _now_iso() if status in {"completed", "failed", "stopped"} else None
        cursor.execute(
            """
            UPDATE simulation_runs
            SET status = ?, stopped_reason = ?, completed_at = COALESCE(?, completed_at)
            WHERE run_id = ?
            """,
            (status, stopped_reason, completed_at, run_id),
        )
        self.conn.commit()

    def set_run_metadata(self, run_id: str, key: str, value: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO run_metadata (run_id, key, value)
            VALUES (?, ?, ?)
            """,
            (run_id, key, value),
        )
        self.conn.commit()

    def get_run_metadata(self, run_id: str, key: str) -> str | None:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT value FROM run_metadata WHERE run_id = ? AND key = ?",
            (run_id, key),
        )
        row = cursor.fetchone()
        return str(row["value"]) if row else None

    def save_simulation_checkpoint(
        self,
        run_id: str,
        timestep: int,
        chunk_index: int,
        status: str,
    ) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO simulation_checkpoints
            (run_id, timestep, chunk_index, status, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, timestep, chunk_index, status, _now_iso()),
        )
        self.conn.commit()

    def get_completed_simulation_chunks(self, run_id: str, timestep: int) -> set[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT chunk_index
            FROM simulation_checkpoints
            WHERE run_id = ? AND timestep = ? AND status = 'done'
            """,
            (run_id, timestep),
        )
        return {int(row["chunk_index"]) for row in cursor.fetchall()}

    def create_chat_session(
        self,
        run_id: str,
        agent_id: str,
        mode: str,
        meta: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO chat_sessions
            (session_id, run_id, agent_id, mode, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sid, run_id, agent_id, mode, _now_iso(), _dumps(meta or {})),
        )
        self.conn.commit()
        return sid

    def append_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: dict[str, Any] | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> int:
        payload = ChatMessagePayload(
            role=role,
            content=content,
            citations=citations or {},
            token_usage=token_usage or {},
        )

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COALESCE(MAX(turn_index), -1) AS max_turn FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )
        turn = int(cursor.fetchone()["max_turn"]) + 1
        cursor.execute(
            """
            INSERT INTO chat_messages
            (session_id, turn_index, role, content, citations_json, token_usage_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn,
                payload.role,
                payload.content,
                _dumps(payload.citations),
                _dumps(payload.token_usage),
                _now_iso(),
            ),
        )
        self.conn.commit()
        return turn

    def get_chat_messages(self, session_id: str) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT turn_index, role, content, citations_json, token_usage_json, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY turn_index
            """,
            (session_id,),
        )
        out: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            out.append(
                {
                    "turn_index": int(row["turn_index"]),
                    "role": row["role"],
                    "content": row["content"],
                    "citations": json.loads(row["citations_json"] or "{}"),
                    "token_usage": json.loads(row["token_usage_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return out

    def run_select(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        sql = query.strip().rstrip(";")
        if limit is not None and " limit " not in sql.lower():
            sql = f"{sql} LIMIT {int(limit)}"
        cursor.execute(sql, params)
        cols = [d[0] for d in cursor.description or []]
        rows = []
        for row in cursor.fetchall():
            rows.append({k: row[idx] for idx, k in enumerate(cols)})
        return rows


def open_study_db(path: Path | str) -> StudyDB:
    """Open ``study.db`` and ensure schema exists."""
    return StudyDB(path)
