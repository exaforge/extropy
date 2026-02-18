"""SQLite-backed agent state management for simulation.

Provides scalable state storage that can handle large populations
without excessive memory usage. Includes support for conviction,
public statements, and memory traces.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..core.models import (
    AgentState,
    ExposureRecord,
    MemoryEntry,
    SimulationEvent,
    TimestepSummary,
)


class StateManager:
    """Manages agent state using SQLite for persistence and scalability.

    All state is stored in an SQLite database, with in-memory caching
    for frequently accessed data.
    """

    def __init__(
        self,
        db_path: Path | str,
        agents: list[dict[str, Any]] | None = None,
        run_id: str = "default",
    ):
        """Initialize state manager with database path.

        Args:
            db_path: Path to SQLite database file
            agents: Optional list of agents to initialize
            run_id: Simulation run scope key
        """
        self.db_path = Path(db_path)
        self.run_id = run_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

        self._create_schema()
        self._upgrade_schema()

        if agents:
            self.initialize_agents(agents)

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        cursor = self.conn.cursor()

        # Agent states table
        cursor.execute(
            """
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
                latest_info_epoch INTEGER DEFAULT -1,
                last_reasoned_info_epoch INTEGER DEFAULT -1,
                updated_at INTEGER DEFAULT 0,
                PRIMARY KEY (run_id, agent_id)
            )
        """
        )

        # Exposure history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exposures (
                run_id TEXT NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestep INTEGER,
                channel TEXT,
                source_agent_id TEXT,
                content TEXT,
                credibility REAL,
                info_epoch INTEGER,
                force_rereason INTEGER DEFAULT 0,
                FOREIGN KEY (run_id, agent_id) REFERENCES agent_states(run_id, agent_id)
            )
        """
        )

        # Memory traces table (max 3 per agent, managed by application)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_traces (
                run_id TEXT NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestep INTEGER,
                sentiment REAL,
                conviction REAL,
                summary TEXT,
                FOREIGN KEY (run_id, agent_id) REFERENCES agent_states(run_id, agent_id)
            )
        """
        )

        # Timeline events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline (
                run_id TEXT NOT NULL,
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestep INTEGER,
                event_type TEXT,
                agent_id TEXT,
                details_json TEXT,
                wall_timestamp TEXT
            )
        """
        )

        # Timestep summaries table
        cursor.execute(
            """
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
            )
        """
        )

        # Create indexes for common queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exposures_agent
            ON exposures(run_id, agent_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exposures_timestep
            ON exposures(run_id, timestep)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exposures_force_epoch
            ON exposures(run_id, agent_id, force_rereason, info_epoch)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timeline_timestep
            ON timeline(run_id, timestep)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_states_aware
            ON agent_states(run_id, aware)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_states_will_share
            ON agent_states(run_id, will_share)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_traces_agent
            ON memory_traces(run_id, agent_id)
        """
        )

        # Shared-to tracking table (one-shot sharing)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS shared_to (
                run_id TEXT NOT NULL,
                source_agent_id TEXT,
                target_agent_id TEXT,
                timestep INTEGER,
                position TEXT,
                PRIMARY KEY (run_id, source_agent_id, target_agent_id)
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_shared_to_source
            ON shared_to(run_id, source_agent_id)
        """
        )

        # Simulation metadata table (for checkpointing)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_metadata (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT
                ,
                PRIMARY KEY (run_id, key)
            )
        """
        )

        self.conn.commit()

    def _upgrade_schema(self) -> None:
        """Add columns that may be missing from older databases."""
        cursor = self.conn.cursor()

        migrations = [
            ("agent_states", "run_id", "TEXT DEFAULT 'default'"),
            ("exposures", "run_id", "TEXT DEFAULT 'default'"),
            ("memory_traces", "run_id", "TEXT DEFAULT 'default'"),
            ("timeline", "run_id", "TEXT DEFAULT 'default'"),
            ("timestep_summaries", "run_id", "TEXT DEFAULT 'default'"),
            ("shared_to", "run_id", "TEXT DEFAULT 'default'"),
            ("simulation_metadata", "run_id", "TEXT DEFAULT 'default'"),
            ("agent_states", "conviction", "REAL"),
            ("agent_states", "public_statement", "TEXT"),
            ("timestep_summaries", "average_conviction", "REAL"),
            ("timestep_summaries", "sentiment_variance", "REAL"),
            ("agent_states", "committed", "INTEGER DEFAULT 0"),
            ("agent_states", "network_hop_depth", "INTEGER"),
            ("agent_states", "public_position", "TEXT"),
            ("agent_states", "public_sentiment", "REAL"),
            ("agent_states", "public_conviction", "REAL"),
            ("agent_states", "private_position", "TEXT"),
            ("agent_states", "private_sentiment", "REAL"),
            ("agent_states", "private_conviction", "REAL"),
            ("agent_states", "private_outcomes_json", "TEXT"),
            ("agent_states", "latest_info_epoch", "INTEGER DEFAULT -1"),
            ("agent_states", "last_reasoned_info_epoch", "INTEGER DEFAULT -1"),
            ("exposures", "info_epoch", "INTEGER"),
            ("exposures", "force_rereason", "INTEGER DEFAULT 0"),
            ("memory_traces", "raw_reasoning", "TEXT"),
            ("memory_traces", "action_intent", "TEXT"),
        ]

        for table, column, col_type in migrations:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            except sqlite3.OperationalError:
                # Column already exists
                pass

        cursor.execute(
            "UPDATE agent_states SET run_id = COALESCE(run_id, 'default') WHERE run_id IS NULL"
        )
        cursor.execute(
            "UPDATE exposures SET run_id = COALESCE(run_id, 'default') WHERE run_id IS NULL"
        )
        cursor.execute(
            "UPDATE memory_traces SET run_id = COALESCE(run_id, 'default') WHERE run_id IS NULL"
        )
        cursor.execute(
            "UPDATE timeline SET run_id = COALESCE(run_id, 'default') WHERE run_id IS NULL"
        )
        cursor.execute(
            "UPDATE timestep_summaries SET run_id = COALESCE(run_id, 'default') WHERE run_id IS NULL"
        )
        cursor.execute(
            "UPDATE shared_to SET run_id = COALESCE(run_id, 'default') WHERE run_id IS NULL"
        )
        cursor.execute(
            "UPDATE simulation_metadata SET run_id = COALESCE(run_id, 'default') WHERE run_id IS NULL"
        )
        cursor.execute(
            "UPDATE agent_states SET latest_info_epoch = COALESCE(latest_info_epoch, -1)"
        )
        cursor.execute(
            "UPDATE agent_states SET last_reasoned_info_epoch = COALESCE(last_reasoned_info_epoch, -1)"
        )
        cursor.execute(
            "UPDATE exposures SET force_rereason = COALESCE(force_rereason, 0)"
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exposures_force_epoch
            ON exposures(run_id, agent_id, force_rereason, info_epoch)
        """
        )

        self.conn.commit()

    @contextmanager
    def transaction(self):
        """Context manager for batching writes into a single SQLite transaction.

        Commits on success, rolls back on exception.
        """
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def initialize_agents(self, agents: list[dict[str, Any]]) -> None:
        """Initialize state rows for all agents.

        Args:
            agents: List of agent dictionaries (must have _id field)
        """
        cursor = self.conn.cursor()

        for agent in agents:
            agent_id = agent.get("_id", str(agent.get("id", "")))
            cursor.execute(
                """
                INSERT OR IGNORE INTO agent_states (run_id, agent_id)
                VALUES (?, ?)
            """,
                (self.run_id, agent_id),
            )

        self.conn.commit()

    def get_agent_state(self, agent_id: str) -> AgentState:
        """Get full state for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            AgentState with all current state
        """
        cursor = self.conn.cursor()

        # Get main state
        cursor.execute(
            """
            SELECT * FROM agent_states WHERE run_id = ? AND agent_id = ?
        """,
            (self.run_id, agent_id),
        )
        row = cursor.fetchone()

        if not row:
            return AgentState(agent_id=agent_id)

        # Get exposure history
        cursor.execute(
            """
            SELECT * FROM exposures
            WHERE run_id = ? AND agent_id = ?
            ORDER BY timestep
        """,
            (self.run_id, agent_id),
        )
        exposure_rows = cursor.fetchall()

        exposures = [
            ExposureRecord(
                timestep=e["timestep"],
                channel=e["channel"],
                source_agent_id=e["source_agent_id"],
                content=e["content"],
                credibility=e["credibility"],
                info_epoch=e["info_epoch"],
                force_rereason=bool(e["force_rereason"]),
            )
            for e in exposure_rows
        ]

        # Parse outcomes JSON
        outcomes = {}
        if row["outcomes_json"]:
            try:
                outcomes = json.loads(row["outcomes_json"])
            except json.JSONDecodeError:
                pass

        private_outcomes = {}
        if row["private_outcomes_json"]:
            try:
                private_outcomes = json.loads(row["private_outcomes_json"])
            except json.JSONDecodeError:
                pass
        if not private_outcomes:
            private_outcomes = outcomes

        # committed column may not exist in older databases
        try:
            committed = bool(row["committed"])
        except (IndexError, KeyError):
            committed = False

        return AgentState(
            agent_id=agent_id,
            aware=bool(row["aware"]),
            exposure_count=row["exposure_count"],
            exposures=exposures,
            last_reasoning_timestep=row["last_reasoning_timestep"],
            position=row["private_position"] or row["position"],
            sentiment=row["private_sentiment"]
            if row["private_sentiment"] is not None
            else row["sentiment"],
            conviction=row["private_conviction"]
            if row["private_conviction"] is not None
            else row["conviction"],
            public_statement=row["public_statement"],
            action_intent=row["action_intent"],
            will_share=bool(row["will_share"]),
            public_position=row["public_position"] or row["position"],
            public_sentiment=(
                row["public_sentiment"]
                if row["public_sentiment"] is not None
                else row["sentiment"]
            ),
            public_conviction=(
                row["public_conviction"]
                if row["public_conviction"] is not None
                else row["conviction"]
            ),
            private_position=row["private_position"] or row["position"],
            private_sentiment=(
                row["private_sentiment"]
                if row["private_sentiment"] is not None
                else row["sentiment"]
            ),
            private_conviction=(
                row["private_conviction"]
                if row["private_conviction"] is not None
                else row["conviction"]
            ),
            private_outcomes=private_outcomes,
            committed=committed,
            outcomes=private_outcomes,
            raw_reasoning=row["raw_reasoning"],
            latest_info_epoch=(
                row["latest_info_epoch"] if row["latest_info_epoch"] is not None else -1
            ),
            last_reasoned_info_epoch=(
                row["last_reasoned_info_epoch"]
                if row["last_reasoned_info_epoch"] is not None
                else -1
            ),
            updated_at=row["updated_at"],
        )

    def get_unaware_agents(self) -> list[str]:
        """Get IDs of agents who haven't been exposed yet."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT agent_id FROM agent_states WHERE run_id = ? AND aware = 0",
            (self.run_id,),
        )
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_aware_agents(self) -> list[str]:
        """Get IDs of agents who are aware of the event."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT agent_id FROM agent_states WHERE run_id = ? AND aware = 1",
            (self.run_id,),
        )
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_sharers(self) -> list[str]:
        """Get IDs of agents who will share."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT agent_id FROM agent_states WHERE run_id = ? AND aware = 1 AND will_share = 1",
            (self.run_id,),
        )
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_all_agent_ids(self) -> list[str]:
        """Get all agent IDs in the database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT agent_id FROM agent_states WHERE run_id = ?", (self.run_id,)
        )
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_network_hop_depth(self, agent_id: str) -> int | None:
        """Get the minimum network hop depth from a seed exposure for an agent."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT network_hop_depth FROM agent_states WHERE run_id = ? AND agent_id = ?",
            (self.run_id, agent_id),
        )
        row = cursor.fetchone()
        if not row:
            return None
        value = row["network_hop_depth"]
        return int(value) if value is not None else None

    def get_agents_to_reason(
        self,
        timestep: int,
        threshold: int,
        forced_agent_ids: set[str] | None = None,
    ) -> list[str]:
        """Get agents who should reason this timestep.

        Agents reason if:
        1. They're aware AND haven't reasoned yet, OR
        2. They have >= threshold unique source agents exposing them since
           last reasoning AND they are not committed (conviction >= firm), OR
        3. They received forced re-reason exposures from a newer info epoch.

        Args:
            timestep: Current timestep
            threshold: Multi-touch threshold
            forced_agent_ids: Optional explicit agent IDs to include this timestep

        Returns:
            List of agent IDs that should reason
        """
        cursor = self.conn.cursor()

        # Agents who are aware but never reasoned
        cursor.execute(
            """
            SELECT agent_id FROM agent_states
            WHERE run_id = ? AND aware = 1 AND last_reasoning_timestep < 0
            ORDER BY agent_id
        """,
            (self.run_id,),
        )
        never_reasoned = [row["agent_id"] for row in cursor.fetchall()]

        # Forced re-reason: agents with new forced exposures since their last reasoned epoch.
        cursor.execute(
            """
            SELECT DISTINCT s.agent_id
            FROM agent_states s
            JOIN exposures e
              ON e.run_id = s.run_id
              AND e.agent_id = s.agent_id
              AND e.timestep > s.last_reasoning_timestep
              AND e.force_rereason = 1
              AND e.info_epoch IS NOT NULL
            WHERE s.run_id = ?
              AND s.aware = 1
              AND e.info_epoch > COALESCE(s.last_reasoned_info_epoch, -1)
            ORDER BY s.agent_id
        """,
            (self.run_id,),
        )
        forced_by_exposure = [row["agent_id"] for row in cursor.fetchall()]

        explicit_forced: list[str] = []
        if forced_agent_ids:
            placeholders = ",".join("?" for _ in forced_agent_ids)
            cursor.execute(
                f"""
                SELECT agent_id
                FROM agent_states
                WHERE run_id = ?
                  AND aware = 1
                  AND agent_id IN ({placeholders})
                ORDER BY agent_id
            """,
                [self.run_id, *sorted(forced_agent_ids)],
            )
            explicit_forced = [row["agent_id"] for row in cursor.fetchall()]

        # Multi-touch: count UNIQUE source agents since last reasoning,
        # excluding committed agents
        cursor.execute(
            """
            SELECT s.agent_id,
                   COUNT(DISTINCT e.source_agent_id) as unique_sources
            FROM agent_states s
            JOIN exposures e
              ON e.run_id = s.run_id
              AND e.agent_id = s.agent_id
              AND e.timestep > s.last_reasoning_timestep
              AND e.source_agent_id IS NOT NULL
            WHERE s.run_id = ?
              AND s.aware = 1
              AND s.last_reasoning_timestep >= 0
              AND s.committed = 0
            GROUP BY s.agent_id
            ORDER BY s.agent_id
        """,
            (self.run_id,),
        )

        multi_touch = []
        for row in cursor.fetchall():
            if row["unique_sources"] >= threshold:
                multi_touch.append(row["agent_id"])

        ordered = never_reasoned + forced_by_exposure + explicit_forced + multi_touch
        deduped: list[str] = []
        seen: set[str] = set()
        for agent_id in ordered:
            if agent_id in seen:
                continue
            seen.add(agent_id)
            deduped.append(agent_id)
        return deduped

    def record_share(
        self, source_id: str, target_id: str, timestep: int, position: str | None
    ) -> None:
        """Record that source_id has attempted to share with target_id.

        Uses INSERT OR REPLACE so a position change overwrites the old record.

        Args:
            source_id: Sharing agent ID
            target_id: Target neighbor ID
            timestep: When the share attempt occurred
            position: Source agent's current position
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO shared_to
            (run_id, source_agent_id, target_agent_id, timestep, position)
            VALUES (?, ?, ?, ?, ?)
        """,
            (self.run_id, source_id, target_id, timestep, position),
        )

    def get_unshared_neighbors(
        self, source_id: str, neighbor_ids: list[str], current_position: str | None
    ) -> list[str]:
        """Return neighbors that source has not yet shared to (or shared with a different position).

        Args:
            source_id: Sharing agent ID
            neighbor_ids: List of candidate neighbor IDs
            current_position: Source agent's current position

        Returns:
            List of neighbor IDs eligible for sharing
        """
        if not neighbor_ids:
            return []

        cursor = self.conn.cursor()
        placeholders = ",".join("?" for _ in neighbor_ids)
        cursor.execute(
            f"""
            SELECT target_agent_id, position
            FROM shared_to
            WHERE run_id = ?
              AND source_agent_id = ?
              AND target_agent_id IN ({placeholders})
        """,
            [self.run_id, source_id] + neighbor_ids,
        )

        already_shared = {
            row["target_agent_id"]: row["position"] for row in cursor.fetchall()
        }

        eligible = []
        for nid in neighbor_ids:
            if nid not in already_shared:
                eligible.append(nid)
            elif already_shared[nid] != current_position:
                # Position changed since last share — allow re-share
                eligible.append(nid)

        return eligible

    def save_metadata(self, key: str, value: str) -> None:
        """Save a metadata key-value pair with immediate commit.

        Args:
            key: Metadata key
            value: Metadata value
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO simulation_metadata (run_id, key, value) VALUES (?, ?, ?)",
            (self.run_id, key, value),
        )
        self.conn.commit()

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value by key.

        Args:
            key: Metadata key

        Returns:
            Value string or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT value FROM simulation_metadata WHERE run_id = ? AND key = ?",
            (self.run_id, key),
        )
        row = cursor.fetchone()
        return row["value"] if row else None

    def delete_metadata(self, key: str) -> None:
        """Delete a metadata key with immediate commit.

        Args:
            key: Metadata key to delete
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM simulation_metadata WHERE run_id = ? AND key = ?",
            (self.run_id, key),
        )
        self.conn.commit()

    def get_last_completed_timestep(self) -> int:
        """Get the last fully completed timestep.

        Returns:
            Max timestep from timestep_summaries, or -1 if none exist.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(timestep) as max_ts FROM timestep_summaries WHERE run_id = ?",
            (self.run_id,),
        )
        row = cursor.fetchone()
        if row and row["max_ts"] is not None:
            return row["max_ts"]
        return -1

    def get_checkpoint_timestep(self) -> int | None:
        """Get the timestep that was started but not completed (crash recovery).

        Returns:
            Timestep number or None if no checkpoint is set.
        """
        value = self.get_metadata("checkpoint_timestep")
        return int(value) if value is not None else None

    def mark_timestep_started(self, timestep: int) -> None:
        """Mark a timestep as started (for crash recovery).

        Args:
            timestep: Timestep being started
        """
        self.save_metadata("checkpoint_timestep", str(timestep))

    def mark_timestep_completed(self, timestep: int) -> None:
        """Mark a timestep as completed, clearing the checkpoint.

        Args:
            timestep: Timestep that completed
        """
        self.delete_metadata("checkpoint_timestep")

    def get_agents_already_reasoned_this_timestep(self, timestep: int) -> set[str]:
        """Get agents that already reasoned in this timestep (for resume).

        Args:
            timestep: Current timestep

        Returns:
            Set of agent IDs that have last_reasoning_timestep == timestep
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT agent_id FROM agent_states WHERE run_id = ? AND last_reasoning_timestep = ?",
            (self.run_id, timestep),
        )
        return {row["agent_id"] for row in cursor.fetchall()}

    def record_exposure(self, agent_id: str, exposure: ExposureRecord) -> None:
        """Record an exposure event for an agent.

        Args:
            agent_id: Agent ID
            exposure: Exposure record to add
        """
        cursor = self.conn.cursor()

        # Insert exposure record
        cursor.execute(
            """
            INSERT INTO exposures (
                run_id,
                agent_id,
                timestep,
                channel,
                source_agent_id,
                content,
                credibility,
                info_epoch,
                force_rereason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                self.run_id,
                agent_id,
                exposure.timestep,
                exposure.channel,
                exposure.source_agent_id,
                exposure.content,
                exposure.credibility,
                exposure.info_epoch,
                1 if exposure.force_rereason else 0,
            ),
        )

        # Estimate minimum hop depth (seed exposure = 0, network = source + 1).
        new_hop_depth: int | None
        if exposure.source_agent_id is None:
            new_hop_depth = 0
        else:
            source_hop = self.get_network_hop_depth(exposure.source_agent_id)
            new_hop_depth = 1 if source_hop is None else source_hop + 1

        # Update agent state
        cursor.execute(
            """
            UPDATE agent_states
            SET aware = 1,
                exposure_count = exposure_count + 1,
                network_hop_depth = CASE
                    WHEN network_hop_depth IS NULL THEN ?
                    WHEN ? IS NULL THEN network_hop_depth
                    ELSE MIN(network_hop_depth, ?)
                END,
                latest_info_epoch = CASE
                    WHEN ? IS NULL THEN COALESCE(latest_info_epoch, -1)
                    WHEN latest_info_epoch IS NULL THEN ?
                    ELSE MAX(latest_info_epoch, ?)
                END,
                updated_at = ?
            WHERE run_id = ?
              AND agent_id = ?
        """,
            (
                new_hop_depth,
                new_hop_depth,
                new_hop_depth,
                exposure.info_epoch,
                exposure.info_epoch,
                exposure.info_epoch,
                exposure.timestep,
                self.run_id,
                agent_id,
            ),
        )

    def apply_conviction_decay(
        self,
        timestep: int,
        decay_rate: float,
        sharing_threshold: float,
        firm_threshold: float,
    ) -> int:
        """Decay conviction for agents that did not reason this timestep.

        The decay only applies above sharing_threshold to avoid infinitesimal
        long-tail updates that prevent practical convergence.
        """
        if decay_rate <= 0:
            return 0

        decay_multiplier = 1.0 - decay_rate
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE agent_states
            SET conviction = conviction * ?,
                private_conviction = CASE
                    WHEN private_conviction IS NOT NULL THEN private_conviction * ?
                    ELSE NULL
                END,
                public_conviction = CASE
                    WHEN public_conviction IS NOT NULL THEN public_conviction * ?
                    ELSE NULL
                END,
                committed = CASE
                    WHEN conviction * ? >= ? THEN committed
                    ELSE 0
                END,
                will_share = CASE
                    WHEN COALESCE(public_conviction, conviction) * ? <= ? THEN 0
                    ELSE will_share
                END,
                updated_at = ?
            WHERE run_id = ?
              AND aware = 1
              AND conviction IS NOT NULL
              AND conviction > ?
              AND last_reasoning_timestep < ?
        """,
            (
                decay_multiplier,
                decay_multiplier,
                decay_multiplier,
                decay_multiplier,
                firm_threshold,
                decay_multiplier,
                sharing_threshold,
                timestep,
                self.run_id,
                sharing_threshold,
                timestep,
            ),
        )
        return cursor.rowcount

    def update_agent_state(
        self, agent_id: str, state: AgentState, timestep: int
    ) -> None:
        """Update agent state after reasoning.

        Args:
            agent_id: Agent ID
            state: New state values
            timestep: Current timestep
        """
        cursor = self.conn.cursor()

        private_outcomes = state.private_outcomes or state.outcomes
        outcomes_json = json.dumps(private_outcomes) if private_outcomes else None
        private_outcomes_json = (
            json.dumps(private_outcomes) if private_outcomes else None
        )

        cursor.execute(
            """
            UPDATE agent_states
            SET position = ?,
                sentiment = ?,
                conviction = ?,
                public_statement = ?,
                action_intent = ?,
                will_share = ?,
                public_position = ?,
                public_sentiment = ?,
                public_conviction = ?,
                private_position = ?,
                private_sentiment = ?,
                private_conviction = ?,
                committed = ?,
                outcomes_json = ?,
                private_outcomes_json = ?,
                raw_reasoning = ?,
                last_reasoning_timestep = ?,
                last_reasoned_info_epoch = CASE
                    WHEN latest_info_epoch IS NULL THEN COALESCE(last_reasoned_info_epoch, -1)
                    ELSE latest_info_epoch
                END,
                updated_at = ?
            WHERE run_id = ?
              AND agent_id = ?
        """,
            (
                state.position,
                state.sentiment,
                state.conviction,
                state.public_statement,
                state.action_intent,
                1 if state.will_share else 0,
                state.public_position,
                state.public_sentiment,
                state.public_conviction,
                state.private_position or state.position,
                state.private_sentiment
                if state.private_sentiment is not None
                else state.sentiment,
                state.private_conviction
                if state.private_conviction is not None
                else state.conviction,
                1 if state.committed else 0,
                outcomes_json,
                private_outcomes_json,
                state.raw_reasoning,
                timestep,
                timestep,
                self.run_id,
                agent_id,
            ),
        )

    def batch_update_states(
        self, updates: list[tuple[str, AgentState]], timestep: int
    ) -> None:
        """Batch update multiple agent states.

        Args:
            updates: List of (agent_id, state) tuples
            timestep: Current timestep
        """
        cursor = self.conn.cursor()

        for agent_id, state in updates:
            private_outcomes = state.private_outcomes or state.outcomes
            outcomes_json = json.dumps(private_outcomes) if private_outcomes else None
            private_outcomes_json = (
                json.dumps(private_outcomes) if private_outcomes else None
            )

            cursor.execute(
                """
                UPDATE agent_states
                SET position = ?,
                    sentiment = ?,
                    conviction = ?,
                    public_statement = ?,
                    action_intent = ?,
                    will_share = ?,
                    public_position = ?,
                    public_sentiment = ?,
                    public_conviction = ?,
                    private_position = ?,
                    private_sentiment = ?,
                    private_conviction = ?,
                    committed = ?,
                    outcomes_json = ?,
                    private_outcomes_json = ?,
                    raw_reasoning = ?,
                    last_reasoning_timestep = ?,
                    last_reasoned_info_epoch = CASE
                        WHEN latest_info_epoch IS NULL THEN COALESCE(last_reasoned_info_epoch, -1)
                        ELSE latest_info_epoch
                    END,
                    updated_at = ?
                WHERE run_id = ?
                  AND agent_id = ?
            """,
                (
                    state.position,
                    state.sentiment,
                    state.conviction,
                    state.public_statement,
                    state.action_intent,
                    1 if state.will_share else 0,
                    state.public_position,
                    state.public_sentiment,
                    state.public_conviction,
                    state.private_position or state.position,
                    state.private_sentiment
                    if state.private_sentiment is not None
                    else state.sentiment,
                    state.private_conviction
                    if state.private_conviction is not None
                    else state.conviction,
                    1 if state.committed else 0,
                    outcomes_json,
                    private_outcomes_json,
                    state.raw_reasoning,
                    timestep,
                    timestep,
                    self.run_id,
                    agent_id,
                ),
            )

    def save_memory_entry(self, agent_id: str, entry: MemoryEntry) -> None:
        """Save a memory trace entry for an agent.

        All entries are retained (no eviction cap).

        Args:
            agent_id: Agent ID
            entry: Memory entry to save
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO memory_traces
                (run_id, agent_id, timestep, sentiment, conviction, summary,
                 raw_reasoning, action_intent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                self.run_id,
                agent_id,
                entry.timestep,
                entry.sentiment,
                entry.conviction,
                entry.summary,
                entry.raw_reasoning,
                entry.action_intent,
            ),
        )

    def get_memory_traces(self, agent_id: str) -> list[MemoryEntry]:
        """Get all memory trace entries for an agent, oldest first.

        Args:
            agent_id: Agent ID

        Returns:
            List of MemoryEntry objects
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM memory_traces
            WHERE run_id = ? AND agent_id = ?
            ORDER BY timestep ASC
        """,
            (self.run_id, agent_id),
        )

        entries = []
        for row in cursor.fetchall():
            # raw_reasoning / action_intent columns may not exist in older DBs
            raw_reasoning = None
            action_intent = None
            try:
                raw_reasoning = row["raw_reasoning"]
            except (IndexError, KeyError):
                pass
            try:
                action_intent = row["action_intent"]
            except (IndexError, KeyError):
                pass
            entries.append(
                MemoryEntry(
                    timestep=row["timestep"],
                    sentiment=row["sentiment"],
                    conviction=row["conviction"],
                    summary=row["summary"],
                    raw_reasoning=raw_reasoning,
                    action_intent=action_intent,
                )
            )
        return entries

    def log_event(self, event: SimulationEvent) -> None:
        """Log a simulation event to the timeline.

        Args:
            event: Event to log
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO timeline (
                run_id,
                timestep,
                event_type,
                agent_id,
                details_json,
                wall_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                self.run_id,
                event.timestep,
                event.event_type.value,
                event.agent_id,
                json.dumps(event.details),
                event.timestamp.isoformat(),
            ),
        )

    def get_exposure_rate(self) -> float:
        """Get fraction of population that is aware."""
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) as total FROM agent_states WHERE run_id = ?",
            (self.run_id,),
        )
        total = cursor.fetchone()["total"]

        if total == 0:
            return 0.0

        cursor.execute(
            "SELECT COUNT(*) as aware FROM agent_states WHERE run_id = ? AND aware = 1",
            (self.run_id,),
        )
        aware = cursor.fetchone()["aware"]

        return aware / total

    def get_position_distribution(self) -> dict[str, int]:
        """Get count of agents per position."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT COALESCE(private_position, position) as position, COUNT(*) as cnt
            FROM agent_states
            WHERE run_id = ?
              AND COALESCE(private_position, position) IS NOT NULL
            GROUP BY COALESCE(private_position, position)
        """,
            (self.run_id,),
        )

        return {row["position"]: row["cnt"] for row in cursor.fetchall()}

    def get_average_sentiment(self) -> float | None:
        """Get average sentiment of aware agents."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT AVG(COALESCE(private_sentiment, sentiment)) as avg_sentiment
            FROM agent_states
            WHERE run_id = ?
              AND COALESCE(private_sentiment, sentiment) IS NOT NULL
        """,
            (self.run_id,),
        )
        row = cursor.fetchone()

        return row["avg_sentiment"] if row else None

    def get_average_conviction(self) -> float | None:
        """Get average conviction of aware agents."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT AVG(COALESCE(private_conviction, conviction)) as avg_conviction
            FROM agent_states
            WHERE run_id = ?
              AND COALESCE(private_conviction, conviction) IS NOT NULL
        """,
            (self.run_id,),
        )
        row = cursor.fetchone()

        return row["avg_conviction"] if row else None

    def get_sentiment_variance(self) -> float | None:
        """Get variance of sentiment across aware agents (for convergence detection)."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT AVG(COALESCE(private_sentiment, sentiment)) as mean_s, COUNT(*) as cnt
            FROM agent_states
            WHERE run_id = ?
              AND COALESCE(private_sentiment, sentiment) IS NOT NULL
        """,
            (self.run_id,),
        )
        row = cursor.fetchone()

        if not row or row["cnt"] < 2:
            return None

        mean = row["mean_s"]
        cursor.execute(
            """
            SELECT AVG(
                (COALESCE(private_sentiment, sentiment) - ?)
                * (COALESCE(private_sentiment, sentiment) - ?)
            ) as variance
            FROM agent_states
            WHERE run_id = ?
              AND COALESCE(private_sentiment, sentiment) IS NOT NULL
        """,
            (mean, mean, self.run_id),
        )
        var_row = cursor.fetchone()
        return var_row["variance"] if var_row else None

    def save_timestep_summary(self, summary: TimestepSummary) -> None:
        """Save a timestep summary.

        Args:
            summary: Timestep summary to save
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO timestep_summaries
            (run_id, timestep, new_exposures, agents_reasoned, shares_occurred,
             state_changes, exposure_rate, position_distribution_json, average_sentiment,
             average_conviction, sentiment_variance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                self.run_id,
                summary.timestep,
                summary.new_exposures,
                summary.agents_reasoned,
                summary.shares_occurred,
                summary.state_changes,
                summary.exposure_rate,
                json.dumps(summary.position_distribution),
                summary.average_sentiment,
                summary.average_conviction,
                summary.sentiment_variance,
            ),
        )

    def get_timestep_summaries(self) -> list[TimestepSummary]:
        """Get all timestep summaries."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT * FROM timestep_summaries
            WHERE run_id = ?
            ORDER BY timestep
        """,
            (self.run_id,),
        )

        summaries = []
        for row in cursor.fetchall():
            position_dist = {}
            if row["position_distribution_json"]:
                try:
                    position_dist = json.loads(row["position_distribution_json"])
                except json.JSONDecodeError:
                    pass

            summaries.append(
                TimestepSummary(
                    timestep=row["timestep"],
                    new_exposures=row["new_exposures"],
                    agents_reasoned=row["agents_reasoned"],
                    shares_occurred=row["shares_occurred"],
                    state_changes=row["state_changes"],
                    exposure_rate=row["exposure_rate"],
                    position_distribution=position_dist,
                    average_sentiment=row["average_sentiment"],
                    average_conviction=row["average_conviction"],
                    sentiment_variance=row["sentiment_variance"],
                )
            )

        return summaries

    def export_final_states(self) -> list[dict[str, Any]]:
        """Export all final agent states as dictionaries.

        Returns:
            List of agent state dictionaries
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM agent_states WHERE run_id = ?", (self.run_id,))
        agent_rows = cursor.fetchall()
        states = []

        cursor.execute(
            """
            SELECT agent_id, COUNT(*) as cnt
            FROM exposures
            WHERE run_id = ?
            GROUP BY agent_id
        """,
            (self.run_id,),
        )
        exposure_counts = {row["agent_id"]: row["cnt"] for row in cursor.fetchall()}

        for row in agent_rows:
            exposure_count = exposure_counts.get(row["agent_id"], 0)

            outcomes = {}
            if row["outcomes_json"]:
                try:
                    outcomes = json.loads(row["outcomes_json"])
                except json.JSONDecodeError:
                    pass

            private_outcomes = {}
            if row["private_outcomes_json"]:
                try:
                    private_outcomes = json.loads(row["private_outcomes_json"])
                except json.JSONDecodeError:
                    pass
            if not private_outcomes:
                private_outcomes = outcomes

            states.append(
                {
                    "agent_id": row["agent_id"],
                    "aware": bool(row["aware"]),
                    "exposure_count": exposure_count,
                    "last_reasoning_timestep": row["last_reasoning_timestep"],
                    "position": row["private_position"] or row["position"],
                    "sentiment": (
                        row["private_sentiment"]
                        if row["private_sentiment"] is not None
                        else row["sentiment"]
                    ),
                    "conviction": (
                        row["private_conviction"]
                        if row["private_conviction"] is not None
                        else row["conviction"]
                    ),
                    "public_position": row["public_position"] or row["position"],
                    "public_sentiment": (
                        row["public_sentiment"]
                        if row["public_sentiment"] is not None
                        else row["sentiment"]
                    ),
                    "public_conviction": (
                        row["public_conviction"]
                        if row["public_conviction"] is not None
                        else row["conviction"]
                    ),
                    "private_position": row["private_position"] or row["position"],
                    "private_sentiment": (
                        row["private_sentiment"]
                        if row["private_sentiment"] is not None
                        else row["sentiment"]
                    ),
                    "private_conviction": (
                        row["private_conviction"]
                        if row["private_conviction"] is not None
                        else row["conviction"]
                    ),
                    "public_statement": row["public_statement"],
                    "action_intent": row["action_intent"],
                    "will_share": bool(row["will_share"]),
                    "outcomes": private_outcomes,
                    "private_outcomes": private_outcomes,
                    "raw_reasoning": row["raw_reasoning"],
                }
            )

        return states

    def export_timeline(self) -> list[dict[str, Any]]:
        """Export all timeline events as dictionaries.

        Returns:
            List of event dictionaries
        """
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT * FROM timeline WHERE run_id = ? ORDER BY timestep, id",
            (self.run_id,),
        )

        events = []
        for row in cursor.fetchall():
            details = {}
            if row["details_json"]:
                try:
                    details = json.loads(row["details_json"])
                except json.JSONDecodeError:
                    pass

            events.append(
                {
                    "timestep": row["timestep"],
                    "event_type": row["event_type"],
                    "agent_id": row["agent_id"],
                    "details": details,
                    "timestamp": row["wall_timestamp"],
                }
            )

        return events

    def get_population_count(self) -> int:
        """Get total number of agents."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM agent_states WHERE run_id = ?",
            (self.run_id,),
        )
        return cursor.fetchone()["cnt"]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
