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

    def __init__(self, db_path: Path | str, agents: list[dict[str, Any]] | None = None):
        """Initialize state manager with database path.

        Args:
            db_path: Path to SQLite database file
            agents: Optional list of agents to initialize
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
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
                agent_id TEXT PRIMARY KEY,
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
                raw_reasoning TEXT,
                updated_at INTEGER DEFAULT 0
            )
        """
        )

        # Exposure history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS exposures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestep INTEGER,
                channel TEXT,
                source_agent_id TEXT,
                content TEXT,
                credibility REAL,
                FOREIGN KEY (agent_id) REFERENCES agent_states(agent_id)
            )
        """
        )

        # Memory traces table (max 3 per agent, managed by application)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestep INTEGER,
                sentiment REAL,
                conviction REAL,
                summary TEXT,
                FOREIGN KEY (agent_id) REFERENCES agent_states(agent_id)
            )
        """
        )

        # Timeline events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS timeline (
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
                timestep INTEGER PRIMARY KEY,
                new_exposures INTEGER,
                agents_reasoned INTEGER,
                shares_occurred INTEGER,
                state_changes INTEGER,
                exposure_rate REAL,
                position_distribution_json TEXT,
                average_sentiment REAL,
                average_conviction REAL,
                sentiment_variance REAL
            )
        """
        )

        # Create indexes for common queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exposures_agent
            ON exposures(agent_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exposures_timestep
            ON exposures(timestep)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timeline_timestep
            ON timeline(timestep)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_states_aware
            ON agent_states(aware)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_states_will_share
            ON agent_states(will_share)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_traces_agent
            ON memory_traces(agent_id)
        """
        )

        # Shared-to tracking table (one-shot sharing)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS shared_to (
                source_agent_id TEXT,
                target_agent_id TEXT,
                timestep INTEGER,
                position TEXT,
                PRIMARY KEY (source_agent_id, target_agent_id)
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_shared_to_source
            ON shared_to(source_agent_id)
        """
        )

        # Simulation metadata table (for checkpointing)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        self.conn.commit()

    def _upgrade_schema(self) -> None:
        """Add columns that may be missing from older databases."""
        cursor = self.conn.cursor()

        migrations = [
            ("agent_states", "conviction", "REAL"),
            ("agent_states", "public_statement", "TEXT"),
            ("timestep_summaries", "average_conviction", "REAL"),
            ("timestep_summaries", "sentiment_variance", "REAL"),
            ("agent_states", "committed", "INTEGER DEFAULT 0"),
        ]

        for table, column, col_type in migrations:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            except sqlite3.OperationalError:
                # Column already exists
                pass

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
                INSERT OR IGNORE INTO agent_states (agent_id)
                VALUES (?)
            """,
                (agent_id,),
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
            SELECT * FROM agent_states WHERE agent_id = ?
        """,
            (agent_id,),
        )
        row = cursor.fetchone()

        if not row:
            return AgentState(agent_id=agent_id)

        # Get exposure history
        cursor.execute(
            """
            SELECT * FROM exposures WHERE agent_id = ? ORDER BY timestep
        """,
            (agent_id,),
        )
        exposure_rows = cursor.fetchall()

        exposures = [
            ExposureRecord(
                timestep=e["timestep"],
                channel=e["channel"],
                source_agent_id=e["source_agent_id"],
                content=e["content"],
                credibility=e["credibility"],
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
            position=row["position"],
            sentiment=row["sentiment"],
            conviction=row["conviction"],
            public_statement=row["public_statement"],
            action_intent=row["action_intent"],
            will_share=bool(row["will_share"]),
            committed=committed,
            outcomes=outcomes,
            raw_reasoning=row["raw_reasoning"],
            updated_at=row["updated_at"],
        )

    def get_unaware_agents(self) -> list[str]:
        """Get IDs of agents who haven't been exposed yet."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT agent_id FROM agent_states WHERE aware = 0")
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_aware_agents(self) -> list[str]:
        """Get IDs of agents who are aware of the event."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT agent_id FROM agent_states WHERE aware = 1")
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_sharers(self) -> list[str]:
        """Get IDs of agents who will share."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT agent_id FROM agent_states WHERE aware = 1 AND will_share = 1"
        )
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_all_agent_ids(self) -> list[str]:
        """Get all agent IDs in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT agent_id FROM agent_states")
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_agents_to_reason(self, timestep: int, threshold: int) -> list[str]:
        """Get agents who should reason this timestep.

        Agents reason if:
        1. They're aware AND haven't reasoned yet, OR
        2. They have >= threshold unique source agents exposing them since
           last reasoning AND they are not committed (conviction >= firm)

        Args:
            timestep: Current timestep
            threshold: Multi-touch threshold

        Returns:
            List of agent IDs that should reason
        """
        cursor = self.conn.cursor()

        # Agents who are aware but never reasoned
        cursor.execute(
            """
            SELECT agent_id FROM agent_states
            WHERE aware = 1 AND last_reasoning_timestep < 0
        """
        )
        never_reasoned = [row["agent_id"] for row in cursor.fetchall()]

        # Multi-touch: count UNIQUE source agents since last reasoning,
        # excluding committed agents
        cursor.execute(
            """
            SELECT s.agent_id,
                   COUNT(DISTINCT e.source_agent_id) as unique_sources
            FROM agent_states s
            JOIN exposures e
              ON e.agent_id = s.agent_id
              AND e.timestep > s.last_reasoning_timestep
              AND e.source_agent_id IS NOT NULL
            WHERE s.aware = 1
              AND s.last_reasoning_timestep >= 0
              AND s.committed = 0
            GROUP BY s.agent_id
        """
        )

        multi_touch = []
        for row in cursor.fetchall():
            if row["unique_sources"] >= threshold:
                multi_touch.append(row["agent_id"])

        return never_reasoned + multi_touch

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
            (source_agent_id, target_agent_id, timestep, position)
            VALUES (?, ?, ?, ?)
        """,
            (source_id, target_id, timestep, position),
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
            WHERE source_agent_id = ?
              AND target_agent_id IN ({placeholders})
        """,
            [source_id] + neighbor_ids,
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
            "INSERT OR REPLACE INTO simulation_metadata (key, value) VALUES (?, ?)",
            (key, value),
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
        cursor.execute("SELECT value FROM simulation_metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None

    def delete_metadata(self, key: str) -> None:
        """Delete a metadata key with immediate commit.

        Args:
            key: Metadata key to delete
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM simulation_metadata WHERE key = ?", (key,))
        self.conn.commit()

    def get_last_completed_timestep(self) -> int:
        """Get the last fully completed timestep.

        Returns:
            Max timestep from timestep_summaries, or -1 if none exist.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(timestep) as max_ts FROM timestep_summaries")
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
            "SELECT agent_id FROM agent_states WHERE last_reasoning_timestep = ?",
            (timestep,),
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
            INSERT INTO exposures (agent_id, timestep, channel, source_agent_id, content, credibility)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                agent_id,
                exposure.timestep,
                exposure.channel,
                exposure.source_agent_id,
                exposure.content,
                exposure.credibility,
            ),
        )

        # Update agent state
        cursor.execute(
            """
            UPDATE agent_states
            SET aware = 1, exposure_count = exposure_count + 1, updated_at = ?
            WHERE agent_id = ?
        """,
            (exposure.timestep, agent_id),
        )

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

        outcomes_json = json.dumps(state.outcomes) if state.outcomes else None

        cursor.execute(
            """
            UPDATE agent_states
            SET position = ?,
                sentiment = ?,
                conviction = ?,
                public_statement = ?,
                action_intent = ?,
                will_share = ?,
                committed = ?,
                outcomes_json = ?,
                raw_reasoning = ?,
                last_reasoning_timestep = ?,
                updated_at = ?
            WHERE agent_id = ?
        """,
            (
                state.position,
                state.sentiment,
                state.conviction,
                state.public_statement,
                state.action_intent,
                1 if state.will_share else 0,
                1 if state.committed else 0,
                outcomes_json,
                state.raw_reasoning,
                timestep,
                timestep,
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
            outcomes_json = json.dumps(state.outcomes) if state.outcomes else None

            cursor.execute(
                """
                UPDATE agent_states
                SET position = ?,
                    sentiment = ?,
                    conviction = ?,
                    public_statement = ?,
                    action_intent = ?,
                    will_share = ?,
                    committed = ?,
                    outcomes_json = ?,
                    raw_reasoning = ?,
                    last_reasoning_timestep = ?,
                    updated_at = ?
                WHERE agent_id = ?
            """,
                (
                    state.position,
                    state.sentiment,
                    state.conviction,
                    state.public_statement,
                    state.action_intent,
                    1 if state.will_share else 0,
                    1 if state.committed else 0,
                    outcomes_json,
                    state.raw_reasoning,
                    timestep,
                    timestep,
                    agent_id,
                ),
            )

    def save_memory_entry(self, agent_id: str, entry: MemoryEntry) -> None:
        """Save a memory trace entry for an agent.

        Maintains a sliding window of max 3 entries per agent.
        Oldest entries are evicted when the limit is reached.

        Args:
            agent_id: Agent ID
            entry: Memory entry to save
        """
        cursor = self.conn.cursor()

        # Insert new entry
        cursor.execute(
            """
            INSERT INTO memory_traces (agent_id, timestep, sentiment, conviction, summary)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                agent_id,
                entry.timestep,
                entry.sentiment,
                entry.conviction,
                entry.summary,
            ),
        )

        # Evict oldest entries beyond 3
        cursor.execute(
            """
            DELETE FROM memory_traces
            WHERE id NOT IN (
                SELECT id FROM memory_traces
                WHERE agent_id = ?
                ORDER BY timestep DESC
                LIMIT 3
            ) AND agent_id = ?
        """,
            (agent_id, agent_id),
        )

    def get_memory_traces(self, agent_id: str) -> list[MemoryEntry]:
        """Get memory trace entries for an agent (max 3, oldest first).

        Args:
            agent_id: Agent ID

        Returns:
            List of MemoryEntry objects
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM memory_traces
            WHERE agent_id = ?
            ORDER BY timestep ASC
        """,
            (agent_id,),
        )

        return [
            MemoryEntry(
                timestep=row["timestep"],
                sentiment=row["sentiment"],
                conviction=row["conviction"],
                summary=row["summary"],
            )
            for row in cursor.fetchall()
        ]

    def log_event(self, event: SimulationEvent) -> None:
        """Log a simulation event to the timeline.

        Args:
            event: Event to log
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO timeline (timestep, event_type, agent_id, details_json, wall_timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
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

        cursor.execute("SELECT COUNT(*) as total FROM agent_states")
        total = cursor.fetchone()["total"]

        if total == 0:
            return 0.0

        cursor.execute("SELECT COUNT(*) as aware FROM agent_states WHERE aware = 1")
        aware = cursor.fetchone()["aware"]

        return aware / total

    def get_position_distribution(self) -> dict[str, int]:
        """Get count of agents per position."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT position, COUNT(*) as cnt
            FROM agent_states
            WHERE position IS NOT NULL
            GROUP BY position
        """
        )

        return {row["position"]: row["cnt"] for row in cursor.fetchall()}

    def get_average_sentiment(self) -> float | None:
        """Get average sentiment of aware agents."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT AVG(sentiment) as avg_sentiment
            FROM agent_states
            WHERE sentiment IS NOT NULL
        """
        )
        row = cursor.fetchone()

        return row["avg_sentiment"] if row else None

    def get_average_conviction(self) -> float | None:
        """Get average conviction of aware agents."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT AVG(conviction) as avg_conviction
            FROM agent_states
            WHERE conviction IS NOT NULL
        """
        )
        row = cursor.fetchone()

        return row["avg_conviction"] if row else None

    def get_sentiment_variance(self) -> float | None:
        """Get variance of sentiment across aware agents (for convergence detection)."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT AVG(sentiment) as mean_s, COUNT(*) as cnt
            FROM agent_states
            WHERE sentiment IS NOT NULL
        """
        )
        row = cursor.fetchone()

        if not row or row["cnt"] < 2:
            return None

        mean = row["mean_s"]
        cursor.execute(
            """
            SELECT AVG((sentiment - ?) * (sentiment - ?)) as variance
            FROM agent_states
            WHERE sentiment IS NOT NULL
        """,
            (mean, mean),
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
            (timestep, new_exposures, agents_reasoned, shares_occurred,
             state_changes, exposure_rate, position_distribution_json, average_sentiment,
             average_conviction, sentiment_variance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
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
            SELECT * FROM timestep_summaries ORDER BY timestep
        """
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

        cursor.execute("SELECT * FROM agent_states")
        agent_rows = cursor.fetchall()
        states = []

        cursor.execute(
            """
            SELECT agent_id, COUNT(*) as cnt
            FROM exposures
            GROUP BY agent_id
        """
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

            states.append(
                {
                    "agent_id": row["agent_id"],
                    "aware": bool(row["aware"]),
                    "exposure_count": exposure_count,
                    "last_reasoning_timestep": row["last_reasoning_timestep"],
                    "position": row["position"],
                    "sentiment": row["sentiment"],
                    "conviction": row["conviction"],
                    "public_statement": row["public_statement"],
                    "action_intent": row["action_intent"],
                    "will_share": bool(row["will_share"]),
                    "outcomes": outcomes,
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

        cursor.execute("SELECT * FROM timeline ORDER BY timestep, id")

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
        cursor.execute("SELECT COUNT(*) as cnt FROM agent_states")
        return cursor.fetchone()["cnt"]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
