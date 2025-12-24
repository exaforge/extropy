"""SQLite-backed agent state management for simulation.

Provides scalable state storage that can handle large populations
without excessive memory usage.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.models import (
    AgentState,
    ExposureRecord,
    SimulationEvent,
    SimulationEventType,
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

        self._create_schema()

        if agents:
            self.initialize_agents(agents)

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        cursor = self.conn.cursor()

        # Agent states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT PRIMARY KEY,
                aware INTEGER DEFAULT 0,
                exposure_count INTEGER DEFAULT 0,
                last_reasoning_timestep INTEGER DEFAULT -1,
                position TEXT,
                sentiment REAL,
                action_intent TEXT,
                will_share INTEGER DEFAULT 0,
                outcomes_json TEXT,
                raw_reasoning TEXT,
                updated_at INTEGER DEFAULT 0
            )
        """)

        # Exposure history table
        cursor.execute("""
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
        """)

        # Timeline events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestep INTEGER,
                event_type TEXT,
                agent_id TEXT,
                details_json TEXT,
                wall_timestamp TEXT
            )
        """)

        # Timestep summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timestep_summaries (
                timestep INTEGER PRIMARY KEY,
                new_exposures INTEGER,
                agents_reasoned INTEGER,
                shares_occurred INTEGER,
                state_changes INTEGER,
                exposure_rate REAL,
                position_distribution_json TEXT,
                average_sentiment REAL
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exposures_agent
            ON exposures(agent_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_exposures_timestep
            ON exposures(timestep)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timeline_timestep
            ON timeline(timestep)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_states_aware
            ON agent_states(aware)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_states_will_share
            ON agent_states(will_share)
        """)

        self.conn.commit()

    def initialize_agents(self, agents: list[dict[str, Any]]) -> None:
        """Initialize state rows for all agents.

        Args:
            agents: List of agent dictionaries (must have _id field)
        """
        cursor = self.conn.cursor()

        for agent in agents:
            agent_id = agent.get("_id", str(agent.get("id", "")))
            cursor.execute("""
                INSERT OR IGNORE INTO agent_states (agent_id)
                VALUES (?)
            """, (agent_id,))

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
        cursor.execute("""
            SELECT * FROM agent_states WHERE agent_id = ?
        """, (agent_id,))
        row = cursor.fetchone()

        if not row:
            return AgentState(agent_id=agent_id)

        # Get exposure history
        cursor.execute("""
            SELECT * FROM exposures WHERE agent_id = ? ORDER BY timestep
        """, (agent_id,))
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

        return AgentState(
            agent_id=agent_id,
            aware=bool(row["aware"]),
            exposure_count=row["exposure_count"],
            exposures=exposures,
            last_reasoning_timestep=row["last_reasoning_timestep"],
            position=row["position"],
            sentiment=row["sentiment"],
            action_intent=row["action_intent"],
            will_share=bool(row["will_share"]),
            outcomes=outcomes,
            raw_reasoning=row["raw_reasoning"],
            updated_at=row["updated_at"],
        )

    def get_agents_by_condition(self, condition: str) -> list[str]:
        """Get agent IDs matching a SQL condition.

        Args:
            condition: SQL WHERE clause (e.g., "aware = 1 AND will_share = 1")

        Returns:
            List of matching agent IDs
        """
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT agent_id FROM agent_states WHERE {condition}
        """)
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_unaware_agents(self) -> list[str]:
        """Get IDs of agents who haven't been exposed yet."""
        return self.get_agents_by_condition("aware = 0")

    def get_aware_agents(self) -> list[str]:
        """Get IDs of agents who are aware of the event."""
        return self.get_agents_by_condition("aware = 1")

    def get_sharers(self) -> list[str]:
        """Get IDs of agents who will share."""
        return self.get_agents_by_condition("aware = 1 AND will_share = 1")

    def get_all_agent_ids(self) -> list[str]:
        """Get all agent IDs in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT agent_id FROM agent_states")
        return [row["agent_id"] for row in cursor.fetchall()]

    def get_agents_to_reason(self, timestep: int, threshold: int) -> list[str]:
        """Get agents who should reason this timestep.

        Agents reason if:
        1. They're aware AND haven't reasoned yet, OR
        2. Their exposure_count increased by >= threshold since last reasoning

        Args:
            timestep: Current timestep
            threshold: Multi-touch threshold

        Returns:
            List of agent IDs that should reason
        """
        cursor = self.conn.cursor()

        # Agents who are aware but never reasoned
        cursor.execute("""
            SELECT agent_id FROM agent_states
            WHERE aware = 1 AND last_reasoning_timestep < 0
        """)
        never_reasoned = [row["agent_id"] for row in cursor.fetchall()]

        # For multi-touch, we need to check exposure counts
        # Get agents who have had exposures since their last reasoning
        cursor.execute("""
            SELECT agent_id, exposure_count, last_reasoning_timestep
            FROM agent_states
            WHERE aware = 1 AND last_reasoning_timestep >= 0
        """)

        multi_touch = []
        for row in cursor.fetchall():
            agent_id = row["agent_id"]
            current_count = row["exposure_count"]
            last_timestep = row["last_reasoning_timestep"]

            # Count exposures at time of last reasoning
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM exposures
                WHERE agent_id = ? AND timestep <= ?
            """, (agent_id, last_timestep))
            count_at_last = cursor.fetchone()["cnt"]

            if current_count - count_at_last >= threshold:
                multi_touch.append(agent_id)

        return never_reasoned + multi_touch

    def record_exposure(
        self, agent_id: str, exposure: ExposureRecord
    ) -> None:
        """Record an exposure event for an agent.

        Args:
            agent_id: Agent ID
            exposure: Exposure record to add
        """
        cursor = self.conn.cursor()

        # Insert exposure record
        cursor.execute("""
            INSERT INTO exposures (agent_id, timestep, channel, source_agent_id, content, credibility)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            agent_id,
            exposure.timestep,
            exposure.channel,
            exposure.source_agent_id,
            exposure.content,
            exposure.credibility,
        ))

        # Update agent state
        cursor.execute("""
            UPDATE agent_states
            SET aware = 1, exposure_count = exposure_count + 1, updated_at = ?
            WHERE agent_id = ?
        """, (exposure.timestep, agent_id))

        self.conn.commit()

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

        cursor.execute("""
            UPDATE agent_states
            SET position = ?,
                sentiment = ?,
                action_intent = ?,
                will_share = ?,
                outcomes_json = ?,
                raw_reasoning = ?,
                last_reasoning_timestep = ?,
                updated_at = ?
            WHERE agent_id = ?
        """, (
            state.position,
            state.sentiment,
            state.action_intent,
            1 if state.will_share else 0,
            outcomes_json,
            state.raw_reasoning,
            timestep,
            timestep,
            agent_id,
        ))

        self.conn.commit()

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

            cursor.execute("""
                UPDATE agent_states
                SET position = ?,
                    sentiment = ?,
                    action_intent = ?,
                    will_share = ?,
                    outcomes_json = ?,
                    raw_reasoning = ?,
                    last_reasoning_timestep = ?,
                    updated_at = ?
                WHERE agent_id = ?
            """, (
                state.position,
                state.sentiment,
                state.action_intent,
                1 if state.will_share else 0,
                outcomes_json,
                state.raw_reasoning,
                timestep,
                timestep,
                agent_id,
            ))

        self.conn.commit()

    def log_event(self, event: SimulationEvent) -> None:
        """Log a simulation event to the timeline.

        Args:
            event: Event to log
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO timeline (timestep, event_type, agent_id, details_json, wall_timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event.timestep,
            event.event_type.value,
            event.agent_id,
            json.dumps(event.details),
            event.timestamp.isoformat(),
        ))

        self.conn.commit()

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

        cursor.execute("""
            SELECT position, COUNT(*) as cnt
            FROM agent_states
            WHERE position IS NOT NULL
            GROUP BY position
        """)

        return {row["position"]: row["cnt"] for row in cursor.fetchall()}

    def get_average_sentiment(self) -> float | None:
        """Get average sentiment of aware agents."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT AVG(sentiment) as avg_sentiment
            FROM agent_states
            WHERE sentiment IS NOT NULL
        """)
        row = cursor.fetchone()

        return row["avg_sentiment"] if row else None

    def save_timestep_summary(self, summary: TimestepSummary) -> None:
        """Save a timestep summary.

        Args:
            summary: Timestep summary to save
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO timestep_summaries
            (timestep, new_exposures, agents_reasoned, shares_occurred,
             state_changes, exposure_rate, position_distribution_json, average_sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.timestep,
            summary.new_exposures,
            summary.agents_reasoned,
            summary.shares_occurred,
            summary.state_changes,
            summary.exposure_rate,
            json.dumps(summary.position_distribution),
            summary.average_sentiment,
        ))

        self.conn.commit()

    def get_timestep_summaries(self) -> list[TimestepSummary]:
        """Get all timestep summaries."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM timestep_summaries ORDER BY timestep
        """)

        summaries = []
        for row in cursor.fetchall():
            position_dist = {}
            if row["position_distribution_json"]:
                try:
                    position_dist = json.loads(row["position_distribution_json"])
                except json.JSONDecodeError:
                    pass

            summaries.append(TimestepSummary(
                timestep=row["timestep"],
                new_exposures=row["new_exposures"],
                agents_reasoned=row["agents_reasoned"],
                shares_occurred=row["shares_occurred"],
                state_changes=row["state_changes"],
                exposure_rate=row["exposure_rate"],
                position_distribution=position_dist,
                average_sentiment=row["average_sentiment"],
            ))

        return summaries

    def export_final_states(self) -> list[dict[str, Any]]:
        """Export all final agent states as dictionaries.

        Returns:
            List of agent state dictionaries
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM agent_states")
        states = []

        for row in cursor.fetchall():
            # Get exposures
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM exposures WHERE agent_id = ?
            """, (row["agent_id"],))
            exposure_count = cursor.fetchone()["cnt"]

            outcomes = {}
            if row["outcomes_json"]:
                try:
                    outcomes = json.loads(row["outcomes_json"])
                except json.JSONDecodeError:
                    pass

            states.append({
                "agent_id": row["agent_id"],
                "aware": bool(row["aware"]),
                "exposure_count": exposure_count,
                "last_reasoning_timestep": row["last_reasoning_timestep"],
                "position": row["position"],
                "sentiment": row["sentiment"],
                "action_intent": row["action_intent"],
                "will_share": bool(row["will_share"]),
                "outcomes": outcomes,
                "raw_reasoning": row["raw_reasoning"],
            })

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

            events.append({
                "timestep": row["timestep"],
                "event_type": row["event_type"],
                "agent_id": row["agent_id"],
                "details": details,
                "timestamp": row["wall_timestamp"],
            })

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
