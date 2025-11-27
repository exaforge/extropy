"""SQLite database operations for Entropy."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import get_settings
from .models import (
    Agent,
    AgentState,
    Cognitive,
    Connection,
    Demographics,
    InformationEnvironment,
    Network,
    ParsedContext,
    Population,
    Psychographics,
    ResearchData,
    SituationSchema,
)


def get_db_path() -> Path:
    """Get database path from settings."""
    return get_settings().db_path_resolved


@contextmanager
def get_connection():
    """Get database connection with context management."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Initialize database schema."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS populations (
                name TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                context_raw TEXT NOT NULL,
                context_parsed TEXT NOT NULL,
                research TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                population_name TEXT NOT NULL,
                demographics TEXT NOT NULL,
                psychographics TEXT NOT NULL,
                cognitive TEXT NOT NULL,
                information_env TEXT NOT NULL,
                situation TEXT NOT NULL,
                network TEXT NOT NULL,
                persona TEXT NOT NULL,
                state TEXT NOT NULL,
                FOREIGN KEY (population_name) REFERENCES populations(name) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agents_population ON agents(population_name)
        """)


def save_population(population: Population) -> None:
    """Save a population to the database."""
    init_db()

    with get_connection() as conn:
        # Save population metadata
        conn.execute(
            """
            INSERT OR REPLACE INTO populations (name, size, context_raw, context_parsed, research, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                population.name,
                population.size,
                population.context_raw,
                population.context_parsed.model_dump_json(),
                population.research.model_dump_json(),
                population.created_at.isoformat(),
            ),
        )

        # Delete existing agents for this population (in case of update)
        conn.execute("DELETE FROM agents WHERE population_name = ?", (population.name,))

        # Save agents
        for agent in population.agents:
            conn.execute(
                """
                INSERT INTO agents (id, population_name, demographics, psychographics, cognitive, 
                                   information_env, situation, network, persona, state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent.id,
                    population.name,
                    agent.demographics.model_dump_json(),
                    agent.psychographics.model_dump_json(),
                    agent.cognitive.model_dump_json(),
                    agent.information_env.model_dump_json(),
                    json.dumps(agent.situation),
                    agent.network.model_dump_json(),
                    agent.persona,
                    agent.state.model_dump_json(),
                ),
            )


def load_population(name: str, include_agents: bool = True) -> Population | None:
    """Load a population from the database."""
    init_db()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM populations WHERE name = ?", (name,)
        ).fetchone()

        if not row:
            return None

        agents = []
        if include_agents:
            agent_rows = conn.execute(
                "SELECT * FROM agents WHERE population_name = ?", (name,)
            ).fetchall()

            for agent_row in agent_rows:
                agents.append(
                    Agent(
                        id=agent_row["id"],
                        demographics=Demographics.model_validate_json(agent_row["demographics"]),
                        psychographics=Psychographics.model_validate_json(agent_row["psychographics"]),
                        cognitive=Cognitive.model_validate_json(agent_row["cognitive"]),
                        information_env=InformationEnvironment.model_validate_json(agent_row["information_env"]),
                        situation=json.loads(agent_row["situation"]),
                        network=Network.model_validate_json(agent_row["network"]),
                        persona=agent_row["persona"],
                        state=AgentState.model_validate_json(agent_row["state"]),
                    )
                )

        return Population(
            name=row["name"],
            size=row["size"],
            context_raw=row["context_raw"],
            context_parsed=ParsedContext.model_validate_json(row["context_parsed"]),
            research=ResearchData.model_validate_json(row["research"]),
            agents=agents,
            created_at=datetime.fromisoformat(row["created_at"]),
        )


def list_populations() -> list[dict]:
    """List all populations (metadata only)."""
    init_db()

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT name, size, context_raw, created_at FROM populations ORDER BY created_at DESC"
        ).fetchall()

        return [
            {
                "name": row["name"],
                "size": row["size"],
                "context": row["context_raw"],
                "created_at": datetime.fromisoformat(row["created_at"]),
            }
            for row in rows
        ]


def delete_population(name: str) -> bool:
    """Delete a population and its agents."""
    init_db()

    with get_connection() as conn:
        # Delete agents first (due to foreign key)
        conn.execute("DELETE FROM agents WHERE population_name = ?", (name,))
        result = conn.execute("DELETE FROM populations WHERE name = ?", (name,))
        return result.rowcount > 0


def population_exists(name: str) -> bool:
    """Check if a population exists."""
    init_db()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM populations WHERE name = ?", (name,)
        ).fetchone()
        return row is not None


def get_sample_agents(name: str, count: int = 5) -> list[Agent]:
    """Get a sample of agents from a population."""
    init_db()

    with get_connection() as conn:
        agent_rows = conn.execute(
            "SELECT * FROM agents WHERE population_name = ? LIMIT ?", (name, count)
        ).fetchall()

        return [
            Agent(
                id=row["id"],
                demographics=Demographics.model_validate_json(row["demographics"]),
                psychographics=Psychographics.model_validate_json(row["psychographics"]),
                cognitive=Cognitive.model_validate_json(row["cognitive"]),
                information_env=InformationEnvironment.model_validate_json(row["information_env"]),
                situation=json.loads(row["situation"]),
                network=Network.model_validate_json(row["network"]),
                persona=row["persona"],
                state=AgentState.model_validate_json(row["state"]),
            )
            for row in agent_rows
        ]

