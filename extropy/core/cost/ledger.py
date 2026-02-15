"""Persistent cost ledger.

Appends session cost summaries to a local SQLite database.
Provides query methods for the `extropy cost` command.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_LEDGER_DIR = Path.home() / ".config" / "extropy"
_LEDGER_FILE = _LEDGER_DIR / "cost_ledger.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cost_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    date TEXT NOT NULL,
    command TEXT NOT NULL,
    scenario TEXT NOT NULL DEFAULT '',
    total_calls INTEGER NOT NULL DEFAULT 0,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost REAL,
    models_json TEXT NOT NULL DEFAULT '{}',
    elapsed_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_cost_entries_date ON cost_entries(date);
CREATE INDEX IF NOT EXISTS idx_cost_entries_command ON cost_entries(command);
"""


class CostEntry(BaseModel):
    """A single cost ledger entry."""

    timestamp: float
    date: str
    command: str
    scenario: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float | None
    models: dict[str, Any]
    elapsed_seconds: float | None


def _get_connection() -> sqlite3.Connection:
    """Get a connection to the ledger database, creating it if needed."""
    _LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_LEDGER_FILE))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def record_session(summary: dict[str, Any]) -> None:
    """Append a session cost summary to the ledger.

    Args:
        summary: Dict from CostTracker.summary()
    """
    if summary.get("total_calls", 0) == 0:
        return

    try:
        conn = _get_connection()
        try:
            now = time.time()
            conn.execute(
                """
                INSERT INTO cost_entries
                    (timestamp, date, command, scenario, total_calls,
                     total_input_tokens, total_output_tokens, total_cost,
                     models_json, elapsed_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    datetime.fromtimestamp(now).strftime("%Y-%m-%d"),
                    summary.get("command", ""),
                    summary.get("scenario", ""),
                    summary.get("total_calls", 0),
                    summary.get("total_input_tokens", 0),
                    summary.get("total_output_tokens", 0),
                    summary.get("total_cost"),
                    json.dumps(summary.get("by_model", {})),
                    summary.get("elapsed_seconds"),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as e:
        logger.debug(f"Failed to record cost to ledger: {e}")


def query_entries(
    days: int | None = 7,
    command: str | None = None,
    limit: int = 100,
) -> list[CostEntry]:
    """Query cost ledger entries.

    Args:
        days: Number of days to look back (None = all time)
        command: Filter by command name (None = all commands)
        limit: Max entries to return

    Returns:
        List of CostEntry, newest first.
    """
    try:
        conn = _get_connection()
    except (sqlite3.Error, OSError):
        return []

    try:
        clauses = []
        params: list[Any] = []

        if days is not None:
            cutoff = time.time() - (days * 86400)
            clauses.append("timestamp >= ?")
            params.append(cutoff)

        if command:
            clauses.append("command = ?")
            params.append(command)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT * FROM cost_entries
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        entries = []
        for row in rows:
            try:
                models = json.loads(row["models_json"])
            except (json.JSONDecodeError, TypeError):
                models = {}

            entries.append(
                CostEntry(
                    timestamp=row["timestamp"],
                    date=row["date"],
                    command=row["command"],
                    scenario=row["scenario"],
                    total_calls=row["total_calls"],
                    total_input_tokens=row["total_input_tokens"],
                    total_output_tokens=row["total_output_tokens"],
                    total_cost=row["total_cost"],
                    models=models,
                    elapsed_seconds=row["elapsed_seconds"],
                )
            )

        return entries
    finally:
        conn.close()


def query_totals(
    days: int | None = 7,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Query aggregated cost totals.

    Args:
        days: Number of days to look back (None = all time)
        group_by: Group results by "command", "date", or "model" (None = totals only)

    Returns:
        Dict with total and optional grouped breakdowns.
    """
    try:
        conn = _get_connection()
    except (sqlite3.Error, OSError):
        return {"total_cost": None, "total_calls": 0}

    try:
        where = ""
        params: list[Any] = []
        if days is not None:
            cutoff = time.time() - (days * 86400)
            where = "WHERE timestamp >= ?"
            params.append(cutoff)

        # Overall totals
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) as sessions,
                SUM(total_calls) as calls,
                SUM(total_input_tokens) as input_tokens,
                SUM(total_output_tokens) as output_tokens,
                SUM(total_cost) as cost
            FROM cost_entries
            {where}
            """,
            params,
        ).fetchone()

        result: dict[str, Any] = {
            "sessions": row["sessions"] or 0,
            "total_calls": row["calls"] or 0,
            "total_input_tokens": row["input_tokens"] or 0,
            "total_output_tokens": row["output_tokens"] or 0,
            "total_cost": round(row["cost"], 4) if row["cost"] else None,
        }

        # Grouped breakdown
        if group_by == "command":
            rows = conn.execute(
                f"""
                SELECT command,
                       COUNT(*) as sessions,
                       SUM(total_calls) as calls,
                       SUM(total_cost) as cost
                FROM cost_entries
                {where}
                GROUP BY command
                ORDER BY cost DESC
                """,
                params,
            ).fetchall()
            result["by_command"] = {
                r["command"]: {
                    "sessions": r["sessions"],
                    "calls": r["calls"] or 0,
                    "cost": round(r["cost"], 4) if r["cost"] else None,
                }
                for r in rows
            }

        elif group_by == "date":
            rows = conn.execute(
                f"""
                SELECT date,
                       COUNT(*) as sessions,
                       SUM(total_calls) as calls,
                       SUM(total_cost) as cost
                FROM cost_entries
                {where}
                GROUP BY date
                ORDER BY date DESC
                """,
                params,
            ).fetchall()
            result["by_date"] = {
                r["date"]: {
                    "sessions": r["sessions"],
                    "calls": r["calls"] or 0,
                    "cost": round(r["cost"], 4) if r["cost"] else None,
                }
                for r in rows
            }

        return result
    finally:
        conn.close()
