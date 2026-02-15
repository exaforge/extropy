"""Results command for DB-first simulation results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer

from ..app import app, console


@app.command("results")
def results_command(
    study_db: Path = typer.Option(..., "--study-db", help="Canonical study DB file"),
    run_id: str | None = typer.Option(None, "--run-id", help="Simulation run id"),
    segment: str | None = typer.Option(
        None, "--segment", "-s", help="Attribute to segment by"
    ),
    timeline: bool = typer.Option(False, "--timeline", "-t", help="Show timeline view"),
    agent: str | None = typer.Option(
        None, "--agent", "-a", help="Show single agent details"
    ),
):
    """Display simulation results from the canonical study DB."""
    if not study_db.exists():
        console.print(f"[red]✗[/red] Study DB not found: {study_db}")
        raise typer.Exit(1)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        if run_id:
            cur.execute(
                """
                SELECT run_id, status, started_at, completed_at, stopped_reason, population_id
                FROM simulation_runs
                WHERE run_id = ?
                """,
                (run_id,),
            )
        else:
            cur.execute(
                """
                SELECT run_id, status, started_at, completed_at, stopped_reason, population_id
                FROM simulation_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
        run_row = cur.fetchone()
        if not run_row:
            console.print("[yellow]No simulation runs found in study DB.[/yellow]")
            raise typer.Exit(0)
        resolved_run_id = str(run_row["run_id"])
        population_id = str(run_row["population_id"])
        console.print(
            f"[dim]run_id={resolved_run_id} status={run_row['status']} "
            f"started_at={run_row['started_at']} completed_at={run_row['completed_at'] or '-'}[/dim]"
        )
        if agent:
            _display_agent(conn, resolved_run_id, population_id, agent)
            return
        if segment:
            _display_segment(conn, resolved_run_id, population_id, segment)
            return
        if timeline:
            _display_timeline(conn, resolved_run_id)
            return
        _display_summary(conn, resolved_run_id)
    finally:
        conn.close()


def _display_summary(conn: sqlite3.Connection, run_id: str) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM agent_states WHERE run_id = ?", (run_id,))
    total = int(cur.fetchone()["cnt"])
    if total == 0:
        console.print("[yellow]No simulation state found in study DB.[/yellow]")
        return

    cur.execute(
        "SELECT COUNT(*) AS cnt FROM agent_states WHERE run_id = ? AND aware = 1",
        (run_id,),
    )
    aware = int(cur.fetchone()["cnt"])

    cur.execute(
        """
        SELECT COALESCE(private_position, position) AS position, COUNT(*) AS cnt
        FROM agent_states
        WHERE run_id = ?
          AND COALESCE(private_position, position) IS NOT NULL
        GROUP BY COALESCE(private_position, position)
        ORDER BY cnt DESC
        """,
        (run_id,),
    )
    rows = cur.fetchall()

    console.print()
    console.print("[bold]Simulation Summary[/bold]")
    console.print(f"Agents: {total}")
    console.print(f"Aware: {aware} ({aware / total:.1%})")
    console.print("Positions:")
    for row in rows:
        pct = int(row["cnt"]) / total
        console.print(f"  - {row['position']}: {row['cnt']} ({pct:.1%})")


def _display_timeline(conn: sqlite3.Connection, run_id: str) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT timestep, new_exposures, agents_reasoned, shares_occurred, exposure_rate
        FROM timestep_summaries
        WHERE run_id = ?
        ORDER BY timestep
        """,
        (run_id,),
    )
    rows = cur.fetchall()
    if not rows:
        console.print("[yellow]No timestep summaries found.[/yellow]")
        return

    console.print()
    console.print("[bold]Timeline[/bold]")
    for row in rows:
        console.print(
            f"t={row['timestep']:>3} | new_exp={row['new_exposures']:>5} | "
            f"reasoned={row['agents_reasoned']:>5} | shares={row['shares_occurred']:>5} | "
            f"exposure={float(row['exposure_rate']):.1%}"
        )


def _display_segment(
    conn: sqlite3.Connection,
    run_id: str,
    population_id: str,
    attribute: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT agent_id, attrs_json FROM agents WHERE population_id = ?",
        (population_id,),
    )
    attr_by_agent: dict[str, str] = {}
    for row in cur.fetchall():
        try:
            attrs = json.loads(row["attrs_json"])
        except json.JSONDecodeError:
            continue
        attr_by_agent[str(row["agent_id"])] = str(attrs.get(attribute, "unknown"))

    if not attr_by_agent:
        console.print("[yellow]No agent attribute records found.[/yellow]")
        return

    cur.execute(
        """
        SELECT agent_id, aware, COALESCE(private_position, position) AS position
        FROM agent_states
        WHERE run_id = ?
        """,
        (run_id,),
    )
    groups: dict[str, dict[str, int]] = {}
    for row in cur.fetchall():
        aid = str(row["agent_id"])
        key = attr_by_agent.get(aid, "unknown")
        if key not in groups:
            groups[key] = {"total": 0, "aware": 0}
        groups[key]["total"] += 1
        if int(row["aware"]) == 1:
            groups[key]["aware"] += 1

    console.print()
    console.print(f"[bold]Segment by {attribute}[/bold]")
    for key, data in sorted(groups.items(), key=lambda x: x[1]["total"], reverse=True):
        total = data["total"]
        aware = data["aware"]
        pct = aware / total if total else 0.0
        console.print(f"  - {key}: {total} agents, aware={aware} ({pct:.1%})")


def _display_agent(
    conn: sqlite3.Connection,
    run_id: str,
    population_id: str,
    agent_id: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM agent_states
        WHERE run_id = ? AND agent_id = ?
        """,
        (run_id, agent_id),
    )
    row = cur.fetchone()
    if not row:
        console.print(
            f"[yellow]Agent not found in simulation state: {agent_id}[/yellow]"
        )
        return

    cur.execute(
        "SELECT attrs_json FROM agents WHERE population_id = ? AND agent_id = ? LIMIT 1",
        (population_id, agent_id),
    )
    attrs_row = cur.fetchone()
    attrs = {}
    if attrs_row:
        try:
            attrs = json.loads(attrs_row["attrs_json"])
        except json.JSONDecodeError:
            attrs = {}

    console.print()
    console.print(f"[bold]Agent {agent_id}[/bold]")
    console.print(f"Aware: {bool(row['aware'])}")
    console.print(f"Position: {row['private_position'] or row['position']}")
    console.print(
        f"Sentiment: {row['private_sentiment'] if row['private_sentiment'] is not None else row['sentiment']}"
    )
    console.print(
        f"Conviction: {row['private_conviction'] if row['private_conviction'] is not None else row['conviction']}"
    )
    if row["public_statement"]:
        console.print(f"Public statement: {row['public_statement']}")
    if row["action_intent"]:
        console.print(f"Action intent: {row['action_intent']}")
    if row["raw_reasoning"]:
        console.print()
        console.print("[bold]Raw Reasoning[/bold]")
        console.print(str(row["raw_reasoning"]))
    if attrs:
        console.print()
        console.print("[bold]Attributes[/bold]")
        for key in sorted(attrs.keys()):
            if key.startswith("_"):
                continue
            console.print(f"  - {key}: {attrs[key]}")
