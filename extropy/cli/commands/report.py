"""Reusable report generation commands."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer

from ..app import app, console

report_app = typer.Typer(help="Generate reusable JSON reports")
app.add_typer(report_app, name="report")


@report_app.command("run")
def report_run(
    study_db: Path = typer.Option(..., "--study-db"),
    run_id: str | None = typer.Option(None, "--run-id"),
    output: Path = typer.Option(..., "--output", "-o"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        if run_id:
            cur.execute(
                "SELECT run_id, population_id FROM simulation_runs WHERE run_id = ?",
                (run_id,),
            )
        else:
            cur.execute(
                """
                SELECT run_id, population_id
                FROM simulation_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
        run_row = cur.fetchone()
        if not run_row:
            console.print("[yellow]No simulation runs found.[/yellow]")
            raise typer.Exit(1)
        resolved_run_id = str(run_row["run_id"])

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM agent_states WHERE run_id = ?",
            (resolved_run_id,),
        )
        total = int(cur.fetchone()["cnt"])
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM agent_states WHERE run_id = ? AND aware = 1",
            (resolved_run_id,),
        )
        aware = int(cur.fetchone()["cnt"])
        cur.execute(
            """
            SELECT COALESCE(private_position, position) AS position, COUNT(*) AS cnt
            FROM agent_states
            WHERE run_id = ?
              AND COALESCE(private_position, position) IS NOT NULL
            GROUP BY COALESCE(private_position, position)
            """,
            (resolved_run_id,),
        )
        positions = {row["position"]: int(row["cnt"]) for row in cur.fetchall()}
    finally:
        conn.close()

    payload = {
        "run_id": resolved_run_id,
        "agent_count": total,
        "aware_count": aware,
        "aware_rate": (aware / total) if total else 0.0,
        "positions": positions,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/green] Wrote run report: {output}")


@report_app.command("network")
def report_network(
    study_db: Path = typer.Option(..., "--study-db"),
    network_id: str = typer.Option("default", "--network-id"),
    output: Path = typer.Option(..., "--output", "-o"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS cnt, AVG(weight) AS avg_w FROM network_edges WHERE network_id = ?",
            (network_id,),
        )
        row = cur.fetchone()
        edge_count = int(row["cnt"]) if row else 0
        avg_weight = float(row["avg_w"]) if row and row["avg_w"] is not None else 0.0

        cur.execute(
            "SELECT edge_type, COUNT(*) AS cnt FROM network_edges WHERE network_id = ? GROUP BY edge_type",
            (network_id,),
        )
        edge_types = {r["edge_type"]: int(r["cnt"]) for r in cur.fetchall()}
    finally:
        conn.close()

    payload = {
        "network_id": network_id,
        "edge_count": edge_count,
        "avg_weight": avg_weight,
        "edge_types": edge_types,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/green] Wrote network report: {output}")
