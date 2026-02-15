"""Explicit exports from study DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer

from ..app import app, console

export_app = typer.Typer(help="Export datasets from study DB")
app.add_typer(export_app, name="export")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


@export_app.command("agents")
def export_agents(
    study_db: Path = typer.Option(..., "--study-db"),
    population_id: str = typer.Option("default", "--population-id"),
    output: Path = typer.Option(..., "--to"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT agent_id, attrs_json FROM agents WHERE population_id = ? ORDER BY agent_id",
            (population_id,),
        )
        rows = []
        for row in cur.fetchall():
            try:
                rows.append(json.loads(row["attrs_json"]))
            except json.JSONDecodeError:
                rows.append({"_id": row["agent_id"]})
    finally:
        conn.close()

    _write_jsonl(output, rows)
    console.print(f"[green]✓[/green] Exported {len(rows)} agents -> {output}")


@export_app.command("edges")
def export_edges(
    study_db: Path = typer.Option(..., "--study-db"),
    network_id: str = typer.Option("default", "--network-id"),
    output: Path = typer.Option(..., "--to"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_id, target_id, weight, edge_type, influence_st, influence_ts
            FROM network_edges
            WHERE network_id = ?
            ORDER BY source_id, target_id
            """,
            (network_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    _write_jsonl(output, rows)
    console.print(f"[green]✓[/green] Exported {len(rows)} edges -> {output}")


@export_app.command("states")
def export_states(
    study_db: Path = typer.Option(..., "--study-db"),
    run_id: str | None = typer.Option(None, "--run-id"),
    output: Path = typer.Option(..., "--to"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        if run_id:
            cur.execute(
                "SELECT run_id FROM simulation_runs WHERE run_id = ?",
                (run_id,),
            )
        else:
            cur.execute(
                "SELECT run_id FROM simulation_runs ORDER BY started_at DESC LIMIT 1"
            )
        run_row = cur.fetchone()
        if not run_row:
            console.print("[yellow]No simulation runs found.[/yellow]")
            raise typer.Exit(1)
        resolved_run_id = str(run_row["run_id"])

        cur.execute(
            "SELECT * FROM agent_states WHERE run_id = ? ORDER BY agent_id",
            (resolved_run_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    _write_jsonl(output, rows)
    console.print(f"[green]✓[/green] Exported {len(rows)} agent states -> {output}")
