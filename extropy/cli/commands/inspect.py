"""Inspect commands for DB-backed artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer

from ...storage import open_study_db
from ..app import app, console

inspect_app = typer.Typer(help="Inspect study DB entities")
app.add_typer(inspect_app, name="inspect")


def _resolve_run(conn: sqlite3.Connection, run_id: str | None) -> sqlite3.Row | None:
    cur = conn.cursor()
    if run_id:
        cur.execute(
            """
            SELECT run_id, population_id, network_id, status, started_at, completed_at
            FROM simulation_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
    else:
        cur.execute(
            """
            SELECT run_id, population_id, network_id, status, started_at, completed_at
            FROM simulation_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
    return cur.fetchone()


@inspect_app.command("summary")
def inspect_summary(
    study_db: Path = typer.Option(..., "--study-db", help="Canonical study DB file"),
    population_id: str = typer.Option("default", "--population-id"),
    network_id: str = typer.Option("default", "--network-id"),
    run_id: str | None = typer.Option(None, "--run-id"),
):
    with open_study_db(study_db) as db:
        agent_count = db.get_agent_count(population_id)
        edge_count = db.get_network_edge_count(network_id)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        run_row = _resolve_run(conn, run_id)
        resolved_run_id = str(run_row["run_id"]) if run_row else None
        if run_row:
            population_id = str(run_row["population_id"])
            network_id = str(run_row["network_id"])

        cur = conn.cursor()
        if resolved_run_id:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM agent_states WHERE run_id = ?",
                (resolved_run_id,),
            )
            sim_agents = int(cur.fetchone()["cnt"])
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM timestep_summaries WHERE run_id = ?",
                (resolved_run_id,),
            )
            timesteps = int(cur.fetchone()["cnt"])
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM timeline WHERE run_id = ?",
                (resolved_run_id,),
            )
            events = int(cur.fetchone()["cnt"])
        else:
            sim_agents = 0
            timesteps = 0
            events = 0
    finally:
        conn.close()

    console.print("[bold]Study Summary[/bold]")
    console.print(f"study_db: {study_db}")
    console.print(f"population_id={population_id} agents={agent_count}")
    console.print(f"network_id={network_id} edges={edge_count}")
    if resolved_run_id:
        console.print(f"run_id={resolved_run_id}")
    console.print(f"simulation.agent_states={sim_agents}")
    console.print(f"simulation.timesteps={timesteps}")
    console.print(f"simulation.events={events}")


@inspect_app.command("agent")
def inspect_agent(
    study_db: Path = typer.Option(..., "--study-db"),
    agent_id: str = typer.Option(..., "--agent-id"),
    run_id: str | None = typer.Option(None, "--run-id"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        run_row = _resolve_run(conn, run_id)
        if not run_row:
            console.print("[yellow]No simulation runs found.[/yellow]")
            return
        resolved_run_id = str(run_row["run_id"])
        population_id = str(run_row["population_id"])

        cur = conn.cursor()
        cur.execute(
            "SELECT attrs_json FROM agents WHERE population_id = ? AND agent_id = ? LIMIT 1",
            (population_id, agent_id),
        )
        attrs_row = cur.fetchone()
        attrs = json.loads(attrs_row["attrs_json"]) if attrs_row else {}

        cur.execute(
            "SELECT * FROM agent_states WHERE run_id = ? AND agent_id = ? LIMIT 1",
            (resolved_run_id, agent_id),
        )
        state = cur.fetchone()

        cur.execute(
            """
            SELECT timestep, event_type, details_json
            FROM timeline
            WHERE run_id = ? AND agent_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (resolved_run_id, agent_id),
        )
        events = cur.fetchall()
    finally:
        conn.close()

    console.print(f"[bold]Agent {agent_id}[/bold]")
    if attrs:
        console.print("[bold]Attributes[/bold]")
        for key in sorted(attrs.keys()):
            if key.startswith("_"):
                continue
            console.print(f"  - {key}: {attrs[key]}")

    if state:
        console.print("[bold]State[/bold]")
        console.print(
            f"  aware={bool(state['aware'])} will_share={bool(state['will_share'])}"
        )
        console.print(
            f"  position={state['private_position'] or state['position']} "
            f"sentiment={state['private_sentiment'] if state['private_sentiment'] is not None else state['sentiment']}"
        )
        if state["raw_reasoning"]:
            console.print("[bold]Raw reasoning[/bold]")
            console.print(str(state["raw_reasoning"]))

    if events:
        console.print("[bold]Recent events[/bold]")
        for row in events:
            details = row["details_json"] or "{}"
            console.print(f"  t={row['timestep']} {row['event_type']} {details}")


@inspect_app.command("network")
def inspect_network(
    study_db: Path = typer.Option(..., "--study-db"),
    network_id: str = typer.Option("default", "--network-id"),
    top: int = typer.Option(10, "--top", min=1),
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
        avg_w = float(row["avg_w"]) if row and row["avg_w"] is not None else 0.0

        cur.execute(
            """
            SELECT source_id, COUNT(*) AS degree
            FROM network_edges
            WHERE network_id = ?
            GROUP BY source_id
            ORDER BY degree DESC
            LIMIT ?
            """,
            (network_id, top),
        )
        top_rows = cur.fetchall()
    finally:
        conn.close()

    console.print(f"[bold]Network {network_id}[/bold]")
    console.print(f"edges={edge_count} avg_weight={avg_w:.4f}")
    if top_rows:
        console.print("top source degrees:")
        for r in top_rows:
            console.print(f"  - {r['source_id']}: {r['degree']}")


@inspect_app.command("network-status")
def inspect_network_status(
    study_db: Path = typer.Option(..., "--study-db"),
    network_run_id: str = typer.Option(..., "--network-run-id"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT phase, current, total, message, updated_at
            FROM network_generation_status
            WHERE network_run_id = ?
            """,
            (network_run_id,),
        )
        status = cur.fetchone()
        cur.execute(
            """
            SELECT restart_index, status, best_score, best_metrics_json
            FROM network_calibration_runs
            WHERE network_run_id = ?
            ORDER BY restart_index DESC
            LIMIT 3
            """,
            (network_run_id,),
        )
        runs = cur.fetchall()
        cur.execute(
            """
            SELECT calibration_run_id, iteration, score, accepted, created_at
            FROM network_calibration_iterations
            WHERE calibration_run_id IN (
                SELECT calibration_run_id
                FROM network_calibration_runs
                WHERE network_run_id = ?
            )
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (network_run_id,),
        )
        iters = cur.fetchall()
    finally:
        conn.close()

    console.print(f"[bold]Network Run Status[/bold] {network_run_id}")
    if status:
        console.print(
            f"phase={status['phase']} progress={status['current']}/{status['total']} updated_at={status['updated_at']}"
        )
        if status["message"]:
            console.print(f"message={status['message']}")
    else:
        console.print("[dim]No live status row found.[/dim]")

    if runs:
        console.print("[bold]Recent calibration runs[/bold]")
        for row in runs:
            metrics = row["best_metrics_json"] or "{}"
            console.print(
                f"  restart={row['restart_index']} status={row['status']} best_score={row['best_score']} metrics={metrics}"
            )
    if iters:
        console.print("[bold]Recent calibration iterations[/bold]")
        for row in iters:
            console.print(
                f"  run={row['calibration_run_id'][:8]} iter={row['iteration']} score={row['score']:.4f} accepted={bool(row['accepted'])} at={row['created_at']}"
            )
