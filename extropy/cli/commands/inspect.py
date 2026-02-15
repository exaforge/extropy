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


@inspect_app.command("summary")
def inspect_summary(
    study_db: Path = typer.Option(..., "--study-db", help="Canonical study DB file"),
    population_id: str = typer.Option("default", "--population-id"),
    network_id: str = typer.Option("default", "--network-id"),
):
    with open_study_db(study_db) as db:
        agent_count = db.get_agent_count(population_id)
        edge_count = db.get_network_edge_count(network_id)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM agent_states")
        sim_agents = int(cur.fetchone()["cnt"])
        cur.execute("SELECT COUNT(*) AS cnt FROM timestep_summaries")
        timesteps = int(cur.fetchone()["cnt"])
        cur.execute("SELECT COUNT(*) AS cnt FROM timeline")
        events = int(cur.fetchone()["cnt"])
    finally:
        conn.close()

    console.print("[bold]Study Summary[/bold]")
    console.print(f"study_db: {study_db}")
    console.print(f"population_id={population_id} agents={agent_count}")
    console.print(f"network_id={network_id} edges={edge_count}")
    console.print(f"simulation.agent_states={sim_agents}")
    console.print(f"simulation.timesteps={timesteps}")
    console.print(f"simulation.events={events}")


@inspect_app.command("agent")
def inspect_agent(
    study_db: Path = typer.Option(..., "--study-db"),
    agent_id: str = typer.Option(..., "--agent-id"),
):
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT attrs_json FROM agents WHERE agent_id = ? LIMIT 1", (agent_id,))
        attrs_row = cur.fetchone()
        attrs = json.loads(attrs_row["attrs_json"]) if attrs_row else {}

        cur.execute("SELECT * FROM agent_states WHERE agent_id = ? LIMIT 1", (agent_id,))
        state = cur.fetchone()

        cur.execute(
            "SELECT timestep, event_type, details_json FROM timeline WHERE agent_id = ? ORDER BY id DESC LIMIT 10",
            (agent_id,),
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
        console.print(f"  aware={bool(state['aware'])} will_share={bool(state['will_share'])}")
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
