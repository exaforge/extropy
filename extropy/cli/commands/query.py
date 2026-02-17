"""Query command for raw data access from the study DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import typer

from ...storage import open_study_db, ReadOnlySQLRequest
from ..app import app, console, is_agent_mode, get_study_path
from ..study import get_study_db
from ..utils import Output

query_app = typer.Typer(help="Query study data")
app.add_typer(query_app, name="query")

_ALLOWED_PREFIXES = ("select", "with", "explain")
_DENYLIST_TOKENS = (
    " insert ",
    " update ",
    " delete ",
    " alter ",
    " drop ",
    " create ",
    " attach ",
    " vacuum ",
    " pragma ",
    " replace ",
    " truncate ",
)


def _get_study_db() -> Path:
    """Resolve study DB path via auto-detection or --study flag."""
    try:
        return get_study_db(get_study_path())
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


def _resolve_run(conn: sqlite3.Connection, run_id: str | None) -> sqlite3.Row | None:
    cur = conn.cursor()
    if run_id:
        cur.execute(
            """
            SELECT run_id, scenario_name, population_id, network_id, status, started_at, completed_at
            FROM simulation_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
    else:
        cur.execute(
            """
            SELECT run_id, scenario_name, population_id, network_id, status, started_at, completed_at
            FROM simulation_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
    return cur.fetchone()


def _resolve_scenario_name(
    conn: sqlite3.Connection, scenario: str | None, run_id: str | None
) -> str:
    """Resolve the scenario_id value for WHERE clauses.

    If --scenario is given explicitly, use it directly.
    Otherwise resolve from the latest simulation_runs.scenario_name.
    """
    if scenario:
        return scenario
    run_row = _resolve_run(conn, run_id)
    if run_row:
        return str(run_row["scenario_name"] or run_row["population_id"])
    return "default"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


# --- Data queries (absorb export) ---


@query_app.command("agents")
def query_agents(
    to: Path | None = typer.Option(None, "--to", help="Write JSONL to file"),
    scenario: str = typer.Option(None, "--scenario", "-s", help="Scenario name"),
    run_id: str | None = typer.Option(None, "--run-id"),
):
    """Dump agent attributes."""
    study_db = _get_study_db()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        scenario_name = _resolve_scenario_name(conn, scenario, run_id)
        cur = conn.cursor()
        cur.execute(
            "SELECT agent_id, attrs_json FROM agents WHERE scenario_id = ? ORDER BY agent_id",
            (scenario_name,),
        )
        rows = []
        for row in cur.fetchall():
            try:
                rows.append(json.loads(row["attrs_json"]))
            except json.JSONDecodeError:
                rows.append({"_id": row["agent_id"]})
    finally:
        conn.close()

    if to:
        _write_jsonl(to, rows)
        if not agent_mode:
            console.print(f"[green]✓[/green] Exported {len(rows)} agents -> {to}")
        else:
            out.set_data("exported", len(rows))
            out.set_data("file", str(to))
    elif agent_mode:
        out.set_data("agents", rows)
        out.set_data("count", len(rows))
    else:
        for row in rows:
            console.print(json.dumps(row, default=str))

    raise typer.Exit(out.finish())


@query_app.command("edges")
def query_edges(
    to: Path | None = typer.Option(None, "--to", help="Write JSONL to file"),
    scenario: str = typer.Option(None, "--scenario", "-s", help="Scenario name"),
    run_id: str | None = typer.Option(None, "--run-id"),
):
    """Dump network edges."""
    study_db = _get_study_db()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        scenario_name = _resolve_scenario_name(conn, scenario, run_id)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source_id, target_id, weight, edge_type, influence_st, influence_ts
            FROM network_edges
            WHERE scenario_id = ?
            ORDER BY source_id, target_id
            """,
            (scenario_name,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    if to:
        _write_jsonl(to, rows)
        if not agent_mode:
            console.print(f"[green]✓[/green] Exported {len(rows)} edges -> {to}")
        else:
            out.set_data("exported", len(rows))
            out.set_data("file", str(to))
    elif agent_mode:
        out.set_data("edges", rows)
        out.set_data("count", len(rows))
    else:
        for row in rows:
            console.print(json.dumps(row, default=str))

    raise typer.Exit(out.finish())


@query_app.command("states")
def query_states(
    run_id: str | None = typer.Option(None, "--run-id"),
    to: Path | None = typer.Option(None, "--to", help="Write JSONL to file"),
):
    """Dump agent states for a simulation run."""
    study_db = _get_study_db()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

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
            if not agent_mode:
                console.print("[yellow]No simulation runs found.[/yellow]")
            else:
                out.warning("No simulation runs found.")
            raise typer.Exit(out.finish() or 1)
        resolved_run_id = str(run_row["run_id"])

        cur.execute(
            "SELECT * FROM agent_states WHERE run_id = ? ORDER BY agent_id",
            (resolved_run_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    if to:
        _write_jsonl(to, rows)
        if not agent_mode:
            console.print(f"[green]✓[/green] Exported {len(rows)} agent states -> {to}")
        else:
            out.set_data("exported", len(rows))
            out.set_data("file", str(to))
            out.set_data("run_id", resolved_run_id)
    elif agent_mode:
        out.set_data("states", rows)
        out.set_data("count", len(rows))
        out.set_data("run_id", resolved_run_id)
    else:
        for row in rows:
            console.print(json.dumps(row, default=str))

    raise typer.Exit(out.finish())


# --- Inspection queries (absorb inspect) ---


@query_app.command("summary")
def query_summary(
    run_id: str | None = typer.Option(None, "--run-id"),
    scenario: str = typer.Option(None, "--scenario", "-s", help="Scenario name"),
):
    """Show study entity counts."""
    study_db = _get_study_db()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        run_row = _resolve_run(conn, run_id)
        resolved_run_id = str(run_row["run_id"]) if run_row else None
        scenario_name = _resolve_scenario_name(conn, scenario, run_id)

        with open_study_db(study_db) as db:
            agent_count = db.get_agent_count_by_scenario(scenario_name)
            edge_count = db.get_network_edge_count_by_scenario(scenario_name)

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

    if agent_mode:
        out.set_data("study_db", str(study_db))
        out.set_data("scenario_name", scenario_name)
        out.set_data("agents", agent_count)
        out.set_data("edges", edge_count)
        if resolved_run_id:
            out.set_data("run_id", resolved_run_id)
        out.set_data("simulation_agent_states", sim_agents)
        out.set_data("simulation_timesteps", timesteps)
        out.set_data("simulation_events", events)
    else:
        console.print("[bold]Study Summary[/bold]")
        console.print(f"study_db: {study_db}")
        console.print(f"scenario={scenario_name} agents={agent_count}")
        console.print(f"edges={edge_count}")
        if resolved_run_id:
            console.print(f"run_id={resolved_run_id}")
        console.print(f"simulation.agent_states={sim_agents}")
        console.print(f"simulation.timesteps={timesteps}")
        console.print(f"simulation.events={events}")

    raise typer.Exit(out.finish())


@query_app.command("network")
def query_network(
    scenario: str = typer.Option(None, "--scenario", "-s", help="Scenario name"),
    run_id: str | None = typer.Option(None, "--run-id"),
    top: int = typer.Option(10, "--top", min=1),
):
    """Show network statistics."""
    study_db = _get_study_db()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        scenario_name = _resolve_scenario_name(conn, scenario, run_id)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS cnt, AVG(weight) AS avg_w FROM network_edges WHERE scenario_id = ?",
            (scenario_name,),
        )
        row = cur.fetchone()
        edge_count = int(row["cnt"]) if row else 0
        avg_w = float(row["avg_w"]) if row and row["avg_w"] is not None else 0.0

        cur.execute(
            """
            SELECT source_id, COUNT(*) AS degree
            FROM network_edges
            WHERE scenario_id = ?
            GROUP BY source_id
            ORDER BY degree DESC
            LIMIT ?
            """,
            (scenario_name, top),
        )
        top_rows = cur.fetchall()
    finally:
        conn.close()

    if agent_mode:
        out.set_data("scenario_name", scenario_name)
        out.set_data("edge_count", edge_count)
        out.set_data("avg_weight", avg_w)
        out.set_data(
            "top_degrees",
            [{"source_id": r["source_id"], "degree": r["degree"]} for r in top_rows],
        )
    else:
        console.print(f"[bold]Network (scenario={scenario_name})[/bold]")
        console.print(f"edges={edge_count} avg_weight={avg_w:.4f}")
        if top_rows:
            console.print("top source degrees:")
            for r in top_rows:
                console.print(f"  - {r['source_id']}: {r['degree']}")

    raise typer.Exit(out.finish())


@query_app.command("network-status")
def query_network_status(
    network_run_id: str = typer.Argument(..., help="Network generation run ID"),
):
    """Show network calibration progress."""
    study_db = _get_study_db()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

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

    if agent_mode:
        out.set_data("network_run_id", network_run_id)
        if status:
            out.set_data(
                "status",
                {
                    "phase": status["phase"],
                    "current": status["current"],
                    "total": status["total"],
                    "message": status["message"],
                    "updated_at": status["updated_at"],
                },
            )
        if runs:
            out.set_data(
                "calibration_runs",
                [
                    {
                        "restart_index": r["restart_index"],
                        "status": r["status"],
                        "best_score": r["best_score"],
                        "best_metrics_json": r["best_metrics_json"],
                    }
                    for r in runs
                ],
            )
        if iters:
            out.set_data(
                "calibration_iterations",
                [
                    {
                        "calibration_run_id": r["calibration_run_id"],
                        "iteration": r["iteration"],
                        "score": r["score"],
                        "accepted": bool(r["accepted"]),
                        "created_at": r["created_at"],
                    }
                    for r in iters
                ],
            )
    else:
        console.print(f"[bold]Network Run Status[/bold] {network_run_id}")
        if status:
            console.print(
                f"phase={status['phase']} progress={status['current']}/{status['total']} updated_at={status['updated_at']}"
            )
            if status["message"]:
                try:
                    payload = json.loads(status["message"])
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    details = ", ".join(f"{k}={v}" for k, v in payload.items())
                    console.print(f"message={details}")
                else:
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

    raise typer.Exit(out.finish())


# --- Raw SQL ---


@query_app.command("sql")
def query_sql(
    sql: str = typer.Argument(..., help="Read-only SQL statement"),
    limit: int = typer.Option(1000, "--limit", min=1),
    format: str = typer.Option("table", "--format", help="table|json|jsonl"),
):
    """Run a read-only SQL query against the study database."""
    study_db = _get_study_db()

    req = ReadOnlySQLRequest(sql=sql, limit=limit)
    normalized = req.sql.strip().lower()
    if not normalized.startswith(_ALLOWED_PREFIXES):
        console.print(
            "[red]✗[/red] Only read-only SELECT/WITH/EXPLAIN queries are allowed"
        )
        raise typer.Exit(1)
    padded = f" {normalized} "
    if ";" in req.sql.strip().rstrip(";"):
        console.print("[red]✗[/red] Multi-statement SQL is not allowed")
        raise typer.Exit(1)
    if any(tok in padded for tok in _DENYLIST_TOKENS):
        console.print("[red]✗[/red] Mutating SQL tokens are not allowed")
        raise typer.Exit(1)

    with open_study_db(study_db) as db:
        try:
            rows = db.run_select(req.sql, limit=req.limit)
        except Exception as e:
            console.print(f"[red]✗[/red] Query failed: {e}")
            raise typer.Exit(1)

    if format == "json":
        console.print_json(data=rows)
        return
    if format == "jsonl":
        for row in rows:
            console.print(json.dumps(row, default=str))
        return

    if not rows:
        console.print("[dim](no rows)[/dim]")
        return

    columns = list(rows[0].keys())
    console.print(" | ".join(columns))
    console.print("-" * max(20, len(" | ".join(columns))))
    for row in rows:
        console.print(" | ".join(str(row.get(c, "")) for c in columns))
