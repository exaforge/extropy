"""Results command for DB-first simulation results."""

from __future__ import annotations

import json
import sqlite3

import typer

from ..app import app, console, is_agent_mode, get_study_path
from ..study import StudyContext, detect_study_folder, parse_version_ref
from ..utils import Output, ExitCode

results_app = typer.Typer(
    help="Display simulation results",
    invoke_without_command=True,
)
app.add_typer(results_app, name="results")


@results_app.callback()
def results_callback(
    ctx: typer.Context,
    scenario: str = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (uses latest run if not specified)",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Simulation run id"),
):
    """Display simulation results from the canonical study DB.

    By default shows a summary of the latest simulation run. Use subcommands
    for different views:

        extropy results                          # summary (default)
        extropy results timeline                 # timestep progression
        extropy results segment age_group        # segment by attribute
        extropy results agent agent_123          # single agent details
    """
    agent_mode = is_agent_mode()
    out = Output(console, json_mode=agent_mode)

    # Resolve study context
    study_path = get_study_path()
    detected = detect_study_folder(study_path)
    if detected is None:
        out.error(
            "Not in a study folder. Use --study to specify or run from a study folder.",
            exit_code=ExitCode.FILE_NOT_FOUND,
        )
        raise typer.Exit(out.finish())

    study_ctx = StudyContext(detected)
    study_db = study_ctx.db_path

    if not study_db.exists():
        out.error(f"Study DB not found: {study_db}")
        raise typer.Exit(1)

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()

        # Resolve scenario if provided
        scenario_name = None
        if scenario:
            scenario_name, _ = _resolve_scenario(study_ctx, scenario, out)

        # Find run
        if run_id:
            cur.execute(
                """
                SELECT run_id, scenario_name, status, started_at, completed_at, stopped_reason, population_id
                FROM simulation_runs
                WHERE run_id = ?
                """,
                (run_id,),
            )
        elif scenario_name:
            cur.execute(
                """
                SELECT run_id, scenario_name, status, started_at, completed_at, stopped_reason, population_id
                FROM simulation_runs
                WHERE scenario_name = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (scenario_name,),
            )
        else:
            cur.execute(
                """
                SELECT run_id, scenario_name, status, started_at, completed_at, stopped_reason, population_id
                FROM simulation_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
        run_row = cur.fetchone()
        if not run_row:
            if scenario_name:
                out.warning(f"No simulation runs found for scenario: {scenario_name}")
            else:
                out.warning("No simulation runs found in study DB.")
            raise typer.Exit(0)

        resolved_run_id = str(run_row["run_id"])
        scenario_name = str(run_row["scenario_name"] or run_row["population_id"])

        if not agent_mode:
            console.print(
                f"[dim]run_id={resolved_run_id} status={run_row['status']} "
                f"started_at={run_row['started_at']} completed_at={run_row['completed_at'] or '-'}[/dim]"
            )

        out.set_data("run_id", resolved_run_id)
        out.set_data("status", run_row["status"])
        out.set_data("started_at", run_row["started_at"])
        out.set_data("completed_at", run_row["completed_at"])

        # Store shared state for subcommands
        ctx.ensure_object(dict)
        ctx.obj["conn"] = conn
        ctx.obj["run_id"] = resolved_run_id
        ctx.obj["scenario_name"] = scenario_name
        ctx.obj["out"] = out
        ctx.obj["agent_mode"] = agent_mode
        ctx.obj["_keep_conn"] = True

        # If no subcommand invoked, show summary
        if ctx.invoked_subcommand is None:
            _display_summary(conn, resolved_run_id, out, agent_mode)
            ctx.obj["_keep_conn"] = False
            conn.close()
            raise typer.Exit(out.finish())
    except typer.Exit:
        if not ctx.obj.get("_keep_conn", False):
            conn.close()
        raise
    except Exception:
        conn.close()
        raise


@results_app.command("summary")
def results_summary(ctx: typer.Context):
    """Show result summary (default view)."""
    obj = ctx.ensure_object(dict)
    conn, run_id, out, agent_mode = (
        obj["conn"],
        obj["run_id"],
        obj["out"],
        obj["agent_mode"],
    )
    try:
        _display_summary(conn, run_id, out, agent_mode)
    finally:
        conn.close()
    raise typer.Exit(out.finish())


@results_app.command("timeline")
def results_timeline(ctx: typer.Context):
    """Show timestep progression."""
    obj = ctx.ensure_object(dict)
    conn, run_id, out, agent_mode = (
        obj["conn"],
        obj["run_id"],
        obj["out"],
        obj["agent_mode"],
    )
    try:
        _display_timeline(conn, run_id, out, agent_mode)
    finally:
        conn.close()
    raise typer.Exit(out.finish())


@results_app.command("segment")
def results_segment(
    ctx: typer.Context,
    attribute: str = typer.Argument(..., help="Attribute to segment by"),
):
    """Segment results by an agent attribute."""
    obj = ctx.ensure_object(dict)
    conn, run_id, scenario_name, out, agent_mode = (
        obj["conn"],
        obj["run_id"],
        obj["scenario_name"],
        obj["out"],
        obj["agent_mode"],
    )
    try:
        _display_segment(conn, run_id, scenario_name, attribute, out, agent_mode)
    finally:
        conn.close()
    raise typer.Exit(out.finish())


@results_app.command("agent")
def results_agent(
    ctx: typer.Context,
    agent_id: str = typer.Argument(..., help="Agent ID to inspect"),
):
    """Show single agent details."""
    obj = ctx.ensure_object(dict)
    conn, run_id, scenario_name, out, agent_mode = (
        obj["conn"],
        obj["run_id"],
        obj["scenario_name"],
        obj["out"],
        obj["agent_mode"],
    )
    try:
        _display_agent(conn, run_id, scenario_name, agent_id, out, agent_mode)
    finally:
        conn.close()
    raise typer.Exit(out.finish())


def _resolve_scenario(
    study_ctx: StudyContext, scenario_ref: str | None, out: Output
) -> tuple[str, int | None]:
    """Resolve scenario name and version."""
    scenarios = study_ctx.list_scenarios()

    if not scenarios:
        out.error("No scenarios found.")
        raise typer.Exit(1)

    if scenario_ref is None:
        if len(scenarios) == 1:
            return scenarios[0], None
        else:
            out.error(
                f"Multiple scenarios found: {', '.join(scenarios)}. "
                "Use -s to specify which one."
            )
            raise typer.Exit(1)

    name, version = parse_version_ref(scenario_ref)
    if name not in scenarios:
        out.error(f"Scenario not found: {name}")
        raise typer.Exit(1)

    return name, version


def _display_summary(
    conn: sqlite3.Connection, run_id: str, out: Output, agent_mode: bool
) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM agent_states WHERE run_id = ?", (run_id,))
    total = int(cur.fetchone()["cnt"])
    if total == 0:
        out.warning("No simulation state found in study DB.")
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

    positions = {str(row["position"]): int(row["cnt"]) for row in rows}

    if agent_mode:
        out.set_data("total_agents", total)
        out.set_data("aware_agents", aware)
        out.set_data("awareness_rate", aware / total if total else 0)
        out.set_data("positions", positions)
    else:
        console.print()
        console.print("[bold]Simulation Summary[/bold]")
        console.print(f"Agents: {total}")
        console.print(f"Aware: {aware} ({aware / total:.1%})")
        console.print("Positions:")
        for row in rows:
            pct = int(row["cnt"]) / total
            console.print(f"  - {row['position']}: {row['cnt']} ({pct:.1%})")


def _display_timeline(
    conn: sqlite3.Connection, run_id: str, out: Output, agent_mode: bool
) -> None:
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
        out.warning("No timestep summaries found.")
        return

    if agent_mode:
        timeline_data = [
            {
                "timestep": row["timestep"],
                "new_exposures": row["new_exposures"],
                "agents_reasoned": row["agents_reasoned"],
                "shares_occurred": row["shares_occurred"],
                "exposure_rate": float(row["exposure_rate"]),
            }
            for row in rows
        ]
        out.set_data("timeline", timeline_data)
    else:
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
    scenario_name: str,
    attribute: str,
    out: Output,
    agent_mode: bool,
) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT agent_id, attrs_json FROM agents WHERE scenario_id = ?",
        (scenario_name,),
    )
    attr_by_agent: dict[str, str] = {}
    for row in cur.fetchall():
        try:
            attrs = json.loads(row["attrs_json"])
        except json.JSONDecodeError:
            continue
        attr_by_agent[str(row["agent_id"])] = str(attrs.get(attribute, "unknown"))

    if not attr_by_agent:
        out.warning("No agent attribute records found.")
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

    if agent_mode:
        segment_data = {
            key: {
                "total": data["total"],
                "aware": data["aware"],
                "awareness_rate": data["aware"] / data["total"] if data["total"] else 0,
            }
            for key, data in groups.items()
        }
        out.set_data("segment_attribute", attribute)
        out.set_data("segments", segment_data)
    else:
        console.print()
        console.print(f"[bold]Segment by {attribute}[/bold]")
        for key, data in sorted(
            groups.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            total = data["total"]
            aware = data["aware"]
            pct = aware / total if total else 0.0
            console.print(f"  - {key}: {total} agents, aware={aware} ({pct:.1%})")


def _display_agent(
    conn: sqlite3.Connection,
    run_id: str,
    scenario_name: str,
    agent_id: str,
    out: Output,
    agent_mode: bool,
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
        out.warning(f"Agent not found in simulation state: {agent_id}")
        return

    cur.execute(
        "SELECT attrs_json FROM agents WHERE scenario_id = ? AND agent_id = ? LIMIT 1",
        (scenario_name, agent_id),
    )
    attrs_row = cur.fetchone()
    attrs = {}
    if attrs_row:
        try:
            attrs = json.loads(attrs_row["attrs_json"])
        except json.JSONDecodeError:
            attrs = {}

    if agent_mode:
        out.set_data("agent_id", agent_id)
        out.set_data("aware", bool(row["aware"]))
        out.set_data("position", row["private_position"] or row["position"])
        out.set_data(
            "sentiment",
            row["private_sentiment"]
            if row["private_sentiment"] is not None
            else row["sentiment"],
        )
        out.set_data(
            "conviction",
            row["private_conviction"]
            if row["private_conviction"] is not None
            else row["conviction"],
        )
        if row["public_statement"]:
            out.set_data("public_statement", row["public_statement"])
        if row["action_intent"]:
            out.set_data("action_intent", row["action_intent"])
        if row["raw_reasoning"]:
            out.set_data("raw_reasoning", row["raw_reasoning"])
        # Filter out internal attributes
        filtered_attrs = {k: v for k, v in attrs.items() if not k.startswith("_")}
        out.set_data("attributes", filtered_attrs)
    else:
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
