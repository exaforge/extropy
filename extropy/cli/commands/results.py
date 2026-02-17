"""Results command for DB-first simulation results."""

from __future__ import annotations

import json
import sqlite3

import typer

from ..app import app, console, is_agent_mode, get_study_path
from ..study import StudyContext, detect_study_folder, parse_version_ref
from ..utils import Output, ExitCode


@app.command("results")
def results_command(
    scenario: str = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (uses latest run if not specified)",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Simulation run id"),
    segment: str | None = typer.Option(
        None, "--segment", help="Attribute to segment by"
    ),
    timeline: bool = typer.Option(False, "--timeline", "-t", help="Show timeline view"),
    agent: str | None = typer.Option(
        None, "--agent", "-a", help="Show single agent details"
    ),
):
    """
    Display simulation results from the canonical study DB.

    By default shows results from the latest simulation run. Use -s to
    filter by scenario, or --run-id for a specific run.

    Examples:
        extropy results                     # latest run
        extropy results -s ai-adoption      # latest run for scenario
        extropy results --timeline          # show timestep progression
        extropy results --segment age_group # segment by attribute
        extropy results --agent agent_123   # show single agent
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
                SELECT run_id, status, started_at, completed_at, stopped_reason, population_id
                FROM simulation_runs
                WHERE run_id = ?
                """,
                (run_id,),
            )
        elif scenario_name:
            # Find latest run for this scenario
            cur.execute(
                """
                SELECT run_id, status, started_at, completed_at, stopped_reason, population_id
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
                SELECT run_id, status, started_at, completed_at, stopped_reason, population_id
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
        population_id = str(run_row["population_id"])

        if not agent_mode:
            console.print(
                f"[dim]run_id={resolved_run_id} status={run_row['status']} "
                f"started_at={run_row['started_at']} completed_at={run_row['completed_at'] or '-'}[/dim]"
            )

        out.set_data("run_id", resolved_run_id)
        out.set_data("status", run_row["status"])
        out.set_data("started_at", run_row["started_at"])
        out.set_data("completed_at", run_row["completed_at"])

        if agent:
            _display_agent(conn, resolved_run_id, population_id, agent, out, agent_mode)
            raise typer.Exit(out.finish())
        if segment:
            _display_segment(
                conn, resolved_run_id, population_id, segment, out, agent_mode
            )
            raise typer.Exit(out.finish())
        if timeline:
            _display_timeline(conn, resolved_run_id, out, agent_mode)
            raise typer.Exit(out.finish())
        _display_summary(conn, resolved_run_id, out, agent_mode)
        raise typer.Exit(out.finish())
    finally:
        conn.close()


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
    population_id: str,
    attribute: str,
    out: Output,
    agent_mode: bool,
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
    population_id: str,
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
