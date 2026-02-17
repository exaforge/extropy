"""Agent chat commands backed by study DB history."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from ...config import get_config
from ...core.llm import simple_call
from ..app import app, console, get_study_path, is_agent_mode
from ..study import get_study_db
from ..utils import Output

chat_app = typer.Typer(help="Chat with simulated agents using DB-backed history")
app.add_typer(chat_app, name="chat")


def _get_study_db() -> Path:
    """Resolve study DB path via auto-detection or --study flag."""
    try:
        return get_study_db(get_study_path())
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _ensure_chat_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL,
            closed_at TEXT,
            meta_json TEXT
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT,
            token_usage_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, turn_index)
        );
        """
    )
    conn.commit()


def _create_chat_session(
    conn: sqlite3.Connection,
    run_id: str,
    agent_id: str,
    mode: str,
    meta: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> str:
    _ensure_chat_tables(conn)
    sid = session_id or str(uuid.uuid4())
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO chat_sessions
        (session_id, run_id, agent_id, mode, created_at, meta_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (sid, run_id, agent_id, mode, _now_iso(), json.dumps(meta or {})),
    )
    conn.commit()
    return sid


def _append_chat_message(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    citations: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> int:
    _ensure_chat_tables(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT COALESCE(MAX(turn_index), -1) AS max_turn FROM chat_messages WHERE session_id = ?",
        (session_id,),
    )
    turn = int(cur.fetchone()["max_turn"]) + 1
    cur.execute(
        """
        INSERT INTO chat_messages
        (session_id, turn_index, role, content, citations_json, token_usage_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            turn,
            role,
            content,
            json.dumps(citations or {}),
            json.dumps(token_usage or {}),
            _now_iso(),
        ),
    )
    conn.commit()
    return turn


def _get_chat_messages(
    conn: sqlite3.Connection, session_id: str
) -> list[dict[str, Any]]:
    _ensure_chat_tables(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT turn_index, role, content, citations_json, token_usage_json, created_at
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY turn_index
        """,
        (session_id,),
    )
    rows = []
    for row in cur.fetchall():
        rows.append(
            {
                "turn_index": int(row["turn_index"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "citations": json.loads(row["citations_json"] or "{}"),
                "token_usage": json.loads(row["token_usage_json"] or "{}"),
                "created_at": str(row["created_at"]),
            }
        )
    return rows


def _load_agent_chat_context(
    conn: sqlite3.Connection,
    run_id: str,
    agent_id: str,
    timeline_n: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT scenario_name, population_id FROM simulation_runs WHERE run_id = ? LIMIT 1",
        (run_id,),
    )
    run_row = cur.fetchone()
    if not run_row:
        return {"run_id": run_id, "agent_id": agent_id, "error": "run_id not found"}, []
    scenario_name = str(run_row["scenario_name"] or run_row["population_id"])

    cur.execute(
        """
        SELECT attrs_json
        FROM agents
        WHERE scenario_id = ? AND agent_id = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (scenario_name, agent_id),
    )
    attrs_row = cur.fetchone()
    attrs = {}
    if attrs_row and attrs_row["attrs_json"]:
        try:
            attrs = json.loads(attrs_row["attrs_json"])
        except json.JSONDecodeError:
            attrs = {}

    cur.execute(
        "SELECT * FROM agent_states WHERE run_id = ? AND agent_id = ? LIMIT 1",
        (run_id, agent_id),
    )
    state_row = cur.fetchone()
    state = dict(state_row) if state_row else {}

    cur.execute(
        """
        SELECT timestep, event_type, details_json
        FROM timeline
        WHERE run_id = ? AND agent_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (run_id, agent_id, timeline_n),
    )
    timeline_rows = [dict(r) for r in cur.fetchall()]

    context = {
        "run_id": run_id,
        "scenario_name": scenario_name,
        "agent_id": agent_id,
        "attributes": attrs,
        "state": state,
        "timeline": list(reversed(timeline_rows)),
    }

    citations = [
        {"table": "agents", "scenario_id": scenario_name, "agent_id": agent_id},
        {"table": "agent_states", "run_id": run_id, "agent_id": agent_id},
        {
            "table": "timeline",
            "run_id": run_id,
            "agent_id": agent_id,
            "limit": timeline_n,
        },
    ]
    return context, citations


def _render_chat_history(history: list[dict[str, Any]], max_turns: int = 12) -> str:
    if not history:
        return "(no prior conversation)"

    rendered: list[str] = []
    for msg in history[-max_turns:]:
        role = "User" if msg.get("role") == "user" else "Agent"
        content = str(msg.get("content") or "").strip().replace("\n", " ")
        if len(content) > 400:
            content = content[:400].rstrip() + "..."
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def _build_agent_chat_prompt(
    context: dict[str, Any],
    user_prompt: str,
    history: list[dict[str, Any]],
) -> str:
    state = context.get("state", {})
    attrs = context.get("attributes", {})
    timeline = context.get("timeline", [])

    # Keep this compact so chat stays cheap/fast while still grounded.
    context_payload = {
        "run_id": context.get("run_id"),
        "scenario_name": context.get("scenario_name"),
        "agent_id": context.get("agent_id"),
        "attributes": attrs,
        "state": {
            k: state.get(k)
            for k in (
                "aware",
                "position",
                "private_position",
                "public_position",
                "sentiment",
                "private_sentiment",
                "public_sentiment",
                "conviction",
                "private_conviction",
                "public_conviction",
                "action_intent",
                "public_statement",
                "raw_reasoning",
            )
            if k in state
        },
        "recent_timeline": timeline[-8:],
    }

    return (
        "You are answering as this simulated person from a completed simulation run.\n"
        "Stay in first person and in character.\n"
        "Use only the provided simulation context and chat history.\n"
        "Do not claim facts outside the run data.\n"
        "Do not mention being an AI model.\n"
        "If the data is missing, say you're unsure based on what you experienced in this run.\n"
        "Keep responses conversational and concise (2-6 sentences unless asked for more).\n\n"
        "SIMULATION CONTEXT (JSON):\n"
        f"{json.dumps(context_payload, indent=2, default=str)}\n\n"
        "CHAT HISTORY:\n"
        f"{_render_chat_history(history)}\n\n"
        "NEW USER QUESTION:\n"
        f"{user_prompt}"
    )


def _generate_agent_chat_reply(
    context: dict[str, Any],
    user_prompt: str,
    history: list[dict[str, Any]],
) -> tuple[str, str]:
    model = get_config().resolve_sim_strong()
    prompt = _build_agent_chat_prompt(context, user_prompt, history)
    schema = {
        "type": "object",
        "properties": {
            "assistant_text": {
                "type": "string",
                "description": "In-character reply from the simulated agent",
            }
        },
        "required": ["assistant_text"],
        "additionalProperties": False,
    }
    response = simple_call(
        prompt=prompt,
        response_schema=schema,
        schema_name="agent_chat_reply",
        model=model,
        log=False,
        max_tokens=500,
    )
    assistant_text = str(response.get("assistant_text", "")).strip()
    if not assistant_text:
        raise ValueError("LLM returned empty assistant_text for chat reply")
    return assistant_text, model


def _print_repl_help() -> None:
    console.print("[dim]Commands: /context, /timeline <n>, /history, /exit[/dim]")


def _resolve_run_and_agent(
    conn: sqlite3.Connection,
    run_id: str | None,
    agent_id: str | None,
) -> tuple[str, str]:
    cur = conn.cursor()
    if run_id:
        cur.execute(
            """
            SELECT run_id, scenario_name, population_id
            FROM simulation_runs
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        )
    else:
        cur.execute(
            """
            SELECT run_id, scenario_name, population_id
            FROM simulation_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
    run_row = cur.fetchone()
    if not run_row:
        raise ValueError("No simulation runs found in study DB")
    resolved_run_id = str(run_row["run_id"])
    scenario_name = str(run_row["scenario_name"] or run_row["population_id"])

    if agent_id:
        cur.execute(
            "SELECT 1 FROM agent_states WHERE run_id = ? AND agent_id = ? LIMIT 1",
            (resolved_run_id, agent_id),
        )
        if cur.fetchone():
            return resolved_run_id, agent_id
        cur.execute(
            "SELECT 1 FROM agents WHERE scenario_id = ? AND agent_id = ? LIMIT 1",
            (scenario_name, agent_id),
        )
        if cur.fetchone():
            return resolved_run_id, agent_id
        raise ValueError(f"agent_id not found for run/scenario: {agent_id}")

    cur.execute(
        "SELECT agent_id FROM agent_states WHERE run_id = ? ORDER BY agent_id LIMIT 1",
        (resolved_run_id,),
    )
    agent_row = cur.fetchone()
    if not agent_row:
        cur.execute(
            "SELECT agent_id FROM agents WHERE scenario_id = ? ORDER BY agent_id LIMIT 1",
            (scenario_name,),
        )
        agent_row = cur.fetchone()
    if not agent_row:
        raise ValueError("No agents found for resolved run")
    return resolved_run_id, str(agent_row["agent_id"])


@chat_app.command("list")
def chat_list(
    limit_runs: int = typer.Option(10, "--limit-runs", min=1, max=100),
    agents_per_run: int = typer.Option(5, "--agents-per-run", min=1, max=25),
):
    """List recent runs with sample agents for quick chat selection."""
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    study_db = _get_study_db()
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT run_id, scenario_name, status, started_at, completed_at, population_id
            FROM simulation_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit_runs,),
        )
        runs = [dict(r) for r in cur.fetchall()]
        for run in runs:
            run_id = str(run["run_id"])
            scenario_name = str(run.get("scenario_name") or run["population_id"])
            cur.execute(
                """
                SELECT agent_id
                FROM agent_states
                WHERE run_id = ?
                ORDER BY agent_id
                LIMIT ?
                """,
                (run_id, agents_per_run),
            )
            sample_agents = [str(r["agent_id"]) for r in cur.fetchall()]
            if not sample_agents:
                cur.execute(
                    """
                    SELECT agent_id
                    FROM agents
                    WHERE scenario_id = ?
                    ORDER BY agent_id
                    LIMIT ?
                    """,
                    (scenario_name, agents_per_run),
                )
                sample_agents = [str(r["agent_id"]) for r in cur.fetchall()]
            run["sample_agents"] = sample_agents
    finally:
        conn.close()

    out.set_data("study_db", str(study_db))
    out.set_data("runs", runs)

    if agent_mode:
        raise typer.Exit(out.finish())

    if not runs:
        console.print("[yellow]No simulation runs found.[/yellow]")
        return
    console.print(f"[bold]Recent Runs[/bold] ({len(runs)})")
    for run in runs:
        agents = ", ".join(run["sample_agents"]) if run["sample_agents"] else "-"
        scenario_name = run.get("scenario_name") or run["population_id"]
        console.print(
            f"- {run['run_id']} status={run['status']} started={run['started_at']} "
            f"scenario={scenario_name} sample_agents=[{agents}]"
        )


@chat_app.callback(invoke_without_command=True)
def chat_interactive(
    ctx: typer.Context,
    run_id: str | None = typer.Option(None, "--run-id"),
    agent_id: str | None = typer.Option(None, "--agent-id"),
    session_id: str | None = typer.Option(None, "--session-id"),
):
    """Interactive chat REPL.

    Example:
        extropy chat --run-id run_123 --agent-id a_42
    """
    if ctx.invoked_subcommand is not None:
        return

    study_db = _get_study_db()
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        resolved_run_id, resolved_agent_id = _resolve_run_and_agent(
            conn, run_id, agent_id
        )
    except ValueError as e:
        conn.close()
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    sid = _create_chat_session(
        conn=conn,
        run_id=resolved_run_id,
        agent_id=resolved_agent_id,
        mode="interactive",
        meta={"entrypoint": "repl"},
        session_id=session_id,
    )

    console.print(f"[bold]Chat session[/bold] {sid}")
    console.print(
        f"[dim]Using run_id={resolved_run_id} agent_id={resolved_agent_id}[/dim]"
    )
    _print_repl_help()

    try:
        while True:
            try:
                prompt = input("chat> ").strip()
            except EOFError:
                break

            if not prompt:
                continue
            if prompt == "/exit":
                break
            if prompt == "/history":
                messages = _get_chat_messages(conn, sid)
                for m in messages:
                    console.print(f"[{m['role']}] {m['content']}")
                continue
            if prompt.startswith("/timeline"):
                parts = prompt.split()
                try:
                    n = int(parts[1]) if len(parts) > 1 else 10
                except ValueError:
                    n = 10
                context, _ = _load_agent_chat_context(
                    conn, resolved_run_id, resolved_agent_id, timeline_n=max(1, n)
                )
                for item in context.get("timeline", []):
                    console.print(
                        f"t={item.get('timestep')} {item.get('event_type')} {item.get('details_json') or '{}'}"
                    )
                continue
            if prompt == "/context":
                context, _ = _load_agent_chat_context(
                    conn, resolved_run_id, resolved_agent_id, timeline_n=10
                )
                console.print_json(data=context)
                continue

            started = time.time()
            context, citations = _load_agent_chat_context(
                conn, resolved_run_id, resolved_agent_id, timeline_n=12
            )
            history = _get_chat_messages(conn, sid)
            try:
                answer, model_used = _generate_agent_chat_reply(
                    context=context,
                    user_prompt=prompt,
                    history=history,
                )
            except Exception as e:
                console.print(f"[red]✗[/red] LLM chat failed: {e}")
                continue
            latency_ms = int((time.time() - started) * 1000)

            _append_chat_message(conn, sid, "user", prompt)
            _append_chat_message(
                conn,
                sid,
                "assistant",
                answer,
                citations={"sources": citations, "model": model_used},
                token_usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": latency_ms,
                },
            )

            console.print(answer)

    finally:
        conn.close()


@chat_app.command("ask")
def chat_ask(
    run_id: str | None = typer.Option(None, "--run-id"),
    agent_id: str | None = typer.Option(None, "--agent-id"),
    prompt: str = typer.Option(..., "--prompt"),
    session_id: str | None = typer.Option(None, "--session-id"),
):
    """Non-interactive chat API for automation.

    Example:
        extropy chat ask --run-id r1 --agent-id a1 --prompt "What changed?"
    """
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    study_db = _get_study_db()
    started = time.time()
    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        resolved_run_id, resolved_agent_id = _resolve_run_and_agent(
            conn, run_id, agent_id
        )
        sid = _create_chat_session(
            conn=conn,
            run_id=resolved_run_id,
            agent_id=resolved_agent_id,
            mode="machine",
            meta={"entrypoint": "ask"},
            session_id=session_id,
        )
        history = _get_chat_messages(conn, sid)
        context, citations = _load_agent_chat_context(
            conn, resolved_run_id, resolved_agent_id, timeline_n=12
        )
        answer, model_used = _generate_agent_chat_reply(
            context=context,
            user_prompt=prompt,
            history=history,
        )
        latency_ms = int((time.time() - started) * 1000)
        user_turn = _append_chat_message(conn, sid, "user", prompt)
        assistant_turn = _append_chat_message(
            conn,
            sid,
            "assistant",
            answer,
            citations={"sources": citations, "model": model_used},
            token_usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": latency_ms,
            },
        )
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)
    finally:
        conn.close()

    out.set_data("session_id", sid)
    out.set_data("run_id", resolved_run_id)
    out.set_data("agent_id", resolved_agent_id)
    out.set_data("user_turn_index", user_turn)
    out.set_data("turn_index", assistant_turn)
    out.set_data("assistant_text", answer)
    out.set_data("citations", {"sources": citations})
    out.set_data("token_usage", {"input_tokens": 0, "output_tokens": 0})
    out.set_data("latency_ms", latency_ms)

    if agent_mode:
        raise typer.Exit(out.finish())

    console.print(answer)
