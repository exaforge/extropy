"""Agent chat commands backed by study DB history."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import typer

from ...storage import open_study_db
from ..app import app, console, get_json_mode

chat_app = typer.Typer(help="Chat with simulated agents using DB-backed history")
app.add_typer(chat_app, name="chat")


def _load_agent_chat_context(
    conn: sqlite3.Connection,
    run_id: str,
    agent_id: str,
    timeline_n: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cur = conn.cursor()

    cur.execute(
        "SELECT attrs_json FROM agents WHERE agent_id = ? ORDER BY rowid DESC LIMIT 1",
        (agent_id,),
    )
    attrs_row = cur.fetchone()
    attrs = {}
    if attrs_row and attrs_row["attrs_json"]:
        try:
            attrs = json.loads(attrs_row["attrs_json"])
        except json.JSONDecodeError:
            attrs = {}

    cur.execute("SELECT * FROM agent_states WHERE agent_id = ? LIMIT 1", (agent_id,))
    state_row = cur.fetchone()
    state = dict(state_row) if state_row else {}

    cur.execute(
        """
        SELECT timestep, event_type, details_json
        FROM timeline
        WHERE agent_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (agent_id, timeline_n),
    )
    timeline_rows = [dict(r) for r in cur.fetchall()]

    context = {
        "run_id": run_id,
        "agent_id": agent_id,
        "attributes": attrs,
        "state": state,
        "timeline": list(reversed(timeline_rows)),
    }

    citations = [
        {"table": "agents", "agent_id": agent_id},
        {"table": "agent_states", "agent_id": agent_id},
        {"table": "timeline", "agent_id": agent_id, "limit": timeline_n},
    ]
    return context, citations


def _summarize_context(context: dict[str, Any], prompt: str) -> str:
    state = context.get("state", {})
    attrs = context.get("attributes", {})
    timeline = context.get("timeline", [])
    agent_id = context.get("agent_id")

    private_position = state.get("private_position") or state.get("position")
    private_sentiment = state.get("private_sentiment")
    if private_sentiment is None:
        private_sentiment = state.get("sentiment")
    private_conviction = state.get("private_conviction")
    if private_conviction is None:
        private_conviction = state.get("conviction")

    lines = [f"Agent `{agent_id}` context snapshot:"]
    if private_position is not None:
        lines.append(f"- Position: {private_position}")
    if private_sentiment is not None:
        lines.append(f"- Sentiment: {private_sentiment:.3f}")
    if private_conviction is not None:
        lines.append(f"- Conviction: {private_conviction:.3f}")

    if state.get("public_statement"):
        lines.append(f"- Public statement: {state['public_statement']}")
    if state.get("raw_reasoning"):
        lines.append(f"- Latest raw reasoning: {state['raw_reasoning']}")

    if attrs:
        top_attrs = [(k, v) for k, v in attrs.items() if not str(k).startswith("_")]
        top_attrs = sorted(top_attrs)[:8]
        if top_attrs:
            lines.append(
                "- Key attributes: " + ", ".join(f"{k}={v}" for k, v in top_attrs)
            )

    if timeline:
        lines.append("- Recent timeline events:")
        for item in timeline[-5:]:
            details = item.get("details_json") or "{}"
            lines.append(
                f"  - t={item.get('timestep')} {item.get('event_type')} details={details}"
            )

    lines.append(f"- Your prompt: {prompt}")
    lines.append(
        "This answer is grounded in persisted DB state and does not mutate simulation state."
    )
    return "\n".join(lines)


def _print_repl_help() -> None:
    console.print("[dim]Commands: /context, /timeline <n>, /history, /exit[/dim]")


@chat_app.callback(invoke_without_command=True)
def chat_interactive(
    ctx: typer.Context,
    study_db: Path | None = typer.Option(None, "--study-db"),
    run_id: str | None = typer.Option(None, "--run-id"),
    agent_id: str | None = typer.Option(None, "--agent-id"),
    session_id: str | None = typer.Option(None, "--session-id"),
):
    """Interactive chat REPL.

    Example:
        extropy chat --study-db study.db --run-id run_123 --agent-id a_42
    """
    if ctx.invoked_subcommand is not None:
        return

    if not study_db or not run_id or not agent_id:
        console.print(
            "[red]✗[/red] interactive chat requires --study-db, --run-id, and --agent-id"
        )
        raise typer.Exit(1)

    if not study_db.exists():
        console.print(f"[red]✗[/red] Study DB not found: {study_db}")
        raise typer.Exit(1)

    with open_study_db(study_db) as db:
        sid = session_id or db.create_chat_session(
            run_id=run_id,
            agent_id=agent_id,
            mode="interactive",
            meta={"entrypoint": "repl"},
        )

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row

    console.print(f"[bold]Chat session[/bold] {sid}")
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
                with open_study_db(study_db) as db:
                    messages = db.get_chat_messages(sid)
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
                    conn, run_id, agent_id, timeline_n=max(1, n)
                )
                for item in context.get("timeline", []):
                    console.print(
                        f"t={item.get('timestep')} {item.get('event_type')} {item.get('details_json') or '{}'}"
                    )
                continue
            if prompt == "/context":
                context, _ = _load_agent_chat_context(
                    conn, run_id, agent_id, timeline_n=10
                )
                console.print_json(data=context)
                continue

            started = time.time()
            context, citations = _load_agent_chat_context(
                conn, run_id, agent_id, timeline_n=12
            )
            answer = _summarize_context(context, prompt)
            latency_ms = int((time.time() - started) * 1000)

            with open_study_db(study_db) as db:
                db.append_chat_message(sid, "user", prompt)
                db.append_chat_message(
                    sid,
                    "assistant",
                    answer,
                    citations={"sources": citations},
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
    study_db: Path = typer.Option(..., "--study-db"),
    run_id: str = typer.Option(..., "--run-id"),
    agent_id: str = typer.Option(..., "--agent-id"),
    prompt: str = typer.Option(..., "--prompt"),
    session_id: str | None = typer.Option(None, "--session-id"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Non-interactive chat API for automation.

    Example:
        extropy chat ask --study-db study.db --run-id r1 --agent-id a1 --prompt "What changed?" --json
    """
    if not study_db.exists():
        console.print(f"[red]✗[/red] Study DB not found: {study_db}")
        raise typer.Exit(1)

    started = time.time()

    with open_study_db(study_db) as db:
        sid = session_id or db.create_chat_session(
            run_id=run_id,
            agent_id=agent_id,
            mode="machine",
            meta={"entrypoint": "ask"},
        )

    conn = sqlite3.connect(str(study_db))
    conn.row_factory = sqlite3.Row
    try:
        context, citations = _load_agent_chat_context(
            conn, run_id, agent_id, timeline_n=12
        )
        answer = _summarize_context(context, prompt)
    finally:
        conn.close()

    latency_ms = int((time.time() - started) * 1000)

    with open_study_db(study_db) as db:
        user_turn = db.append_chat_message(sid, "user", prompt)
        assistant_turn = db.append_chat_message(
            sid,
            "assistant",
            answer,
            citations={"sources": citations},
            token_usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": latency_ms,
            },
        )

    payload = {
        "session_id": sid,
        "user_turn_index": user_turn,
        "turn_index": assistant_turn,
        "assistant_text": answer,
        "citations": {"sources": citations},
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "latency_ms": latency_ms,
    }

    if json_output or get_json_mode():
        console.print_json(data=payload)
    else:
        console.print(answer)
