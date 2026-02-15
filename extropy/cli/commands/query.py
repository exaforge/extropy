"""Ad-hoc read-only query command."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ...storage import open_study_db, ReadOnlySQLRequest
from ..app import app, console

query_app = typer.Typer(help="Read-only SQL query tools")
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


@query_app.command("sql")
def query_sql(
    study_db: Path = typer.Option(..., "--study-db"),
    sql: str = typer.Option(..., "--sql", help="Read-only SQL statement"),
    limit: int = typer.Option(1000, "--limit", min=1),
    format: str = typer.Option("table", "--format", help="table|json|jsonl"),
):
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
