"""Migration commands for DB-first runtime artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from ...storage import open_study_db
from ..app import app, console

migrate_app = typer.Typer(help="Migrate legacy artifacts to DB-first schema")
app.add_typer(migrate_app, name="migrate")


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@migrate_app.command("legacy")
def migrate_legacy_artifacts(
    study_db: Path = typer.Option(..., "--study-db", help="Target canonical study DB"),
    agents_file: Path | None = typer.Option(
        None, "--agents-file", help="Legacy agents JSON"
    ),
    network_file: Path | None = typer.Option(
        None, "--network-file", help="Legacy network JSON"
    ),
    population_spec: Path | None = typer.Option(
        None,
        "--population-spec",
        help="Optional population spec YAML source used for sample provenance",
    ),
    population_id: str = typer.Option("default", "--population-id"),
    network_id: str = typer.Option("default", "--network-id"),
):
    """Ingest legacy `agents.json`/`network.json` into `study.db`."""
    if agents_file is None and network_file is None:
        console.print(
            "[red]✗[/red] Provide at least one of --agents-file or --network-file"
        )
        raise typer.Exit(1)

    with open_study_db(study_db) as db:
        if population_spec is not None:
            if not population_spec.exists():
                console.print(
                    f"[red]✗[/red] population spec not found: {population_spec}"
                )
                raise typer.Exit(1)
            db.save_population_spec(
                population_id=population_id,
                spec_yaml=population_spec.read_text(encoding="utf-8"),
                source_path=str(population_spec),
            )

        if agents_file is not None:
            if not agents_file.exists():
                console.print(f"[red]✗[/red] agents file not found: {agents_file}")
                raise typer.Exit(1)
            agents_data = _load_json(agents_file)
            if not isinstance(agents_data, list):
                console.print("[red]✗[/red] agents JSON must be a list")
                raise typer.Exit(1)
            sample_run_id = db.save_sample_result(
                population_id=population_id,
                agents=agents_data,
                meta={
                    "source": "legacy_migration",
                    "source_file": str(agents_file),
                },
                seed=None,
            )
            console.print(
                f"[green]✓[/green] Imported {len(agents_data)} agents "
                f"(population_id={population_id}, sample_run_id={sample_run_id})"
            )

        if network_file is not None:
            if not network_file.exists():
                console.print(f"[red]✗[/red] network file not found: {network_file}")
                raise typer.Exit(1)
            network_data = _load_json(network_file)
            if not isinstance(network_data, dict):
                console.print("[red]✗[/red] network JSON must be an object")
                raise typer.Exit(1)

            raw_edges = network_data.get("edges", [])
            if not isinstance(raw_edges, list):
                console.print("[red]✗[/red] network.edges must be a list")
                raise typer.Exit(1)

            network_run_id = db.save_network_result(
                population_id=population_id,
                network_id=network_id,
                config=network_data.get("config", {}),
                result_meta=network_data.get("meta", {}),
                edges=raw_edges,
                seed=None,
                candidate_mode="legacy",
                network_metrics=network_data.get("metrics"),
            )
            console.print(
                f"[green]✓[/green] Imported {len(raw_edges)} edges "
                f"(network_id={network_id}, network_run_id={network_run_id})"
            )

    console.print(f"[green]✓[/green] Migration complete: {study_db}")


@migrate_app.command("scenario")
def migrate_scenario_yaml(
    input_path: Path = typer.Option(..., "--input", help="Legacy scenario YAML"),
    study_db: Path = typer.Option(..., "--study-db", help="Canonical study DB path"),
    population_id: str = typer.Option("default", "--population-id"),
    network_id: str = typer.Option("default", "--network-id"),
    output: Path | None = typer.Option(None, "--output", "-o"),
):
    """Rewrite a legacy scenario YAML to DB-first metadata fields."""
    if not input_path.exists():
        console.print(f"[red]✗[/red] Scenario file not found: {input_path}")
        raise typer.Exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    meta = data.get("meta")
    if not isinstance(meta, dict):
        console.print("[red]✗[/red] Invalid scenario YAML: missing meta object")
        raise typer.Exit(1)

    had_legacy = "agents_file" in meta or "network_file" in meta
    meta.pop("agents_file", None)
    meta.pop("network_file", None)
    meta["study_db"] = str(study_db)
    meta["population_id"] = population_id
    meta["network_id"] = network_id
    data["meta"] = meta

    out = output or input_path.with_suffix(".db-first.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    if had_legacy:
        console.print(f"[green]✓[/green] Migrated legacy scenario -> {out}")
    else:
        console.print(f"[green]✓[/green] Rewrote scenario metadata -> {out}")
