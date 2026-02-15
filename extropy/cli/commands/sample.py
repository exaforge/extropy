"""Sample command for generating agents from population specs."""

import time
from pathlib import Path

import typer

from ...core.models import PopulationSpec
from ...population.validator import validate_spec
from ..app import app, console, get_json_mode
from ..utils import (
    Output,
    ExitCode,
    format_elapsed,
    format_validation_for_json,
    format_sampling_stats_for_json,
)


@app.command("sample")
def sample_command(
    spec_file: Path = typer.Argument(
        ..., help="Population spec YAML file to sample from"
    ),
    study_db: Path = typer.Option(..., "--study-db", help="Canonical study database"),
    population_id: str = typer.Option(
        "default", "--population-id", help="Population identifier inside study DB"
    ),
    count: int | None = typer.Option(
        None, "--count", "-n", help="Number of agents (default: spec.meta.size)"
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
    report: bool = typer.Option(
        False, "--report", "-r", help="Show distribution summaries and stats"
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip validator errors"
    ),
):
    """
    Generate agents from a population spec.

    Samples from a PopulationSpec YAML file, producing a population of agents
    with attributes matching the spec's distributions and dependencies.

    EXIT CODES:
        0 = Success
        1 = Validation error
        2 = File not found
        3 = Sampling error

    EXAMPLES:
        extropy sample surgeons.yaml --study-db study.db
        extropy sample surgeons.yaml --study-db study.db --population-id main --seed 42
        extropy sample surgeons.yaml --study-db study.db --count 1000 --report
        extropy --json sample surgeons.yaml --study-db study.db --report
    """
    from ...population.sampler import (
        sample_population,
        SamplingError,
    )
    from ...storage import open_study_db

    out = Output(console, json_mode=get_json_mode())
    start_time = time.time()
    out.blank()

    # Load Spec
    if not spec_file.exists():
        out.error(
            f"Spec file not found: {spec_file}",
            exit_code=ExitCode.FILE_NOT_FOUND,
            suggestion=f"Check the file path: {spec_file.absolute()}",
        )
        raise typer.Exit(out.finish())

    if not get_json_mode():
        with console.status("[cyan]Loading spec...[/cyan]"):
            try:
                spec = PopulationSpec.from_yaml(spec_file)
            except Exception as e:
                out.error(
                    f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR
                )
                raise typer.Exit(out.finish())
    else:
        try:
            spec = PopulationSpec.from_yaml(spec_file)
        except Exception as e:
            out.error(f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR)
            raise typer.Exit(out.finish())

    effective_count = count if count is not None else spec.meta.size
    out.success(
        f"Loaded: [bold]{spec.meta.description}[/bold] "
        f"({len(spec.attributes)} attributes, sampling {effective_count} agents)",
        spec_file=str(spec_file),
        description=spec.meta.description,
        attribute_count=len(spec.attributes),
        agent_count=effective_count,
    )

    # Validation Gate
    out.blank()
    if not get_json_mode():
        with console.status("[cyan]Validating spec...[/cyan]"):
            validation_result = validate_spec(spec)
    else:
        validation_result = validate_spec(spec)

    out.set_data("validation", format_validation_for_json(validation_result))

    if not validation_result.valid:
        if skip_validation:
            out.warning(
                f"Spec has {len(validation_result.errors)} error(s) - skipping validation (--skip-validation)"
            )
            if not get_json_mode():
                for err in validation_result.errors[:5]:
                    out.text(f"  [red]✗[/red] {err.location}: {err.message}")
                if len(validation_result.errors) > 5:
                    out.text(
                        f"  [dim]... and {len(validation_result.errors) - 5} more[/dim]"
                    )
        else:
            out.error(
                f"Spec has {len(validation_result.errors)} error(s)",
                exit_code=ExitCode.VALIDATION_ERROR,
            )

            if not get_json_mode():
                error_rows = []
                for err in validation_result.errors[:10]:
                    error_rows.append([err.location, err.message[:50]])

                out.table(
                    "Validation Errors",
                    ["Attribute", "Message"],
                    error_rows,
                    styles=["red", None],
                )

                if len(validation_result.errors) > 10:
                    out.text(
                        f"  [dim]... and {len(validation_result.errors) - 10} more[/dim]"
                    )
                out.blank()
                out.text("[dim]Use --skip-validation to sample anyway[/dim]")

            raise typer.Exit(out.finish())
    else:
        if validation_result.warnings:
            out.success(
                f"Spec validated with {len(validation_result.warnings)} warning(s)"
            )
        else:
            out.success("Spec validated")

    # Sampling
    out.blank()
    sampling_start = time.time()
    result = None
    sampling_error = None

    show_progress = effective_count >= 100 and not get_json_mode()

    if show_progress:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Sampling agents...[/cyan]"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Sampling", total=effective_count)

            def on_progress(current: int, total: int):
                progress.update(task, completed=current)

            try:
                result = sample_population(
                    spec, count=effective_count, seed=seed, on_progress=on_progress
                )
            except SamplingError as e:
                sampling_error = e
    else:
        if not get_json_mode():
            with console.status("[cyan]Sampling agents...[/cyan]"):
                try:
                    result = sample_population(spec, count=effective_count, seed=seed)
                except SamplingError as e:
                    sampling_error = e
        else:
            try:
                result = sample_population(spec, count=effective_count, seed=seed)
            except SamplingError as e:
                sampling_error = e

    if sampling_error:
        out.error(
            f"Sampling failed: {sampling_error}",
            exit_code=ExitCode.SAMPLING_ERROR,
            suggestion="Check attribute dependencies and formula syntax",
        )
        raise typer.Exit(out.finish())

    sampling_elapsed = time.time() - sampling_start
    out.success(
        f"Sampled {len(result.agents)} agents ({format_elapsed(sampling_elapsed)}, seed={result.meta['seed']})",
        sampled_count=len(result.agents),
        seed=result.meta["seed"],
        sampling_time_seconds=sampling_elapsed,
    )

    # Report
    if get_json_mode() or report:
        out.set_data("stats", format_sampling_stats_for_json(result.stats, spec))

    if report and not get_json_mode():
        out.header("SAMPLING REPORT")

        # Numeric attributes
        numeric_attrs = [a for a in spec.attributes if a.type in ("int", "float")]
        if numeric_attrs:
            numeric_rows = []
            for attr in numeric_attrs[:20]:
                mean = result.stats.attribute_means.get(attr.name, 0)
                std = result.stats.attribute_stds.get(attr.name, 0)
                numeric_rows.append([attr.name, f"{mean:.2f}", f"{std:.2f}"])
            out.table(
                "Numeric Attributes",
                ["Attribute", "Mean (μ)", "Std (σ)"],
                numeric_rows,
                styles=["cyan", None, "dim"],
            )
            if len(numeric_attrs) > 20:
                out.text(f"  [dim]... and {len(numeric_attrs) - 20} more[/dim]")
            out.blank()

        # Categorical attributes
        cat_attrs = [a for a in spec.attributes if a.type == "categorical"]
        if cat_attrs:
            cat_rows = []
            for attr in cat_attrs[:15]:
                counts = result.stats.categorical_counts.get(attr.name, {})
                total = sum(counts.values()) or 1
                top_3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
                dist_str = ", ".join(f"{k}: {v / total:.0%}" for k, v in top_3)
                cat_rows.append([attr.name, dist_str])
            out.table(
                "Categorical Attributes",
                ["Attribute", "Top Values (%)"],
                cat_rows,
                styles=["cyan", None],
            )
            if len(cat_attrs) > 15:
                out.text(f"  [dim]... and {len(cat_attrs) - 15} more[/dim]")
            out.blank()

        # Boolean attributes
        bool_attrs = [a for a in spec.attributes if a.type == "boolean"]
        if bool_attrs:
            bool_rows = []
            for attr in bool_attrs[:15]:
                counts = result.stats.boolean_counts.get(attr.name, {True: 0, False: 0})
                total = sum(counts.values()) or 1
                pct_true = counts.get(True, 0) / total
                bool_rows.append([attr.name, f"{pct_true:.1%}"])
            out.table(
                "Boolean Attributes",
                ["Attribute", "True %"],
                bool_rows,
                styles=["cyan", None],
            )
            out.blank()

    # Save to canonical DB
    out.blank()
    if not get_json_mode():
        with console.status(f"[cyan]Saving to study DB: {study_db}...[/cyan]"):
            with open_study_db(study_db) as db:
                db.save_population_spec(
                    population_id=population_id,
                    spec_yaml=spec_file.read_text(encoding="utf-8"),
                    source_path=str(spec_file),
                )
                sample_run_id = db.save_sample_result(
                    population_id=population_id,
                    agents=result.agents,
                    meta=result.meta,
                    seed=result.meta.get("seed"),
                )
    else:
        with open_study_db(study_db) as db:
            db.save_population_spec(
                population_id=population_id,
                spec_yaml=spec_file.read_text(encoding="utf-8"),
                source_path=str(spec_file),
            )
            sample_run_id = db.save_sample_result(
                population_id=population_id,
                agents=result.agents,
                meta=result.meta,
                seed=result.meta.get("seed"),
            )

    elapsed = time.time() - start_time

    out.set_data("study_db", str(study_db))
    out.set_data("population_id", population_id)
    out.set_data("sample_run_id", sample_run_id)
    out.set_data("total_time_seconds", elapsed)

    out.divider()
    out.success(
        f"Saved {len(result.agents)} agents to [bold]{study_db}[/bold] "
        f"(population_id={population_id}, sample_run_id={sample_run_id})"
    )
    out.text(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    out.divider()

    raise typer.Exit(out.finish())
