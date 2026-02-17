"""Sample command for generating agents from scenario specs."""

import time
from pathlib import Path

import typer

from ...core.models import PopulationSpec
from ...core.models.scenario import ScenarioSpec
from ...population.validator import validate_spec
from ..app import app, console, is_agent_mode, get_study_path
from ..study import StudyContext, detect_study_folder, parse_version_ref
from ..utils import (
    Output,
    ExitCode,
    format_elapsed,
    format_validation_for_json,
    format_sampling_stats_for_json,
)


@app.command("sample")
def sample_command(
    scenario: str = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (auto-selects if only one exists)",
    ),
    count: int = typer.Option(
        ...,
        "--count",
        "-n",
        help="Number of agents to sample",
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
    Sample agents from a scenario's merged population spec.

    Loads the scenario's base population and extended attributes, merges them,
    validates the merged spec, and samples the requested number of agents.

    Prerequisites:
        - Population spec must exist
        - Scenario spec must exist
        - Persona config must exist for the scenario

    EXIT CODES:
        0 = Success
        1 = Validation error
        2 = File not found
        3 = Sampling error

    Examples:
        extropy sample -s ai-adoption -n 500
        extropy sample -s ai-adoption -n 1000 --seed 42 --report
        extropy sample -n 500  # auto-selects scenario if only one exists
    """
    from ...population.sampler import sample_population, SamplingError

    agent_mode = is_agent_mode()
    out = Output(console, json_mode=agent_mode)
    start_time = time.time()
    out.blank()

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

    # Resolve scenario
    scenario_name, scenario_version = _resolve_scenario(study_ctx, scenario, out)

    # Pre-flight: Check persona config exists
    try:
        study_ctx.get_persona_path(scenario_name)
    except FileNotFoundError:
        out.error(
            f"No persona config found for scenario: {scenario_name}. "
            "Run 'extropy persona -s {scenario_name}' first.",
            exit_code=ExitCode.FILE_NOT_FOUND,
        )
        raise typer.Exit(out.finish())

    # Load scenario spec
    try:
        scenario_path = study_ctx.get_scenario_path(scenario_name, scenario_version)
        scenario_spec = ScenarioSpec.from_yaml(scenario_path)
    except FileNotFoundError:
        out.error(f"Scenario not found: {scenario_name}")
        raise typer.Exit(1)
    except Exception as e:
        out.error(f"Failed to load scenario: {e}")
        raise typer.Exit(1)

    # Load base population spec
    base_pop_ref = scenario_spec.meta.base_population
    if not base_pop_ref:
        out.error("Scenario has no base_population reference")
        raise typer.Exit(1)

    pop_name, pop_version = _parse_base_population_ref(base_pop_ref)
    pop_path = study_ctx.get_population_path(pop_name, pop_version)

    try:
        pop_spec = PopulationSpec.from_yaml(pop_path)
    except Exception as e:
        out.error(f"Failed to load population spec: {e}")
        raise typer.Exit(1)

    # Merge attributes: base population + scenario extended attributes
    merged_attributes = list(pop_spec.attributes)
    extended_attrs = scenario_spec.extended_attributes or []
    merged_attributes.extend(extended_attrs)

    # Compute merged sampling order
    merged_sampling_order = list(pop_spec.sampling_order)
    for attr in extended_attrs:
        if attr.name not in merged_sampling_order:
            merged_sampling_order.append(attr.name)

    # Create merged spec for sampling
    merged_spec = PopulationSpec(
        meta=pop_spec.meta,
        grounding=pop_spec.grounding,
        attributes=merged_attributes,
        sampling_order=merged_sampling_order,
    )

    out.success(
        f"Loaded scenario: [bold]{scenario_name}[/bold] "
        f"({len(merged_attributes)} attributes: {len(pop_spec.attributes)} base + {len(extended_attrs)} extended)",
        scenario=scenario_name,
        base_population=base_pop_ref,
        attribute_count=len(merged_attributes),
        agent_count=count,
    )

    # Validation Gate
    out.blank()
    if not agent_mode:
        with console.status("[cyan]Validating merged spec...[/cyan]"):
            validation_result = validate_spec(merged_spec)
    else:
        validation_result = validate_spec(merged_spec)

    out.set_data("validation", format_validation_for_json(validation_result))

    if not validation_result.valid:
        if skip_validation:
            out.warning(
                f"Spec has {len(validation_result.errors)} error(s) - skipping validation"
            )
        else:
            out.error(
                f"Merged spec has {len(validation_result.errors)} error(s)",
                exit_code=ExitCode.VALIDATION_ERROR,
            )
            if not agent_mode:
                for err in validation_result.errors[:5]:
                    out.text(f"  [red]✗[/red] {err.location}: {err.message}")
                if len(validation_result.errors) > 5:
                    out.text(
                        f"  [dim]... and {len(validation_result.errors) - 5} more[/dim]"
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

    show_progress = count >= 100 and not agent_mode

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
            task = progress.add_task("Sampling", total=count)

            def on_progress(current: int, total: int):
                progress.update(task, completed=current)

            try:
                result = sample_population(
                    merged_spec, count=count, seed=seed, on_progress=on_progress
                )
            except SamplingError as e:
                sampling_error = e
    else:
        if not agent_mode:
            with console.status("[cyan]Sampling agents...[/cyan]"):
                try:
                    result = sample_population(merged_spec, count=count, seed=seed)
                except SamplingError as e:
                    sampling_error = e
        else:
            try:
                result = sample_population(merged_spec, count=count, seed=seed)
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
    if agent_mode or report:
        out.set_data("stats", format_sampling_stats_for_json(result.stats, merged_spec))

    if report and not agent_mode:
        _show_sampling_report(out, result, merged_spec)

    # Household report (if applicable)
    households = getattr(result, "_households", [])
    if households and report and not agent_mode:
        _show_household_report(out, households, result)

    if households and agent_mode:
        out.set_data("household_count", len(households))
        out.set_data(
            "household_type_distribution",
            result.meta.get("household_type_distribution", {}),
        )

    # Save to canonical DB with scenario_id
    out.blank()
    db_path = study_ctx.db_path

    if not agent_mode:
        with console.status(f"[cyan]Saving to study DB: {db_path}...[/cyan]"):
            sample_run_id = _save_to_db(
                db_path, scenario_name, pop_spec, pop_path, result, households
            )
    else:
        sample_run_id = _save_to_db(
            db_path, scenario_name, pop_spec, pop_path, result, households
        )

    elapsed = time.time() - start_time

    out.set_data("study_db", str(db_path))
    out.set_data("scenario_id", scenario_name)
    out.set_data("sample_run_id", sample_run_id)
    out.set_data("total_time_seconds", elapsed)

    out.divider()
    out.success(
        f"Saved {len(result.agents)} agents to [bold]{db_path}[/bold] "
        f"(scenario_id={scenario_name}, sample_run_id={sample_run_id})"
    )
    out.text(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    out.divider()

    raise typer.Exit(out.finish())


def _resolve_scenario(
    study_ctx: StudyContext, scenario_ref: str | None, out: Output
) -> tuple[str, int | None]:
    """Resolve scenario name and version."""
    scenarios = study_ctx.list_scenarios()

    if not scenarios:
        out.error("No scenarios found. Run 'extropy scenario' first.")
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


def _parse_base_population_ref(ref: str) -> tuple[str, int | None]:
    """Parse base_population reference like 'population.v2'."""
    import re

    match = re.match(r"^(.+)\.v(\d+)$", ref)
    if match:
        return match.group(1), int(match.group(2))
    return ref, None


def _save_to_db(
    db_path: Path,
    scenario_name: str,
    pop_spec: PopulationSpec,
    pop_path: Path,
    result,
    households: list,
) -> str:
    """Save sampling results to study database."""
    from ...storage import open_study_db

    with open_study_db(db_path) as db:
        # Save population spec (using scenario_name as population_id for backwards compat)
        db.save_population_spec(
            population_id=scenario_name,
            spec_yaml=pop_path.read_text(encoding="utf-8"),
            source_path=str(pop_path),
        )

        # Save agents with scenario_id
        sample_run_id = db.save_sample_result(
            population_id=scenario_name,
            agents=result.agents,
            meta=result.meta,
            seed=result.meta.get("seed"),
            scenario_id=scenario_name,
        )

        if households:
            db.save_households(
                population_id=scenario_name,
                sample_run_id=sample_run_id,
                households=households,
            )

    return sample_run_id


def _show_sampling_report(out: Output, result, spec: PopulationSpec) -> None:
    """Display sampling statistics report."""
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
            ["Attribute", "Mean", "Std"],
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
            ["Attribute", "Top Values"],
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


def _show_household_report(out: Output, households: list, result) -> None:
    """Display household statistics report."""
    out.header("HOUSEHOLD REPORT")
    type_counts: dict[str, int] = {}
    for hh in households:
        ht = hh["household_type"]
        type_counts[ht] = type_counts.get(ht, 0) + 1
    hh_rows = [[htype, str(cnt)] for htype, cnt in sorted(type_counts.items())]
    out.table(
        "Household Types",
        ["Type", "Count"],
        hh_rows,
        styles=["cyan", None],
    )
    out.text(
        f"  Total households: {len(households)}, Total agents: {len(result.agents)}"
    )
    out.blank()
