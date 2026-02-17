"""Scenario command for creating scenario specs with extended attributes."""

import time
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ...core.models import PopulationSpec
from ...population.spec_builder import (
    select_attributes,
    hydrate_attributes,
    bind_constraints,
)
from ...utils import topological_sort, CircularDependencyError
from ..app import app, console, is_agent_mode, get_study_path
from ..study import (
    StudyContext,
    detect_study_folder,
    parse_population_ref,
)
from ..display import display_extend_attributes
from ..utils import format_elapsed, Output, ExitCode


@app.command("scenario")
def scenario_command(
    description: str = typer.Argument(
        ..., help="Scenario description (what event/situation to simulate)"
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Scenario name (creates scenario/{name}/scenario.v1.yaml)",
    ),
    population_ref: str | None = typer.Argument(
        None,
        help="Population version reference: @pop:v1, @pop:latest, or path to YAML",
    ),
    rebase: str | None = typer.Option(
        None,
        "--rebase",
        help="Rebase existing scenario to new population version (e.g. @pop:v2)",
    ),
    timeline: str = typer.Option(
        "auto",
        "--timeline",
        help="Timeline mode: auto (LLM decides), static (single event), evolving (multi-event)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
):
    """
    Create a scenario with extended attributes and event configuration.

    This command:
    1. Discovers scenario-specific attributes (behavioral, psychographic)
    2. Researches distributions for new attributes
    3. Creates event/exposure/outcome configuration

    Examples:
        # Create new scenario
        extropy scenario "AI diagnostic tool adoption" -o ai-adoption

        # Pin population version
        extropy scenario "vaccine mandate" -o vaccine @pop:v1

        # Rebase existing scenario to new population
        extropy scenario ai-adoption --rebase @pop:v2
    """
    from ...core.cost.tracker import CostTracker

    CostTracker.get().set_context(command="scenario")

    start_time = time.time()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    if not agent_mode:
        console.print()

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

    # Handle rebase mode (quick operation, no LLM calls)
    if rebase:
        _handle_rebase(study_ctx, output, rebase, out, agent_mode)
        return

    # Resolve population spec
    pop_spec, pop_version = _resolve_population(
        study_ctx, population_ref, out, agent_mode
    )

    # Resolve output path with auto-versioning
    scenario_name = output
    scenario_dir = study_ctx.get_scenario_dir(scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    next_ver = study_ctx.get_next_scenario_version(scenario_name)
    scenario_path = scenario_dir / f"scenario.v{next_ver}.yaml"

    if not agent_mode:
        console.print(
            f"[dim]Creating scenario: {scenario_name}/scenario.v{next_ver}.yaml[/dim]"
        )
        console.print(f"[dim]Base population: population.v{pop_version}.yaml[/dim]")
        console.print()

    # Step 1: Attribute Selection (Extend Mode)
    selection_start = time.time()
    new_attributes = None
    selection_done = Event()
    selection_error = None

    def do_selection():
        nonlocal new_attributes, selection_error
        try:
            new_attributes = select_attributes(
                description=description,
                geography=pop_spec.meta.geography,
                context=pop_spec.attributes,
            )
        except Exception as e:
            selection_error = e
        finally:
            selection_done.set()

    selection_thread = Thread(target=do_selection, daemon=True)
    selection_thread.start()

    spinner = Spinner("dots", text="Discovering scenario attributes...", style="cyan")
    with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
        while not selection_done.is_set():
            elapsed = time.time() - selection_start
            spinner.update(
                text=f"Discovering scenario attributes... {format_elapsed(elapsed)}"
            )
            time.sleep(0.1)

    selection_elapsed = time.time() - selection_start

    if selection_error:
        out.error(f"Attribute selection failed: {selection_error}")
        raise typer.Exit(1)

    if not agent_mode:
        console.print(
            f"[green]✓[/green] Found {len(new_attributes)} NEW attributes ({format_elapsed(selection_elapsed)})"
        )

    # Human Checkpoint #1
    if not agent_mode:
        display_extend_attributes(
            len(pop_spec.attributes), new_attributes, pop_spec.meta.geography
        )

        if not yes:
            choice = (
                typer.prompt("[Y] Proceed  [n] Cancel", default="Y", show_default=False)
                .strip()
                .lower()
            )
            if choice == "n":
                console.print("[dim]Cancelled.[/dim]")
                raise typer.Exit(0)

    # Early cycle detection
    try:
        base_names = {a.name for a in pop_spec.attributes}
        deps = {a.name: a.depends_on for a in new_attributes}
        deps_filtered = {
            name: [d for d in ds if d not in base_names] for name, ds in deps.items()
        }
        topological_sort(deps_filtered)
    except CircularDependencyError as e:
        out.error(f"Circular dependency: {e}")
        raise typer.Exit(1)

    # Step 2: Distribution Research
    if not agent_mode:
        console.print()

    hydration_start = time.time()
    hydrated = None
    sources = []
    warnings = []
    hydration_done = Event()
    hydration_error = None
    current_step = ["2a", "Starting..."]

    def on_progress(step: str, status: str, count: int | None):
        current_step[0] = step
        current_step[1] = status

    household_config = None
    name_config = None

    def do_hydration():
        nonlocal \
            hydrated, \
            sources, \
            warnings, \
            hydration_error, \
            household_config, \
            name_config
        try:
            hydrated, sources, warnings, household_config, name_config = (
                hydrate_attributes(
                    attributes=new_attributes,
                    description=f"{pop_spec.meta.description} + {description}",
                    geography=pop_spec.meta.geography,
                    context=pop_spec.attributes,
                    on_progress=on_progress,
                )
            )
        except Exception as e:
            hydration_error = e
        finally:
            hydration_done.set()

    hydration_thread = Thread(target=do_hydration, daemon=True)
    hydration_thread.start()

    spinner = Spinner("dots", text="Starting...", style="cyan")
    with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
        while not hydration_done.is_set():
            elapsed = time.time() - hydration_start
            step, status = current_step
            spinner.update(text=f"Step {step}: {status} {format_elapsed(elapsed)}")
            time.sleep(0.1)

    hydration_elapsed = time.time() - hydration_start

    if hydration_error:
        out.error(f"Distribution research failed: {hydration_error}")
        raise typer.Exit(1)

    if not agent_mode:
        console.print(
            f"[green]✓[/green] Researched distributions ({format_elapsed(hydration_elapsed)}, {len(sources)} sources)"
        )

        if warnings:
            console.print(f"[yellow]⚠[/yellow] {len(warnings)} validation warning(s):")
            for w in warnings[:5]:
                console.print(f"  [dim]- {w}[/dim]")
            if len(warnings) > 5:
                console.print(f"  [dim]... and {len(warnings) - 5} more[/dim]")

    # Step 3: Constraint Binding
    with console.status("[cyan]Binding constraints...[/cyan]"):
        try:
            bound_attrs, sampling_order, bind_warnings = bind_constraints(
                hydrated, context=pop_spec.attributes
            )
        except CircularDependencyError as e:
            out.error(f"Circular dependency detected: {e}")
            raise typer.Exit(1)
        except Exception as e:
            out.error(f"Constraint binding failed: {e}")
            raise typer.Exit(1)

    if not agent_mode:
        console.print("[green]✓[/green] Constraints bound")

        if bind_warnings:
            console.print(
                f"[yellow]⚠[/yellow] {len(bind_warnings)} binding warning(s):"
            )
            for w in bind_warnings[:5]:
                console.print(f"  [dim]- {w}[/dim]")
            if len(bind_warnings) > 5:
                console.print(f"  [dim]... and {len(bind_warnings) - 5} more[/dim]")

    # Step 4: Create Scenario Spec
    # For now, we'll create a simpler scenario spec format that stores:
    # - Extended attributes
    # - Reference to base population
    # - Scenario configuration (event, exposure, outcomes) will be generated on demand

    from ...scenario import create_scenario_spec

    if not agent_mode:
        console.print()

    # Run scenario pipeline
    current_step = ["4/5", "Creating scenario config..."]
    pipeline_done = Event()
    pipeline_error = None
    result_spec = None
    validation_result = None

    def on_scenario_progress(step: str, status: str):
        current_step[0] = step
        current_step[1] = status

    def run_pipeline():
        nonlocal result_spec, validation_result, pipeline_error
        try:
            timeline_mode = None if timeline == "auto" else timeline
            # We need to create a merged spec first to pass to scenario creation
            # For now, create scenario without agents (they'll be sampled later)
            result_spec, validation_result = create_scenario_spec(
                description=description,
                population_spec=pop_spec,
                extended_attributes=bound_attrs,
                on_progress=on_scenario_progress,
                timeline_mode=timeline_mode,
            )
        except Exception as e:
            pipeline_error = e
        finally:
            pipeline_done.set()

    pipeline_thread = Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()

    spinner = Spinner("dots", text="Creating scenario config...", style="cyan")
    with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
        while not pipeline_done.is_set():
            elapsed = time.time() - start_time
            step, status = current_step
            spinner.update(text=f"Step {step}: {status} {format_elapsed(elapsed)}")
            time.sleep(0.1)

    if pipeline_error:
        out.error(f"Scenario creation failed: {pipeline_error}")
        raise typer.Exit(1)

    if not agent_mode:
        console.print("[green]✓[/green] Scenario spec ready")

    # Set metadata
    result_spec.meta.name = scenario_name
    result_spec.meta.base_population = f"population.v{pop_version}"
    result_spec.extended_attributes = bound_attrs

    # Display Summary
    if not agent_mode:
        _display_scenario_summary(result_spec)

    # Validation
    if validation_result and validation_result.errors:
        invalid_path = scenario_path.with_suffix(".invalid.yaml")
        result_spec.to_yaml(invalid_path)
        if not agent_mode:
            console.print(
                f"[yellow]⚠[/yellow] Scenario saved to [bold]{invalid_path}[/bold] for review"
            )
        out.error("Scenario validation failed")
        raise typer.Exit(1)

    # Human Checkpoint #2
    if not agent_mode and not yes:
        choice = (
            typer.prompt("[Y] Save  [n] Cancel", default="Y", show_default=False)
            .strip()
            .lower()
        )
        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Save
    result_spec.to_yaml(scenario_path)
    elapsed = time.time() - start_time

    if agent_mode:
        out.success(
            "Scenario saved",
            output=str(scenario_path),
            scenario_name=scenario_name,
            base_population=f"population.v{pop_version}",
            extended_attributes=len(bound_attrs),
            elapsed_seconds=elapsed,
        )
        raise typer.Exit(out.finish())
    else:
        console.print()
        console.print("═" * 60)
        console.print(
            f"[green]✓[/green] Scenario saved to [bold]{scenario_path}[/bold]"
        )
        console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
        console.print("═" * 60)


def _resolve_population(
    study_ctx: StudyContext,
    population_ref: str | None,
    out: Output,
    agent_mode: bool,
) -> tuple[PopulationSpec, int]:
    """Resolve population spec from reference or latest.

    Returns:
        Tuple of (PopulationSpec, version_number)
    """
    if population_ref is None:
        # Use latest
        version = study_ctx.get_latest_population_version()
        if version is None:
            out.error(
                "No population spec found. Run 'extropy spec' first.",
                exit_code=ExitCode.FILE_NOT_FOUND,
            )
            raise typer.Exit(out.finish())
    elif population_ref.startswith("@pop:"):
        # Parse @pop:vN or @pop:latest
        try:
            _, version = parse_population_ref(population_ref)
        except ValueError as e:
            out.error(str(e), exit_code=ExitCode.VALIDATION_ERROR)
            raise typer.Exit(out.finish())
        if version is None:
            version = study_ctx.get_latest_population_version()
            if version is None:
                out.error("No population spec found.")
                raise typer.Exit(1)
    else:
        # Explicit path - not supported in new flow
        out.error(
            "Use @pop:vN or @pop:latest to reference population versions.",
            exit_code=ExitCode.VALIDATION_ERROR,
        )
        raise typer.Exit(out.finish())

    pop_path = study_ctx.get_population_path(version=version)
    if not pop_path.exists():
        out.error(f"Population spec not found: {pop_path}")
        raise typer.Exit(1)

    try:
        pop_spec = PopulationSpec.from_yaml(pop_path)
    except Exception as e:
        out.error(f"Failed to load population spec: {e}")
        raise typer.Exit(1)

    return pop_spec, version


def _handle_rebase(
    study_ctx: StudyContext,
    scenario_name: str,
    new_pop_ref: str,
    out: Output,
    agent_mode: bool,
) -> None:
    """Handle --rebase flag for cheap population updates."""
    # Load existing scenario
    try:
        scenario_path = study_ctx.get_scenario_path(scenario_name)
    except FileNotFoundError:
        out.error(f"Scenario not found: {scenario_name}")
        raise typer.Exit(1)

    from ...core.models.scenario import ScenarioSpec

    try:
        scenario_spec = ScenarioSpec.from_yaml(scenario_path)
    except Exception as e:
        out.error(f"Failed to load scenario: {e}")
        raise typer.Exit(1)

    # Resolve new population
    pop_spec, pop_version = _resolve_population(study_ctx, new_pop_ref, out, agent_mode)

    # Safety check: ensure scenario's attribute dependencies exist in new population
    base_attr_names = {a.name for a in pop_spec.attributes}
    if (
        hasattr(scenario_spec, "extended_attributes")
        and scenario_spec.extended_attributes
    ):
        for attr in scenario_spec.extended_attributes:
            for dep in getattr(attr, "depends_on", []) or []:
                if dep not in base_attr_names:
                    out.error(
                        f"Unsafe rebase: scenario attribute '{attr.name}' depends on "
                        f"'{dep}' which doesn't exist in population.v{pop_version}"
                    )
                    raise typer.Exit(1)

    # Create new version with updated reference
    next_ver = study_ctx.get_next_scenario_version(scenario_name)
    new_path = study_ctx.get_scenario_dir(scenario_name) / f"scenario.v{next_ver}.yaml"

    scenario_spec.meta.base_population = f"population.v{pop_version}"
    scenario_spec.to_yaml(new_path)

    if agent_mode:
        out.success(
            "Scenario rebased",
            output=str(new_path),
            scenario_name=scenario_name,
            new_base_population=f"population.v{pop_version}",
        )
        raise typer.Exit(out.finish())
    else:
        console.print(
            f"[green]✓[/green] Rebased {scenario_name} to population.v{pop_version}"
        )
        console.print(f"[dim]Created {new_path.name}[/dim]")


def _display_scenario_summary(spec) -> None:
    """Display scenario spec summary."""
    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " SCENARIO SPEC READY".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    # Event info
    if spec.event:
        event = spec.event
        content_preview = (
            event.content[:50] + "..." if len(event.content) > 50 else event.content
        )
        console.print(f'[bold]Event:[/bold] {event.type.value} — "{content_preview}"')
        console.print(
            f"[bold]Source:[/bold] {event.source} (credibility: {event.credibility:.2f})"
        )
        console.print()

    # Exposure info
    if spec.seed_exposure:
        console.print("[bold]Exposure Channels:[/bold]")
        for ch in spec.seed_exposure.channels:
            console.print(f"  - {ch.name} ({ch.reach})")
        console.print()

    # Extended attributes
    if hasattr(spec, "extended_attributes") and spec.extended_attributes:
        console.print(
            f"[bold]Extended Attributes:[/bold] {len(spec.extended_attributes)}"
        )
        for attr in spec.extended_attributes[:5]:
            console.print(f"  - {attr.name} ({attr.type.value})")
        if len(spec.extended_attributes) > 5:
            console.print(
                f"  [dim]... and {len(spec.extended_attributes) - 5} more[/dim]"
            )
        console.print()

    # Outcomes info
    if spec.outcomes:
        console.print("[bold]Outcomes:[/bold]")
        for outcome in spec.outcomes.suggested_outcomes:
            type_info = outcome.type.value
            if outcome.options:
                type_info += f": {', '.join(outcome.options[:3])}"
                if len(outcome.options) > 3:
                    type_info += ", ..."
            console.print(f"  - {outcome.name} ({type_info})")
        console.print()

    # Simulation info
    if spec.simulation:
        sim = spec.simulation
        console.print(
            f"[bold]Simulation:[/bold] {sim.max_timesteps} {sim.timestep_unit.value}s"
        )
        console.print()
