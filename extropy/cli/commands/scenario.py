"""Scenario command for creating scenario specs with extended attributes."""

import time
from threading import Event, Thread

import click
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
from ..utils import (
    format_elapsed,
    Output,
    ExitCode,
    get_next_invalid_artifact_path,
    save_invalid_json_artifact,
)


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
    timestep_unit: str | None = typer.Option(
        None,
        "--timestep-unit",
        help="Override timestep unit: hour, day, week, month, year",
        click_type=click.Choice(["hour", "day", "week", "month", "year"]),
    ),
    max_timesteps: int | None = typer.Option(
        None,
        "--max-timesteps",
        help="Override maximum number of timesteps",
        min=1,
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    use_defaults: bool = typer.Option(
        False, "--use-defaults", help="Accept default values for sufficiency questions"
    ),
):
    """
    Create a scenario with extended attributes and event configuration.

    This command:
    1. Checks scenario description sufficiency
    2. Discovers scenario-specific attributes (behavioral, psychographic)
    3. Researches distributions for new attributes
    4. Creates event/exposure/timeline/outcome configuration

    Examples:
        # Create new scenario
        extropy scenario "AI diagnostic tool adoption" -o ai-adoption

        # With explicit timestep overrides
        extropy scenario "AI replaces jobs over 6 months" -o ai-jobs --timestep-unit month --max-timesteps 6

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

    # Step 0: Scenario Sufficiency Check
    from ...scenario.sufficiency import (
        check_scenario_sufficiency,
        check_scenario_sufficiency_with_answers,
    )

    sufficiency_result = None
    if not agent_mode:
        with console.status("[cyan]Checking scenario sufficiency...[/cyan]"):
            try:
                sufficiency_result = check_scenario_sufficiency(description)
            except Exception as e:
                console.print(f"[red]x[/red] Error checking sufficiency: {e}")
                raise typer.Exit(1)
    else:
        try:
            sufficiency_result = check_scenario_sufficiency(description)
        except Exception as e:
            out.error(
                f"Error checking sufficiency: {e}",
                exit_code=ExitCode.VALIDATION_ERROR,
            )
            raise typer.Exit(out.finish())

    if not sufficiency_result.sufficient:
        questions = sufficiency_result.questions

        # Try using defaults if flag is set
        if use_defaults and questions:
            default_answers = {
                q.id: q.default for q in questions if q.default is not None
            }
            if default_answers:
                try:
                    sufficiency_result = check_scenario_sufficiency_with_answers(
                        description, default_answers
                    )
                except Exception as e:
                    if not agent_mode:
                        console.print(f"[red]x[/red] Error: {e}")
                    else:
                        out.error(str(e))
                    raise typer.Exit(1)

    if not sufficiency_result.sufficient:
        questions = sufficiency_result.questions

        if agent_mode:
            resume_cmd = (
                f'extropy scenario "{description}" -o {scenario_name} --use-defaults'
            )
            out.needs_clarification(
                questions=questions,
                resume_command=resume_cmd,
            )
            raise typer.Exit(out.finish())

        if questions:
            console.print("[yellow]Additional info needed:[/yellow]")
            collected_answers: dict[str, str | int] = {}
            for q in questions:
                default_hint = f" (default: {q.default})" if q.default else ""
                if q.type == "single_choice" and q.options:
                    console.print(f"\n{q.question}{default_hint}")
                    for i, opt in enumerate(q.options, 1):
                        console.print(f"  {i}. {opt}")
                    answer = typer.prompt("Choice", default=str(q.default or ""))
                    # Handle numeric selection
                    try:
                        idx = int(answer) - 1
                        if 0 <= idx < len(q.options):
                            answer = q.options[idx]
                    except ValueError:
                        pass
                    collected_answers[q.id] = answer
                elif q.type == "number":
                    answer = typer.prompt(
                        q.question, default=str(q.default or ""), type=int
                    )
                    collected_answers[q.id] = answer
                else:
                    answer = typer.prompt(q.question, default=str(q.default or ""))
                    collected_answers[q.id] = answer

            # Enrich description with answers
            for key, value in collected_answers.items():
                description = f"{description} | {key}: {value}"

            console.print()
            with console.status("[cyan]Re-checking scenario sufficiency...[/cyan]"):
                try:
                    sufficiency_result = check_scenario_sufficiency(description)
                except Exception as e:
                    console.print(f"[red]x[/red] Error: {e}")
                    raise typer.Exit(1)

            if not sufficiency_result.sufficient:
                console.print("[red]x[/red] Still insufficient after clarification")
                raise typer.Exit(1)
        else:
            console.print(
                "[red]x[/red] Scenario description is insufficient. "
                "Please provide more detail about the event, duration, and outcomes."
            )
            raise typer.Exit(1)

    if not agent_mode:
        parts = []
        if sufficiency_result.inferred_duration:
            parts.append(f"duration: {sufficiency_result.inferred_duration}")
        if sufficiency_result.inferred_timestep_unit:
            parts.append(f"unit: {sufficiency_result.inferred_timestep_unit}")
        if sufficiency_result.inferred_scenario_type:
            parts.append(f"type: {sufficiency_result.inferred_scenario_type}")
        if sufficiency_result.has_explicit_outcomes:
            parts.append("explicit outcomes detected")
        detail = f" ({', '.join(parts)})" if parts else ""
        console.print(f"[green]✓[/green] Scenario description sufficient{detail}")

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

    if not new_attributes:
        invalid_path = save_invalid_json_artifact(
            {
                "stage": "scenario",
                "error": "No scenario-specific extended attributes discovered",
                "description": description,
                "base_population": f"population.v{pop_version}",
            },
            scenario_path,
            extension=".json",
        )
        out.error(
            f"Scenario requires non-empty extended_attributes. Saved: {invalid_path}",
            exit_code=ExitCode.SCENARIO_ERROR,
        )
        raise typer.Exit(out.finish())

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

    def do_hydration():
        nonlocal hydrated, sources, warnings, hydration_error, household_config
        try:
            hydrated, sources, warnings, household_config = hydrate_attributes(
                attributes=new_attributes,
                description=f"{pop_spec.meta.description} + {description}",
                geography=pop_spec.meta.geography,
                context=pop_spec.attributes,
                include_household=True,
                on_progress=on_progress,
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
    from ...scenario.validator import validate_scenario

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
            resolved_timestep_unit = (
                timestep_unit or sufficiency_result.inferred_timestep_unit
            )
            # We need to create a merged spec first to pass to scenario creation
            # For now, create scenario without agents (they'll be sampled later)
            result_spec, validation_result = create_scenario_spec(
                description=description,
                population_spec=pop_spec,
                extended_attributes=bound_attrs,
                household_config=household_config,
                agent_focus_mode=sufficiency_result.inferred_agent_focus_mode,
                on_progress=on_scenario_progress,
                timeline_mode=timeline_mode,
                timestep_unit_override=resolved_timestep_unit,
                max_timesteps_override=max_timesteps,
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
        invalid_path = save_invalid_json_artifact(
            {
                "stage": "scenario",
                "error": str(pipeline_error),
                "description": description,
                "base_population": f"population.v{pop_version}",
            },
            scenario_path,
            extension=".json",
        )
        if not agent_mode:
            console.print(
                f"[yellow]⚠[/yellow] Scenario invalid artifact saved: [bold]{invalid_path}[/bold]"
            )
        out.error(f"Scenario creation failed: {pipeline_error}")
        raise typer.Exit(1)

    if not agent_mode:
        console.print("[green]✓[/green] Scenario spec ready")

    # Set metadata
    result_spec.meta.name = scenario_name
    result_spec.meta.base_population = f"population.v{pop_version}"

    # Re-validate after metadata finalization.
    # The initial validation occurs inside create_scenario_spec before CLI-level
    # metadata (including base_population) is attached.
    validation_result = validate_scenario(
        spec=result_spec,
        population_spec=pop_spec,
        agent_count=None,
        network=None,
    )

    # Display Summary
    if not agent_mode:
        _display_scenario_summary(result_spec)

    # Validation
    if validation_result and validation_result.errors:
        invalid_path = get_next_invalid_artifact_path(scenario_path)
        result_spec.to_yaml(invalid_path)
        if not agent_mode:
            console.print(
                f"[yellow]⚠[/yellow] Scenario saved to [bold]{invalid_path}[/bold] for review"
            )
        out.error("Scenario validation failed", exit_code=ExitCode.SCENARIO_ERROR)
        raise typer.Exit(out.finish())

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
            console.print(f"  - {ch.name}")
        console.print()

    # Extended attributes
    if hasattr(spec, "extended_attributes") and spec.extended_attributes:
        console.print(
            f"[bold]Extended Attributes:[/bold] {len(spec.extended_attributes)}"
        )
        for attr in spec.extended_attributes[:5]:
            console.print(f"  - {attr.name} ({attr.type})")
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
