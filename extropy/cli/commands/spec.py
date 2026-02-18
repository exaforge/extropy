"""Spec command for generating population specs from descriptions."""

import json
import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ...population.spec_builder import (
    check_sufficiency,
    check_sufficiency_with_answers,
    select_attributes,
    hydrate_attributes,
    bind_constraints,
    build_spec,
)
from ...utils import topological_sort, CircularDependencyError
from ...population.validator import validate_spec
from ..app import app, console, is_agent_mode, get_study_path
from ..study import (
    StudyContext,
    create_study_folder,
    detect_study_folder,
)
from ..display import (
    display_discovered_attributes,
    display_spec_summary,
    display_validation_result,
)
from ..utils import format_elapsed, Output, ExitCode


@app.command("spec")
def spec_command(
    description: str = typer.Argument(
        ..., help="Natural language population description"
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path: study folder (creates population.v1.yaml), "
        "folder/name (creates folder/name.v1.yaml), or explicit .yaml file",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    answers: str | None = typer.Option(
        None,
        "--answers",
        help="JSON with pre-supplied clarification answers (for agent mode)",
    ),
    use_defaults: bool = typer.Option(
        False,
        "--use-defaults",
        help="Use defaults for ambiguous values instead of prompting",
    ),
):
    """
    Generate a population spec from a description.

    Discovers attributes, researches distributions, and saves
    a complete PopulationSpec to YAML.

    Examples:
        # Create new study folder with population.v1.yaml
        extropy spec "500 German surgeons" -o surgeons

        # Create with custom name (surgeons/hospital-staff.v1.yaml)
        extropy spec "German surgeons" -o surgeons/hospital-staff

        # Iterate on existing (from within study folder)
        cd surgeons && extropy spec "add income distribution"
        # Creates population.v2.yaml

        # Explicit file path
        extropy spec "farmers" -o my-spec.yaml
    """
    from ...core.cost.tracker import CostTracker

    CostTracker.get().set_context(command="spec")

    start_time = time.time()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

    if not agent_mode:
        console.print()

    # Resolve output path with auto-versioning
    study_ctx: StudyContext | None = None
    resolved_output: Path
    pop_name = "population"

    if output is None:
        # No -o: must be in a study folder, auto-version
        study_path = get_study_path()
        detected = detect_study_folder(study_path)
        if detected is None:
            out.error(
                "Not in a study folder. Use -o to specify output path or create a new study.",
                exit_code=ExitCode.FILE_NOT_FOUND,
            )
            raise typer.Exit(out.finish())
        study_ctx = StudyContext(detected)
        next_ver = study_ctx.get_next_population_version(pop_name)
        resolved_output = study_ctx.root / f"{pop_name}.v{next_ver}.yaml"
        if not agent_mode:
            console.print(f"[dim]Auto-versioning to {resolved_output.name}[/dim]")
    elif output.suffix in (".yaml", ".yml"):
        # Explicit YAML file path
        resolved_output = output
    elif "/" in str(output) or output.parent != Path("."):
        # folder/name format: create study folder with custom population name
        study_root = output.parent
        pop_name = output.name
        study_ctx = create_study_folder(study_root)
        next_ver = study_ctx.get_next_population_version(pop_name)
        resolved_output = study_ctx.root / f"{pop_name}.v{next_ver}.yaml"
        if not agent_mode:
            console.print(f"[dim]Created study folder: {study_ctx.root}[/dim]")
    else:
        # Just a name: create study folder with that name
        study_ctx = create_study_folder(output)
        next_ver = study_ctx.get_next_population_version(pop_name)
        resolved_output = study_ctx.root / f"{pop_name}.v{next_ver}.yaml"
        if not agent_mode:
            console.print(f"[dim]Created study folder: {study_ctx.root}[/dim]")

    # Parse answers if provided
    parsed_answers: dict[str, str | int] = {}
    if answers:
        try:
            parsed_answers = json.loads(answers)
        except json.JSONDecodeError as e:
            out.error(
                f"Invalid JSON in --answers: {e}", exit_code=ExitCode.VALIDATION_ERROR
            )
            raise typer.Exit(out.finish())

    # Step 0: Context Sufficiency Check
    sufficiency_result = None
    if not agent_mode:
        with console.status("[cyan]Checking context sufficiency...[/cyan]"):
            try:
                if parsed_answers:
                    sufficiency_result = check_sufficiency_with_answers(
                        description, parsed_answers
                    )
                else:
                    sufficiency_result = check_sufficiency(description)
            except Exception as e:
                console.print(f"[red]✗[/red] Error checking sufficiency: {e}")
                raise typer.Exit(1)
    else:
        try:
            if parsed_answers:
                sufficiency_result = check_sufficiency_with_answers(
                    description, parsed_answers
                )
            else:
                sufficiency_result = check_sufficiency(description)
        except Exception as e:
            out.error(
                f"Error checking sufficiency: {e}", exit_code=ExitCode.VALIDATION_ERROR
            )
            raise typer.Exit(out.finish())

    # Handle insufficient context
    if not sufficiency_result.sufficient:
        questions = sufficiency_result.questions

        # Try using defaults if flag is set
        if use_defaults and questions:
            default_answers = {
                q.id: q.default for q in questions if q.default is not None
            }
            if default_answers:
                all_answers = {**parsed_answers, **default_answers}
                try:
                    sufficiency_result = check_sufficiency_with_answers(
                        description, all_answers
                    )
                except Exception as e:
                    if agent_mode:
                        out.error(
                            f"Error with defaults: {e}",
                            exit_code=ExitCode.VALIDATION_ERROR,
                        )
                        raise typer.Exit(out.finish())
                    else:
                        console.print(f"[red]✗[/red] Error with defaults: {e}")
                        raise typer.Exit(1)

    # Still insufficient after defaults?
    if not sufficiency_result.sufficient:
        questions = sufficiency_result.questions

        if agent_mode:
            # Agent mode: return structured questions with exit code 2
            resume_cmd = f"extropy spec \"{description}\" -o {resolved_output} --answers '{{...}}'"
            out.needs_clarification(
                questions=questions,
                resume_command=resume_cmd,
            )
            raise typer.Exit(out.finish())
        else:
            # Human mode: interactive prompts
            console.print("[yellow]⚠[/yellow] Description needs clarification:")
            collected_answers: dict[str, str | int] = dict(parsed_answers)

            for q in questions:
                console.print()
                console.print(f"[bold]{q.question}[/bold]")

                if q.type == "single_choice" and q.options:
                    for i, opt in enumerate(q.options, 1):
                        default_marker = (
                            " [dim](default)[/dim]" if opt == q.default else ""
                        )
                        console.print(f"  [{i}] {opt}{default_marker}")

                    default_idx = (
                        q.options.index(q.default) + 1 if q.default in q.options else 1
                    )
                    choice = typer.prompt(
                        f"Select [1-{len(q.options)}]",
                        default=str(default_idx),
                        show_default=False,
                    )
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(q.options):
                            collected_answers[q.id] = q.options[idx]
                        else:
                            collected_answers[q.id] = q.options[0]
                    except ValueError:
                        # User typed text instead of number, use as-is
                        collected_answers[q.id] = choice
                elif q.type == "number":
                    val = typer.prompt(
                        "Enter number",
                        default=str(q.default) if q.default else "",
                    )
                    try:
                        collected_answers[q.id] = int(val)
                    except ValueError:
                        collected_answers[q.id] = val
                else:  # text
                    val = typer.prompt(
                        "Enter value",
                        default=str(q.default) if q.default else "",
                    )
                    collected_answers[q.id] = val

            # Re-check with collected answers
            console.print()
            with console.status("[cyan]Re-checking context sufficiency...[/cyan]"):
                try:
                    sufficiency_result = check_sufficiency_with_answers(
                        description, collected_answers
                    )
                except Exception as e:
                    console.print(f"[red]✗[/red] Error: {e}")
                    raise typer.Exit(1)

            if not sufficiency_result.sufficient:
                console.print("[red]✗[/red] Still insufficient after clarification:")
                for q in sufficiency_result.clarifications_needed:
                    console.print(f"  • {q}")
                raise typer.Exit(1)

    geography = sufficiency_result.geography
    agent_focus = sufficiency_result.agent_focus
    geo_str = f", {geography}" if geography else ""
    focus_str = f", focus: {agent_focus}" if agent_focus else ""

    if not agent_mode:
        console.print(f"[green]✓[/green] Context sufficient{geo_str}{focus_str}")

    # Step 1: Attribute Selection
    console.print()
    selection_start = time.time()
    attributes = None
    selection_done = Event()
    selection_error = None

    def do_selection():
        nonlocal attributes, selection_error
        try:
            attributes = select_attributes(description, geography)
        except Exception as e:
            selection_error = e
        finally:
            selection_done.set()

    selection_thread = Thread(target=do_selection, daemon=True)
    selection_thread.start()

    spinner = Spinner("dots", text="Discovering attributes...", style="cyan")
    with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
        while not selection_done.is_set():
            elapsed = time.time() - selection_start
            spinner.update(text=f"Discovering attributes... {format_elapsed(elapsed)}")
            time.sleep(0.1)

    selection_elapsed = time.time() - selection_start

    if selection_error:
        console.print(f"[red]✗[/red] Attribute selection failed: {selection_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Found {len(attributes)} attributes ({format_elapsed(selection_elapsed)})"
    )

    # Human Checkpoint #1 - skip in agent mode (agents decide by running the command)
    if not agent_mode:
        display_discovered_attributes(attributes, geography)

        if not yes:
            choice = (
                typer.prompt("[Y] Proceed  [n] Cancel", default="Y", show_default=False)
                .strip()
                .lower()
            )
            if choice == "n":
                console.print("[dim]Cancelled.[/dim]")
                raise typer.Exit(0)

    # Early cycle detection - check before expensive hydration
    try:
        deps = {a.name: a.depends_on for a in attributes}
        topological_sort(deps)
    except CircularDependencyError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print("[dim]Please review attribute dependencies.[/dim]")
        raise typer.Exit(1)

    # Step 2: Distribution Research
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

    name_config = None

    def do_hydration():
        nonlocal hydrated, sources, warnings, hydration_error, name_config
        try:
            hydrated, sources, warnings, _household_config, name_config = (
                hydrate_attributes(
                    attributes, description, geography, on_progress=on_progress
                )
            )
            # Note: household_config is ignored at spec level - it's set per-scenario
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
        console.print(f"[red]✗[/red] Distribution research failed: {hydration_error}")
        raise typer.Exit(1)

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
            bound_attrs, sampling_order, bind_warnings = bind_constraints(hydrated)
        except CircularDependencyError as e:
            console.print(f"[red]✗[/red] Circular dependency detected: {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]✗[/red] Constraint binding failed: {e}")
            raise typer.Exit(1)

    console.print("[green]✓[/green] Constraints bound, sampling order determined")

    if bind_warnings:
        console.print(f"[yellow]⚠[/yellow] {len(bind_warnings)} binding warning(s):")
        for w in bind_warnings[:5]:
            console.print(f"  [dim]- {w}[/dim]")
        if len(bind_warnings) > 5:
            console.print(f"  [dim]... and {len(bind_warnings) - 5} more[/dim]")

    # Step 4: Build Spec
    with console.status("[cyan]Building spec...[/cyan]"):
        population_spec = build_spec(
            description=description,
            geography=geography,
            attributes=bound_attrs,
            sampling_order=sampling_order,
            sources=sources,
            agent_focus=agent_focus,
            name_config=name_config,
        )

    console.print("[green]✓[/green] Spec assembled")

    # Validation Gate
    with console.status("[cyan]Validating spec...[/cyan]"):
        validation_result = validate_spec(population_spec)

    if not display_validation_result(validation_result):
        # Save with .invalid.yaml suffix so work isn't lost
        invalid_path = resolved_output.with_suffix(".invalid.yaml")
        population_spec.to_yaml(invalid_path)
        console.print()
        console.print(
            f"[yellow]⚠[/yellow] Spec saved to [bold]{invalid_path}[/bold] for manual review"
        )
        console.print("[red]Spec validation failed. Please fix the errors above.[/red]")
        raise typer.Exit(1)

    # Note: Persona template generation happens in extend, not base spec
    # This ensures the template includes scenario attributes

    # Human Checkpoint #2 - skip in agent mode
    if not agent_mode:
        display_spec_summary(population_spec)

        if not yes:
            choice = (
                typer.prompt(
                    "[Y] Save spec  [n] Cancel", default="Y", show_default=False
                )
                .strip()
                .lower()
            )
            if choice == "n":
                console.print("[dim]Cancelled.[/dim]")
                raise typer.Exit(0)

    # Save
    population_spec.to_yaml(resolved_output)
    elapsed = time.time() - start_time

    if agent_mode:
        out.success(
            "Spec saved",
            output=str(resolved_output),
            geography=geography,
            agent_focus=agent_focus,
            attribute_count=len(population_spec.attributes),
            elapsed_seconds=elapsed,
            study_folder=str(study_ctx.root) if study_ctx else None,
        )
        raise typer.Exit(out.finish())
    else:
        console.print()
        console.print("═" * 60)
        console.print(f"[green]✓[/green] Spec saved to [bold]{resolved_output}[/bold]")
        console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
        console.print("═" * 60)
