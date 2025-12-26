"""Spec command for generating population specs from descriptions."""

import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live

from ...population.architect import (
    check_sufficiency,
    select_attributes,
    hydrate_attributes,
    bind_constraints,
    build_spec,
)
from ...population.architect.binder import CircularDependencyError
from ...population.validator import validate_spec
from ..app import app, console
from ..display import (
    display_discovered_attributes,
    display_spec_summary,
    display_validation_result,
    generate_and_review_persona_template,
)
from ..utils import format_elapsed


@app.command("spec")
def spec_command(
    description: str = typer.Argument(
        ..., help="Natural language population description"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output YAML file path"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
):
    """
    Generate a population spec from a description.

    Discovers attributes, researches distributions, and saves
    a complete PopulationSpec to YAML.

    Example:
        entropy spec "500 German surgeons" -o surgeons.yaml
        entropy spec "1000 Indian smallholder farmers" -o farmers.yaml
    """
    start_time = time.time()
    console.print()

    # Step 0: Context Sufficiency Check
    sufficiency_result = None
    with console.status("[cyan]Checking context sufficiency...[/cyan]"):
        try:
            sufficiency_result = check_sufficiency(description)
        except Exception as e:
            console.print(f"[red]✗[/red] Error checking sufficiency: {e}")
            raise typer.Exit(1)

    if not sufficiency_result.sufficient:
        console.print("[red]✗[/red] Description needs clarification:")
        for q in sufficiency_result.clarifications_needed:
            console.print(f"  • {q}")
        raise typer.Exit(1)

    size = sufficiency_result.size
    geography = sufficiency_result.geography
    geo_str = f", {geography}" if geography else ""
    console.print(f"[green]✓[/green] Context sufficient ({size} agents{geo_str})")

    # Step 1: Attribute Selection
    console.print()
    selection_start = time.time()
    attributes = None
    selection_done = Event()
    selection_error = None

    def do_selection():
        nonlocal attributes, selection_error
        try:
            attributes = select_attributes(description, size, geography)
        except Exception as e:
            selection_error = e
        finally:
            selection_done.set()

    selection_thread = Thread(target=do_selection, daemon=True)
    selection_thread.start()

    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    spinner_index = 0
    with Live(console=console, refresh_per_second=1, transient=True) as live:
        while not selection_done.is_set():
            elapsed = time.time() - selection_start
            spinner = spinner_chars[spinner_index % len(spinner_chars)]
            live.update(
                f"[cyan]{spinner}[/cyan] Discovering attributes... {format_elapsed(elapsed)}"
            )
            spinner_index += 1
            time.sleep(0.1)

    selection_elapsed = time.time() - selection_start

    if selection_error:
        console.print(f"[red]✗[/red] Attribute selection failed: {selection_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Found {len(attributes)} attributes ({format_elapsed(selection_elapsed)})"
    )

    # Human Checkpoint #1
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

    def do_hydration():
        nonlocal hydrated, sources, warnings, hydration_error
        try:
            hydrated, sources, warnings = hydrate_attributes(
                attributes, description, geography, on_progress=on_progress
            )
        except Exception as e:
            hydration_error = e
        finally:
            hydration_done.set()

    hydration_thread = Thread(target=do_hydration, daemon=True)
    hydration_thread.start()

    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    spinner_index = 0
    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while not hydration_done.is_set():
            elapsed = time.time() - hydration_start
            step, status = current_step
            spinner = spinner_chars[spinner_index % len(spinner_chars)]
            live.update(
                f"[cyan]{spinner}[/cyan] Step {step}: {status} {format_elapsed(elapsed)}"
            )
            spinner_index += 1
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
            size=size,
            geography=geography,
            attributes=bound_attrs,
            sampling_order=sampling_order,
            sources=sources,
        )

    console.print("[green]✓[/green] Spec assembled")

    # Validation Gate
    with console.status("[cyan]Validating spec...[/cyan]"):
        validation_result = validate_spec(population_spec)

    if not display_validation_result(validation_result):
        console.print()
        console.print("[red]Spec validation failed. Please fix the errors above.[/red]")
        raise typer.Exit(1)

    # Step 5: Persona Template Generation
    persona_template = generate_and_review_persona_template(population_spec, yes)
    if persona_template:
        population_spec.meta.persona_template = persona_template
        console.print("[green]✓[/green] Persona template added to spec")

    # Human Checkpoint #2
    display_spec_summary(population_spec)

    if not yes:
        choice = (
            typer.prompt("[Y] Save spec  [n] Cancel", default="Y", show_default=False)
            .strip()
            .lower()
        )
        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Save
    population_spec.to_yaml(output)
    elapsed = time.time() - start_time

    console.print()
    console.print("═" * 60)
    console.print(f"[green]✓[/green] Spec saved to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)
