"""Extend command for layering scenario attributes on base population."""

import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ...population.spec_builder import (
    select_attributes,
    hydrate_attributes,
    bind_constraints,
    build_spec,
)
from ...utils import topological_sort, CircularDependencyError
from ...core.models import PopulationSpec
from ...population.validator import validate_spec
from ..app import app, console
from ..display import (
    display_extend_attributes,
    display_spec_summary,
    display_validation_result,
    generate_and_review_persona_template,
)
from ..utils import format_elapsed


@app.command("extend")
def extend_command(
    base_spec: Path = typer.Argument(..., help="Base population spec YAML file"),
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario description"),
    output: Path = typer.Option(..., "--output", "-o", help="Output merged spec YAML"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
):
    """
    Layer scenario-specific attributes on a base population.

    Loads an existing population spec and adds behavioral attributes
    for a specific scenario. The new attributes can depend on base
    attributes (e.g., age, income) for realistic correlations.

    Example:
        entropy extend surgeons_base.yaml -s "AI diagnostic tool adoption" -o surgeons_ai.yaml
        entropy extend farmers.yaml -s "Drought response behavior" -o farmers_drought.yaml
    """
    start_time = time.time()
    console.print()

    # Load Base Spec
    if not base_spec.exists():
        console.print(f"[red]✗[/red] Base spec not found: {base_spec}")
        raise typer.Exit(1)

    with console.status("[cyan]Loading base spec...[/cyan]"):
        try:
            base = PopulationSpec.from_yaml(base_spec)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load base spec: {e}")
            raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Loaded base: [bold]{base.meta.description}[/bold] ({len(base.attributes)} attributes)"
    )

    # Step 1: Attribute Selection (Extend Mode)
    console.print()
    selection_start = time.time()
    new_attributes = None
    selection_done = Event()
    selection_error = None

    def do_selection():
        nonlocal new_attributes, selection_error
        try:
            new_attributes = select_attributes(
                description=scenario,
                size=base.meta.size,
                geography=base.meta.geography,
                context=base.attributes,
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
        console.print(f"[red]✗[/red] Attribute selection failed: {selection_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Found {len(new_attributes)} NEW attributes ({format_elapsed(selection_elapsed)})"
    )

    # Human Checkpoint #1
    display_extend_attributes(
        len(base.attributes), new_attributes, base.meta.geography
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

    # Early cycle detection - check new attributes + base context
    try:
        # Build combined dependency map (new attrs can depend on base attrs)
        base_names = {a.name for a in base.attributes}
        deps = {a.name: a.depends_on for a in new_attributes}
        # Filter deps to only include new attrs (base attrs are already sampled)
        deps_filtered = {
            name: [d for d in ds if d not in base_names] for name, ds in deps.items()
        }
        topological_sort(deps_filtered)
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

    def do_hydration():
        nonlocal hydrated, sources, warnings, hydration_error
        try:
            hydrated, sources, warnings = hydrate_attributes(
                attributes=new_attributes,
                description=f"{base.meta.description} + {scenario}",
                geography=base.meta.geography,
                context=base.attributes,
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
            bound_attrs, sampling_order, bind_warnings = bind_constraints(
                hydrated, context=base.attributes
            )
        except CircularDependencyError as e:
            console.print(f"[red]✗[/red] Circular dependency detected: {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]✗[/red] Constraint binding failed: {e}")
            raise typer.Exit(1)

    console.print("[green]✓[/green] Constraints bound")

    if bind_warnings:
        console.print(f"[yellow]⚠[/yellow] {len(bind_warnings)} binding warning(s):")
        for w in bind_warnings[:5]:
            console.print(f"  [dim]- {w}[/dim]")
        if len(bind_warnings) > 5:
            console.print(f"  [dim]... and {len(bind_warnings) - 5} more[/dim]")

    # Step 4: Build and Merge
    with console.status("[cyan]Building and merging specs...[/cyan]"):
        extension_spec = build_spec(
            description=scenario,
            size=base.meta.size,
            geography=base.meta.geography,
            attributes=bound_attrs,
            sampling_order=sampling_order,
            sources=sources,
        )
        merged_spec = base.merge(extension_spec)

    console.print(
        f"[green]✓[/green] Merged: {len(base.attributes)} base + {len(bound_attrs)} extension = {len(merged_spec.attributes)} total"
    )

    # Validation Gate
    with console.status("[cyan]Validating merged spec...[/cyan]"):
        validation_result = validate_spec(merged_spec)

    if not display_validation_result(validation_result):
        # Save with .invalid.yaml suffix so work isn't lost
        invalid_path = output.with_suffix(".invalid.yaml")
        merged_spec.to_yaml(invalid_path)
        console.print()
        console.print(
            f"[yellow]⚠[/yellow] Spec saved to [bold]{invalid_path}[/bold] for manual review"
        )
        console.print("[red]Spec validation failed. Please fix the errors above.[/red]")
        raise typer.Exit(1)

    # Step 5: Persona Template Generation
    # Generate template for merged spec (includes all base + extension attributes)
    persona_template = generate_and_review_persona_template(merged_spec, yes)
    if persona_template:
        merged_spec.meta.persona_template = persona_template
        console.print("[green]✓[/green] Persona template added to spec")

    # Store scenario description for use by scenario command
    merged_spec.meta.scenario_description = scenario

    # Human Checkpoint #2
    display_spec_summary(merged_spec)

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
    merged_spec.to_yaml(output)
    elapsed = time.time() - start_time

    console.print()
    console.print("═" * 60)
    console.print(f"[green]✓[/green] Merged spec saved to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)
