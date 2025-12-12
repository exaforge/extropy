"""CLI for Entropy - Architect Layer."""

import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.console import Console
from rich.live import Live

from .architect import (
    check_sufficiency,
    select_attributes,
    hydrate_attributes,
    bind_constraints,
    build_spec,
)
from .architect.binder import CircularDependencyError
from .models import DiscoveredAttribute, PopulationSpec

app = typer.Typer(
    name="entropy",
    help="Generate population specs for agent-based simulation.",
    no_args_is_help=True,
)

console = Console()


def _grounding_indicator(level: str) -> str:
    """Get colored grounding indicator."""
    indicators = {
        "strong": "[green]🟢 Strong[/green]",
        "medium": "[yellow]🟡 Medium[/yellow]",
        "low": "[red]🔴 Low[/red]",
    }
    return indicators.get(level, "[dim]Unknown[/dim]")


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as Xm Ys or Xs."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    return f"{seconds:.0f}s"


def _display_discovered_attributes(
    attributes: list[DiscoveredAttribute], geography: str | None
) -> None:
    """Display discovered attributes grouped by category."""
    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " DISCOVERED ATTRIBUTES".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    # Group by category
    by_category = {
        "universal": [],
        "population_specific": [],
        "context_specific": [],
        "personality": [],
    }
    for attr in attributes:
        by_category[attr.category].append(attr)

    category_labels = {
        "universal": "Universal",
        "population_specific": "Population-specific",
        "context_specific": "Context-specific",
        "personality": "Personality",
    }

    for cat, cat_label in category_labels.items():
        cat_attrs = by_category[cat]
        if not cat_attrs:
            continue

        console.print(f"[bold]{cat_label} ({len(cat_attrs)}):[/bold]")
        for attr in cat_attrs:
            type_str = f"[dim]({attr.type})[/dim]"
            dep_str = ""
            if attr.depends_on:
                dep_str = f" [cyan]← depends on: {', '.join(attr.depends_on)}[/cyan]"
            console.print(f"  • {attr.name} {type_str}{dep_str}")
        console.print()


def _display_spec_summary(spec: PopulationSpec) -> None:
    """Display spec summary before saving."""
    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " SPEC READY".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    console.print(f"[bold]{spec.meta.description}[/bold] ({spec.meta.size} agents)")
    console.print(
        f"Grounding: {_grounding_indicator(spec.grounding.overall)} ({spec.grounding.sources_count} sources)"
    )
    console.print()

    # Show attributes with grounding
    console.print("[bold]Attributes:[/bold]")
    for attr in spec.attributes[:12]:
        level_icon = {"strong": "🟢", "medium": "🟡", "low": "🔴"}.get(
            attr.grounding.level, "⚪"
        )

        # Format distribution info
        dist_info = ""
        if attr.sampling.distribution:
            dist = attr.sampling.distribution
            # Check BetaDistribution first (has alpha/beta) - must come before min/max check
            # since BetaDistribution has min/max attributes that can be None
            if hasattr(dist, "alpha") and hasattr(dist, "beta"):
                dist_info = f"β(α={dist.alpha:.1f}, β={dist.beta:.1f})"
            elif hasattr(dist, "mean") and dist.mean is not None:
                dist_info = f"μ={dist.mean:.0f}"
                if dist.min is not None and dist.max is not None:
                    dist_info = f"{dist.min:.0f}-{dist.max:.0f}, {dist_info}"
            elif hasattr(dist, "mean_formula") and dist.mean_formula:
                # Formula-based distribution (conditional)
                dist_info = f"μ={dist.mean_formula}"
            elif hasattr(dist, "options") and dist.options:
                # Check options is truthy before slicing
                opts = dist.options[:2]
                dist_info = f"{', '.join(opts)}{'...' if len(dist.options) > 2 else ''}"
            elif hasattr(dist, "min") and hasattr(dist, "max"):
                # Only format if both min and max are not None
                if dist.min is not None and dist.max is not None:
                    dist_info = f"{dist.min:.1f}-{dist.max:.1f}"
            elif hasattr(dist, "probability_true"):
                # Check probability_true is not None before formatting
                if dist.probability_true is not None:
                    dist_info = f"P={dist.probability_true:.0%}"

        method_str = f"[dim]— {attr.grounding.method}[/dim]"
        console.print(
            f"  {level_icon} {attr.name} [dim]({dist_info})[/dim] {method_str}"
        )

    if len(spec.attributes) > 12:
        console.print(f"  [dim]... and {len(spec.attributes) - 12} more[/dim]")

    console.print()
    console.print(f"[bold]Sampling order:[/bold]")
    order_preview = " → ".join(spec.sampling_order[:6])
    if len(spec.sampling_order) > 6:
        order_preview += " → ..."
    console.print(f"  {order_preview}")
    console.print()


def _display_overlay_attributes(
    base_count: int,
    new_attributes: list[DiscoveredAttribute],
    geography: str | None,
) -> None:
    """Display overlay attributes with base context."""
    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " NEW SCENARIO ATTRIBUTES".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    console.print(f"[dim]Base population: {base_count} existing attributes[/dim]")
    console.print()

    for attr in new_attributes:
        type_str = f"[dim]({attr.type})[/dim]"
        dep_str = ""
        if attr.depends_on:
            dep_str = f" [cyan]← depends on: {', '.join(attr.depends_on)}[/cyan]"
        console.print(f"  • {attr.name} {type_str}{dep_str}")

    console.print()


@app.command("overlay")
def overlay_command(
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
        entropy overlay surgeons_base.yaml -s "AI diagnostic tool adoption" -o surgeons_ai.yaml
        entropy overlay farmers.yaml -s "Drought response behavior" -o farmers_drought.yaml
    """
    start_time = time.time()
    console.print()

    # =========================================================================
    # Load Base Spec
    # =========================================================================

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

    # =========================================================================
    # Step 1: Attribute Selection (Overlay Mode)
    # =========================================================================

    console.print()
    selection_start = time.time()
    new_attributes = None
    selection_done = Event()
    selection_error = None

    def do_selection():
        nonlocal new_attributes, selection_error
        try:
            # Pass base attributes as context - selector won't rediscover them
            new_attributes = select_attributes(
                description=scenario,
                size=base.meta.size,
                geography=base.meta.geography,
                context=base.attributes,  # KEY: pass context
            )
        except Exception as e:
            selection_error = e
        finally:
            selection_done.set()

    selection_thread = Thread(target=do_selection, daemon=True)
    selection_thread.start()

    with Live(console=console, refresh_per_second=1, transient=True) as live:
        while not selection_done.is_set():
            elapsed = time.time() - selection_start
            live.update(
                f"[cyan]⠋[/cyan] Discovering scenario attributes... {_format_elapsed(elapsed)}"
            )
            time.sleep(0.1)

    selection_elapsed = time.time() - selection_start

    if selection_error:
        console.print(f"[red]✗[/red] Attribute selection failed: {selection_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Found {len(new_attributes)} NEW attributes ({_format_elapsed(selection_elapsed)})"
    )

    # =========================================================================
    # Human Checkpoint #1: Confirm New Attributes
    # =========================================================================

    _display_overlay_attributes(
        len(base.attributes), new_attributes, base.meta.geography
    )

    if not yes:
        choice = (
            typer.prompt(
                "[Y] Proceed  [n] Cancel",
                default="Y",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # =========================================================================
    # Step 2: Distribution Research (with Context)
    # =========================================================================

    console.print()
    hydration_start = time.time()
    hydrated = None
    sources = []
    warnings = []
    hydration_done = Event()
    hydration_error = None
    current_step = ["2a", "Starting..."]  # [step, status] - mutable for thread access

    def on_progress(step: str, status: str, count: int | None):
        """Update current step for display."""
        current_step[0] = step
        current_step[1] = status

    def do_hydration():
        nonlocal hydrated, sources, warnings, hydration_error
        try:
            # Pass base attributes as context - hydrator can reference them in formulas
            hydrated, sources, warnings = hydrate_attributes(
                attributes=new_attributes,
                description=f"{base.meta.description} + {scenario}",
                geography=base.meta.geography,
                context=base.attributes,  # KEY: pass context
                on_progress=on_progress,
            )
        except Exception as e:
            hydration_error = e
        finally:
            hydration_done.set()

    hydration_thread = Thread(target=do_hydration, daemon=True)
    hydration_thread.start()

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while not hydration_done.is_set():
            elapsed = time.time() - hydration_start
            step, status = current_step
            live.update(
                f"[cyan]⠋[/cyan] Step {step}: {status} {_format_elapsed(elapsed)}"
            )
            time.sleep(0.1)

    hydration_elapsed = time.time() - hydration_start

    if hydration_error:
        console.print(f"[red]✗[/red] Distribution research failed: {hydration_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Researched distributions ({_format_elapsed(hydration_elapsed)}, {len(sources)} sources)"
    )

    # Show validation warnings if any
    if warnings:
        console.print(f"[yellow]⚠[/yellow] {len(warnings)} validation warning(s):")
        for w in warnings[:5]:  # Show first 5
            console.print(f"  [dim]- {w}[/dim]")
        if len(warnings) > 5:
            console.print(f"  [dim]... and {len(warnings) - 5} more[/dim]")

    # =========================================================================
    # Step 3: Constraint Binding (with Context)
    # =========================================================================

    with console.status("[cyan]Binding constraints...[/cyan]"):
        try:
            # Pass base attributes as context for cross-layer dependencies
            bound_attrs, sampling_order, bind_warnings = bind_constraints(
                hydrated,
                context=base.attributes,  # KEY: pass context
            )
        except CircularDependencyError as e:
            console.print(f"[red]✗[/red] Circular dependency detected: {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]✗[/red] Constraint binding failed: {e}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Constraints bound")

    # Show binding warnings if any
    if bind_warnings:
        console.print(f"[yellow]⚠[/yellow] {len(bind_warnings)} binding warning(s):")
        for w in bind_warnings[:5]:  # Show first 5
            console.print(f"  [dim]- {w}[/dim]")
        if len(bind_warnings) > 5:
            console.print(f"  [dim]... and {len(bind_warnings) - 5} more[/dim]")

    # =========================================================================
    # Step 4: Build Overlay Spec and Merge
    # =========================================================================

    with console.status("[cyan]Building and merging specs...[/cyan]"):
        # Build overlay spec
        overlay_spec = build_spec(
            description=scenario,
            size=base.meta.size,
            geography=base.meta.geography,
            attributes=bound_attrs,
            sampling_order=sampling_order,
            sources=sources,
        )

        # Merge base + overlay
        merged_spec = base.merge(overlay_spec)

    console.print(
        f"[green]✓[/green] Merged: {len(base.attributes)} base + {len(bound_attrs)} overlay = {len(merged_spec.attributes)} total"
    )

    # =========================================================================
    # Human Checkpoint #2: Confirm and Save
    # =========================================================================

    _display_spec_summary(merged_spec)

    if not yes:
        choice = (
            typer.prompt(
                "[Y] Save spec  [n] Cancel",
                default="Y",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Save to YAML
    merged_spec.to_yaml(output)

    elapsed = time.time() - start_time

    console.print()
    console.print("═" * 60)
    console.print(f"[green]✓[/green] Merged spec saved to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {_format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)


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

    # =========================================================================
    # Step 0: Context Sufficiency Check
    # =========================================================================

    sufficiency_result = None
    with console.status("[cyan]Checking context sufficiency...[/cyan]"):
        try:
            sufficiency_result = check_sufficiency(description)
        except Exception as e:
            console.print(f"[red]✗[/red] Error checking sufficiency: {e}")
            raise typer.Exit(1)

    if not sufficiency_result.sufficient:
        console.print(f"[red]✗[/red] Description needs clarification:")
        for q in sufficiency_result.clarifications_needed:
            console.print(f"  • {q}")
        raise typer.Exit(1)

    size = sufficiency_result.size
    geography = sufficiency_result.geography
    geo_str = f", {geography}" if geography else ""
    console.print(f"[green]✓[/green] Context sufficient ({size} agents{geo_str})")

    # =========================================================================
    # Step 1: Attribute Selection
    # =========================================================================

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

    with Live(console=console, refresh_per_second=1, transient=True) as live:
        while not selection_done.is_set():
            elapsed = time.time() - selection_start
            live.update(
                f"[cyan]⠋[/cyan] Discovering attributes... {_format_elapsed(elapsed)}"
            )
            time.sleep(0.1)

    selection_elapsed = time.time() - selection_start

    if selection_error:
        console.print(f"[red]✗[/red] Attribute selection failed: {selection_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Found {len(attributes)} attributes ({_format_elapsed(selection_elapsed)})"
    )

    # =========================================================================
    # Human Checkpoint #1: Confirm Attributes
    # =========================================================================

    _display_discovered_attributes(attributes, geography)

    if not yes:
        choice = (
            typer.prompt(
                "[Y] Proceed  [n] Cancel",
                default="Y",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # =========================================================================
    # Step 2: Distribution Research (Hydration)
    # =========================================================================

    console.print()
    hydration_start = time.time()
    hydrated = None
    sources = []
    warnings = []
    hydration_done = Event()
    hydration_error = None
    current_step = ["2a", "Starting..."]  # [step, status] - mutable for thread access

    def on_progress(step: str, status: str, count: int | None):
        """Update current step for display."""
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

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while not hydration_done.is_set():
            elapsed = time.time() - hydration_start
            step, status = current_step
            live.update(
                f"[cyan]⠋[/cyan] Step {step}: {status} {_format_elapsed(elapsed)}"
            )
            time.sleep(0.1)

    hydration_elapsed = time.time() - hydration_start

    if hydration_error:
        console.print(f"[red]✗[/red] Distribution research failed: {hydration_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Researched distributions ({_format_elapsed(hydration_elapsed)}, {len(sources)} sources)"
    )

    # Show validation warnings if any
    if warnings:
        console.print(f"[yellow]⚠[/yellow] {len(warnings)} validation warning(s):")
        for w in warnings[:5]:  # Show first 5
            console.print(f"  [dim]- {w}[/dim]")
        if len(warnings) > 5:
            console.print(f"  [dim]... and {len(warnings) - 5} more[/dim]")

    # =========================================================================
    # Step 3: Constraint Binding
    # =========================================================================

    with console.status("[cyan]Binding constraints...[/cyan]"):
        try:
            bound_attrs, sampling_order, bind_warnings = bind_constraints(hydrated)
        except CircularDependencyError as e:
            console.print(f"[red]✗[/red] Circular dependency detected: {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]✗[/red] Constraint binding failed: {e}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Constraints bound, sampling order determined")

    # Show binding warnings if any
    if bind_warnings:
        console.print(f"[yellow]⚠[/yellow] {len(bind_warnings)} binding warning(s):")
        for w in bind_warnings[:5]:  # Show first 5
            console.print(f"  [dim]- {w}[/dim]")
        if len(bind_warnings) > 5:
            console.print(f"  [dim]... and {len(bind_warnings) - 5} more[/dim]")

    # =========================================================================
    # Step 4: Build Spec
    # =========================================================================

    with console.status("[cyan]Building spec...[/cyan]"):
        population_spec = build_spec(
            description=description,
            size=size,
            geography=geography,
            attributes=bound_attrs,
            sampling_order=sampling_order,
            sources=sources,
        )

    console.print(f"[green]✓[/green] Spec assembled")

    # =========================================================================
    # Human Checkpoint #2: Confirm and Save
    # =========================================================================

    _display_spec_summary(population_spec)

    if not yes:
        choice = (
            typer.prompt(
                "[Y] Save spec  [n] Cancel",
                default="Y",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Save to YAML
    population_spec.to_yaml(output)

    elapsed = time.time() - start_time

    console.print()
    console.print("═" * 60)
    console.print(f"[green]✓[/green] Spec saved to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {_format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)


if __name__ == "__main__":
    app()
