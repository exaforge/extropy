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
from .spec import DiscoveredAttribute, PopulationSpec

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


def _display_discovered_attributes(attributes: list[DiscoveredAttribute], geography: str | None) -> None:
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
    console.print(f"Grounding: {_grounding_indicator(spec.grounding.overall)} ({spec.grounding.sources_count} sources)")
    console.print()
    
    # Show attributes with grounding
    console.print("[bold]Attributes:[/bold]")
    for attr in spec.attributes[:12]:
        level_icon = {"strong": "🟢", "medium": "🟡", "low": "🔴"}.get(attr.grounding.level, "⚪")
        
        # Format distribution info
        dist_info = ""
        if attr.sampling.distribution:
            dist = attr.sampling.distribution
            if hasattr(dist, "mean") and hasattr(dist, "std"):
                dist_info = f"μ={dist.mean:.0f}"
                if dist.min is not None and dist.max is not None:
                    dist_info = f"{dist.min:.0f}-{dist.max:.0f}, {dist_info}"
            elif hasattr(dist, "options"):
                opts = dist.options[:2]
                dist_info = f"{', '.join(opts)}{'...' if len(dist.options) > 2 else ''}"
            elif hasattr(dist, "min") and hasattr(dist, "max"):
                dist_info = f"{dist.min:.1f}-{dist.max:.1f}"
            elif hasattr(dist, "probability_true"):
                dist_info = f"P={dist.probability_true:.0%}"
        
        method_str = f"[dim]— {attr.grounding.method}[/dim]"
        console.print(f"  {level_icon} {attr.name} [dim]({dist_info})[/dim] {method_str}")
    
    if len(spec.attributes) > 12:
        console.print(f"  [dim]... and {len(spec.attributes) - 12} more[/dim]")
    
    console.print()
    console.print(f"[bold]Sampling order:[/bold]")
    order_preview = " → ".join(spec.sampling_order[:6])
    if len(spec.sampling_order) > 6:
        order_preview += " → ..."
    console.print(f"  {order_preview}")
    console.print()


@app.command("spec")
def spec_command(
    description: str = typer.Argument(..., help="Natural language population description"),
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
            live.update(f"[cyan]⠋[/cyan] Discovering attributes... {_format_elapsed(elapsed)}")
            time.sleep(0.1)
    
    selection_elapsed = time.time() - selection_start
    
    if selection_error:
        console.print(f"[red]✗[/red] Attribute selection failed: {selection_error}")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Found {len(attributes)} attributes ({_format_elapsed(selection_elapsed)})")
    
    # =========================================================================
    # Human Checkpoint #1: Confirm Attributes
    # =========================================================================
    
    _display_discovered_attributes(attributes, geography)
    
    if not yes:
        choice = typer.prompt(
            "[Y] Proceed  [n] Cancel",
            default="Y",
            show_default=False,
        ).strip().lower()
        
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
    hydration_done = Event()
    hydration_error = None
    
    def do_hydration():
        nonlocal hydrated, sources, hydration_error
        try:
            hydrated, sources = hydrate_attributes(attributes, description, geography)
        except Exception as e:
            hydration_error = e
        finally:
            hydration_done.set()
    
    hydration_thread = Thread(target=do_hydration, daemon=True)
    hydration_thread.start()
    
    with Live(console=console, refresh_per_second=1, transient=True) as live:
        while not hydration_done.is_set():
            elapsed = time.time() - hydration_start
            live.update(f"[cyan]⠋[/cyan] Researching distributions... {_format_elapsed(elapsed)}")
            time.sleep(0.1)
    
    hydration_elapsed = time.time() - hydration_start
    
    if hydration_error:
        console.print(f"[red]✗[/red] Distribution research failed: {hydration_error}")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Researched distributions ({_format_elapsed(hydration_elapsed)}, {len(sources)} sources)")
    
    # =========================================================================
    # Step 3: Constraint Binding
    # =========================================================================
    
    with console.status("[cyan]Binding constraints...[/cyan]"):
        try:
            bound_attrs, sampling_order = bind_constraints(hydrated)
        except CircularDependencyError as e:
            console.print(f"[red]✗[/red] Circular dependency detected: {e}")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]✗[/red] Constraint binding failed: {e}")
            raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Constraints bound, sampling order determined")
    
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
        choice = typer.prompt(
            "[Y] Save spec  [n] Cancel",
            default="Y",
            show_default=False,
        ).strip().lower()
        
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
