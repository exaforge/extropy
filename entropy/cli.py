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
from .validator import validate_spec, Severity, fix_modifier_conditions, fix_spec_file

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


def _display_validation_result(result, strict: bool = False) -> bool:
    """Display validation result and return True if should proceed.

    Args:
        result: ValidationResult from validate_spec()
        strict: If True, treat warnings as errors

    Returns:
        True if validation passed (no errors, or only warnings and not strict)
    """
    if result.valid and not result.warnings:
        console.print("[green]✓[/green] Spec validated")
        return True

    if result.errors:
        console.print(f"[red]✗[/red] Spec has {len(result.errors)} error(s)")
        for err in result.errors[:10]:
            loc = err.attribute
            if err.modifier_index is not None:
                loc = f"{err.attribute}[{err.modifier_index}]"
            console.print(f"  [red]✗[/red] {loc}: {err.message}")
            if err.suggestion:
                console.print(f"    [dim]→ {err.suggestion}[/dim]")
        if len(result.errors) > 10:
            console.print(f"  [dim]... and {len(result.errors) - 10} more error(s)[/dim]")
        return False

    if result.warnings:
        if strict:
            console.print(f"[red]✗[/red] Spec has {len(result.warnings)} warning(s) (strict mode)")
            for warn in result.warnings[:5]:
                loc = warn.attribute
                if warn.modifier_index is not None:
                    loc = f"{warn.attribute}[{warn.modifier_index}]"
                console.print(f"  [yellow]⚠[/yellow] {loc}: {warn.message}")
            if len(result.warnings) > 5:
                console.print(f"  [dim]... and {len(result.warnings) - 5} more warning(s)[/dim]")
            return False
        else:
            console.print(f"[green]✓[/green] Spec validated with {len(result.warnings)} warning(s)")
            for warn in result.warnings[:3]:
                loc = warn.attribute
                if warn.modifier_index is not None:
                    loc = f"{warn.attribute}[{warn.modifier_index}]"
                console.print(f"  [yellow]⚠[/yellow] {loc}: {warn.message}")
            if len(result.warnings) > 3:
                console.print(f"  [dim]... and {len(result.warnings) - 3} more warning(s)[/dim]")
            return True

    return True


@app.command("validate")
def validate_command(
    spec_file: Path = typer.Argument(..., help="Population spec YAML file to validate"),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors"),
):
    """
    Validate a population spec for structural correctness.

    Checks for type/modifier compatibility, range violations, weight validity,
    distribution parameters, dependencies, conditions, formulas, duplicates,
    and strategy consistency.

    Example:
        entropy validate surgeons.yaml
        entropy validate surgeons.yaml --strict
    """
    console.print()

    if not spec_file.exists():
        console.print(f"[red]✗[/red] Spec file not found: {spec_file}")
        raise typer.Exit(1)

    with console.status("[cyan]Loading spec...[/cyan]"):
        try:
            spec = PopulationSpec.from_yaml(spec_file)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load spec: {e}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Loaded: [bold]{spec.meta.description}[/bold] ({len(spec.attributes)} attributes)")
    console.print()

    with console.status("[cyan]Validating spec...[/cyan]"):
        result = validate_spec(spec)

    if not _display_validation_result(result, strict):
        raise typer.Exit(1)

    # Show summary
    console.print()
    if result.errors:
        console.print(f"[red]Validation failed[/red]: {len(result.errors)} error(s)")
    elif result.warnings and strict:
        console.print(f"[red]Validation failed[/red]: {len(result.warnings)} warning(s) in strict mode")
    else:
        console.print("[green]Validation passed[/green]")


@app.command("fix")
def fix_command(
    spec_file: Path = typer.Argument(..., help="Population spec YAML file to fix"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file (defaults to overwriting input)"),
    confidence: float = typer.Option(0.6, "--confidence", "-c", help="Minimum fuzzy match confidence (0-1)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show fixes without applying them"),
):
    """
    Auto-fix modifier condition option references.

    Fixes a common LLM error where modifier 'when' conditions reference
    categorical options with inconsistent naming (e.g., 'University hospital'
    instead of 'University_hospital').

    Uses fuzzy matching to identify and correct option references in modifier
    conditions. The --confidence flag controls how strict the matching is.

    Example:
        entropy fix surgeons.yaml                    # Fix in place
        entropy fix surgeons.yaml -o fixed.yaml     # Save to new file
        entropy fix surgeons.yaml --dry-run         # Preview fixes
        entropy fix surgeons.yaml -c 0.8            # Stricter matching
    """
    console.print()

    if not spec_file.exists():
        console.print(f"[red]✗[/red] Spec file not found: {spec_file}")
        raise typer.Exit(1)

    # Load and fix
    with console.status("[cyan]Loading spec and analyzing conditions...[/cyan]"):
        try:
            spec = PopulationSpec.from_yaml(spec_file)
            result = fix_modifier_conditions(spec, min_confidence=confidence)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load/fix spec: {e}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Loaded: [bold]{spec.meta.description}[/bold] ({len(spec.attributes)} attributes)")
    console.print()

    if result.fix_count == 0:
        console.print("[green]✓[/green] No fixes needed - all option references are correct")
        if result.unfixable:
            console.print()
            console.print(f"[yellow]⚠[/yellow] {len(result.unfixable)} potential issue(s) below confidence threshold:")
            for issue in result.unfixable:
                console.print(f"  [dim]{issue}[/dim]")
        return

    # Show fixes
    console.print(f"[cyan]Found {result.fix_count} fix(es):[/cyan]")
    console.print()

    for fix in result.fixes:
        console.print(f"  {fix.attribute}[{fix.modifier_index}]:")
        console.print(f"    [red]- '{fix.original_value}'[/red]")
        console.print(f"    [green]+ '{fix.fixed_value}'[/green] [dim](confidence: {fix.confidence:.0%})[/dim]")

    if result.unfixable:
        console.print()
        console.print(f"[yellow]⚠[/yellow] {len(result.unfixable)} issue(s) below confidence threshold:")
        for issue in result.unfixable:
            console.print(f"  [dim]{issue}[/dim]")

    console.print()

    if dry_run:
        console.print("[dim]Dry run - no changes made[/dim]")
        return

    # Save fixed spec
    out_path = output or spec_file
    result.spec.to_yaml(out_path)

    console.print("═" * 60)
    console.print(f"[green]✓[/green] Fixed spec saved to [bold]{out_path}[/bold]")
    console.print("═" * 60)


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
    # Validation Gate
    # =========================================================================

    with console.status("[cyan]Validating merged spec...[/cyan]"):
        validation_result = validate_spec(merged_spec)

    if not _display_validation_result(validation_result):
        console.print()
        console.print("[red]Spec validation failed. Please fix the errors above.[/red]")
        raise typer.Exit(1)

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
    # Validation Gate
    # =========================================================================

    with console.status("[cyan]Validating spec...[/cyan]"):
        validation_result = validate_spec(population_spec)

    if not _display_validation_result(validation_result):
        console.print()
        console.print("[red]Spec validation failed. Please fix the errors above.[/red]")
        raise typer.Exit(1)

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


@app.command("sample")
def sample_command(
    spec_file: Path = typer.Argument(..., help="Population spec YAML file to sample from"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path (.json or .db)"),
    count: int | None = typer.Option(None, "--count", "-n", help="Number of agents (default: spec.meta.size)"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or sqlite"),
    report: bool = typer.Option(False, "--report", "-r", help="Show distribution summaries and stats"),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Skip validator errors"),
):
    """
    Generate agents from a population spec.

    Samples from a PopulationSpec YAML file, producing a population of agents
    with attributes matching the spec's distributions and dependencies.

    Example:
        entropy sample surgeons.yaml -o agents.json
        entropy sample surgeons.yaml -n 500 -o agents.json --seed 42
        entropy sample surgeons.yaml -n 1000 -o agents.db --format sqlite
        entropy sample surgeons.yaml -o agents.json --report
    """
    from .sampler import sample_population, save_json, save_sqlite, SamplingError

    start_time = time.time()
    console.print()

    # =========================================================================
    # Load Spec
    # =========================================================================

    if not spec_file.exists():
        console.print(f"[red]✗[/red] Spec file not found: {spec_file}")
        raise typer.Exit(1)

    with console.status("[cyan]Loading spec...[/cyan]"):
        try:
            spec = PopulationSpec.from_yaml(spec_file)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load spec: {e}")
            raise typer.Exit(1)

    effective_count = count if count is not None else spec.meta.size
    console.print(
        f"[green]✓[/green] Loaded: [bold]{spec.meta.description}[/bold] "
        f"({len(spec.attributes)} attributes, sampling {effective_count} agents)"
    )

    # =========================================================================
    # Validation Gate
    # =========================================================================

    console.print()
    with console.status("[cyan]Validating spec...[/cyan]"):
        validation_result = validate_spec(spec)

    if not validation_result.valid:
        if skip_validation:
            console.print(
                f"[yellow]⚠[/yellow] Spec has {len(validation_result.errors)} error(s) - "
                f"skipping validation (--skip-validation)"
            )
            for err in validation_result.errors[:5]:
                console.print(f"  [red]✗[/red] {err.attribute}: {err.message}")
            if len(validation_result.errors) > 5:
                console.print(f"  [dim]... and {len(validation_result.errors) - 5} more[/dim]")
        else:
            console.print(f"[red]✗[/red] Spec has {len(validation_result.errors)} error(s)")
            for err in validation_result.errors[:10]:
                console.print(f"  [red]✗[/red] {err.attribute}: {err.message}")
                if err.suggestion:
                    console.print(f"    [dim]→ {err.suggestion}[/dim]")
            if len(validation_result.errors) > 10:
                console.print(f"  [dim]... and {len(validation_result.errors) - 10} more[/dim]")
            console.print()
            console.print("[dim]Use --skip-validation to sample anyway[/dim]")
            raise typer.Exit(1)
    else:
        if validation_result.warnings:
            console.print(
                f"[green]✓[/green] Spec validated with {len(validation_result.warnings)} warning(s)"
            )
        else:
            console.print("[green]✓[/green] Spec validated")

    # =========================================================================
    # Sampling
    # =========================================================================

    console.print()
    sampling_start = time.time()
    result = None
    sampling_error = None

    # Show progress for larger populations
    show_progress = effective_count >= 100

    if show_progress:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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
                    spec,
                    count=effective_count,
                    seed=seed,
                    on_progress=on_progress,
                )
            except SamplingError as e:
                sampling_error = e
    else:
        with console.status("[cyan]Sampling agents...[/cyan]"):
            try:
                result = sample_population(
                    spec,
                    count=effective_count,
                    seed=seed,
                )
            except SamplingError as e:
                sampling_error = e

    if sampling_error:
        console.print(f"[red]✗[/red] Sampling failed: {sampling_error}")
        raise typer.Exit(1)

    sampling_elapsed = time.time() - sampling_start
    console.print(
        f"[green]✓[/green] Sampled {len(result.agents)} agents "
        f"({_format_elapsed(sampling_elapsed)}, seed={result.meta['seed']})"
    )

    # =========================================================================
    # Report (optional)
    # =========================================================================

    if report:
        console.print()
        console.print("┌" + "─" * 58 + "┐")
        console.print("│" + " SAMPLING REPORT".center(58) + "│")
        console.print("└" + "─" * 58 + "┘")
        console.print()

        # Numeric attribute stats
        numeric_attrs = [a for a in spec.attributes if a.type in ("int", "float")]
        if numeric_attrs:
            console.print("[bold]Numeric Attributes:[/bold]")
            for attr in numeric_attrs[:15]:
                mean = result.stats.attribute_means.get(attr.name, 0)
                std = result.stats.attribute_stds.get(attr.name, 0)
                console.print(f"  {attr.name}: μ={mean:.2f}, σ={std:.2f}")
            if len(numeric_attrs) > 15:
                console.print(f"  [dim]... and {len(numeric_attrs) - 15} more[/dim]")
            console.print()

        # Categorical attribute stats
        cat_attrs = [a for a in spec.attributes if a.type == "categorical"]
        if cat_attrs:
            console.print("[bold]Categorical Attributes:[/bold]")
            for attr in cat_attrs[:10]:
                counts = result.stats.categorical_counts.get(attr.name, {})
                total = sum(counts.values()) or 1
                top_3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
                dist_str = ", ".join(f"{k}:{v/total:.0%}" for k, v in top_3)
                console.print(f"  {attr.name}: {dist_str}")
            if len(cat_attrs) > 10:
                console.print(f"  [dim]... and {len(cat_attrs) - 10} more[/dim]")
            console.print()

        # Boolean attribute stats
        bool_attrs = [a for a in spec.attributes if a.type == "boolean"]
        if bool_attrs:
            console.print("[bold]Boolean Attributes:[/bold]")
            for attr in bool_attrs:
                counts = result.stats.boolean_counts.get(attr.name, {True: 0, False: 0})
                total = sum(counts.values()) or 1
                pct_true = counts.get(True, 0) / total
                console.print(f"  {attr.name}: {pct_true:.1%} true")
            console.print()

        # Modifier triggers
        triggered_mods = {k: v for k, v in result.stats.modifier_triggers.items() if any(v.values())}
        if triggered_mods:
            console.print("[bold]Modifier Triggers:[/bold]")
            for attr_name, triggers in list(triggered_mods.items())[:10]:
                attr = spec.get_attribute(attr_name)
                if attr:
                    for idx, count in triggers.items():
                        if count > 0 and idx < len(attr.sampling.modifiers):
                            mod = attr.sampling.modifiers[idx]
                            console.print(f"  {attr_name}[{idx}] '{mod.when}': {count} times")
            console.print()

        # Constraint violations
        if result.stats.constraint_violations:
            console.print("[bold]Expression Constraint Violations:[/bold]")
            for expr, count in list(result.stats.constraint_violations.items())[:10]:
                console.print(f"  [yellow]⚠[/yellow] {expr}: {count} agents")
            console.print()

    # =========================================================================
    # Save Output
    # =========================================================================

    console.print()
    output_format = format.lower()

    # Auto-detect format from extension if not explicitly set
    if output.suffix.lower() == ".db":
        output_format = "sqlite"
    elif output.suffix.lower() == ".json":
        output_format = "json"

    with console.status(f"[cyan]Saving to {output_format}...[/cyan]"):
        if output_format == "sqlite":
            save_sqlite(result, output)
        else:
            save_json(result, output)

    elapsed = time.time() - start_time

    console.print("═" * 60)
    console.print(f"[green]✓[/green] Saved {len(result.agents)} agents to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {_format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)


@app.command("network")
def network_command(
    agents_file: Path = typer.Argument(..., help="Agents JSON file to generate network from"),
    output: Path = typer.Option(..., "--output", "-o", help="Output network JSON file"),
    avg_degree: float = typer.Option(20.0, "--avg-degree", help="Target average degree (connections per agent)"),
    rewire_prob: float = typer.Option(0.05, "--rewire-prob", help="Watts-Strogatz rewiring probability"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Print validation metrics"),
    no_metrics: bool = typer.Option(False, "--no-metrics", help="Skip computing node metrics (faster)"),
):
    """
    Generate a social network from sampled agents.

    Creates edges between agents based on attribute similarity, with
    degree correction for high-influence agents and Watts-Strogatz
    rewiring for small-world properties.

    Example:
        entropy network agents.json -o network.json
        entropy network agents.json -o network.json --avg-degree 25 --validate
        entropy network agents.json -o network.json --seed 42
    """
    from .network import (
        generate_network,
        generate_network_with_metrics,
        load_agents_json,
        NetworkConfig,
        validate_network,
    )

    start_time = time.time()
    console.print()

    # =========================================================================
    # Load Agents
    # =========================================================================

    if not agents_file.exists():
        console.print(f"[red]✗[/red] Agents file not found: {agents_file}")
        raise typer.Exit(1)

    with console.status("[cyan]Loading agents...[/cyan]"):
        try:
            agents = load_agents_json(agents_file)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load agents: {e}")
            raise typer.Exit(1)

    console.print(f"[green]✓[/green] Loaded {len(agents)} agents from [bold]{agents_file}[/bold]")

    # =========================================================================
    # Generate Network
    # =========================================================================

    config = NetworkConfig(
        avg_degree=avg_degree,
        rewire_prob=rewire_prob,
        seed=seed,
    )

    console.print()
    generation_start = time.time()
    current_stage = ["Initializing", 0, 0]

    def on_progress(stage: str, current: int, total: int):
        current_stage[0] = stage
        current_stage[1] = current
        current_stage[2] = total

    result = None
    generation_error = None

    from threading import Thread, Event
    generation_done = Event()

    def do_generation():
        nonlocal result, generation_error
        try:
            if no_metrics:
                result = generate_network(agents, config, on_progress)
            else:
                result = generate_network_with_metrics(agents, config, on_progress)
        except Exception as e:
            generation_error = e
        finally:
            generation_done.set()

    gen_thread = Thread(target=do_generation, daemon=True)
    gen_thread.start()

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while not generation_done.is_set():
            elapsed = time.time() - generation_start
            stage, current, total = current_stage
            if total > 0:
                pct = current / total * 100
                live.update(
                    f"[cyan]⠋[/cyan] {stage}... {current}/{total} ({pct:.0f}%) {_format_elapsed(elapsed)}"
                )
            else:
                live.update(f"[cyan]⠋[/cyan] {stage}... {_format_elapsed(elapsed)}")
            time.sleep(0.1)

    generation_elapsed = time.time() - generation_start

    if generation_error:
        console.print(f"[red]✗[/red] Network generation failed: {generation_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Generated network: {result.meta['edge_count']} edges, "
        f"avg degree {result.meta['avg_degree']:.1f} ({_format_elapsed(generation_elapsed)})"
    )

    # =========================================================================
    # Validation (optional)
    # =========================================================================

    if validate:
        console.print()
        console.print("┌" + "─" * 58 + "┐")
        console.print("│" + " NETWORK VALIDATION".center(58) + "│")
        console.print("└" + "─" * 58 + "┘")
        console.print()

        if result.network_metrics:
            metrics = result.network_metrics
            console.print(f"  Nodes: {metrics.node_count}")
            console.print(f"  Edges: {metrics.edge_count}")
            console.print(f"  Avg Degree: {metrics.avg_degree:.2f}")
            console.print(f"  Clustering: {metrics.clustering_coefficient:.3f}")
            if metrics.avg_path_length:
                console.print(f"  Avg Path Length: {metrics.avg_path_length:.2f}")
            else:
                console.print("  Avg Path Length: [dim]N/A (disconnected)[/dim]")
            console.print(f"  Modularity: {metrics.modularity:.3f}")
            console.print(f"  Largest Component: {metrics.largest_component_ratio:.1%}")
            console.print(f"  Degree Assortativity: {metrics.degree_assortativity:.3f}")

            is_valid, warnings = metrics.is_valid()
            console.print()
            if is_valid:
                console.print("[green]✓[/green] All metrics within expected ranges")
            else:
                console.print(f"[yellow]⚠[/yellow] {len(warnings)} metric(s) outside expected range:")
                for w in warnings:
                    console.print(f"  [yellow]•[/yellow] {w}")
        else:
            console.print("[dim]Metrics not computed (use without --no-metrics)[/dim]")

        # Edge type distribution
        console.print()
        console.print("[bold]Edge Types:[/bold]")
        edge_types: dict[str, int] = {}
        for edge in result.edges:
            t = edge.edge_type
            edge_types[t] = edge_types.get(t, 0) + 1
        for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            pct = count / len(result.edges) * 100 if result.edges else 0
            console.print(f"  {edge_type}: {count} ({pct:.1f}%)")

    # =========================================================================
    # Save Output
    # =========================================================================

    console.print()
    with console.status(f"[cyan]Saving to {output}...[/cyan]"):
        result.save_json(output)

    elapsed = time.time() - start_time

    console.print("═" * 60)
    console.print(f"[green]✓[/green] Network saved to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {_format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)


@app.command("scenario")
def scenario_command(
    description: str = typer.Argument(..., help="Natural language scenario description"),
    population: Path = typer.Option(..., "--population", "-p", help="Population spec YAML file"),
    agents: Path = typer.Option(..., "--agents", "-a", help="Sampled agents JSON file"),
    network: Path = typer.Option(..., "--network", "-n", help="Network JSON file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output scenario YAML file"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
):
    """
    Create a scenario spec from a natural language description.

    Generates a complete scenario specification including:
    - Event definition (type, content, source, credibility)
    - Seed exposure rules (how agents learn about the event)
    - Interaction model (how agents discuss and respond)
    - Spread configuration (how information propagates)
    - Outcome definitions (what to measure)

    Example:
        entropy scenario "Netflix announces $3 price increase" \\
            -p population.yaml -a agents.json -n network.json -o scenario.yaml
    """
    from .scenario import create_scenario, ScenarioSpec

    start_time = time.time()
    console.print()

    # =========================================================================
    # Validate Input Files
    # =========================================================================

    if not population.exists():
        console.print(f"[red]✗[/red] Population spec not found: {population}")
        raise typer.Exit(1)

    if not agents.exists():
        console.print(f"[red]✗[/red] Agents file not found: {agents}")
        raise typer.Exit(1)

    if not network.exists():
        console.print(f"[red]✗[/red] Network file not found: {network}")
        raise typer.Exit(1)

    console.print(f"Creating scenario for: [bold]{description}[/bold]")
    console.print()

    # =========================================================================
    # Run Pipeline
    # =========================================================================

    current_step = ["1/5", "Starting..."]
    pipeline_done = Event()
    pipeline_error = None
    result_spec = None
    validation_result = None

    def on_progress(step: str, status: str):
        current_step[0] = step
        current_step[1] = status

    def run_pipeline():
        nonlocal result_spec, validation_result, pipeline_error
        try:
            result_spec, validation_result = create_scenario(
                description=description,
                population_spec_path=population,
                agents_path=agents,
                network_path=network,
                output_path=None,  # Don't save yet
                on_progress=on_progress,
            )
        except Exception as e:
            pipeline_error = e
        finally:
            pipeline_done.set()

    pipeline_thread = Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while not pipeline_done.is_set():
            elapsed = time.time() - start_time
            step, status = current_step
            live.update(
                f"[cyan]⠋[/cyan] Step {step}: {status} {_format_elapsed(elapsed)}"
            )
            time.sleep(0.1)

    if pipeline_error:
        console.print(f"[red]✗[/red] Scenario creation failed: {pipeline_error}")
        raise typer.Exit(1)

    console.print("[green]✓[/green] Scenario spec ready")

    # =========================================================================
    # Display Summary
    # =========================================================================

    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " SCENARIO SPEC READY".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    # Event info
    event = result_spec.event
    content_preview = event.content[:50] + "..." if len(event.content) > 50 else event.content
    console.print(f"[bold]Event:[/bold] {event.type.value} — \"{content_preview}\"")
    console.print(f"[bold]Source:[/bold] {event.source} (credibility: {event.credibility:.2f})")
    console.print()

    # Exposure info
    console.print("[bold]Exposure Channels:[/bold]")
    for ch in result_spec.seed_exposure.channels:
        console.print(f"  • {ch.name} ({ch.reach})")
    console.print()

    console.print(f"[bold]Seed Exposure Rules:[/bold] {len(result_spec.seed_exposure.rules)}")
    for rule in result_spec.seed_exposure.rules[:3]:
        when_preview = rule.when[:30] + "..." if len(rule.when) > 30 else rule.when
        console.print(f"  • {rule.channel}: {when_preview} at t={rule.timestep}")
    if len(result_spec.seed_exposure.rules) > 3:
        console.print(f"  [dim]... and {len(result_spec.seed_exposure.rules) - 3} more[/dim]")
    console.print()

    # Interaction info
    interaction = result_spec.interaction
    model_str = interaction.primary_model.value
    if interaction.secondary_model:
        model_str += f" + {interaction.secondary_model.value}"
    console.print(f"[bold]Interaction Model:[/bold] {model_str}")
    console.print(f"[bold]Share Probability:[/bold] {result_spec.spread.share_probability:.2f}")
    console.print()

    # Outcomes info
    console.print("[bold]Outcomes:[/bold]")
    for outcome in result_spec.outcomes.suggested_outcomes:
        type_info = outcome.type.value
        if outcome.options:
            type_info += f": {', '.join(outcome.options[:3])}"
            if len(outcome.options) > 3:
                type_info += ", ..."
        elif outcome.range:
            type_info += f": {outcome.range[0]} to {outcome.range[1]}"
        console.print(f"  • {outcome.name} ({type_info})")
    console.print()

    # Simulation info
    sim = result_spec.simulation
    console.print(f"[bold]Simulation:[/bold] {sim.max_timesteps} {sim.timestep_unit.value}s")
    console.print()

    # =========================================================================
    # Validation Results
    # =========================================================================

    if validation_result.errors:
        console.print(f"[red]✗[/red] Validation: {len(validation_result.errors)} error(s)")
        for err in validation_result.errors[:5]:
            console.print(f"  [red]✗[/red] {err.location}: {err.message}")
            if err.suggestion:
                console.print(f"    [dim]→ {err.suggestion}[/dim]")
        if len(validation_result.errors) > 5:
            console.print(f"  [dim]... and {len(validation_result.errors) - 5} more[/dim]")
    elif validation_result.warnings:
        console.print(f"[green]✓[/green] Validation passed with {len(validation_result.warnings)} warning(s)")
        for warn in validation_result.warnings[:3]:
            console.print(f"  [yellow]⚠[/yellow] {warn.location}: {warn.message}")
    else:
        console.print("[green]✓[/green] Validation passed")

    # =========================================================================
    # Human Checkpoint
    # =========================================================================

    console.print()
    if not yes:
        choice = (
            typer.prompt(
                "[Y] Save  [n] Cancel",
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
    result_spec.to_yaml(output)

    elapsed = time.time() - start_time

    console.print()
    console.print("═" * 60)
    console.print(f"[green]✓[/green] Scenario saved to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {_format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)


@app.command("validate-scenario")
def validate_scenario_command(
    scenario_file: Path = typer.Argument(..., help="Scenario spec YAML file to validate"),
):
    """
    Validate a scenario spec against its referenced files.

    Checks that all attribute references, channel references, and file
    paths are valid. Reports errors and warnings.

    Example:
        entropy validate-scenario scenario.yaml
    """
    from .scenario import load_and_validate_scenario

    console.print()

    if not scenario_file.exists():
        console.print(f"[red]✗[/red] Scenario file not found: {scenario_file}")
        raise typer.Exit(1)

    console.print(f"Validating scenario: [bold]{scenario_file}[/bold]")
    console.print()

    try:
        spec, result = load_and_validate_scenario(scenario_file)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load scenario: {e}")
        raise typer.Exit(1)

    # Show file references
    console.print("[bold]File References:[/bold]")

    pop_path = Path(spec.meta.population_spec)
    if pop_path.exists():
        console.print(f"  [green]✓[/green] Population spec: {spec.meta.population_spec}")
    else:
        console.print(f"  [red]✗[/red] Population spec: {spec.meta.population_spec} (not found)")

    agents_path = Path(spec.meta.agents_file)
    if agents_path.exists():
        console.print(f"  [green]✓[/green] Agents file: {spec.meta.agents_file}")
    else:
        console.print(f"  [red]✗[/red] Agents file: {spec.meta.agents_file} (not found)")

    network_path = Path(spec.meta.network_file)
    if network_path.exists():
        console.print(f"  [green]✓[/green] Network file: {spec.meta.network_file}")
    else:
        console.print(f"  [red]✗[/red] Network file: {spec.meta.network_file} (not found)")

    console.print()

    # Show validation results
    if result.errors:
        console.print(f"[red]✗[/red] {len(result.errors)} error(s) found:")
        for err in result.errors:
            console.print(f"  [red]✗[/red] [{err.category}] {err.location}: {err.message}")
            if err.suggestion:
                console.print(f"    [dim]→ {err.suggestion}[/dim]")
        console.print()
        console.print("[red]Validation failed[/red]")
        raise typer.Exit(1)

    if result.warnings:
        console.print(f"[yellow]⚠[/yellow] {len(result.warnings)} warning(s):")
        for warn in result.warnings:
            console.print(f"  [yellow]⚠[/yellow] [{warn.category}] {warn.location}: {warn.message}")
        console.print()

    console.print("[green]✓ Scenario spec is valid[/green]")


@app.command("simulate")
def simulate_command(
    scenario_file: Path = typer.Argument(..., help="Scenario spec YAML file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output results directory"),
    model: str = typer.Option("gpt-5-mini", "--model", "-m", help="LLM model for agent reasoning"),
    threshold: int = typer.Option(3, "--threshold", "-t", help="Multi-touch threshold for re-reasoning"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
):
    """
    Run a simulation from a scenario spec.

    Executes the scenario against its population, simulating opinion
    dynamics with agent reasoning, network propagation, and state evolution.

    Example:
        entropy simulate scenario.yaml -o results/
        entropy simulate scenario.yaml -o results/ --model gpt-5-nano --seed 42
    """
    from .simulation import run_simulation

    start_time = time.time()
    console.print()

    # Validate input file
    if not scenario_file.exists():
        console.print(f"[red]✗[/red] Scenario file not found: {scenario_file}")
        raise typer.Exit(1)

    console.print(f"Simulating: [bold]{scenario_file}[/bold]")
    console.print(f"Output: {output}")
    console.print(f"Model: {model} | Threshold: {threshold}")
    if seed:
        console.print(f"Seed: {seed}")
    console.print()

    # Progress tracking
    current_progress = [0, 0, "Starting..."]

    def on_progress(timestep: int, max_timesteps: int, status: str):
        current_progress[0] = timestep
        current_progress[1] = max_timesteps
        current_progress[2] = status

    # Run simulation
    simulation_done = Event()
    simulation_error = None
    result = None

    def do_simulation():
        nonlocal result, simulation_error
        try:
            result = run_simulation(
                scenario_path=scenario_file,
                output_dir=output,
                model=model,
                multi_touch_threshold=threshold,
                random_seed=seed,
                on_progress=on_progress if not quiet else None,
            )
        except Exception as e:
            simulation_error = e
        finally:
            simulation_done.set()

    simulation_thread = Thread(target=do_simulation, daemon=True)
    simulation_thread.start()

    if not quiet:
        with Live(console=console, refresh_per_second=2, transient=True) as live:
            while not simulation_done.is_set():
                elapsed = time.time() - start_time
                timestep, max_ts, status = current_progress
                if max_ts > 0:
                    pct = timestep / max_ts * 100
                    live.update(
                        f"[cyan]⠋[/cyan] Timestep {timestep}/{max_ts} ({pct:.0f}%) | {status} | {_format_elapsed(elapsed)}"
                    )
                else:
                    live.update(f"[cyan]⠋[/cyan] {status} | {_format_elapsed(elapsed)}")
                time.sleep(0.1)
    else:
        simulation_done.wait()

    if simulation_error:
        console.print(f"[red]✗[/red] Simulation failed: {simulation_error}")
        raise typer.Exit(1)

    elapsed = time.time() - start_time

    # Display summary
    console.print()
    console.print("═" * 60)
    console.print(f"[green]✓[/green] Simulation complete")
    console.print("═" * 60)
    console.print()

    console.print(f"Duration: {_format_elapsed(elapsed)} ({result.total_timesteps} timesteps)")
    if result.stopped_reason:
        console.print(f"Stopped: {result.stopped_reason}")
    console.print(f"Reasoning calls: {result.total_reasoning_calls:,}")
    console.print(f"Final exposure rate: {result.final_exposure_rate:.1%}")
    console.print()

    # Show outcome distributions
    if result.outcome_distributions:
        console.print("[bold]Outcome Distributions:[/bold]")
        for outcome_name, distribution in result.outcome_distributions.items():
            if isinstance(distribution, dict):
                if "mean" in distribution:
                    console.print(f"  {outcome_name}: mean={distribution['mean']:.2f}")
                else:
                    top_3 = sorted(distribution.items(), key=lambda x: -x[1])[:3]
                    dist_str = ", ".join(f"{k}:{v:.1%}" for k, v in top_3)
                    console.print(f"  {outcome_name}: {dist_str}")
        console.print()

    console.print(f"Results saved to: [bold]{output}[/bold]")


@app.command("results")
def results_command(
    results_dir: Path = typer.Argument(..., help="Results directory from simulation"),
    segment: str | None = typer.Option(None, "--segment", "-s", help="Attribute to segment by"),
    timeline: bool = typer.Option(False, "--timeline", "-t", help="Show timeline view"),
    agent: str | None = typer.Option(None, "--agent", "-a", help="Show single agent details"),
):
    """
    Display simulation results.

    Load and display results from a completed simulation run.

    Example:
        entropy results results/               # Summary view
        entropy results results/ --segment age # Breakdown by age
        entropy results results/ --timeline    # Timeline view
        entropy results results/ --agent agent_001  # Single agent
    """
    from .results import (
        load_results,
        display_summary,
        display_segment_breakdown,
        display_timeline,
        display_agent,
    )

    console.print()

    if not results_dir.exists():
        console.print(f"[red]✗[/red] Results directory not found: {results_dir}")
        raise typer.Exit(1)

    try:
        reader = load_results(results_dir)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load results: {e}")
        raise typer.Exit(1)

    # Dispatch to appropriate view
    if agent:
        display_agent(console, reader, agent)
    elif segment:
        display_segment_breakdown(console, reader, segment)
    elif timeline:
        display_timeline(console, reader)
    else:
        display_summary(console, reader)


if __name__ == "__main__":
    app()
