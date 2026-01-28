"""Display helpers for CLI output."""

from rich.table import Table
from rich.tree import Tree

from ..core.models import DiscoveredAttribute, PopulationSpec
from .app import console
from .utils import grounding_indicator


def display_discovered_attributes(
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


def display_spec_summary(spec: PopulationSpec) -> None:
    """Display spec summary before saving."""
    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " SPEC READY".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    console.print(f"[bold]{spec.meta.description}[/bold] ({spec.meta.size} agents)")
    console.print(
        f"Grounding: {grounding_indicator(spec.grounding.overall)} ({spec.grounding.sources_count} sources)"
    )
    console.print()

    # Show attributes with grounding in a table
    attr_rows = []
    for attr in spec.attributes[:15]:
        level_icon = {"strong": "🟢", "medium": "🟡", "low": "🔴"}.get(
            attr.grounding.level, "⚪"
        )

        # Format distribution info
        dist_info = ""
        if attr.sampling.distribution:
            dist = attr.sampling.distribution
            # Check BetaDistribution first (has alpha/beta)
            if hasattr(dist, "alpha") and hasattr(dist, "beta"):
                dist_info = f"β(α={dist.alpha:.1f}, β={dist.beta:.1f})"
            elif hasattr(dist, "mean") and dist.mean is not None:
                dist_info = f"μ={dist.mean:.0f}"
                if dist.min is not None and dist.max is not None:
                    dist_info = f"{dist.min:.0f}-{dist.max:.0f}, {dist_info}"
            elif hasattr(dist, "mean_formula") and dist.mean_formula:
                dist_info = f"μ={dist.mean_formula}"
            elif hasattr(dist, "options") and dist.options:
                opts = dist.options[:2]
                dist_info = f"{', '.join(opts)}{'...' if len(dist.options) > 2 else ''}"
            elif hasattr(dist, "min") and hasattr(dist, "max"):
                if dist.min is not None and dist.max is not None:
                    dist_info = f"{dist.min:.1f}-{dist.max:.1f}"
            elif hasattr(dist, "probability_true"):
                if dist.probability_true is not None:
                    dist_info = f"P={dist.probability_true:.0%}"

        attr_rows.append(
            [
                f"{level_icon} {attr.name}",
                attr.type,
                dist_info[:25] if dist_info else "-",
                attr.grounding.method[:20] if attr.grounding.method else "-",
            ]
        )

    table = Table(title="Attributes", show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Distribution")
    table.add_column("Grounding", style="dim")
    for row in attr_rows:
        table.add_row(*row)
    console.print(table)

    if len(spec.attributes) > 15:
        console.print(f"  [dim]... and {len(spec.attributes) - 15} more[/dim]")

    console.print()

    # Show sampling order as a dependency tree
    console.print("[bold]Sampling Order (Dependencies):[/bold]")

    # Build dependency info
    attrs_with_deps = []
    for attr_name in spec.sampling_order[:12]:
        attr = spec.get_attribute(attr_name)
        if attr and attr.sampling.depends_on:
            attrs_with_deps.append((attr_name, attr.sampling.depends_on))
        elif attr:
            attrs_with_deps.append((attr_name, []))

    # Create a Rich Tree for dependencies
    tree = Tree("📋 [bold]Sampling Order[/bold]")
    for name, deps in attrs_with_deps:
        if deps:
            branch = tree.add(f"[cyan]{name}[/cyan]")
            for dep in deps:
                branch.add(f"[dim]← {dep}[/dim]")
        else:
            tree.add(f"[green]{name}[/green]")

    if len(spec.sampling_order) > 12:
        tree.add(f"[dim]... and {len(spec.sampling_order) - 12} more[/dim]")

    console.print(tree)
    console.print()


def display_extend_attributes(
    base_count: int,
    new_attributes: list[DiscoveredAttribute],
    geography: str | None,
) -> None:
    """Display extend attributes with base context."""
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


def display_validation_result(result, strict: bool = False) -> bool:
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
            loc = err.location
            if err.modifier_index is not None:
                loc = f"{err.location}[{err.modifier_index}]"
            console.print(f"  [red]✗[/red] {loc}: {err.message}")
            if err.suggestion:
                console.print(f"    [dim]→ {err.suggestion}[/dim]")
        if len(result.errors) > 10:
            console.print(
                f"  [dim]... and {len(result.errors) - 10} more error(s)[/dim]"
            )
        return False

    if result.warnings:
        if strict:
            console.print(
                f"[red]✗[/red] Spec has {len(result.warnings)} warning(s) (strict mode)"
            )
            for warn in result.warnings[:5]:
                loc = warn.location
                if warn.modifier_index is not None:
                    loc = f"{warn.location}[{warn.modifier_index}]"
                console.print(f"  [yellow]⚠[/yellow] {loc}: {warn.message}")
            if len(result.warnings) > 5:
                console.print(
                    f"  [dim]... and {len(result.warnings) - 5} more warning(s)[/dim]"
                )
            return False
        else:
            console.print(
                f"[green]✓[/green] Spec validated with {len(result.warnings)} warning(s)"
            )
            for warn in result.warnings[:3]:
                loc = warn.location
                if warn.modifier_index is not None:
                    loc = f"{warn.location}[{warn.modifier_index}]"
                console.print(f"  [yellow]⚠[/yellow] {loc}: {warn.message}")
            if len(result.warnings) > 3:
                console.print(
                    f"  [dim]... and {len(result.warnings) - 3} more warning(s)[/dim]"
                )
            return True

    return True


def create_sample_agent(spec: PopulationSpec) -> dict:
    """Create a sample agent dict with representative values for preview."""
    agent = {}

    for attr in spec.attributes:
        if attr.type == "int":
            # Use mean if available, else reasonable default
            if attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "mean") and dist.mean is not None:
                    agent[attr.name] = int(dist.mean)
                elif hasattr(dist, "min") and hasattr(dist, "max"):
                    if dist.min is not None and dist.max is not None:
                        agent[attr.name] = int((dist.min + dist.max) / 2)
                    else:
                        agent[attr.name] = 35  # default
                else:
                    agent[attr.name] = 35
            else:
                agent[attr.name] = 35

        elif attr.type == "float":
            if attr.sampling.distribution:
                dist = attr.sampling.distribution
                if hasattr(dist, "mean") and dist.mean is not None:
                    agent[attr.name] = dist.mean
                elif hasattr(dist, "alpha") and hasattr(dist, "beta"):
                    # Beta distribution - use alpha / (alpha + beta) as expected value
                    agent[attr.name] = dist.alpha / (dist.alpha + dist.beta)
                elif hasattr(dist, "min") and hasattr(dist, "max"):
                    if dist.min is not None and dist.max is not None:
                        agent[attr.name] = (dist.min + dist.max) / 2
                    else:
                        agent[attr.name] = 0.5
                else:
                    agent[attr.name] = 0.5
            else:
                agent[attr.name] = 0.5

        elif attr.type == "categorical":
            if attr.sampling.distribution and hasattr(
                attr.sampling.distribution, "options"
            ):
                options = attr.sampling.distribution.options
                if options:
                    agent[attr.name] = options[0]

        elif attr.type == "boolean":
            if attr.sampling.distribution and hasattr(
                attr.sampling.distribution, "probability_true"
            ):
                agent[attr.name] = attr.sampling.distribution.probability_true > 0.5
            else:
                agent[attr.name] = True

    return agent


def _preview_full_persona(spec: PopulationSpec, sample_agent: dict, template: str) -> str:
    """Generate full persona preview with template + characteristics list."""
    from ..simulation.persona import (
        format_agent,
        render_persona,
        extract_template_attrs,
        build_characteristics_list,
    )

    # Format agent values
    formatted = format_agent(sample_agent, spec)

    # Render template
    narrative = render_persona(formatted, template)

    # Get used attrs and build characteristics
    used_attrs = extract_template_attrs(template)
    characteristics = build_characteristics_list(spec, formatted, used_attrs)

    return f"""## Who You Are

{narrative}

## Your Characteristics

{characteristics}"""


def generate_and_review_persona_template(
    spec: PopulationSpec,
    yes: bool = False,
) -> str | None:
    """Generate and interactively review a persona template.

    Args:
        spec: Population specification
        yes: If True, skip confirmation prompts

    Returns:
        Approved template string, or None if skipped/cancelled
    """
    import time
    from threading import Event, Thread

    import typer
    from rich.live import Live

    from ..population.spec_builder import (
        generate_persona_template,
        validate_persona_template,
        refine_persona_template,
    )
    from .utils import format_elapsed

    console.print()

    # Generate initial template
    template = None
    template_done = Event()
    template_error = None
    gen_start = time.time()

    def do_template_gen():
        nonlocal template, template_error
        try:
            template = generate_persona_template(spec, log=True)
        except Exception as e:
            template_error = e
        finally:
            template_done.set()

    template_thread = Thread(target=do_template_gen, daemon=True)
    template_thread.start()

    with Live(console=console, refresh_per_second=1, transient=True) as live:
        while not template_done.is_set():
            elapsed = time.time() - gen_start
            live.update(
                f"[cyan]⠋[/cyan] Generating persona template... {format_elapsed(elapsed)}"
            )
            time.sleep(0.1)

    gen_elapsed = time.time() - gen_start

    if template_error:
        console.print(
            f"[yellow]⚠[/yellow] Could not generate persona template: {template_error}"
        )
        console.print("[dim]Falling back to built-in persona generation[/dim]")
        return None

    if not template:
        console.print("[yellow]⚠[/yellow] Empty template generated")
        console.print("[dim]Falling back to built-in persona generation[/dim]")
        return None

    console.print(
        f"[green]✓[/green] Template generated ({format_elapsed(gen_elapsed)})"
    )
    console.print()

    # Display the template
    console.print("[bold]Narrative Intro Template:[/bold]")
    console.print()
    # Show template with syntax highlighting
    for line in template.split("\n"):
        if line.strip():
            console.print(f"  [dim]{line}[/dim]")
    console.print()

    # Create a sample agent for preview
    sample_agent = create_sample_agent(spec)

    # Validate template first
    is_valid, result = validate_persona_template(template, sample_agent)
    if not is_valid:
        console.print(f"[yellow]⚠[/yellow] Template validation warning: {result}")
        console.print()
    else:
        # Show full persona preview (template + characteristics list)
        console.print("[bold]Full Persona Preview:[/bold]")
        console.print()
        full_preview = _preview_full_persona(spec, sample_agent, template)
        for line in full_preview.split("\n"):
            console.print(f"  [green]{line}[/green]")
        console.print()

    if yes:
        return template

    # Interactive review loop
    max_refinements = 3
    refinement_count = 0

    while True:
        choice = (
            typer.prompt(
                "[Y] Use template  [s] Skip (use fallback)  [r] Regenerate  [f] Give feedback",
                default="Y",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "y":
            return template
        elif choice == "s":
            console.print(
                "[dim]Skipping persona template, will use built-in generation[/dim]"
            )
            return None
        elif choice == "r":
            console.print()
            with console.status("[cyan]Regenerating template...[/cyan]"):
                try:
                    template = generate_persona_template(spec, log=True)
                except Exception as e:
                    console.print(f"[red]✗[/red] Regeneration failed: {e}")
                    continue

            console.print("[green]✓[/green] Template regenerated")
            console.print()
            for line in template.split("\n"):
                if line.strip():
                    console.print(f"  [dim]{line}[/dim]")
            console.print()

            is_valid, result = validate_persona_template(template, sample_agent)
            if is_valid:
                console.print("[bold]Full Persona Preview:[/bold]")
                console.print()
                full_preview = _preview_full_persona(spec, sample_agent, template)
                for line in full_preview.split("\n"):
                    console.print(f"  [green]{line}[/green]")
                console.print()

        elif choice == "f":
            if refinement_count >= max_refinements:
                console.print("[yellow]Maximum refinements reached[/yellow]")
                continue

            feedback = typer.prompt("What would you like to change?")
            console.print()

            with console.status("[cyan]Refining template...[/cyan]"):
                try:
                    template = refine_persona_template(
                        template, spec, feedback, log=True
                    )
                except Exception as e:
                    console.print(f"[red]✗[/red] Refinement failed: {e}")
                    continue

            refinement_count += 1
            console.print("[green]✓[/green] Template refined")
            console.print()
            for line in template.split("\n"):
                if line.strip():
                    console.print(f"  [dim]{line}[/dim]")
            console.print()

            is_valid, result = validate_persona_template(template, sample_agent)
            if is_valid:
                console.print("[bold]Full Persona Preview:[/bold]")
                console.print()
                full_preview = _preview_full_persona(spec, sample_agent, template)
                for line in full_preview.split("\n"):
                    console.print(f"  [green]{line}[/green]")
                console.print()
