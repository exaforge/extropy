"""Fix command for auto-fixing spec issues."""

from pathlib import Path

import typer

from ...core.models import PopulationSpec
from ...population.validator import fix_modifier_conditions
from ..app import app, console


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
