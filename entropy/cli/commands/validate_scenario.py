"""Validate-scenario command for validating scenario specs."""

from pathlib import Path

import typer

from ..app import app, console


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
    from ...scenario import load_and_validate_scenario

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
