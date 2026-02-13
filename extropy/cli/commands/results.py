"""Results command for displaying simulation results."""

from pathlib import Path

import typer

from ..app import app, console


@app.command("results")
def results_command(
    results_dir: Path = typer.Argument(..., help="Results directory from simulation"),
    segment: str | None = typer.Option(
        None, "--segment", "-s", help="Attribute to segment by"
    ),
    timeline: bool = typer.Option(False, "--timeline", "-t", help="Show timeline view"),
    agent: str | None = typer.Option(
        None, "--agent", "-a", help="Show single agent details"
    ),
):
    """
    Display simulation results.

    Load and display results from a completed simulation run.

    Example:
        extropy results results/               # Summary view
        extropy results results/ --segment age # Breakdown by age
        extropy results results/ --timeline    # Timeline view
        extropy results results/ --agent agent_001  # Single agent
    """
    from ...results import (
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
