"""Scenario command for creating scenario specs from descriptions."""

import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ..app import app, console
from ..utils import format_elapsed


@app.command("scenario")
def scenario_command(
    description: str = typer.Argument(
        ..., help="Natural language scenario description"
    ),
    population: Path = typer.Option(
        ..., "--population", "-p", help="Population spec YAML file"
    ),
    agents: Path = typer.Option(..., "--agents", "-a", help="Sampled agents JSON file"),
    network: Path = typer.Option(..., "--network", "-n", help="Network JSON file"),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output scenario YAML file"
    ),
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
    from ...scenario import create_scenario

    start_time = time.time()
    console.print()

    # Validate Input Files
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

    # Run Pipeline
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

    spinner = Spinner("dots", text="Starting...", style="cyan")
    with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
        while not pipeline_done.is_set():
            elapsed = time.time() - start_time
            step, status = current_step
            spinner.update(text=f"Step {step}: {status} {format_elapsed(elapsed)}")
            time.sleep(0.1)

    if pipeline_error:
        console.print(f"[red]✗[/red] Scenario creation failed: {pipeline_error}")
        raise typer.Exit(1)

    console.print("[green]✓[/green] Scenario spec ready")

    # Display Summary
    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " SCENARIO SPEC READY".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    # Event info
    event = result_spec.event
    content_preview = (
        event.content[:50] + "..." if len(event.content) > 50 else event.content
    )
    console.print(f'[bold]Event:[/bold] {event.type.value} — "{content_preview}"')
    console.print(
        f"[bold]Source:[/bold] {event.source} (credibility: {event.credibility:.2f})"
    )
    console.print()

    # Exposure info
    console.print("[bold]Exposure Channels:[/bold]")
    for ch in result_spec.seed_exposure.channels:
        console.print(f"  • {ch.name} ({ch.reach})")
    console.print()

    console.print(
        f"[bold]Seed Exposure Rules:[/bold] {len(result_spec.seed_exposure.rules)}"
    )
    for rule in result_spec.seed_exposure.rules[:3]:
        when_preview = rule.when[:30] + "..." if len(rule.when) > 30 else rule.when
        console.print(f"  • {rule.channel}: {when_preview} at t={rule.timestep}")
    if len(result_spec.seed_exposure.rules) > 3:
        console.print(
            f"  [dim]... and {len(result_spec.seed_exposure.rules) - 3} more[/dim]"
        )
    console.print()

    # Interaction info
    interaction = result_spec.interaction
    model_str = interaction.primary_model.value
    if interaction.secondary_model:
        model_str += f" + {interaction.secondary_model.value}"
    console.print(f"[bold]Interaction Model:[/bold] {model_str}")
    console.print(
        f"[bold]Share Probability:[/bold] {result_spec.spread.share_probability:.2f}"
    )
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
    console.print(
        f"[bold]Simulation:[/bold] {sim.max_timesteps} {sim.timestep_unit.value}s"
    )
    console.print()

    # Validation Results
    if validation_result.errors:
        console.print(
            f"[red]✗[/red] Validation: {len(validation_result.errors)} error(s)"
        )
        for err in validation_result.errors[:5]:
            console.print(f"  [red]✗[/red] {err.location}: {err.message}")
            if err.suggestion:
                console.print(f"    [dim]→ {err.suggestion}[/dim]")
        if len(validation_result.errors) > 5:
            console.print(
                f"  [dim]... and {len(validation_result.errors) - 5} more[/dim]"
            )
    elif validation_result.warnings:
        console.print(
            f"[green]✓[/green] Validation passed with {len(validation_result.warnings)} warning(s)"
        )
        for warn in validation_result.warnings[:3]:
            console.print(f"  [yellow]⚠[/yellow] {warn.location}: {warn.message}")
    else:
        console.print("[green]✓[/green] Validation passed")

    # Human Checkpoint
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
    console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)
