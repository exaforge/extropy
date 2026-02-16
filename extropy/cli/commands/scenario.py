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
    population: Path = typer.Option(
        ..., "--population", "-p", help="Population spec YAML file"
    ),
    study_db: Path = typer.Option(..., "--study-db", help="Canonical study DB file"),
    population_id: str = typer.Option(
        "default", "--population-id", help="Population ID in study DB"
    ),
    network_id: str = typer.Option(
        "default", "--network-id", help="Network ID in study DB"
    ),
    description: str | None = typer.Option(
        None,
        "--description",
        "-d",
        help="Scenario description (defaults to spec metadata)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (defaults to {population_stem}.scenario.yaml)",
    ),
    timeline: str = typer.Option(
        "auto",
        "--timeline",
        help="Timeline mode: auto (LLM decides), static (single event), evolving (multi-event)",
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
        extropy scenario -p population.yaml --study-db study.db
        extropy scenario -p pop.yaml --study-db study.db --population-id main --network-id main -d "Custom description" -o custom.yaml
    """
    from ...core.models import PopulationSpec
    from ...scenario import create_scenario
    from ...storage import open_study_db

    start_time = time.time()
    console.print()

    # Validate Input Files
    if not population.exists():
        console.print(f"[red]✗[/red] Population spec not found: {population}")
        raise typer.Exit(1)

    if not study_db.exists():
        console.print(f"[red]✗[/red] Study DB not found: {study_db}")
        raise typer.Exit(1)

    with open_study_db(study_db) as db:
        if db.get_agent_count(population_id) == 0:
            console.print(
                f"[red]✗[/red] No agents found for population_id '{population_id}' in {study_db}"
            )
            raise typer.Exit(1)
        if db.get_network_edge_count(network_id) == 0:
            console.print(
                f"[red]✗[/red] No network edges found for network_id '{network_id}' in {study_db}"
            )
            raise typer.Exit(1)

    # Load population spec to get scenario description if not provided
    try:
        pop_spec = PopulationSpec.from_yaml(population)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load population spec: {e}")
        raise typer.Exit(1)

    # Use description from CLI or fall back to spec metadata
    scenario_desc = description or pop_spec.meta.scenario_description
    if not scenario_desc:
        console.print("[red]✗[/red] No scenario description found.")
        console.print(
            "  Either provide --description or use a spec created with 'extropy extend'."
        )
        raise typer.Exit(1)

    # Auto-name output if not provided
    output_path = output or population.with_suffix(".scenario.yaml")

    console.print(f"Creating scenario for: [bold]{scenario_desc}[/bold]")
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
            # Convert timeline mode (auto -> None for LLM decision)
            timeline_mode = None if timeline == "auto" else timeline
            result_spec, validation_result = create_scenario(
                description=scenario_desc,
                population_spec_path=population,
                study_db_path=study_db,
                population_id=population_id,
                network_id=network_id,
                output_path=None,  # Don't save yet
                on_progress=on_progress,
                timeline_mode=timeline_mode,
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

    # Timeline info (Phase C)
    if result_spec.timeline:
        console.print(f"[bold]Timeline:[/bold] {len(result_spec.timeline)} events")
        for te in result_spec.timeline[:3]:
            desc = te.description or te.event.content[:40]
            console.print(f"  • t={te.timestep}: {desc}")
        if len(result_spec.timeline) > 3:
            console.print(f"  [dim]... and {len(result_spec.timeline) - 3} more[/dim]")
        console.print()
    else:
        console.print("[bold]Timeline:[/bold] static (single event)")
        console.print()

    # Background context (Phase C)
    if result_spec.background_context:
        ctx_preview = (
            result_spec.background_context[:60] + "..."
            if len(result_spec.background_context) > 60
            else result_spec.background_context
        )
        console.print(f"[bold]Background:[/bold] {ctx_preview}")
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

    result_spec.meta.population_spec = str(population)
    result_spec.meta.study_db = str(study_db)
    result_spec.meta.population_id = population_id
    result_spec.meta.network_id = network_id

    # Save to YAML
    result_spec.to_yaml(output_path)

    elapsed = time.time() - start_time

    console.print()
    console.print("═" * 60)
    console.print(f"[green]✓[/green] Scenario saved to [bold]{output_path}[/bold]")
    console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)
