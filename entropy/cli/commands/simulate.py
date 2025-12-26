"""Simulate command for running simulations from scenario specs."""

import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ..app import app, console
from ..utils import format_elapsed


@app.command("simulate")
def simulate_command(
    scenario_file: Path = typer.Argument(..., help="Scenario spec YAML file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output results directory"),
    model: str = typer.Option(
        "gpt-5-mini", "--model", "-m", help="LLM model for agent reasoning"
    ),
    threshold: int = typer.Option(
        3, "--threshold", "-t", help="Multi-touch threshold for re-reasoning"
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
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
    from ...simulation import run_simulation

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
        spinner = Spinner("dots", text="Starting...", style="cyan")
        with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
            while not simulation_done.is_set():
                elapsed = time.time() - start_time
                timestep, max_ts, status = current_progress
                if max_ts > 0:
                    pct = timestep / max_ts * 100
                    spinner.update(
                        text=f"Timestep {timestep}/{max_ts} ({pct:.0f}%) | {status} | {format_elapsed(elapsed)}"
                    )
                else:
                    spinner.update(text=f"{status} | {format_elapsed(elapsed)}")
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
    console.print("[green]✓[/green] Simulation complete")
    console.print("═" * 60)
    console.print()

    console.print(
        f"Duration: {format_elapsed(elapsed)} ({result.total_timesteps} timesteps)"
    )
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
