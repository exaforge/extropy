"""Simulate command for running simulations from scenario specs."""

import logging
import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.logging import RichHandler
from rich.spinner import Spinner

from ..app import app, console
from ..utils import format_elapsed


def setup_logging(verbose: bool = False, debug: bool = False):
    """Configure logging for simulation."""
    level = logging.WARNING
    if verbose:
        level = logging.INFO
    if debug:
        level = logging.DEBUG

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
        force=True,
    )

    # Also set specific loggers
    for name in ["entropy.simulation", "entropy.core.llm"]:
        logging.getLogger(name).setLevel(level)


@app.command("simulate")
def simulate_command(
    scenario_file: Path = typer.Argument(..., help="Scenario spec YAML file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output results directory"),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="LLM model for agent reasoning (empty = use config default)",
    ),
    pivotal_model: str = typer.Option(
        "",
        "--pivotal-model",
        help="Model for pivotal/first-pass reasoning (default: same as --model)",
    ),
    routine_model: str = typer.Option(
        "",
        "--routine-model",
        help="Cheap model for classification pass (default: provider cheap tier)",
    ),
    threshold: int = typer.Option(
        3, "--threshold", "-t", help="Multi-touch threshold for re-reasoning"
    ),
    rate_tier: int | None = typer.Option(
        None, "--rate-tier", help="Rate limit tier (1-4, higher = more generous limits)"
    ),
    rpm_override: int | None = typer.Option(
        None, "--rpm-override", help="Override requests per minute limit"
    ),
    tpm_override: int | None = typer.Option(
        None, "--tpm-override", help="Override tokens per minute limit"
    ),
    chunk_size: int = typer.Option(
        50, "--chunk-size", help="Agents per reasoning chunk for checkpointing"
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
    persona_config: Path | None = typer.Option(
        None,
        "--persona",
        "-p",
        help="PersonaConfig YAML for embodied personas (auto-detected if not specified)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
    debug: bool = typer.Option(
        False, "--debug", help="Show debug-level logs (very verbose)"
    ),
):
    """
    Run a simulation from a scenario spec.

    Executes the scenario against its population, simulating opinion
    dynamics with agent reasoning, network propagation, and state evolution.

    If a persona config exists at <population>.persona.yaml, it will be
    used automatically for embodied first-person personas.

    Example:
        entropy simulate scenario.yaml -o results/
        entropy simulate scenario.yaml -o results/ --model gpt-5-nano --seed 42
        entropy simulate scenario.yaml -o results/ --persona population.persona.yaml
    """
    from ...simulation import run_simulation

    # Setup logging based on verbosity
    setup_logging(verbose=verbose, debug=debug)

    start_time = time.time()
    console.print()

    # Validate input file
    if not scenario_file.exists():
        console.print(f"[red]✗[/red] Scenario file not found: {scenario_file}")
        raise typer.Exit(1)

    from ...config import get_config

    config = get_config()

    # Resolve models from CLI args > config > defaults
    effective_model = model or config.simulation.model
    effective_pivotal = pivotal_model or config.simulation.pivotal_model
    effective_routine = routine_model or config.simulation.routine_model
    effective_tier = rate_tier or config.simulation.rate_tier
    effective_rpm = rpm_override or config.simulation.rpm_override
    effective_tpm = tpm_override or config.simulation.tpm_override

    display_model = effective_model or f"({config.simulation.provider} default)"
    display_provider = config.simulation.provider

    console.print(f"Simulating: [bold]{scenario_file}[/bold]")
    console.print(f"Output: {output}")
    console.print(
        f"Provider: {display_provider} | Model: {display_model} | Threshold: {threshold}"
    )
    if effective_pivotal or effective_routine:
        parts = []
        if effective_pivotal:
            parts.append(f"Pivotal: {effective_pivotal}")
        if effective_routine:
            parts.append(f"Routine: {effective_routine}")
        console.print(" | ".join(parts))
    if effective_tier:
        console.print(f"Rate tier: {effective_tier}")
    if effective_rpm or effective_tpm:
        parts = []
        if effective_rpm:
            parts.append(f"RPM: {effective_rpm}")
        if effective_tpm:
            parts.append(f"TPM: {effective_tpm}")
        console.print(f"Rate overrides: {' | '.join(parts)}")
    if seed:
        console.print(f"Seed: {seed}")
    if verbose or debug:
        console.print(f"Logging: {'DEBUG' if debug else 'VERBOSE'}")
    console.print()

    # Progress tracking
    current_progress = [0, 0, "Starting..."]

    def on_progress(timestep: int, max_timesteps: int, status: str):
        current_progress[0] = timestep
        current_progress[1] = max_timesteps
        current_progress[2] = status

    # When verbose/debug, run synchronously to see all logs clearly
    if verbose or debug:
        console.print("[dim]Running with verbose logging (no spinner)...[/dim]")
        console.print()
        try:
            result = run_simulation(
                scenario_path=scenario_file,
                output_dir=output,
                model=effective_model,
                pivotal_model=effective_pivotal,
                routine_model=effective_routine,
                multi_touch_threshold=threshold,
                random_seed=seed,
                on_progress=on_progress,
                persona_config_path=persona_config,
                rate_tier=effective_tier,
                rpm_override=effective_rpm,
                tpm_override=effective_tpm,
                chunk_size=chunk_size,
            )
            simulation_error = None
        except Exception as e:
            simulation_error = e
            result = None
    else:
        # Run simulation in thread with spinner
        simulation_done = Event()
        simulation_error = None
        result = None

        def do_simulation():
            nonlocal result, simulation_error
            try:
                result = run_simulation(
                    scenario_path=scenario_file,
                    output_dir=output,
                    model=effective_model,
                    pivotal_model=effective_pivotal,
                    routine_model=effective_routine,
                    multi_touch_threshold=threshold,
                    random_seed=seed,
                    on_progress=on_progress if not quiet else None,
                    persona_config_path=persona_config,
                    rate_tier=effective_tier,
                    rpm_override=effective_rpm,
                    tpm_override=effective_tpm,
                    chunk_size=chunk_size,
                )
            except Exception as e:
                simulation_error = e
            finally:
                simulation_done.set()

        simulation_thread = Thread(target=do_simulation, daemon=True)
        simulation_thread.start()

        if not quiet:
            spinner = Spinner("dots", text="Starting...", style="cyan")
            with Live(
                spinner, console=console, refresh_per_second=12.5, transient=True
            ):
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
