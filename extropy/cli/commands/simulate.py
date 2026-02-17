"""Simulate command for running simulations from scenario specs."""

import logging
import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.logging import RichHandler
from rich.spinner import Spinner
from rich.text import Text

from ..app import app, console, is_agent_mode, get_study_path
from ..study import StudyContext, detect_study_folder, resolve_scenario
from ..utils import format_elapsed, Output, ExitCode


_BAR_WIDTH = 20


def _build_progress_display(snap: dict, elapsed: float) -> Text:
    """Build a Rich Text renderable showing live simulation progress.

    Shows a header line with timestep/agent progress and distribution bars
    for each position, sorted by count descending.

    Args:
        snap: Snapshot dict from SimulationProgress.snapshot()
        elapsed: Elapsed seconds since simulation start

    Returns:
        Rich Text object for Live display
    """
    text = Text()

    agents_done = snap.get("agents_done", 0)
    agents_total = snap.get("agents_total", 0)
    timestep = snap.get("timestep", 0)
    max_ts = snap.get("max_timesteps", 0)
    exposure_rate = snap.get("exposure_rate", 0.0)

    # Header line
    if max_ts > 0 and agents_total > 0:
        pct = agents_done / agents_total * 100 if agents_total > 0 else 0
        header = (
            f"Timestep {timestep + 1}/{max_ts} | "
            f"{agents_done}/{agents_total} agents ({pct:.0f}%) | "
            f"Exposure: {exposure_rate:.1%} | "
            f"{format_elapsed(elapsed)}"
        )
    elif max_ts > 0:
        header = f"Timestep {timestep + 1}/{max_ts} | {format_elapsed(elapsed)}"
    else:
        header = f"Starting... | {format_elapsed(elapsed)}"

    text.append(header, style="cyan bold")

    # Distribution bars
    counts = snap.get("position_counts", {})
    if counts:
        total = sum(counts.values()) or 1
        max_label = 40
        # Find longest position name for alignment (capped)
        max_name_len = min(max(len(name) for name in counts), max_label)

        for position, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            filled = round(pct / 100 * _BAR_WIDTH)
            bar = "\u2588" * filled + "\u2591" * (_BAR_WIDTH - filled)
            label = position[:max_label] if len(position) > max_label else position
            text.append("\n")
            text.append(f"\n  {label:<{max_name_len}}  {pct:>3.0f}% ", style="bold")
            text.append(bar, style="cyan")

    return text


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
    for name in ["extropy.simulation", "extropy.core.llm"]:
        logging.getLogger(name).setLevel(level)


@app.command("simulate")
def simulate_command(
    scenario: str = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (auto-selects if only one exists)",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output results directory (defaults to results/)"
    ),
    strong: str = typer.Option(
        "",
        "--strong",
        help="Strong model for Pass 1 (provider/model format)",
    ),
    fast: str = typer.Option(
        "",
        "--fast",
        help="Fast model for Pass 2 (provider/model format)",
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
    checkpoint_every_chunks: int = typer.Option(
        1,
        "--checkpoint-every-chunks",
        min=1,
        help="Persist simulation chunk checkpoints every N chunks",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Explicit run id (required with --resume)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an existing run from study DB checkpoints",
    ),
    writer_queue_size: int = typer.Option(
        256,
        "--writer-queue-size",
        min=1,
        help="Max reasoning chunks buffered before DB writer backpressure",
    ),
    db_write_batch_size: int = typer.Option(
        100,
        "--db-write-batch-size",
        min=1,
        help="Number of chunks applied per DB writer transaction",
    ),
    retention_lite: bool = typer.Option(
        False,
        "--retention-lite",
        help="Reduce retained payload volume (drops full raw reasoning text)",
    ),
    resource_mode: str = typer.Option(
        "auto",
        "--resource-mode",
        help="Resource tuning mode: auto | manual",
    ),
    safe_auto_workers: bool = typer.Option(
        True,
        "--safe-auto-workers/--unsafe-auto-workers",
        help="Conservative auto tuning for laptop/VM environments",
    ),
    max_memory_gb: float | None = typer.Option(
        None,
        "--max-memory-gb",
        min=0.5,
        help="Optional memory budget cap for auto resource tuning",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
    merged_pass: bool = typer.Option(
        False,
        "--merged-pass",
        help="Use single merged reasoning pass instead of two-pass (experimental)",
    ),
    fidelity: str = typer.Option(
        "medium",
        "--fidelity",
        "-f",
        help="Fidelity level: low (no conversations), medium (2 turns), high (3 turns)",
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

    Prerequisites:
        - Population spec must exist
        - Scenario spec must exist
        - Persona config must exist for the scenario
        - Agents must be sampled
        - Network must be generated

    Example:
        extropy simulate -s ai-adoption
        extropy simulate -s ai-adoption --seed 42 --strong gpt-4o
        extropy simulate -s ai-adoption --fidelity high
    """
    from ...simulation import run_simulation
    from ...simulation.progress import SimulationProgress
    from ...core.models.scenario import ScenarioSpec
    from ...storage import open_study_db
    from ...utils import ResourceGovernor
    from ...core.cost.tracker import CostTracker

    CostTracker.get().set_context(command="simulate")

    # Setup logging based on verbosity
    setup_logging(verbose=verbose, debug=debug)

    agent_mode = is_agent_mode()
    out = Output(console, json_mode=agent_mode)
    start_time = time.time()

    if not agent_mode:
        console.print()

    # Resolve study context
    study_path = get_study_path()
    detected = detect_study_folder(study_path)
    if detected is None:
        out.error(
            "Not in a study folder. Use --study to specify or run from a study folder.",
            exit_code=ExitCode.FILE_NOT_FOUND,
        )
        raise typer.Exit(out.finish())

    study_ctx = StudyContext(detected)
    study_db = study_ctx.db_path

    # Resolve scenario
    try:
        scenario_name, scenario_version = resolve_scenario(study_ctx, scenario)
    except ValueError as e:
        out.error(str(e), exit_code=ExitCode.FILE_NOT_FOUND)
        raise typer.Exit(out.finish())

    # Pre-flight: Check agents exist for this scenario
    with open_study_db(study_db) as db:
        agent_count = db.get_agent_count_by_scenario(scenario_name)
    if agent_count == 0:
        out.error(
            f"No agents found for scenario: {scenario_name}. "
            f"Run 'extropy sample -s {scenario_name} -n COUNT' first.",
            exit_code=ExitCode.FILE_NOT_FOUND,
        )
        raise typer.Exit(out.finish())

    # Pre-flight: Check network exists for this scenario
    with open_study_db(study_db) as db:
        edge_count = db.get_network_edge_count_by_scenario(scenario_name)
    if edge_count == 0:
        out.error(
            f"No network found for scenario: {scenario_name}. "
            f"Run 'extropy network -s {scenario_name}' first.",
            exit_code=ExitCode.FILE_NOT_FOUND,
        )
        raise typer.Exit(out.finish())

    # Pre-flight: Check persona config exists
    try:
        persona_path = study_ctx.get_persona_path(scenario_name)
    except FileNotFoundError:
        out.error(
            f"No persona config found for scenario: {scenario_name}. "
            f"Run 'extropy persona -s {scenario_name}' first.",
            exit_code=ExitCode.FILE_NOT_FOUND,
        )
        raise typer.Exit(out.finish())

    # Load scenario spec (validates it exists)
    try:
        scenario_path = study_ctx.get_scenario_path(scenario_name, scenario_version)
        ScenarioSpec.from_yaml(scenario_path)  # Validate it loads
    except FileNotFoundError:
        out.error(f"Scenario not found: {scenario_name}")
        raise typer.Exit(1)
    except Exception as e:
        out.error(f"Failed to load scenario: {e}")
        raise typer.Exit(1)

    # Resolve output directory
    if output is None:
        output = study_ctx.root / "results" / scenario_name
    output.mkdir(parents=True, exist_ok=True)

    # Validate flags
    if resume and not run_id:
        out.error("--resume requires --run-id")
        raise typer.Exit(1)
    if resource_mode not in {"auto", "manual"}:
        out.error("--resource-mode must be 'auto' or 'manual'")
        raise typer.Exit(1)

    from ...config import get_config

    config = get_config()

    # Resolve models from CLI args > config > defaults
    effective_strong = strong or config.resolve_sim_strong()
    effective_fast = fast or config.resolve_sim_fast()
    effective_tier = rate_tier or config.simulation.rate_tier
    effective_rpm = rpm_override or config.simulation.rpm_override
    effective_tpm = tpm_override or config.simulation.tpm_override

    out.success(
        f"Loaded scenario: [bold]{scenario_name}[/bold] "
        f"({agent_count} agents, {edge_count} edges)",
        scenario=scenario_name,
        agent_count=agent_count,
        edge_count=edge_count,
    )

    if not agent_mode:
        console.print(f"Output: {output}")
        console.print(
            f"Strong: {effective_strong} | Fast: {effective_fast} | Threshold: {threshold}"
        )
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

    governor = ResourceGovernor(
        resource_mode=resource_mode,
        safe_auto_workers=safe_auto_workers,
        max_memory_gb=max_memory_gb,
    )
    tuned_chunk_size = governor.recommend_chunk_size(
        requested_chunk_size=chunk_size,
        min_chunk_size=8,
        max_chunk_size=2000,
    )
    if resource_mode == "auto" and not agent_mode:
        snap = governor.snapshot()
        console.print(
            f"Resources(auto): cpu={snap.cpu_count} mem={snap.total_memory_gb:.1f}GB "
            f"budget={snap.memory_budget_gb:.1f}GB chunk={tuned_chunk_size}"
        )
    if (verbose or debug) and not agent_mode:
        console.print(f"Logging: {'DEBUG' if debug else 'VERBOSE'}")
    if not agent_mode:
        console.print()

    # Shared progress state for live display
    progress_state = SimulationProgress()

    # Progress tracking (timestep-level callback)
    current_progress = [0, 0, "Starting..."]

    def on_progress(timestep: int, max_timesteps: int, status: str):
        current_progress[0] = timestep
        current_progress[1] = max_timesteps
        current_progress[2] = status

    # Run simulation
    simulation_error = None
    result = None

    # When verbose/debug, run synchronously to see all logs clearly
    if verbose or debug:
        if not agent_mode:
            console.print("[dim]Running with verbose logging (no spinner)...[/dim]")
            console.print()
        try:
            result = run_simulation(
                scenario_path=scenario_path,
                output_dir=output,
                study_db_path=study_db,
                strong=effective_strong,
                fast=effective_fast,
                multi_touch_threshold=threshold,
                random_seed=seed,
                on_progress=on_progress,
                persona_config_path=persona_path,
                rate_tier=effective_tier,
                rpm_override=effective_rpm,
                tpm_override=effective_tpm,
                chunk_size=tuned_chunk_size,
                progress=progress_state,
                run_id=run_id,
                resume=resume,
                checkpoint_every_chunks=checkpoint_every_chunks,
                retention_lite=retention_lite,
                writer_queue_size=writer_queue_size,
                db_write_batch_size=db_write_batch_size,
                resource_governor=governor,
                merged_pass=merged_pass,
                fidelity=fidelity,
            )
        except Exception as e:
            simulation_error = e
    elif agent_mode:
        # Agent mode: no spinner, just run
        try:
            result = run_simulation(
                scenario_path=scenario_path,
                output_dir=output,
                study_db_path=study_db,
                strong=effective_strong,
                fast=effective_fast,
                multi_touch_threshold=threshold,
                random_seed=seed,
                on_progress=on_progress if not quiet else None,
                persona_config_path=persona_path,
                rate_tier=effective_tier,
                rpm_override=effective_rpm,
                tpm_override=effective_tpm,
                chunk_size=tuned_chunk_size,
                progress=progress_state,
                run_id=run_id,
                resume=resume,
                checkpoint_every_chunks=checkpoint_every_chunks,
                retention_lite=retention_lite,
                writer_queue_size=writer_queue_size,
                db_write_batch_size=db_write_batch_size,
                resource_governor=governor,
                merged_pass=merged_pass,
                fidelity=fidelity,
            )
        except Exception as e:
            simulation_error = e
    else:
        # Run simulation in thread with live display
        simulation_done = Event()

        def do_simulation():
            nonlocal result, simulation_error
            try:
                result = run_simulation(
                    scenario_path=scenario_path,
                    output_dir=output,
                    study_db_path=study_db,
                    strong=effective_strong,
                    fast=effective_fast,
                    multi_touch_threshold=threshold,
                    random_seed=seed,
                    on_progress=on_progress if not quiet else None,
                    persona_config_path=persona_path,
                    rate_tier=effective_tier,
                    rpm_override=effective_rpm,
                    tpm_override=effective_tpm,
                    chunk_size=tuned_chunk_size,
                    progress=progress_state,
                    run_id=run_id,
                    resume=resume,
                    checkpoint_every_chunks=checkpoint_every_chunks,
                    retention_lite=retention_lite,
                    writer_queue_size=writer_queue_size,
                    db_write_batch_size=db_write_batch_size,
                    resource_governor=governor,
                    merged_pass=merged_pass,
                    fidelity=fidelity,
                )
            except Exception as e:
                simulation_error = e
            finally:
                simulation_done.set()

        simulation_thread = Thread(target=do_simulation, daemon=True)
        simulation_thread.start()

        if not quiet:
            with Live(
                Spinner("dots", text="Starting...", style="cyan"),
                console=console,
                refresh_per_second=4,
                transient=True,
            ) as live:
                while not simulation_done.is_set():
                    elapsed = time.time() - start_time
                    snap = progress_state.snapshot()
                    live.update(_build_progress_display(snap, elapsed))
                    time.sleep(0.25)
        else:
            simulation_done.wait()

    if simulation_error:
        out.error(f"Simulation failed: {simulation_error}")
        raise typer.Exit(1)

    elapsed = time.time() - start_time

    # Set data for JSON output
    out.set_data("scenario_id", scenario_name)
    out.set_data("study_db", str(study_db))
    out.set_data("output_dir", str(output))
    out.set_data("total_time_seconds", elapsed)
    out.set_data("total_timesteps", result.total_timesteps)
    out.set_data("total_reasoning_calls", result.total_reasoning_calls)
    out.set_data("final_exposure_rate", result.final_exposure_rate)
    if result.stopped_reason:
        out.set_data("stopped_reason", result.stopped_reason)
    if result.outcome_distributions:
        out.set_data("outcome_distributions", result.outcome_distributions)

    # Display summary
    if not agent_mode:
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
                        console.print(
                            f"  {outcome_name}: mean={distribution['mean']:.2f}"
                        )
                    else:
                        top_3 = sorted(distribution.items(), key=lambda x: -x[1])[:3]
                        dist_str = ", ".join(f"{k}:{v:.1%}" for k, v in top_3)
                        console.print(f"  {outcome_name}: {dist_str}")
            console.print()

        console.print(f"Results saved to: [bold]{output}[/bold]")

    raise typer.Exit(out.finish())


