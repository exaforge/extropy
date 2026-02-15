"""Estimate command for predicting simulation costs before running."""

from pathlib import Path

import typer

from ..app import app, console


@app.command("estimate")
def estimate_command(
    scenario_file: Path = typer.Argument(..., help="Scenario spec YAML file"),
    study_db: Path = typer.Option(..., "--study-db", help="Canonical study DB file"),
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-timestep breakdown"
    ),
):
    """
    Estimate simulation cost without running it.

    Loads the scenario and population files, runs a simplified propagation
    model, and predicts LLM calls, tokens, and USD cost. No API keys required.

    Example:
        extropy estimate scenario.yaml --study-db study.db
        extropy estimate scenario.yaml --study-db study.db --model gpt-5-mini
        extropy estimate scenario.yaml --study-db study.db --pivotal-model gpt-5 --routine-model gpt-5-mini -v
    """
    from ...config import get_config
    from ...core.models import ScenarioSpec, PopulationSpec
    from ...simulation.estimator import estimate_simulation_cost
    from ...storage import open_study_db

    # Validate input file
    if not scenario_file.exists():
        console.print(f"[red]x[/red] Scenario file not found: {scenario_file}")
        raise typer.Exit(1)
    if not study_db.exists():
        console.print(f"[red]x[/red] Study DB not found: {study_db}")
        raise typer.Exit(1)

    # Load scenario
    try:
        scenario = ScenarioSpec.from_yaml(scenario_file)
    except Exception as e:
        console.print(f"[red]x[/red] Failed to load scenario: {e}")
        raise typer.Exit(1)

    # Load population spec
    pop_path = Path(scenario.meta.population_spec)
    if not pop_path.is_absolute():
        pop_path = scenario_file.parent / pop_path
    if not pop_path.exists():
        console.print(f"[red]x[/red] Population spec not found: {pop_path}")
        raise typer.Exit(1)
    population_spec = PopulationSpec.from_yaml(pop_path)

    with open_study_db(study_db) as db:
        agents = db.get_agents(scenario.meta.population_id)
        network = db.get_network(scenario.meta.network_id)
    if not agents:
        console.print(
            f"[red]x[/red] Population ID not found in study DB: {scenario.meta.population_id}"
        )
        raise typer.Exit(1)
    if not network.get("edges"):
        console.print(
            f"[red]x[/red] Network ID not found in study DB: {scenario.meta.network_id}"
        )
        raise typer.Exit(1)

    # Resolve config
    config = get_config()
    provider = config.simulation.provider

    eff_model = model or config.simulation.model
    eff_pivotal = pivotal_model or config.simulation.pivotal_model or eff_model
    eff_routine = routine_model or config.simulation.routine_model

    # Run estimation
    est = estimate_simulation_cost(
        scenario=scenario,
        population_spec=population_spec,
        agents=agents,
        network=network,
        provider=provider,
        pivotal_model=eff_pivotal,
        routine_model=eff_routine,
        multi_touch_threshold=threshold,
    )

    # Display results
    console.print()
    console.print("[bold]Simulation Cost Estimate[/bold]")
    console.print()

    early_stop = (
        f" (early stop at ~{est.per_timestep[-1]['exposure_rate']:.0%} exposure)"
        if est.effective_timesteps < est.max_timesteps and est.per_timestep
        else ""
    )
    console.print(
        f"Population: {est.population_size} agents | "
        f"Avg degree: {est.avg_degree:.1f} | "
        f"Max timesteps: {est.max_timesteps}"
    )
    console.print(f"Effective timesteps: ~{est.effective_timesteps}{early_stop}")
    console.print()

    # Models section
    console.print("[bold]Models[/bold]")
    _print_model_line(
        console, "Pass 1 (pivotal)", est.pivotal_model, est.pivotal_pricing
    )
    _print_model_line(
        console, "Pass 2 (routine)", est.routine_model, est.routine_pricing
    )
    console.print()

    # Calls table
    console.print("[bold]Estimated LLM Calls[/bold]")
    console.print(
        f"  {'':16s} {'Calls':>8s}   {'Input Tok':>12s}   {'Output Tok':>12s}"
    )
    console.print(
        f"  {'Pass 1':16s} {est.pass1_calls:>8,}   "
        f"{'~' + f'{est.pass1_input_tokens:,}':>12s}   "
        f"{'~' + f'{est.pass1_output_tokens:,}':>12s}"
    )
    console.print(
        f"  {'Pass 2':16s} {est.pass2_calls:>8,}   "
        f"{'~' + f'{est.pass2_input_tokens:,}':>12s}   "
        f"{'~' + f'{est.pass2_output_tokens:,}':>12s}"
    )
    total_input = est.pass1_input_tokens + est.pass2_input_tokens
    total_output = est.pass1_output_tokens + est.pass2_output_tokens
    console.print(
        f"  {'Total':16s} {est.pass1_calls + est.pass2_calls:>8,}   "
        f"{'~' + f'{total_input:,}':>12s}   "
        f"{'~' + f'{total_output:,}':>12s}"
    )
    console.print()

    # Cost section
    console.print("[bold]Estimated Cost[/bold]")
    if est.pass1_cost is not None:
        console.print(f"  Pass 1:  ${est.pass1_cost:.2f}")
    else:
        console.print(
            f"  Pass 1:  [dim]pricing not available for {est.pivotal_model}[/dim]"
        )

    if est.pass2_cost is not None:
        console.print(f"  Pass 2:  ${est.pass2_cost:.2f}")
    else:
        console.print(
            f"  Pass 2:  [dim]pricing not available for {est.routine_model}[/dim]"
        )

    if est.total_cost is not None:
        console.print(f"  [bold]Total:   ${est.total_cost:.2f}[/bold]")
    console.print()

    console.print(
        "[dim]Estimates are approximate. Actual costs vary with prompt length "
        "and simulation dynamics.[/dim]"
    )

    # Verbose per-timestep breakdown
    if verbose and est.per_timestep:
        console.print()
        console.print("[bold]Per-Timestep Breakdown[/bold]")
        console.print(
            f"  {'Step':>6s}   {'Exposure':>9s}   {'New Exp':>8s}   {'Reasoning':>10s}"
        )
        for row in est.per_timestep:
            if row["new_exposures"] > 0 or row["reasoning_calls"] > 0:
                console.print(
                    f"  {row['timestep']:>6d}   "
                    f"{row['exposure_rate']:>8.1%}   "
                    f"{row['new_exposures']:>8d}   "
                    f"{row['reasoning_calls']:>10d}"
                )
    console.print()


def _print_model_line(console, label: str, model: str, pricing):
    """Print a model info line with optional pricing."""
    if pricing:
        console.print(
            f"  {label}:  {model}  "
            f"(${pricing.input_per_mtok:.2f} / ${pricing.output_per_mtok:.2f} per MTok in/out)"
        )
    else:
        console.print(f"  {label}:  {model}  [dim](pricing not available)[/dim]")
