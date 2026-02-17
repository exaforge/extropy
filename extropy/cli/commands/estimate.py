"""Estimate command for predicting simulation costs before running."""

import typer

from ..app import app, console, get_study_path
from ..study import StudyContext, detect_study_folder, resolve_scenario
from ..utils import Output, ExitCode


@app.command("estimate")
def estimate_command(
    scenario: str = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (auto-selects if only one exists)",
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-timestep breakdown"
    ),
):
    """
    Estimate simulation cost without running it.

    Loads the scenario and population files, runs a simplified propagation
    model, and predicts LLM calls, tokens, and USD cost. No API keys required.

    Example:
        extropy estimate -s ai-adoption
        extropy estimate -s ai-adoption --strong openai/gpt-5
        extropy estimate -s ai-adoption --strong openai/gpt-5 --fast openai/gpt-5-mini -v
    """
    from ...config import get_config
    from ...core.models import ScenarioSpec, PopulationSpec
    from ...simulation.estimator import estimate_simulation_cost
    from ...storage import open_study_db
    from ...core.cost.tracker import CostTracker

    CostTracker.get().set_context(command="estimate")

    out = Output(console=console)

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

    if not study_db.exists():
        out.error(f"Study DB not found: {study_db}", exit_code=ExitCode.FILE_NOT_FOUND)
        raise typer.Exit(out.finish())

    # Resolve scenario
    try:
        scenario_name, scenario_version = resolve_scenario(study_ctx, scenario)
    except ValueError as e:
        out.error(str(e), exit_code=ExitCode.FILE_NOT_FOUND)
        raise typer.Exit(out.finish())

    # Load scenario spec
    try:
        scenario_path = study_ctx.get_scenario_path(scenario_name, scenario_version)
        scenario_spec = ScenarioSpec.from_yaml(scenario_path)
    except FileNotFoundError:
        out.error(
            f"Scenario not found: {scenario_name}", exit_code=ExitCode.FILE_NOT_FOUND
        )
        raise typer.Exit(out.finish())
    except Exception as e:
        out.error(f"Failed to load scenario: {e}")
        raise typer.Exit(out.finish())

    # Load population spec (resolve from base_population or population_spec)
    try:
        pop_name, pop_version = scenario_spec.meta.get_population_ref()
    except ValueError as e:
        out.error(str(e))
        raise typer.Exit(1)

    pop_path = study_ctx.get_population_path(pop_name, pop_version)
    if not pop_path.exists():
        out.error(f"Population spec not found: {pop_path}")
        raise typer.Exit(1)
    try:
        population_spec = PopulationSpec.from_yaml(pop_path)
    except Exception as e:
        out.error(f"Failed to load population spec: {e}")
        raise typer.Exit(1)

    # Load agents and network from study DB using scenario ID
    with open_study_db(study_db) as db:
        agents = db.get_agents_by_scenario(scenario_name)
        network = db.get_network(scenario_name)

    if not agents:
        out.error(
            f"No agents found for scenario '{scenario_name}'. "
            f"Run 'extropy sample -s {scenario_name}' first.",
        )
        raise typer.Exit(1)
    if not network.get("edges"):
        out.error(
            f"No network found for scenario '{scenario_name}'. "
            f"Run 'extropy network -s {scenario_name}' first.",
        )
        raise typer.Exit(1)

    # Resolve config
    config = get_config()
    effective_strong = strong or config.resolve_sim_strong()
    effective_fast = fast or config.resolve_sim_fast()

    # Run estimation
    est = estimate_simulation_cost(
        scenario=scenario_spec,
        population_spec=population_spec,
        agents=agents,
        network=network,
        strong_model=effective_strong,
        fast_model=effective_fast,
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
        console, "Pass 1 (strong)", est.pivotal_model, est.pivotal_pricing
    )
    _print_model_line(console, "Pass 2 (fast)", est.routine_model, est.routine_pricing)
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
