"""Network command for generating social networks from agents."""

import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ..app import app, console
from ..utils import format_elapsed


@app.command("network")
def network_command(
    agents_file: Path = typer.Argument(
        ..., help="Agents JSON file to generate network from"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output network JSON file"),
    population: Path | None = typer.Option(
        None,
        "--population",
        "-p",
        help="Population spec YAML — generates network config via LLM",
    ),
    network_config: Path | None = typer.Option(
        None,
        "--network-config",
        "-c",
        help="Custom network config YAML file",
    ),
    save_config: Path | None = typer.Option(
        None,
        "--save-config",
        help="Save the (generated or loaded) network config to YAML",
    ),
    avg_degree: float = typer.Option(
        20.0, "--avg-degree", help="Target average degree (connections per agent)"
    ),
    rewire_prob: float = typer.Option(
        0.05, "--rewire-prob", help="Watts-Strogatz rewiring probability"
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
    validate: bool = typer.Option(
        False, "--validate", "-v", help="Print validation metrics"
    ),
    no_metrics: bool = typer.Option(
        False, "--no-metrics", help="Skip computing node metrics (faster)"
    ),
):
    """
    Generate a social network from sampled agents.

    Creates edges between agents based on attribute similarity, with
    degree correction for high-influence agents and Watts-Strogatz
    rewiring for small-world properties.

    Network config resolution:
      1. --network-config file → load from YAML
      2. Auto-detect {population_stem}.network-config.yaml if it exists
      3. --population spec → generate config via LLM
      4. None of the above → empty config (flat network, no similarity structure)

    Example:
        extropy network agents.json -o network.json
        extropy network agents.json -o network.json -p population.yaml
        extropy network agents.json -o network.json -c network-config.yaml
        extropy network agents.json -o network.json -p population.yaml --save-config my-config.yaml
    """
    from ...population.network import (
        generate_network,
        generate_network_with_metrics,
        load_agents_json,
        NetworkConfig,
        generate_network_config,
    )
    from ...core.models import PopulationSpec

    start_time = time.time()
    console.print()

    # Load Agents
    if not agents_file.exists():
        console.print(f"[red]✗[/red] Agents file not found: {agents_file}")
        raise typer.Exit(1)

    with console.status("[cyan]Loading agents...[/cyan]"):
        try:
            agents = load_agents_json(agents_file)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load agents: {e}")
            raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Loaded {len(agents)} agents from [bold]{agents_file}[/bold]"
    )

    # =========================================================================
    # Resolve network config
    # =========================================================================

    config = None

    # 1. Explicit --network-config file
    if network_config:
        if not network_config.exists():
            console.print(f"[red]✗[/red] Network config not found: {network_config}")
            raise typer.Exit(1)
        with console.status("[cyan]Loading network config...[/cyan]"):
            config = NetworkConfig.from_yaml(network_config)
        console.print(
            f"[green]✓[/green] Loaded network config from [bold]{network_config}[/bold] "
            f"({len(config.attribute_weights)} weights, {len(config.edge_type_rules)} rules)"
        )

    # 2. Auto-detect {population_stem}.network-config.yaml
    if config is None and population:
        auto_config_path = population.parent / f"{population.stem}.network-config.yaml"
        if auto_config_path.exists():
            with console.status("[cyan]Loading auto-detected network config...[/cyan]"):
                config = NetworkConfig.from_yaml(auto_config_path)
            console.print(
                f"[green]✓[/green] Auto-detected config: [bold]{auto_config_path}[/bold] "
                f"({len(config.attribute_weights)} weights, {len(config.edge_type_rules)} rules)"
            )

    # 3. --population spec → LLM generation
    if config is None and population:
        if not population.exists():
            console.print(f"[red]✗[/red] Population spec not found: {population}")
            raise typer.Exit(1)
        with console.status("[cyan]Loading population spec...[/cyan]"):
            pop_spec = PopulationSpec.from_yaml(population)
        console.print(
            f"[green]✓[/green] Loaded population: [bold]{pop_spec.meta.description}[/bold]"
        )
        console.print()

        # Sample a few agents for context
        agents_sample = agents[:5] if len(agents) >= 5 else agents

        with console.status("[cyan]Generating network config via LLM...[/cyan]"):
            config = generate_network_config(pop_spec, agents_sample)
        console.print(
            f"[green]✓[/green] Generated network config: "
            f"{len(config.attribute_weights)} weights, "
            f"{len(config.edge_type_rules)} edge rules, "
            f"{len(config.influence_factors)} influence factors"
        )

    # 4. No config source → empty config (flat network)
    if config is None:
        config = NetworkConfig()
        console.print(
            "[yellow]![/yellow] No network config — generating flat network "
            "(use -p or -c for meaningful social structure)"
        )

    # Apply CLI overrides
    config = config.model_copy(
        update={
            "avg_degree": avg_degree,
            "rewire_prob": rewire_prob,
            "seed": seed if seed is not None else config.seed,
        }
    )

    # Save config if requested
    if save_config:
        config.to_yaml(save_config)
        console.print(
            f"[green]✓[/green] Saved network config to [bold]{save_config}[/bold]"
        )

    console.print()
    generation_start = time.time()
    current_stage = ["Initializing", 0, 0]

    def on_progress(stage: str, current: int, total: int):
        current_stage[0] = stage
        current_stage[1] = current
        current_stage[2] = total

    result = None
    generation_error = None
    generation_done = Event()

    def do_generation():
        nonlocal result, generation_error
        try:
            if no_metrics:
                result = generate_network(agents, config, on_progress)
            else:
                result = generate_network_with_metrics(agents, config, on_progress)
        except Exception as e:
            generation_error = e
        finally:
            generation_done.set()

    gen_thread = Thread(target=do_generation, daemon=True)
    gen_thread.start()

    spinner = Spinner("dots", text="Initializing...", style="cyan")
    with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
        while not generation_done.is_set():
            elapsed = time.time() - generation_start
            stage, current, total = current_stage
            if total > 0:
                pct = current / total * 100
                spinner.update(
                    text=f"{stage}... {current}/{total} ({pct:.0f}%) {format_elapsed(elapsed)}"
                )
            else:
                spinner.update(text=f"{stage}... {format_elapsed(elapsed)}")
            time.sleep(0.1)

    generation_elapsed = time.time() - generation_start

    if generation_error:
        console.print(f"[red]✗[/red] Network generation failed: {generation_error}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Generated network: {result.meta['edge_count']} edges, "
        f"avg degree {result.meta['avg_degree']:.1f} ({format_elapsed(generation_elapsed)})"
    )

    # Validation (optional)
    if validate:
        console.print()
        console.print("┌" + "─" * 58 + "┐")
        console.print("│" + " NETWORK VALIDATION".center(58) + "│")
        console.print("└" + "─" * 58 + "┘")
        console.print()

        if result.network_metrics:
            metrics = result.network_metrics
            console.print(f"  Nodes: {metrics.node_count}")
            console.print(f"  Edges: {metrics.edge_count}")
            console.print(f"  Avg Degree: {metrics.avg_degree:.2f}")
            console.print(f"  Clustering: {metrics.clustering_coefficient:.3f}")
            if metrics.avg_path_length:
                console.print(f"  Avg Path Length: {metrics.avg_path_length:.2f}")
            else:
                console.print("  Avg Path Length: [dim]N/A (disconnected)[/dim]")
            console.print(f"  Modularity: {metrics.modularity:.3f}")
            console.print(f"  Largest Component: {metrics.largest_component_ratio:.1%}")
            console.print(f"  Degree Assortativity: {metrics.degree_assortativity:.3f}")

            is_valid, warnings = metrics.is_valid()
            console.print()
            if is_valid:
                console.print("[green]✓[/green] All metrics within expected ranges")
            else:
                console.print(
                    f"[yellow]⚠[/yellow] {len(warnings)} metric(s) outside expected range:"
                )
                for w in warnings:
                    console.print(f"  [yellow]•[/yellow] {w}")
        else:
            console.print("[dim]Metrics not computed (use without --no-metrics)[/dim]")

        # Edge type distribution
        console.print()
        console.print("[bold]Edge Types:[/bold]")
        edge_types: dict[str, int] = {}
        for edge in result.edges:
            t = edge.edge_type
            edge_types[t] = edge_types.get(t, 0) + 1
        for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            pct = count / len(result.edges) * 100 if result.edges else 0
            console.print(f"  {edge_type}: {count} ({pct:.1f}%)")

    # Save Output
    console.print()
    with console.status(f"[cyan]Saving to {output}...[/cyan]"):
        result.save_json(output)

    elapsed = time.time() - start_time

    console.print("═" * 60)
    console.print(f"[green]✓[/green] Network saved to [bold]{output}[/bold]")
    console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)
