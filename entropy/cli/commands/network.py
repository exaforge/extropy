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

    Example:
        entropy network agents.json -o network.json
        entropy network agents.json -o network.json --avg-degree 25 --validate
        entropy network agents.json -o network.json --seed 42
    """
    from ...population.network import (
        generate_network,
        generate_network_with_metrics,
        load_agents_json,
        NetworkConfig,
    )

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

    # Generate Network
    config = NetworkConfig(avg_degree=avg_degree, rewire_prob=rewire_prob, seed=seed)

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
