"""CLI display formatting for simulation results.

Provides rich console output for displaying simulation results
in a user-friendly format.
"""

from typing import Any

from rich.console import Console
from rich.table import Table

from ..core.models import (
    SimulationSummary,
    SegmentAggregate,
    TimelinePoint,
    AgentFinalState,
)
from .reader import ResultsReader


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.1f}%"


def format_sentiment(value: float | None) -> str:
    """Format sentiment value."""
    if value is None:
        return "N/A"
    return f"{value:+.2f}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def make_bar(value: float, width: int = 20) -> str:
    """Create a simple progress bar."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def display_summary(console: Console, reader: ResultsReader) -> None:
    """Display simulation summary.

    Args:
        console: Rich console
        reader: Results reader
    """
    summary = reader.get_summary()
    meta = reader.get_meta()

    if not summary:
        console.print("[yellow]No summary available[/yellow]")
        return

    console.print()
    console.print("═" * 60)
    console.print(f"[bold]SIMULATION RESULTS: {meta.scenario_name}[/bold]")
    console.print("═" * 60)
    console.print()

    # Basic stats
    console.print(f"Population: {summary.population_size:,} agents")
    console.print(f"Duration: {summary.total_timesteps} timesteps")
    console.print(f"Model: {summary.model_used}")
    console.print()

    # Exposure stats
    console.print("[bold]EXPOSURE[/bold]")
    console.print("─" * 40)
    console.print(f"Final exposure rate: {format_percentage(summary.final_exposure_rate)}")
    console.print(f"Total exposures: {summary.total_exposures:,}")
    console.print(f"Reasoning calls: {summary.total_reasoning_calls:,}")
    console.print()

    # Outcome distributions
    if summary.outcome_distributions:
        console.print("[bold]OUTCOMES[/bold]")
        console.print("─" * 40)

        for outcome_name, distribution in summary.outcome_distributions.items():
            console.print(f"[bold]{outcome_name}:[/bold]")

            if isinstance(distribution, dict):
                # Check if it's a numeric distribution (has mean/std)
                if "mean" in distribution:
                    console.print(f"  mean: {distribution['mean']:.2f}")
                    if "std" in distribution:
                        console.print(f"  std: {distribution['std']:.2f}")
                else:
                    # Categorical distribution
                    for option, pct in sorted(distribution.items(), key=lambda x: -x[1]):
                        bar = make_bar(pct)
                        console.print(f"  {option:25s} {format_percentage(pct):>6s}  {bar}")

            console.print()


def display_segment_breakdown(
    console: Console,
    reader: ResultsReader,
    attribute: str,
) -> None:
    """Display breakdown by a segment attribute.

    Args:
        console: Rich console
        reader: Results reader
        attribute: Attribute to segment by
    """
    segments = reader.compute_segment(attribute)

    if not segments:
        console.print(f"[yellow]No data for attribute: {attribute}[/yellow]")
        return

    console.print()
    console.print(f"[bold]BREAKDOWN BY: {attribute}[/bold]")
    console.print("─" * 60)
    console.print()

    for segment in segments:
        console.print(f"[bold]{segment.segment_value}[/bold] ({segment.agent_count} agents)")

        # Show position distribution
        if segment.position_distribution:
            positions = sorted(
                segment.position_distribution.items(),
                key=lambda x: -x[1]
            )
            pos_str = ", ".join(
                f"{k}: {format_percentage(v)}"
                for k, v in positions[:3]
            )
            console.print(f"  positions: {pos_str}")

        # Show sentiment
        if segment.average_sentiment is not None:
            console.print(f"  sentiment: {format_sentiment(segment.average_sentiment)}")

        console.print()


def display_timeline(
    console: Console,
    reader: ResultsReader,
    step: int = 10,
) -> None:
    """Display timeline of simulation.

    Args:
        console: Rich console
        reader: Results reader
        step: Show every N timesteps
    """
    timeline = reader.get_timeline()

    if not timeline:
        console.print("[yellow]No timeline data available[/yellow]")
        return

    console.print()
    console.print("[bold]TIMELINE[/bold]")
    console.print("─" * 60)

    # Create table
    table = Table()
    table.add_column("Timestep", justify="right")
    table.add_column("Exposure", justify="right")
    table.add_column("Sentiment", justify="right")
    table.add_column("Reasoned", justify="right")
    table.add_column("Shares", justify="right")

    for point in timeline[::step]:
        table.add_row(
            str(point.timestep),
            format_percentage(point.exposure_rate),
            format_sentiment(point.average_sentiment),
            str(point.agents_reasoned),
            str(point.cumulative_shares),
        )

    console.print(table)


def display_agent(
    console: Console,
    reader: ResultsReader,
    agent_id: str,
) -> None:
    """Display details for a single agent.

    Args:
        console: Rich console
        reader: Results reader
        agent_id: Agent ID to display
    """
    state = reader.get_agent_state(agent_id)

    if not state:
        console.print(f"[red]Agent not found: {agent_id}[/red]")
        return

    console.print()
    console.print(f"[bold]AGENT: {agent_id}[/bold]")
    console.print("─" * 60)
    console.print()

    # Attributes
    console.print("[bold]Attributes:[/bold]")
    for key, value in list(state.attributes.items())[:10]:
        console.print(f"  {key}: {value}")
    if len(state.attributes) > 10:
        console.print(f"  [dim]... and {len(state.attributes) - 10} more[/dim]")

    console.print()

    # State
    console.print("[bold]Final State:[/bold]")
    console.print(f"  Aware: {'Yes' if state.aware else 'No'}")
    console.print(f"  Exposures: {state.exposure_count}")
    console.print(f"  Position: {state.position or 'N/A'}")
    console.print(f"  Sentiment: {format_sentiment(state.sentiment)}")
    console.print(f"  Will share: {'Yes' if state.will_share else 'No'}")
    console.print(f"  Reasoning count: {state.reasoning_count}")

    # Reasoning
    if state.raw_reasoning:
        console.print()
        console.print("[bold]Reasoning:[/bold]")
        console.print(f"  [italic]{state.raw_reasoning}[/italic]")

    # Other outcomes
    if state.outcomes:
        console.print()
        console.print("[bold]Outcomes:[/bold]")
        for key, value in state.outcomes.items():
            if key not in ("reasoning", "will_share"):
                console.print(f"  {key}: {value}")


def display_quick_stats(
    console: Console,
    reader: ResultsReader,
) -> None:
    """Display quick statistics.

    Args:
        console: Rich console
        reader: Results reader
    """
    summary = reader.get_position_summary()

    console.print()
    console.print(f"Total agents: {summary['total']}")
    console.print(f"Aware: {summary['aware']} ({format_percentage(summary['aware'] / summary['total'] if summary['total'] > 0 else 0)})")
    console.print()

    if summary["positions"]:
        console.print("[bold]Position Distribution:[/bold]")
        for position, count in sorted(summary["positions"].items(), key=lambda x: -x[1]):
            pct = count / summary["total"] if summary["total"] > 0 else 0
            bar = make_bar(pct)
            console.print(f"  {position:25s} {count:5d}  {format_percentage(pct):>6s}  {bar}")


def display_help(console: Console) -> None:
    """Display help for results commands.

    Args:
        console: Rich console
    """
    console.print()
    console.print("[bold]Available commands:[/bold]")
    console.print()
    console.print("  entropy results <dir>                     Summary view")
    console.print("  entropy results <dir> --segment <attr>    Breakdown by attribute")
    console.print("  entropy results <dir> --timeline          Timeline view")
    console.print("  entropy results <dir> --agent <id>        Single agent details")
    console.print()
