"""CLI for Entropy."""

import random
import time
from threading import Event, Thread
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

from . import db
from .population import (
    Distributions,
    generate_network,
    parse_context,
    sample_agents,
    synthesize_personas,
    ProgressCallbacks,
)
from .search import conduct_research
from .models import Population

app = typer.Typer(
    name="entropy",
    help="Simulate how populations respond to scenarios.",
    no_args_is_help=True,
)

console = Console()


def grounding_indicator(level: str) -> str:
    """Get colored grounding indicator."""
    indicators = {
        "strong": "[green]🟢 Strong[/green]",
        "medium": "[yellow]🟡 Medium[/yellow]",
        "low": "[red]🔴 Low[/red]",
    }
    return indicators.get(level, "[dim]Unknown[/dim]")


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as Xm Ys or Xs."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    return f"{seconds:.0f}s"


def _display_research_summary(research, parsed_context) -> None:
    """Display rich research summary for confirmation."""
    console.print()
    console.print("╭" + "─" * 58 + "╮")
    console.print("│" + " RESEARCH SUMMARY".center(58) + "│")
    console.print("╰" + "─" * 58 + "╯")
    console.print()
    
    # Demographics
    console.print("[bold]Demographics discovered:[/bold]")
    demo_keys = []
    for key, value in research.demographics.items():
        if value:
            demo_keys.append(key.replace("_distribution", ""))
    console.print(f"  {', '.join(demo_keys)}")
    
    # Situation schema
    console.print()
    console.print(f"[bold]Situation attributes ({len(research.situation_schema)} per-agent):[/bold]")
    for schema in research.situation_schema[:8]:
        type_info = f"[dim]({schema.field_type})[/dim]"
        range_info = ""
        if schema.min_value is not None and schema.max_value is not None:
            range_info = f" [dim][{schema.min_value:.0f}-{schema.max_value:.0f}][/dim]"
        elif schema.options:
            opts = schema.options[:3]
            range_info = f" [dim][{', '.join(opts)}{'...' if len(schema.options) > 3 else ''}][/dim]"
        console.print(f"  • {schema.name} {type_info}{range_info}")
    
    if len(research.situation_schema) > 8:
        console.print(f"  [dim]... and {len(research.situation_schema) - 8} more[/dim]")
    
    # Grounding
    console.print()
    console.print(f"[bold]Grounding:[/bold] {grounding_indicator(research.grounding_level)}")
    console.print(f"  Sources consulted: {len(research.sources)}")
    
    console.print()


@app.command()
def create(
    context: str = typer.Argument(..., help="Natural language population description"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the population"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Create a new population from natural language context.

    Example:
        entropy create "2000 Netflix subscribers in the US" --name netflix_us
    """
    # Check if population already exists
    if db.population_exists(name):
        console.print(f"[red]Error:[/red] Population '{name}' already exists. Use a different name or delete it first.")
        raise typer.Exit(1)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    start_time = time.time()
    
    # Step 1: Parse context (fast)
    console.print()
    parsed = None
    parse_error = None
    with console.status("[cyan]Parsing context...[/cyan]"):
        try:
            parsed = parse_context(context)
        except Exception as e:
            parse_error = e
    
    if parse_error or not parsed:
        console.print(f"[red]✗[/red] Error parsing context: {parse_error}")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Parsing context... {parsed.context_entity or parsed.base_population} ({parsed.size} agents)")
    
    # Step 2: Research with live timer
    console.print()
    research_start = time.time()
    research = None
    research_done = Event()
    research_error = None
    
    def do_research():
        nonlocal research, research_error
        try:
            research = conduct_research(parsed)
        except Exception as e:
            research_error = e
        finally:
            research_done.set()
    
    # Start research in background thread
    research_thread = Thread(target=do_research, daemon=True)
    research_thread.start()
    
    # Live update timer while research runs
    with Live(console=console, refresh_per_second=1, transient=True) as live:
        while not research_done.is_set():
            elapsed = time.time() - research_start
            live.update(f"[cyan]⠋[/cyan] Researching demographics & situation... {_format_elapsed(elapsed)}")
            time.sleep(0.1)
    
    research_elapsed = time.time() - research_start
    
    if research_error:
        console.print(f"\r[red]✗[/red] Research failed: {research_error}")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Researching demographics & situation... {_format_elapsed(research_elapsed)} ({len(research.sources)} sources)")
    
    # Step 3: Display summary and confirm
    _display_research_summary(research, parsed)
    
    if not yes:
        proceed = typer.confirm("Proceed with agent generation?", default=True)
        if not proceed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)
    
    console.print()
    
    # Step 4: Build distributions
    with console.status("[cyan]Building distributions...[/cyan]"):
        distributions = Distributions(research, parsed)
    console.print(f"[green]✓[/green] Building distributions")
    
    # Step 5: Sample agents with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]Generating {parsed.size} agents...", total=parsed.size)
        
        def agent_progress(n: int):
            progress.update(task, completed=n)
        
        agents = sample_agents(distributions, parsed.size, progress_callback=agent_progress)
    
    console.print(f"[green]✓[/green] Generating {parsed.size} agents")
    
    # Step 6: Generate network
    with console.status("[cyan]Building social network...[/cyan]"):
        agents = generate_network(agents)
    console.print(f"[green]✓[/green] Building social network")
    
    # Step 7: Synthesize personas
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]Synthesizing personas...", total=parsed.size)
        
        def persona_progress(n: int):
            progress.update(task, completed=n)
        
        agents = synthesize_personas(agents, parsed, progress_callback=persona_progress)
    
    console.print(f"[green]✓[/green] Synthesizing personas")
    
    # Create population object
    population = Population(
        name=name,
        size=parsed.size,
        context_raw=context,
        context_parsed=parsed,
        research=research,
        agents=agents,
    )
    
    # Save to database
    with console.status("[cyan]Saving population...[/cyan]"):
        db.save_population(population)
    console.print(f"[green]✓[/green] Saving population")

    elapsed = time.time() - start_time

    # Display final summary
    console.print()
    console.print("═" * 60)
    console.print(f'[bold]Population "{name}" created[/bold]')
    console.print("═" * 60)
    console.print()

    # Stats table
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Key", style="dim")
    stats_table.add_column("Value")

    stats_table.add_row("Agents:", f"[bold]{population.size:,}[/bold]")
    stats_table.add_row("Grounding:", grounding_indicator(population.research.grounding_level))
    source_preview = ", ".join(s[:30] + "..." if len(s) > 30 else s for s in population.research.sources[:3]) if population.research.sources else "None"
    stats_table.add_row("Sources:", f"{len(population.research.sources)} ({source_preview})")
    stats_table.add_row("Time:", f"[cyan]{_format_elapsed(elapsed)}[/cyan]")

    console.print(stats_table)

    # Situation schema (dynamically generated)
    if population.research.situation_schema:
        console.print()
        console.print("[bold]Situation schema (generated):[/bold]")
        for schema in population.research.situation_schema[:6]:
            type_info = f"[dim]({schema.field_type})[/dim]"
            range_info = ""
            if schema.min_value is not None and schema.max_value is not None:
                range_info = f" [{schema.min_value}-{schema.max_value}]"
            elif schema.options:
                range_info = f" [{', '.join(schema.options[:3])}...]"
            console.print(f"  • {schema.name} {type_info}{range_info}")

    # Sample agent
    if population.agents:
        console.print()
        console.print("[bold]Sample agent:[/bold]")
        agent = population.agents[0]
        demo = agent.demographics
        psych = agent.psychographics

        console.print(f"  ID: [cyan]{agent.id}[/cyan]")
        console.print(f"  Age: {demo.age} | Gender: {demo.gender.upper()[0]} | Location: {demo.location.get('urban_rural', 'urban').capitalize()} {demo.location.get('state', 'US')} | Income: ${demo.income:,}")
        console.print(f"  Openness: {psych.openness:.2f} | Extraversion: {psych.extraversion:.2f}")

        # Situation summary
        if agent.situation:
            sit_parts = []
            for k, v in list(agent.situation.items())[:4]:
                if isinstance(v, float):
                    sit_parts.append(f"{k.replace('_', ' ')}: {v:.1f}")
                elif isinstance(v, list):
                    sit_parts.append(f"{k.replace('_', ' ')}: {len(v)} items")
                else:
                    sit_parts.append(f"{k.replace('_', ' ')}: {v}")
            console.print(f"  Situation: {', '.join(sit_parts)}")

        console.print(f"  Connections: [cyan]{len(agent.network.connections)}[/cyan] agents")

    console.print()
    console.print("═" * 60)
    console.print(f"[dim]Run 'entropy inspect {name}' for full details.[/dim]")


@app.command("list")
def list_populations():
    """List all populations."""
    populations = db.list_populations()

    if not populations:
        console.print("[dim]No populations found. Create one with 'entropy create'.[/dim]")
        return

    table = Table(title="Populations", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Context")
    table.add_column("Created", style="dim")

    for pop in populations:
        context_short = pop["context"][:40] + "..." if len(pop["context"]) > 40 else pop["context"]
        created = pop["created_at"].strftime("%Y-%m-%d %H:%M")
        table.add_row(pop["name"], f"{pop['size']:,}", context_short, created)

    console.print(table)


@app.command()
def inspect(
    name: str = typer.Argument(..., help="Name of the population to inspect"),
    sample: int = typer.Option(5, "--sample", "-s", help="Number of sample agents to show"),
):
    """Inspect a population in detail."""
    population = db.load_population(name, include_agents=False)

    if not population:
        console.print(f"[red]Error:[/red] Population '{name}' not found.")
        raise typer.Exit(1)

    # Header
    console.print()
    console.print(Panel(f"[bold]{population.name}[/bold]", expand=False))

    # Basic info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="dim")
    info_table.add_column("Value")

    info_table.add_row("Context:", population.context_raw)
    info_table.add_row("Size:", f"{population.size:,} agents")
    info_table.add_row("Grounding:", grounding_indicator(population.research.grounding_level))
    info_table.add_row("Created:", population.created_at.strftime("%Y-%m-%d %H:%M:%S"))

    console.print(info_table)

    # Parsed context
    console.print()
    console.print("[bold]Parsed Context:[/bold]")
    ctx = population.context_parsed
    console.print(f"  Base population: {ctx.base_population}")
    console.print(f"  Context type: {ctx.context_type}")
    console.print(f"  Entity: {ctx.context_entity or 'N/A'}")
    console.print(f"  Geography: {ctx.geography or 'N/A'}")

    # Sources
    if population.research.sources:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for source in population.research.sources[:10]:
            console.print(f"  • {source}")

    # Situation schema
    if population.research.situation_schema:
        console.print()
        console.print("[bold]Situation Schema:[/bold]")
        schema_table = Table(show_header=True, box=None)
        schema_table.add_column("Attribute")
        schema_table.add_column("Type", style="dim")
        schema_table.add_column("Description")

        for schema in population.research.situation_schema:
            schema_table.add_row(schema.name, schema.field_type, schema.description[:50] + "..." if len(schema.description) > 50 else schema.description)

        console.print(schema_table)

    # Sample agents
    if sample > 0:
        console.print()
        console.print(f"[bold]Sample Agents ({sample}):[/bold]")

        sample_agents = db.get_sample_agents(name, sample)

        for agent in sample_agents:
            console.print()
            demo = agent.demographics
            console.print(f"  [cyan]Agent {agent.id}[/cyan]")
            console.print(f"    {demo.age}yo {demo.gender} | {demo.occupation} | ${demo.income:,}")
            console.print(f"    {demo.location.get('urban_rural', '')} {demo.location.get('state', '')} | {demo.education}")

            psych = agent.psychographics
            console.print(f"    Big Five: O={psych.openness:.2f} C={psych.conscientiousness:.2f} E={psych.extraversion:.2f} A={psych.agreeableness:.2f} N={psych.neuroticism:.2f}")

            if agent.situation:
                sit_str = " | ".join(f"{k}: {v}" for k, v in list(agent.situation.items())[:3])
                console.print(f"    Situation: {sit_str}")

            console.print(f"    Connections: {len(agent.network.connections)} | Influence: {agent.network.influence_score:.2f}")


@app.command()
def delete(
    name: str = typer.Argument(..., help="Name of the population to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a population."""
    if not db.population_exists(name):
        console.print(f"[red]Error:[/red] Population '{name}' not found.")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete population '{name}'?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    db.delete_population(name)
    console.print(f"[green]✓[/green] Population '{name}' deleted.")


# Phase 2/3 placeholder commands


@app.command()
def scenario(
    description: str = typer.Argument(..., help="Scenario description"),
    population: str = typer.Option(..., "--population", "-p", help="Population name"),
    name: str = typer.Option(..., "--name", "-n", help="Scenario name"),
):
    """[Phase 2] Create a scenario for a population."""
    console.print("[yellow]Phase 2 not yet implemented.[/yellow]")
    console.print(f"Would create scenario '{name}' for population '{population}':")
    console.print(f"  {description}")


@app.command()
def simulate(
    population: str = typer.Argument(..., help="Population name"),
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name"),
    mode: str = typer.Option("single", "--mode", "-m", help="Simulation mode: single or continuous"),
):
    """[Phase 3] Run a simulation."""
    console.print("[yellow]Phase 3 not yet implemented.[/yellow]")
    console.print(f"Would simulate scenario '{scenario}' on population '{population}' in {mode} mode.")


@app.command()
def results(
    population: str = typer.Argument(..., help="Population name"),
    by: Optional[str] = typer.Option(None, "--by", help="Group results by attribute"),
    timeline: bool = typer.Option(False, "--timeline", help="Show timeline"),
):
    """[Phase 3] View simulation results."""
    console.print("[yellow]Phase 3 not yet implemented.[/yellow]")
    console.print(f"Would show results for population '{population}'.")


if __name__ == "__main__":
    app()

