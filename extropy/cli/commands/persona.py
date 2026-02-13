"""Persona command for generating persona rendering configuration."""

import json
import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ...core.models import PopulationSpec
from ..app import app, console
from ..utils import (
    format_elapsed,
)


@app.command("persona")
def persona_command(
    spec_file: Path = typer.Argument(..., help="Population spec YAML file"),
    agents_file: Path = typer.Option(
        None, "--agents", "-a", help="Sampled agents JSON file (for population stats)"
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for persona config (default: adds to spec)",
    ),
    preview: bool = typer.Option(
        True, "--preview/--no-preview", help="Show a sample persona before saving"
    ),
    agent_index: int = typer.Option(
        0, "--agent", help="Which agent to use for preview (default: first)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Preview existing persona config without regenerating",
    ),
):
    """
    Generate persona rendering configuration for a population.

    Creates a PersonaConfig that defines how to render agent attributes
    into first-person personas. The config is generated once via LLM,
    then applied to all agents via templates (no per-agent LLM calls).

    Pipeline:
        Step 1: Classify attributes and create groups
        Step 2: Generate boolean phrasings
        Step 3: Generate categorical phrasings
        Step 4: Generate relative phrasings
        Step 5: Generate concrete phrasings

    Use --show to preview an existing persona config without regenerating.

    EXIT CODES:
        0 = Success
        1 = Validation error
        2 = File not found
        3 = Generation error

    EXAMPLES:
        extropy persona population.yaml --agents agents.json
        extropy persona population.yaml -a agents.json -o persona_config.yaml
        extropy persona population.yaml -a agents.json --agent 42 -y
        extropy persona population.yaml -a agents.json --show  # preview existing
    """
    from ...population.persona import (
        generate_persona_config,
        preview_persona,
        PersonaConfigError,
    )

    start_time = time.time()
    console.print()

    # Load Spec
    if not spec_file.exists():
        console.print(f"[red]✗[/red] Spec file not found: {spec_file}")
        raise typer.Exit(2)

    with console.status("[cyan]Loading population spec...[/cyan]"):
        try:
            spec = PopulationSpec.from_yaml(spec_file)
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load spec: {e}")
            raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Loaded: [bold]{spec.meta.description}[/bold] ({len(spec.attributes)} attributes)"
    )

    # Load Agents (optional but recommended)
    agents = None
    if agents_file:
        if not agents_file.exists():
            console.print(f"[red]✗[/red] Agents file not found: {agents_file}")
            raise typer.Exit(2)

        with console.status("[cyan]Loading agents...[/cyan]"):
            try:
                with open(agents_file, "r") as f:
                    agents_data = json.load(f)

                # Handle both raw list and {meta, agents} format
                if isinstance(agents_data, dict) and "agents" in agents_data:
                    agents = agents_data["agents"]
                elif isinstance(agents_data, list):
                    agents = agents_data
                else:
                    raise ValueError("Unexpected agents file format")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load agents: {e}")
                raise typer.Exit(1)

        console.print(f"[green]✓[/green] Loaded {len(agents)} agents")
    else:
        console.print(
            "[yellow]⚠[/yellow] No agents file - population stats will use defaults"
        )

    # Handle --show mode: preview existing config without regenerating
    if show:
        from ...population.persona import PersonaConfig

        # Find existing config
        if output and output.exists():
            config_path = output
        else:
            config_path = spec_file.with_suffix(".persona.yaml")

        if not config_path.exists():
            console.print(f"[red]✗[/red] No persona config found at {config_path}")
            console.print("[dim]Run without --show to generate one.[/dim]")
            raise typer.Exit(2)

        try:
            config = PersonaConfig.from_file(str(config_path))
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load persona config: {e}")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Loaded persona config from {config_path}")
        console.print()

        if not agents:
            console.print("[red]✗[/red] Need --agents to preview personas")
            raise typer.Exit(1)

        if agent_index >= len(agents):
            agent_index = 0
        sample_agent = agents[agent_index]
        agent_id = sample_agent.get("_id", str(agent_index))

        console.print(f"[bold]Persona for Agent {agent_id}:[/bold]")
        console.print()

        persona_text = preview_persona(sample_agent, config, max_width=78)

        for line in persona_text.split("\n"):
            if line.startswith("##"):
                console.print(f"[bold cyan]{line}[/bold cyan]")
            elif line.strip():
                console.print(line)
            else:
                console.print()

        raise typer.Exit(0)

    # Generate Config with spinner
    console.print()
    gen_start = time.time()
    config = None
    gen_error = None
    gen_done = Event()
    current_step = ["1", "Starting..."]

    def on_progress(step: str, status: str):
        current_step[0] = step
        current_step[1] = status

    def do_generation():
        nonlocal config, gen_error
        try:
            config = generate_persona_config(
                spec=spec,
                agents=agents,
                log=True,
                on_progress=on_progress,
            )
        except PersonaConfigError as e:
            gen_error = e
        except Exception as e:
            gen_error = e
        finally:
            gen_done.set()

    gen_thread = Thread(target=do_generation, daemon=True)
    gen_thread.start()

    spinner = Spinner("dots", text="Starting...", style="cyan")
    with Live(spinner, console=console, refresh_per_second=12.5, transient=True):
        while not gen_done.is_set():
            elapsed = time.time() - gen_start
            step, status = current_step
            spinner.update(text=f"Step {step}: {status} ({format_elapsed(elapsed)})")
            time.sleep(0.1)

    gen_elapsed = time.time() - gen_start

    if gen_error:
        console.print(f"[red]✗[/red] Failed to generate persona config: {gen_error}")
        raise typer.Exit(3)

    console.print(
        f"[green]✓[/green] Generated persona configuration ({format_elapsed(gen_elapsed)})"
    )

    # Show summary
    console.print()
    console.print("┌" + "─" * 58 + "┐")
    console.print("│" + " PERSONA CONFIGURATION".center(58) + "│")
    console.print("└" + "─" * 58 + "┘")
    console.print()

    # Treatment summary
    concrete_count = sum(
        1 for t in config.treatments if t.treatment.value == "concrete"
    )
    relative_count = sum(
        1 for t in config.treatments if t.treatment.value == "relative"
    )
    console.print(f"  Concrete (keep values): {concrete_count} attributes")
    console.print(f"  Relative (use positioning): {relative_count} attributes")
    console.print()

    # Groups
    console.print("  [bold]Groupings:[/bold]")
    for group in config.groups:
        console.print(f"    • {group.label} ({len(group.attributes)} attributes)")
    console.print()

    # Phrasings summary
    console.print("  [bold]Phrasings:[/bold]")
    console.print(f"    • {len(config.phrasings.boolean)} boolean")
    console.print(f"    • {len(config.phrasings.categorical)} categorical")
    console.print(f"    • {len(config.phrasings.relative)} relative")
    console.print(f"    • {len(config.phrasings.concrete)} concrete")
    console.print()

    # Preview
    if preview and agents:
        if agent_index >= len(agents):
            agent_index = 0
        sample_agent = agents[agent_index]
        agent_id = sample_agent.get("_id", str(agent_index))

        console.print(f"  [bold]Sample Persona (Agent {agent_id}):[/bold]")
        console.print()

        persona_text = preview_persona(sample_agent, config, max_width=74)

        # Display persona with indentation
        for line in persona_text.split("\n"):
            if line.startswith("##"):
                console.print(f"    [bold cyan]{line}[/bold cyan]")
            elif line.strip():
                console.print(f"    {line}")
            else:
                console.print()

        console.print()

    # Confirmation
    if not yes:
        choice = (
            typer.prompt(
                "[Y] Save config  [r] Regenerate  [n] Cancel",
                default="Y",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

        if choice == "r":
            # Regenerate
            console.print()
            gen_start = time.time()
            gen_done.clear()

            gen_thread = Thread(target=do_generation, daemon=True)
            gen_thread.start()

            spinner = Spinner("dots", text="Regenerating...", style="cyan")
            with Live(
                spinner, console=console, refresh_per_second=12.5, transient=True
            ):
                while not gen_done.is_set():
                    elapsed = time.time() - gen_start
                    step, status = current_step
                    spinner.update(
                        text=f"Step {step}: {status} ({format_elapsed(elapsed)})"
                    )
                    time.sleep(0.1)

            if gen_error:
                console.print(f"[red]✗[/red] Regeneration failed: {gen_error}")
                raise typer.Exit(3)

            console.print(
                f"[green]✓[/green] Regenerated ({format_elapsed(time.time() - gen_start)})"
            )

    # Save
    if output:
        config_path = output
    else:
        config_path = spec_file.with_suffix(".persona.yaml")

    try:
        config.to_file(str(config_path))
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to save: {e}")
        raise typer.Exit(1)

    elapsed = time.time() - start_time

    console.print()
    console.print("═" * 60)
    console.print(
        f"[green]✓[/green] Persona config saved to [bold]{config_path}[/bold]"
    )
    console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
    console.print("═" * 60)
