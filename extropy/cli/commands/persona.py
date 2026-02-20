"""Persona command for generating persona rendering configuration."""

import time
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ...core.models import PopulationSpec
from ...core.models.scenario import ScenarioSpec
from ..app import app, console, is_agent_mode, get_study_path
from ..study import StudyContext, detect_study_folder, resolve_scenario
from ..utils import (
    format_elapsed,
    Output,
    ExitCode,
    format_validation_for_json,
    get_next_invalid_artifact_path,
    save_invalid_json_artifact,
)


@app.command("persona")
def persona_command(
    scenario: str = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (auto-selects if only one exists)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (default: scenario/{name}/persona.vN.yaml)",
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
        help="Preview existing persona config without regenerating",
    ),
):
    """
    Generate persona rendering configuration for a scenario.

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

    Examples:
        # Generate for a scenario (auto-versions)
        extropy persona -s ai-adoption

        # Pin scenario version
        extropy persona -s ai-adoption@v1

        # Preview existing config
        extropy persona -s ai-adoption --show
    """
    from ...population.persona import (
        generate_persona_config,
        PersonaConfigError,
        validate_persona_config,
    )
    from ...core.cost.tracker import CostTracker

    CostTracker.get().set_context(command="persona")

    start_time = time.time()
    agent_mode = is_agent_mode()
    out = Output(console=console, json_mode=agent_mode)

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

    # Resolve scenario
    try:
        scenario_name, scenario_version = resolve_scenario(study_ctx, scenario)
    except ValueError as e:
        out.error(str(e), exit_code=ExitCode.FILE_NOT_FOUND)
        raise typer.Exit(out.finish())

    # Load scenario spec
    try:
        scenario_path = study_ctx.get_scenario_path(scenario_name, scenario_version)
    except FileNotFoundError:
        out.error(f"Scenario not found: {scenario_name}")
        raise typer.Exit(1)

    try:
        scenario_spec = ScenarioSpec.from_yaml(scenario_path)
    except Exception as e:
        out.error(f"Failed to load scenario: {e}")
        raise typer.Exit(1)

    # Load base population spec
    try:
        pop_name, pop_version = scenario_spec.meta.get_population_ref()
    except ValueError as e:
        out.error(str(e))
        raise typer.Exit(1)

    if pop_version is not None:
        pop_path = study_ctx.get_population_path(pop_name, pop_version)
    else:
        # Legacy scenario — population_spec is a file path
        pop_path = Path(pop_name)
        if not pop_path.is_absolute():
            pop_path = study_ctx.root / pop_path

    try:
        pop_spec = PopulationSpec.from_yaml(pop_path)
    except Exception as e:
        out.error(f"Failed to load population spec: {e}")
        raise typer.Exit(1)

    # Merge attributes: base population + scenario extended attributes
    merged_attributes = list(pop_spec.attributes)
    if scenario_spec.extended_attributes:
        merged_attributes.extend(scenario_spec.extended_attributes)

    # Create a merged spec for persona generation
    merged_spec = PopulationSpec(
        meta=pop_spec.meta,
        grounding=pop_spec.grounding,
        attributes=merged_attributes,
        sampling_order=pop_spec.sampling_order,
    )

    if not agent_mode:
        console.print(
            f"[green]✓[/green] Loaded scenario: [bold]{scenario_name}[/bold] "
            f"({len(merged_attributes)} attributes: {len(pop_spec.attributes)} base + "
            f"{len(scenario_spec.extended_attributes or [])} extended)"
        )

    # Resolve output path
    scenario_dir = study_ctx.get_scenario_dir(scenario_name)
    if output:
        persona_path = output
    else:
        next_ver = study_ctx.get_next_persona_version(scenario_name)
        persona_path = scenario_dir / f"persona.v{next_ver}.yaml"

    # Handle --show mode: preview existing config
    if show:
        _handle_show_mode(
            study_ctx, scenario_name, persona_path, merged_spec, agent_index, out
        )
        return

    if not agent_mode:
        console.print(f"[dim]Output: {persona_path.name}[/dim]")
        console.print()

    # Generate Config with spinner
    gen_start = time.time()
    config = None
    gen_error = None
    gen_done = Event()
    current_step = ["1", "Starting..."]

    def on_progress(step: str, status: str):
        current_step[0] = step
        current_step[1] = status

    # Try to load agents for population_stats (available if re-running after sample)
    agents_for_stats = None
    if study_ctx.db_path.exists():
        try:
            from ...storage import open_study_db

            with open_study_db(study_ctx.db_path) as db:
                agents_for_stats = db.get_agents_by_scenario(scenario_name)
                if not agents_for_stats:
                    agents_for_stats = None
        except Exception:
            pass  # No agents yet — stats will be computed at simulation time

    def do_generation():
        nonlocal config, gen_error
        try:
            config = generate_persona_config(
                spec=merged_spec,
                agents=agents_for_stats,
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
        invalid_path = save_invalid_json_artifact(
            {
                "stage": "persona",
                "scenario": scenario_name,
                "error": str(gen_error),
            },
            persona_path.with_suffix(".json"),
            stem=persona_path.stem,
            extension=".json",
        )
        out.error(
            f"Failed to generate persona config: {gen_error}. Saved: {invalid_path}",
            exit_code=ExitCode.VALIDATION_ERROR,
        )
        raise typer.Exit(out.finish())

    validation_result = validate_persona_config(merged_spec, config)
    out.set_data("validation", format_validation_for_json(validation_result))
    if validation_result.errors:
        invalid_path = get_next_invalid_artifact_path(persona_path)
        config.to_yaml(invalid_path)
        out.error(
            f"Persona config validation failed ({len(validation_result.errors)} errors). Saved: {invalid_path}",
            exit_code=ExitCode.VALIDATION_ERROR,
        )
        if not agent_mode:
            for err in validation_result.errors[:10]:
                console.print(f"[red]✗[/red] {err.location}: {err.message}")
            if len(validation_result.errors) > 10:
                console.print(
                    f"[dim]... and {len(validation_result.errors) - 10} more[/dim]"
                )
        raise typer.Exit(out.finish())
    if validation_result.warnings and not agent_mode:
        console.print(
            f"[yellow]⚠[/yellow] Persona validation warnings: {len(validation_result.warnings)}"
        )
        for warn in validation_result.warnings[:5]:
            console.print(f"  [dim]- {warn.location}: {warn.message}[/dim]")

    if not agent_mode:
        console.print(
            f"[green]✓[/green] Generated persona configuration ({format_elapsed(gen_elapsed)})"
        )
        _display_config_summary(config)

    # Confirmation
    if not agent_mode and not yes:
        choice = (
            typer.prompt(
                "[Y] Save config  [n] Cancel",
                default="Y",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "n":
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(0)

    # Save
    try:
        config.to_yaml(persona_path)
    except Exception as e:
        out.error(f"Failed to save: {e}")
        raise typer.Exit(1)

    elapsed = time.time() - start_time

    if agent_mode:
        out.success(
            "Persona config saved",
            output=str(persona_path),
            scenario=scenario_name,
            attribute_count=len(merged_attributes),
            elapsed_seconds=elapsed,
        )
        raise typer.Exit(out.finish())
    else:
        console.print()
        console.print("═" * 60)
        console.print(
            f"[green]✓[/green] Persona config saved to [bold]{persona_path}[/bold]"
        )
        console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
        console.print("═" * 60)


def _handle_show_mode(
    study_ctx: StudyContext,
    scenario_name: str,
    persona_path: Path,
    merged_spec: PopulationSpec,
    agent_index: int,
    out: Output,
) -> None:
    """Handle --show mode: preview existing persona config."""
    from ...population.persona import PersonaConfig

    # Find existing config
    try:
        config_path = study_ctx.get_persona_path(scenario_name)
    except FileNotFoundError:
        if persona_path.exists():
            config_path = persona_path
        else:
            out.error(f"No persona config found for scenario: {scenario_name}")
            console.print("[dim]Run without --show to generate one.[/dim]")
            raise typer.Exit(2)

    try:
        config = PersonaConfig.from_yaml(config_path)
    except Exception as e:
        out.error(f"Failed to load persona config: {e}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Loaded persona config from {config_path}")
    console.print()

    _display_config_summary(config)

    console.print(
        "[dim]Note: To preview with actual agents, use 'extropy persona -s {scenario} --show' "
        "after sampling.[/dim]"
    )
    raise typer.Exit(0)


def _display_config_summary(config) -> None:
    """Display persona config summary."""
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
        console.print(f"    - {group.label} ({len(group.attributes)} attributes)")
    console.print()

    # Phrasings summary
    console.print("  [bold]Phrasings:[/bold]")
    console.print(f"    - {len(config.phrasings.boolean)} boolean")
    console.print(f"    - {len(config.phrasings.categorical)} categorical")
    console.print(f"    - {len(config.phrasings.relative)} relative")
    console.print(f"    - {len(config.phrasings.concrete)} concrete")
    console.print()
