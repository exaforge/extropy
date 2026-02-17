"""Network command for generating social networks from agents."""

import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

import typer
from rich.live import Live
from rich.spinner import Spinner

from ..app import app, console, is_agent_mode, get_study_path
from ..study import StudyContext, detect_study_folder, parse_version_ref
from ..utils import format_elapsed, Output, ExitCode


@app.command("network")
def network_command(
    scenario: str = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario name (auto-selects if only one exists)",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Optional JSON export path (non-canonical)"
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
    generate_config: bool = typer.Option(
        False,
        "--generate-config",
        help="Generate network config via LLM from population spec",
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
    quality_profile: str = typer.Option(
        "balanced",
        "--quality-profile",
        help="Quality profile: fast | balanced | strict",
    ),
    candidate_mode: str = typer.Option(
        "blocked",
        "--candidate-mode",
        help="Similarity candidate mode: exact | blocked",
    ),
    candidate_pool_multiplier: float = typer.Option(
        12.0,
        "--candidate-pool-multiplier",
        help="Blocked mode candidate pool size as multiple of avg_degree",
    ),
    block_attr: list[str] | None = typer.Option(
        None,
        "--block-attr",
        help="Blocking attribute (repeatable). If omitted, auto-selects top attributes",
    ),
    similarity_workers: int = typer.Option(
        1,
        "--similarity-workers",
        min=1,
        help="Worker processes for similarity computation",
    ),
    similarity_chunk_size: int = typer.Option(
        64,
        "--similarity-chunk-size",
        min=8,
        help="Row chunk size for similarity worker tasks",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="DB path for similarity checkpointing (must be study.db)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume similarity and calibration checkpoints from study.db",
    ),
    resume_similarity: bool = typer.Option(
        False,
        "--resume-similarity",
        hidden=True,
        help="Compatibility alias for --resume",
    ),
    resume_checkpoint: bool = typer.Option(
        False,
        "--resume-checkpoint",
        hidden=True,
        help="Compatibility alias for --resume",
    ),
    resume_calibration: bool = typer.Option(
        False,
        "--resume-calibration",
        hidden=True,
        help="Compatibility alias for --resume",
    ),
    checkpoint_every: int = typer.Option(
        250,
        "--checkpoint-every",
        min=1,
        help="Write checkpoint every N processed similarity rows",
    ),
    resource_mode: str = typer.Option(
        "auto",
        "--resource-mode",
        help="Resource tuning mode: auto | manual",
    ),
    safe_auto_workers: bool = typer.Option(
        True,
        "--safe-auto-workers/--unsafe-auto-workers",
        help="When auto mode is enabled, keep worker count conservative for laptops/VMs",
    ),
    max_memory_gb: float | None = typer.Option(
        None,
        "--max-memory-gb",
        min=0.5,
        help="Optional memory budget cap for auto resource tuning",
    ),
    topology_gate: str | None = typer.Option(
        None,
        "--topology-gate",
        hidden=True,
        help="Advanced: strict | warn",
    ),
    max_calibration_minutes: int | None = typer.Option(
        None,
        "--max-calibration-minutes",
        hidden=True,
        min=1,
        help="Advanced: max calibration runtime budget in minutes",
    ),
    calibration_restarts: int | None = typer.Option(
        None,
        "--calibration-restarts",
        hidden=True,
        min=1,
        help="Advanced: number of calibration restarts",
    ),
    allow_quarantine: bool | None = typer.Option(
        None,
        "--allow-quarantine/--no-allow-quarantine",
        hidden=True,
        help="Advanced: store rejected strict-run artifact under quarantine network id",
    ),
    quarantine_suffix: str = typer.Option(
        "rejected",
        "--quarantine-suffix",
        hidden=True,
        help="Advanced: suffix for quarantined network IDs",
    ),
    auto_save_generated_config: bool = typer.Option(
        True,
        "--auto-save-generated-config/--no-auto-save-generated-config",
        hidden=True,
        help="Auto-save generated config when using --generate-config",
    ),
):
    """
    Generate a social network from sampled agents.

    Creates edges between agents based on attribute similarity, with
    degree correction for high-influence agents and Watts-Strogatz
    rewiring for small-world properties.

    Network config resolution:
      1. --network-config file → load from YAML
      2. Auto-detect scenario/name/*.network-config.yaml if it exists
      3. --generate-config → generate config via LLM from scenario
      4. None of the above → empty config (flat network, no similarity structure)

    Prerequisites:
        - Agents must be sampled (run 'extropy sample' first)

    Examples:
        extropy network -s ai-adoption
        extropy network -s ai-adoption --avg-degree 15 --seed 42
        extropy network -s ai-adoption --generate-config
    """
    from ...population.network import (
        generate_network,
        generate_network_with_metrics,
        NetworkConfig,
        generate_network_config,
    )
    from ...core.models import PopulationSpec
    from ...core.models.scenario import ScenarioSpec
    from ...storage import open_study_db
    from ...utils import ResourceGovernor

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
    scenario_name, scenario_version = _resolve_scenario(study_ctx, scenario, out)

    # Pre-flight: Check agents exist for this scenario
    with open_study_db(study_db) as db:
        agent_count = db.get_agent_count_by_scenario(scenario_name)
    if agent_count == 0:
        out.error(
            f"No agents found for scenario: {scenario_name}. "
            "Run 'extropy sample -s {scenario_name} -n COUNT' first.",
            exit_code=ExitCode.FILE_NOT_FOUND,
        )
        raise typer.Exit(out.finish())

    # Load scenario spec for network config generation
    try:
        scenario_path = study_ctx.get_scenario_path(scenario_name, scenario_version)
        scenario_spec = ScenarioSpec.from_yaml(scenario_path)
    except FileNotFoundError:
        out.error(f"Scenario not found: {scenario_name}")
        raise typer.Exit(1)
    except Exception as e:
        out.error(f"Failed to load scenario: {e}")
        raise typer.Exit(1)

    # Load base population spec for config generation (only if needed)
    merged_spec = None
    if generate_config:
        base_pop_ref = scenario_spec.meta.base_population
        if base_pop_ref:
            pop_name, pop_version = _parse_base_population_ref(base_pop_ref)
            pop_path = study_ctx.get_population_path(pop_name, pop_version)
            try:
                pop_spec = PopulationSpec.from_yaml(pop_path)
            except Exception as e:
                out.error(f"Failed to load population spec: {e}")
                raise typer.Exit(1)

            # Merge attributes for config generation
            merged_attributes = list(pop_spec.attributes)
            if scenario_spec.extended_attributes:
                merged_attributes.extend(scenario_spec.extended_attributes)
            merged_spec = PopulationSpec(
                meta=pop_spec.meta,
                grounding=pop_spec.grounding,
                attributes=merged_attributes,
                sampling_order=pop_spec.sampling_order,
            )

    # Checkpoint handling
    resume_requested = (
        resume or resume_similarity or resume_calibration or resume_checkpoint
    )
    if (
        checkpoint is not None
        and checkpoint.expanduser().resolve() != study_db.expanduser().resolve()
    ):
        out.error("--checkpoint must point to the same canonical file as study.db")
        raise typer.Exit(1)
    checkpoint_db = study_db if (resume_requested or checkpoint is not None) else None

    # Load agents
    if not agent_mode:
        with console.status("[cyan]Loading agents...[/cyan]"):
            with open_study_db(study_db) as db:
                agents = db.get_agents_by_scenario(scenario_name)
    else:
        with open_study_db(study_db) as db:
            agents = db.get_agents_by_scenario(scenario_name)

    out.success(
        f"Loaded {len(agents)} agents from [bold]{study_db}[/bold] "
        f"(scenario_id={scenario_name})",
        scenario=scenario_name,
        agent_count=len(agents),
    )

    # =========================================================================
    # Resolve network config
    # =========================================================================

    config = None
    generated_from_population = False
    scenario_dir = study_ctx.get_scenario_dir(scenario_name)

    # 1. Explicit --network-config file
    if network_config:
        if not network_config.exists():
            out.error(f"Network config not found: {network_config}")
            raise typer.Exit(1)
        if not agent_mode:
            with console.status("[cyan]Loading network config...[/cyan]"):
                config = NetworkConfig.from_yaml(network_config)
        else:
            config = NetworkConfig.from_yaml(network_config)
        out.success(
            f"Loaded network config from [bold]{network_config}[/bold] "
            f"({len(config.attribute_weights)} weights, {len(config.edge_type_rules)} rules)"
        )

    # 2. Auto-detect scenario/name/*.network-config.yaml
    if config is None:
        auto_config_paths = list(scenario_dir.glob("*.network-config.yaml"))
        if auto_config_paths:
            auto_config_path = sorted(auto_config_paths)[-1]  # latest
            if not agent_mode:
                with console.status(
                    "[cyan]Loading auto-detected network config...[/cyan]"
                ):
                    config = NetworkConfig.from_yaml(auto_config_path)
            else:
                config = NetworkConfig.from_yaml(auto_config_path)
            out.success(
                f"Auto-detected config: [bold]{auto_config_path}[/bold] "
                f"({len(config.attribute_weights)} weights, {len(config.edge_type_rules)} rules)"
            )

    # 3. --generate-config → LLM generation
    if config is None and generate_config:
        if merged_spec is None:
            out.error(
                "Cannot generate config: scenario has no base_population reference"
            )
            raise typer.Exit(1)

        if not agent_mode:
            out.blank()

        # Sample a few agents for context
        agents_sample = agents[:5] if len(agents) >= 5 else agents

        if not agent_mode:
            with console.status("[cyan]Generating network config via LLM...[/cyan]"):
                config = generate_network_config(merged_spec, agents_sample)
        else:
            config = generate_network_config(merged_spec, agents_sample)
        generated_from_population = True

        out.success(
            f"Generated network config: "
            f"{len(config.attribute_weights)} weights, "
            f"{len(config.edge_type_rules)} edge rules, "
            f"{len(config.influence_factors)} influence factors"
        )

    # 4. No config source → empty config (flat network)
    if config is None:
        config = NetworkConfig()
        out.warning(
            "No network config — generating flat network "
            "(use -c or --generate-config for meaningful social structure)"
        )

    # Apply CLI overrides
    base_updates = {
        "avg_degree": avg_degree,
        "rewire_prob": rewire_prob,
        "seed": seed if seed is not None else config.seed,
        "candidate_mode": candidate_mode,
        "candidate_pool_multiplier": candidate_pool_multiplier,
        "blocking_attributes": block_attr or config.blocking_attributes,
        "similarity_workers": similarity_workers,
        "similarity_chunk_size": similarity_chunk_size,
        "checkpoint_every_rows": checkpoint_every,
        "quality_profile": quality_profile,
        "auto_save_generated_config": auto_save_generated_config,
        "quarantine_suffix": quarantine_suffix,
    }
    config = config.model_copy(update=base_updates).apply_quality_profile_defaults(
        force=True
    )

    advanced_updates = {}
    if topology_gate is not None:
        advanced_updates["topology_gate"] = topology_gate
    if max_calibration_minutes is not None:
        advanced_updates["max_calibration_minutes"] = max_calibration_minutes
    if calibration_restarts is not None:
        advanced_updates["calibration_restarts"] = calibration_restarts
    if allow_quarantine is not None:
        advanced_updates["allow_quarantine"] = allow_quarantine
    if advanced_updates:
        config = config.model_copy(update=advanced_updates)

    if config.quality_profile not in {"fast", "balanced", "strict"}:
        out.error(
            f"Invalid --quality-profile '{config.quality_profile}' "
            "(expected: fast | balanced | strict)"
        )
        raise typer.Exit(1)

    if resource_mode not in {"auto", "manual"}:
        out.error("--resource-mode must be 'auto' or 'manual'")
        raise typer.Exit(1)

    governor = ResourceGovernor(
        resource_mode=resource_mode,
        safe_auto_workers=safe_auto_workers,
        max_memory_gb=max_memory_gb,
    )
    tuned_workers = governor.recommend_workers(
        requested_workers=config.similarity_workers,
        memory_per_worker_gb=0.75,
    )
    tuned_chunk = governor.recommend_chunk_size(
        requested_chunk_size=config.similarity_chunk_size,
        min_chunk_size=8,
        max_chunk_size=2048,
    )

    config = config.model_copy(
        update={
            "similarity_workers": tuned_workers,
            "similarity_chunk_size": tuned_chunk,
        }
    )

    if config.candidate_mode not in {"exact", "blocked"}:
        out.error(
            f"Invalid --candidate-mode '{config.candidate_mode}' "
            "(expected: exact | blocked)"
        )
        raise typer.Exit(1)
    if config.topology_gate not in {"strict", "warn"}:
        out.error(
            f"Invalid --topology-gate '{config.topology_gate}' "
            "(expected: strict | warn)"
        )
        raise typer.Exit(1)

    # Save config if requested
    if save_config:
        config.to_yaml(save_config)
        out.success(f"Saved network config to [bold]{save_config}[/bold]")

    if (
        generated_from_population
        and auto_save_generated_config
        and network_config is None
    ):
        seed_label = str(config.seed) if config.seed is not None else "noseed"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        auto_path = scenario_dir / f"network-config.seed{seed_label}.{ts}.yaml"
        if save_config is None or save_config.resolve() != auto_path.resolve():
            config.to_yaml(auto_path)
            out.success(f"Auto-saved generated config to [bold]{auto_path}[/bold]")

    if not agent_mode:
        console.print(
            f"[dim]Mode: {config.candidate_mode} | workers={config.similarity_workers} "
            f"| profile={config.quality_profile} | checkpoint={'on' if checkpoint_db else 'off'}[/dim]"
        )
        if resource_mode == "auto":
            snap = governor.snapshot()
            console.print(
                f"[dim]Auto resources: cpu={snap.cpu_count}, "
                f"total_mem={snap.total_memory_gb:.1f}GB, budget={snap.memory_budget_gb:.1f}GB[/dim]"
            )
        console.print()

    generation_start = time.time()
    network_run_id = str(uuid.uuid4())
    current_stage = ["Initializing", 0, 0]

    if not agent_mode:
        console.print(f"[dim]network_run_id={network_run_id}[/dim]")

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
                result = generate_network(
                    agents,
                    config,
                    on_progress,
                    checkpoint_path=checkpoint_db,
                    resume_from_checkpoint=resume_requested,
                    study_db_path=study_db,
                    network_run_id=network_run_id,
                    resume_calibration=resume_requested,
                )
            else:
                result = generate_network_with_metrics(
                    agents,
                    config,
                    on_progress,
                    checkpoint_path=checkpoint_db,
                    resume_from_checkpoint=resume_requested,
                    study_db_path=study_db,
                    network_run_id=network_run_id,
                    resume_calibration=resume_requested,
                )
        except Exception as e:
            generation_error = e
        finally:
            generation_done.set()

    gen_thread = Thread(target=do_generation, daemon=True)
    gen_thread.start()

    if not agent_mode:
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
    else:
        generation_done.wait()

    generation_elapsed = time.time() - generation_start

    if generation_error:
        out.error(f"Network generation failed: {generation_error}")
        raise typer.Exit(1)

    out.success(
        f"Generated network: {result.meta['edge_count']} edges, "
        f"avg degree {result.meta['avg_degree']:.1f} ({format_elapsed(generation_elapsed)})",
        edge_count=result.meta["edge_count"],
        avg_degree=result.meta["avg_degree"],
        network_run_id=network_run_id,
    )

    # Validation (optional)
    if validate and not agent_mode:
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

    if validate and agent_mode:
        out.set_data(
            "network_metrics",
            result.network_metrics.model_dump(mode="json")
            if result.network_metrics
            else None,
        )
        edge_types: dict[str, int] = {}
        for edge in result.edges:
            t = edge.edge_type
            edge_types[t] = edge_types.get(t, 0) + 1
        out.set_data("edge_type_distribution", edge_types)

    quality_meta = result.meta.get("quality", {})
    accepted = bool(quality_meta.get("accepted", True))
    strict_failed = (
        config.topology_gate == "strict" and not accepted and len(agents) >= 50
    )

    # Save canonical output to study DB (or quarantine on strict failure)
    if not agent_mode:
        console.print()

    # Use scenario_name as network_id for canonical output
    target_network_id = scenario_name

    if not agent_mode:
        with console.status(f"[cyan]Saving network to {study_db}...[/cyan]"):
            _save_network(
                study_db,
                scenario_name,
                target_network_id,
                config,
                result,
                network_run_id,
                strict_failed,
                quality_meta,
            )
    else:
        _save_network(
            study_db,
            scenario_name,
            target_network_id,
            config,
            result,
            network_run_id,
            strict_failed,
            quality_meta,
        )

    if strict_failed and config.allow_quarantine:
        target_network_id = (
            f"{scenario_name}__{config.quarantine_suffix}__{network_run_id[:12]}"
        )
        gate_deltas = quality_meta.get("gate_deltas", {})
        out.error(
            "Topology gate strict failed. Saved quarantined artifact; canonical network not overwritten."
        )
        if not agent_mode:
            console.print(
                f"[yellow]![/yellow] Quarantined network_id={target_network_id}"
            )
            console.print(
                f"[red]✗[/red] Failed gates with best metrics: {quality_meta.get('best_metrics', {})}"
            )
            if gate_deltas:
                console.print(f"[dim]Gate deltas: {gate_deltas}[/dim]")
            console.print(
                f"[dim]inspect via: extropy inspect network-status --network-run-id {network_run_id}[/dim]"
            )
        raise typer.Exit(1)
    if strict_failed and not config.allow_quarantine:
        out.error(
            f"Topology gate strict failed; no output saved for network_id={scenario_name}"
        )
        raise typer.Exit(1)

    if output is not None:
        if not agent_mode:
            with console.status(f"[cyan]Exporting JSON to {output}...[/cyan]"):
                result.save_json(output)
        else:
            result.save_json(output)

    elapsed = time.time() - start_time

    out.set_data("study_db", str(study_db))
    out.set_data("scenario_id", scenario_name)
    out.set_data("network_id", target_network_id)
    out.set_data("network_run_id", network_run_id)
    out.set_data("total_time_seconds", elapsed)

    if not agent_mode:
        console.print("═" * 60)
        console.print(
            f"[green]✓[/green] Network saved to [bold]{study_db}[/bold] "
            f"(scenario_id={scenario_name})"
        )
        console.print(
            f"[dim]Inspect status: extropy inspect network-status --network-run-id {network_run_id}[/dim]"
        )
        if output is not None:
            console.print(f"[dim]Exported JSON: {output}[/dim]")
        console.print(f"[dim]Total time: {format_elapsed(elapsed)}[/dim]")
        console.print("═" * 60)

    raise typer.Exit(out.finish())


def _resolve_scenario(
    study_ctx: StudyContext, scenario_ref: str | None, out: Output
) -> tuple[str, int | None]:
    """Resolve scenario name and version."""
    scenarios = study_ctx.list_scenarios()

    if not scenarios:
        out.error("No scenarios found. Run 'extropy scenario' first.")
        raise typer.Exit(1)

    if scenario_ref is None:
        if len(scenarios) == 1:
            return scenarios[0], None
        else:
            out.error(
                f"Multiple scenarios found: {', '.join(scenarios)}. "
                "Use -s to specify which one."
            )
            raise typer.Exit(1)

    name, version = parse_version_ref(scenario_ref)
    if name not in scenarios:
        out.error(f"Scenario not found: {name}")
        raise typer.Exit(1)

    return name, version


def _parse_base_population_ref(ref: str) -> tuple[str, int | None]:
    """Parse base_population reference like 'population.v2'."""
    import re

    match = re.match(r"^(.+)\.v(\d+)$", ref)
    if match:
        return match.group(1), int(match.group(2))
    return ref, None


def _save_network(
    study_db: Path,
    scenario_name: str,
    network_id: str,
    config,
    result,
    network_run_id: str,
    strict_failed: bool,
    quality_meta: dict,
) -> None:
    """Save network to study database."""
    from ...storage import open_study_db

    with open_study_db(study_db) as db:
        network_metrics = (
            result.network_metrics.model_dump(mode="json")
            if result.network_metrics
            else None
        )
        target_network_id = network_id
        if strict_failed and config.allow_quarantine:
            target_network_id = (
                f"{network_id}__{config.quarantine_suffix}__{network_run_id[:12]}"
            )
            result.meta["outcome"] = "rejected_quarantined"
            result.meta["canonical_network_id"] = network_id
        elif strict_failed:
            result.meta["outcome"] = "rejected"
        elif quality_meta.get("accepted", True):
            result.meta["outcome"] = "accepted"
        else:
            result.meta["outcome"] = "accepted_with_warnings"

        db.save_network_result(
            population_id=scenario_name,  # Keep for backwards compat
            network_id=target_network_id,
            config=config.model_dump(mode="json"),
            result_meta=result.meta,
            edges=[e.to_dict() for e in result.edges],
            seed=config.seed,
            candidate_mode=config.candidate_mode,
            network_metrics=network_metrics,
            network_run_id=network_run_id,
            scenario_id=scenario_name,  # New scenario-centric key
        )
