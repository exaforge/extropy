"""Validate command for population specs and scenario specs."""

import re
from pathlib import Path

import typer

from ...core.models import PopulationSpec
from ...population.validator import validate_spec
from ..app import app, console, get_json_mode, is_agent_mode
from ..utils import Output, ExitCode, format_validation_for_json
from ...utils import topological_sort


def _is_json_output() -> bool:
    """Check if JSON output is enabled (via --json flag or agent mode config)."""
    return get_json_mode() or is_agent_mode()


def _is_scenario_file(path: Path) -> bool:
    """Check if file is a scenario spec based on naming convention."""
    name = path.name
    # Legacy patterns
    if name.endswith(".scenario.yaml") or name.endswith(".scenario.yml"):
        return True
    if name in {"scenario.yaml", "scenario.yml"}:
        return True
    # Versioned pattern: scenario.v{N}.yaml or scenario.v{N}.yml
    if re.match(r"^scenario\.v\d+\.ya?ml$", name):
        return True
    return False


def _is_persona_file(path: Path) -> bool:
    """Check if file is a persona config based on naming convention."""
    name = path.name
    if name in {"persona.yaml", "persona.yml"}:
        return True
    if re.match(r"^persona\.v\d+\.ya?ml$", name):
        return True
    return False


def _validate_population_spec(spec_file: Path, strict: bool, out: Output) -> int:
    """Validate a population spec."""
    # Load spec
    if not _is_json_output():
        with console.status("[cyan]Loading spec...[/cyan]"):
            try:
                spec = PopulationSpec.from_yaml(spec_file)
            except Exception as e:
                out.error(
                    f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR
                )
                return out.finish()
    else:
        try:
            spec = PopulationSpec.from_yaml(spec_file)
        except Exception as e:
            out.error(f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR)
            return out.finish()

    out.success(
        f"Loaded: [bold]{spec.meta.description}[/bold] ({len(spec.attributes)} attributes)",
        spec_file=str(spec_file),
        description=spec.meta.description,
        attribute_count=len(spec.attributes),
    )
    out.blank()

    # Validate spec
    if not _is_json_output():
        with console.status("[cyan]Validating spec...[/cyan]"):
            result = validate_spec(spec)
    else:
        result = validate_spec(spec)

    # Add validation result to JSON output
    out.set_data("validation", format_validation_for_json(result))

    # Handle errors
    if result.errors:
        out.error(
            f"Spec has {len(result.errors)} error(s)",
            exit_code=ExitCode.VALIDATION_ERROR,
        )

        if not _is_json_output():
            error_rows = []
            for err in result.errors[:15]:
                loc = err.location
                if err.modifier_index is not None:
                    loc = f"{err.location}[{err.modifier_index}]"
                error_rows.append([loc, err.category, err.message[:60]])

            if error_rows:
                out.table(
                    "Errors",
                    ["Location", "Category", "Message"],
                    error_rows,
                    styles=["red", "dim", None],
                )

            if len(result.errors) > 15:
                out.text(
                    f"  [dim]... and {len(result.errors) - 15} more error(s)[/dim]"
                )

            out.blank()
            out.text("[bold]Suggestions:[/bold]")
            for err in result.errors[:3]:
                if err.suggestion:
                    out.text(f"  [dim]→ {err.location}: {err.suggestion}[/dim]")

        return out.finish()

    # Handle warnings (with strict mode)
    if result.warnings:
        if strict:
            out.error(
                f"Spec has {len(result.warnings)} warning(s) (strict mode)",
                exit_code=ExitCode.VALIDATION_ERROR,
            )

            if not _is_json_output():
                warning_rows = []
                for warn in result.warnings[:10]:
                    loc = warn.location
                    if warn.modifier_index is not None:
                        loc = f"{warn.location}[{warn.modifier_index}]"
                    warning_rows.append([loc, warn.category, warn.message[:60]])

                out.table(
                    "Warnings",
                    ["Location", "Category", "Message"],
                    warning_rows,
                    styles=["yellow", "dim", None],
                )

            return out.finish()
        else:
            out.success(f"Spec validated with {len(result.warnings)} warning(s)")

            if not _is_json_output():
                for warn in result.warnings[:3]:
                    loc = warn.location
                    if warn.modifier_index is not None:
                        loc = f"{warn.location}[{warn.modifier_index}]"
                    out.warning(f"{loc}: {warn.message}")

                if len(result.warnings) > 3:
                    out.text(
                        f"  [dim]... and {len(result.warnings) - 3} more warning(s)[/dim]"
                    )
    else:
        out.success("Spec validated")

    out.blank()
    out.divider()
    out.text("[green]Validation passed[/green]")
    out.divider()

    return out.finish()


def _validate_scenario_spec(spec_file: Path, out: Output) -> int:
    """Validate a scenario spec."""
    from ...scenario import load_and_validate_scenario

    # Load and validate
    if not _is_json_output():
        with console.status("[cyan]Loading scenario spec...[/cyan]"):
            try:
                spec, result = load_and_validate_scenario(spec_file)
            except Exception as e:
                out.error(
                    f"Failed to load scenario: {e}", exit_code=ExitCode.VALIDATION_ERROR
                )
                return out.finish()
    else:
        try:
            spec, result = load_and_validate_scenario(spec_file)
        except Exception as e:
            out.error(
                f"Failed to load scenario: {e}", exit_code=ExitCode.VALIDATION_ERROR
            )
            return out.finish()

    out.success(
        f"Loaded scenario: [bold]{spec.meta.name}[/bold]",
        spec_file=str(spec_file),
        name=spec.meta.name,
    )
    out.blank()

    # Show file references (human mode only)
    if not _is_json_output():
        out.text("[bold]Scenario Details:[/bold]")

        # New flow: base_population
        if spec.meta.base_population:
            out.text(f"  [cyan]•[/cyan] base_population: {spec.meta.base_population}")
            if spec.extended_attributes:
                out.text(
                    f"  [cyan]•[/cyan] extended_attributes: {len(spec.extended_attributes)}"
                )

        # Legacy flow: population_spec + study_db
        if spec.meta.population_spec:
            from ...utils import resolve_relative_to

            pop_path = resolve_relative_to(spec.meta.population_spec, spec_file)
            if pop_path.exists():
                out.text(f"  [green]✓[/green] Population: {spec.meta.population_spec}")
            else:
                out.text(
                    f"  [red]✗[/red] Population: {spec.meta.population_spec} (not found)"
                )

        if spec.meta.study_db:
            from ...utils import resolve_relative_to

            study_db_path = resolve_relative_to(spec.meta.study_db, spec_file)
            if study_db_path.exists():
                out.text(f"  [green]✓[/green] Study DB: {spec.meta.study_db}")
            else:
                out.text(f"  [red]✗[/red] Study DB: {spec.meta.study_db} (not found)")
            out.text(f"  [cyan]•[/cyan] population_id: {spec.meta.population_id}")
            out.text(f"  [cyan]•[/cyan] network_id: {spec.meta.network_id}")

        out.blank()

    # Handle errors
    if result.errors:
        out.error(
            f"Scenario has {len(result.errors)} error(s)",
            exit_code=ExitCode.VALIDATION_ERROR,
        )

        if not _is_json_output():
            for err in result.errors[:10]:
                out.text(
                    f"  [red]✗[/red] [{err.category}] {err.location}: {err.message}"
                )
                if err.suggestion:
                    out.text(f"    [dim]→ {err.suggestion}[/dim]")

            if len(result.errors) > 10:
                out.text(f"  [dim]... and {len(result.errors) - 10} more[/dim]")

        return out.finish()

    # Handle warnings
    if result.warnings:
        out.success(f"Scenario validated with {len(result.warnings)} warning(s)")

        if not _is_json_output():
            for warn in result.warnings[:5]:
                out.warning(f"[{warn.category}] {warn.location}: {warn.message}")

            if len(result.warnings) > 5:
                out.text(f"  [dim]... and {len(result.warnings) - 5} more[/dim]")
    else:
        out.success("Scenario validated")

    out.blank()
    out.divider()
    out.text("[green]Validation passed[/green]")
    out.divider()

    return out.finish()


def _resolve_merged_population_for_persona(persona_file: Path) -> PopulationSpec:
    """Resolve merged population spec (base + extension) for persona validation."""
    from ...core.models import PopulationSpec
    from ...core.models.scenario import ScenarioSpec

    scenario_dir = persona_file.parent
    scenario_files_with_versions: list[tuple[int, Path]] = []
    for path in scenario_dir.glob("scenario.v*.yaml"):
        match = re.match(r"^scenario\.v(\d+)\.yaml$", path.name)
        if not match:
            continue
        scenario_files_with_versions.append((int(match.group(1)), path))

    scenario_files_with_versions.sort(key=lambda item: item[0])
    if not scenario_files_with_versions:
        raise FileNotFoundError(
            f"No scenario.vN.yaml found next to persona file: {persona_file}"
        )

    # Choose latest scenario version in this directory.
    scenario_path = scenario_files_with_versions[-1][1]
    scenario_spec = ScenarioSpec.from_yaml(scenario_path)

    pop_name, pop_version = scenario_spec.meta.get_population_ref()
    if pop_version is None:
        raise ValueError(
            "Scenario must reference a versioned base_population for persona validation"
        )

    # Study layout: <study>/scenario/<name>/persona.vN.yaml
    study_root = scenario_dir.parent.parent
    pop_path = study_root / f"{pop_name}.v{pop_version}.yaml"
    if not pop_path.exists():
        raise FileNotFoundError(f"Referenced population spec not found: {pop_path}")

    pop_spec = PopulationSpec.from_yaml(pop_path)
    ext_attrs = scenario_spec.extended_attributes or []
    merged_attrs = list(pop_spec.attributes) + ext_attrs
    merged_deps: dict[str, list[str]] = {
        attr.name: list(attr.sampling.depends_on or []) for attr in merged_attrs
    }
    merged_names = set(merged_deps.keys())
    merged_deps = {
        name: [dep for dep in deps if dep in merged_names]
        for name, deps in merged_deps.items()
    }
    merged_order = topological_sort(merged_deps)
    return PopulationSpec(
        meta=pop_spec.meta,
        grounding=pop_spec.grounding,
        attributes=merged_attrs,
        sampling_order=merged_order,
    )


def _validate_persona_spec(spec_file: Path, out: Output) -> int:
    """Validate a persona config against merged scenario attributes."""
    from ...population.persona import PersonaConfig, validate_persona_config

    if not _is_json_output():
        with console.status("[cyan]Loading persona config...[/cyan]"):
            try:
                config = PersonaConfig.from_yaml(spec_file)
                merged_spec = _resolve_merged_population_for_persona(spec_file)
                result = validate_persona_config(merged_spec, config)
            except Exception as e:
                out.error(
                    f"Failed to validate persona: {e}",
                    exit_code=ExitCode.VALIDATION_ERROR,
                )
                return out.finish()
    else:
        try:
            config = PersonaConfig.from_yaml(spec_file)
            merged_spec = _resolve_merged_population_for_persona(spec_file)
            result = validate_persona_config(merged_spec, config)
        except Exception as e:
            out.error(
                f"Failed to validate persona: {e}", exit_code=ExitCode.VALIDATION_ERROR
            )
            return out.finish()

    out.success(
        f"Loaded persona config ({len(config.treatments)} treatments)",
        spec_file=str(spec_file),
        treatment_count=len(config.treatments),
    )
    out.blank()
    out.set_data("validation", format_validation_for_json(result))

    if result.errors:
        out.error(
            f"Persona has {len(result.errors)} error(s)",
            exit_code=ExitCode.VALIDATION_ERROR,
        )
        if not _is_json_output():
            for err in result.errors[:10]:
                out.text(
                    f"  [red]✗[/red] [{err.category}] {err.location}: {err.message}"
                )
            if len(result.errors) > 10:
                out.text(f"  [dim]... and {len(result.errors) - 10} more[/dim]")
        return out.finish()

    if result.warnings:
        out.success(f"Persona validated with {len(result.warnings)} warning(s)")
        if not _is_json_output():
            for warn in result.warnings[:5]:
                out.warning(f"[{warn.category}] {warn.location}: {warn.message}")
    else:
        out.success("Persona validated")

    out.blank()
    out.divider()
    out.text("[green]Validation passed[/green]")
    out.divider()
    return out.finish()


@app.command("validate")
def validate_command(
    spec_file: Path = typer.Argument(
        ..., help="Spec file to validate (.yaml or .scenario.yaml)"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Treat warnings as errors (population specs only)"
    ),
):
    """
    Validate a population spec, scenario spec, or persona config.

    Auto-detects file type based on naming:
    - *.scenario.yaml → scenario spec validation
    - persona.vN.yaml → persona config validation
    - *.yaml → population spec validation

    EXIT CODES:
        0 = Success (valid spec)
        1 = Validation error (invalid spec)
        3 = File not found

    EXAMPLES:
        extropy validate surgeons.yaml              # Population spec
        extropy validate surgeons.scenario.yaml     # Scenario spec
        extropy validate surgeons.yaml --strict     # Treat warnings as errors
    """
    out = Output(console=console, json_mode=_is_json_output())
    out.blank()

    # Check file exists
    if not spec_file.exists():
        out.error(
            f"File not found: {spec_file}",
            exit_code=ExitCode.FILE_NOT_FOUND,
            suggestion=f"Check the file path: {spec_file.absolute()}",
        )
        raise typer.Exit(out.finish())

    # Route to appropriate validator
    if _is_scenario_file(spec_file):
        exit_code = _validate_scenario_spec(spec_file, out)
    elif _is_persona_file(spec_file):
        exit_code = _validate_persona_spec(spec_file, out)
    else:
        exit_code = _validate_population_spec(spec_file, strict, out)

    raise typer.Exit(exit_code)
