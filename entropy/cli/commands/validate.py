"""Validate command for population specs and scenario specs."""

from pathlib import Path

import typer

from ...core.models import PopulationSpec
from ...population.validator import validate_spec
from ..app import app, console, get_json_mode
from ..utils import Output, ExitCode, format_validation_for_json


def _is_scenario_file(path: Path) -> bool:
    """Check if file is a scenario spec based on naming convention."""
    return path.name.endswith(".scenario.yaml") or path.name.endswith(".scenario.yml")


def _validate_population_spec(spec_file: Path, strict: bool, out: Output) -> int:
    """Validate a population spec."""
    # Load spec
    if not get_json_mode():
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
    if not get_json_mode():
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

        if not get_json_mode():
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

            if not get_json_mode():
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

            if not get_json_mode():
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
    if not get_json_mode():
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
    if not get_json_mode():
        out.text("[bold]File References:[/bold]")

        pop_path = Path(spec.meta.population_spec)
        if pop_path.exists():
            out.text(f"  [green]✓[/green] Population: {spec.meta.population_spec}")
        else:
            out.text(
                f"  [red]✗[/red] Population: {spec.meta.population_spec} (not found)"
            )

        agents_path = Path(spec.meta.agents_file)
        if agents_path.exists():
            out.text(f"  [green]✓[/green] Agents: {spec.meta.agents_file}")
        else:
            out.text(f"  [red]✗[/red] Agents: {spec.meta.agents_file} (not found)")

        network_path = Path(spec.meta.network_file)
        if network_path.exists():
            out.text(f"  [green]✓[/green] Network: {spec.meta.network_file}")
        else:
            out.text(f"  [red]✗[/red] Network: {spec.meta.network_file} (not found)")

        out.blank()

    # Handle errors
    if result.errors:
        out.error(
            f"Scenario has {len(result.errors)} error(s)",
            exit_code=ExitCode.VALIDATION_ERROR,
        )

        if not get_json_mode():
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

        if not get_json_mode():
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
    Validate a population spec or scenario spec.

    Auto-detects file type based on naming:
    - *.scenario.yaml → scenario spec validation
    - *.yaml → population spec validation

    EXIT CODES:
        0 = Success (valid spec)
        1 = Validation error (invalid spec)
        2 = File not found

    EXAMPLES:
        entropy validate surgeons.yaml              # Population spec
        entropy validate surgeons.scenario.yaml     # Scenario spec
        entropy validate surgeons.yaml --strict     # Treat warnings as errors
    """
    out = Output(console, json_mode=get_json_mode())
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
    else:
        exit_code = _validate_population_spec(spec_file, strict, out)

    raise typer.Exit(exit_code)
