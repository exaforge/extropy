"""Validate command for population specs."""

from pathlib import Path

import typer

from ...core.models import PopulationSpec
from ...population.validator import validate_spec
from ..app import app, console, get_json_mode
from ..utils import Output, ExitCode, format_validation_for_json


@app.command("validate")
def validate_command(
    spec_file: Path = typer.Argument(..., help="Population spec YAML file to validate"),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors"),
):
    """
    Validate a population spec for structural correctness.

    Checks for type/modifier compatibility, range violations, weight validity,
    distribution parameters, dependencies, conditions, formulas, duplicates,
    and strategy consistency.

    EXIT CODES:
        0 = Success (valid spec)
        1 = Validation error (invalid spec)
        2 = File not found

    EXAMPLES:
        entropy validate surgeons.yaml
        entropy validate surgeons.yaml --strict
        entropy --json validate surgeons.yaml  # Machine-readable output
    """
    out = Output(console, json_mode=get_json_mode())
    out.blank()

    # Check file exists
    if not spec_file.exists():
        out.error(
            f"Spec file not found: {spec_file}",
            exit_code=ExitCode.FILE_NOT_FOUND,
            suggestion=f"Check the file path: {spec_file.absolute()}",
        )
        raise typer.Exit(out.finish())

    # Load spec
    if not get_json_mode():
        with console.status("[cyan]Loading spec...[/cyan]"):
            try:
                spec = PopulationSpec.from_yaml(spec_file)
            except Exception as e:
                out.error(f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR)
                raise typer.Exit(out.finish())
    else:
        try:
            spec = PopulationSpec.from_yaml(spec_file)
        except Exception as e:
            out.error(f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR)
            raise typer.Exit(out.finish())

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
        out.error(f"Spec has {len(result.errors)} error(s)", exit_code=ExitCode.VALIDATION_ERROR)

        if not get_json_mode():
            # Show error table for human mode
            error_rows = []
            for err in result.errors[:15]:
                loc = err.attribute
                if err.modifier_index is not None:
                    loc = f"{err.attribute}[{err.modifier_index}]"
                error_rows.append([loc, err.category, err.message[:60]])

            if error_rows:
                out.table(
                    "Errors",
                    ["Location", "Category", "Message"],
                    error_rows,
                    styles=["red", "dim", None],
                )

            if len(result.errors) > 15:
                out.text(f"  [dim]... and {len(result.errors) - 15} more error(s)[/dim]")

            # Show suggestions for first few errors
            out.blank()
            out.text("[bold]Suggestions:[/bold]")
            for err in result.errors[:3]:
                if err.suggestion:
                    out.text(f"  [dim]→ {err.attribute}: {err.suggestion}[/dim]")

        raise typer.Exit(out.finish())

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
                    loc = warn.attribute
                    if warn.modifier_index is not None:
                        loc = f"{warn.attribute}[{warn.modifier_index}]"
                    warning_rows.append([loc, warn.category, warn.message[:60]])

                out.table(
                    "Warnings",
                    ["Location", "Category", "Message"],
                    warning_rows,
                    styles=["yellow", "dim", None],
                )

            raise typer.Exit(out.finish())
        else:
            out.success(f"Spec validated with {len(result.warnings)} warning(s)")

            if not get_json_mode():
                for warn in result.warnings[:3]:
                    loc = warn.attribute
                    if warn.modifier_index is not None:
                        loc = f"{warn.attribute}[{warn.modifier_index}]"
                    out.warning(f"{loc}: {warn.message}")

                if len(result.warnings) > 3:
                    out.text(f"  [dim]... and {len(result.warnings) - 3} more warning(s)[/dim]")
    else:
        out.success("Spec validated")

    # Show summary
    out.blank()
    out.divider()
    out.text("[green]Validation passed[/green]")
    out.divider()

    raise typer.Exit(out.finish())
