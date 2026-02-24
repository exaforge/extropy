"""CLI utilities for dual-mode output (human-friendly + machine-readable).

This module provides utilities for CLI commands to support both:
- Human mode (default): Rich formatting with colors, tables, progress bars
- Machine mode (--json): Structured JSON output for AI coding tools

Example:
    from ..cli.utils import Output, ExitCode

    @app.command()
    def my_command():
        out = Output(console=console, json_mode=get_json_mode())
        out.success("Loaded spec", spec_name="surgeons.yaml", count=500)
        out.table("Attributes", ["Name", "Type"], [["age", "int"], ["income", "float"]])
        return out.finish()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, PrivateAttr, ConfigDict
from rich.console import Console
from rich.table import Table


class ExitCode:
    """Standardized exit codes for CLI commands.

    AI tools can check $? and know exactly what failed:
        0 = Success
        1 = Validation error (fix spec/input first)
        2 = Needs clarification (agent mode: answer questions and retry)
        3 = File not found
        4 = Sampling error
        5 = Network generation error
        6 = Simulation error
        7 = Scenario error
        10 = User cancelled
    """

    SUCCESS = 0
    VALIDATION_ERROR = 1
    NEEDS_CLARIFICATION = 2
    FILE_NOT_FOUND = 3
    SAMPLING_ERROR = 4
    NETWORK_ERROR = 5
    SIMULATION_ERROR = 6
    SCENARIO_ERROR = 7
    USER_CANCELLED = 10


class Output(BaseModel):
    """Dual-mode output handler for CLI commands.

    In human mode: Uses Rich for pretty terminal output with colors and formatting.
    In JSON mode: Collects structured data and outputs JSON at the end.

    Usage:
        out = Output(console=console, json_mode=False)
        out.success("Loaded spec", count=500)
        out.warning("Some warning")
        out.table("Stats", ["Attr", "Mean"], [["age", "43.2"]])
        exit_code = out.finish()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    console: Console
    json_mode: bool = False

    _data: dict[str, Any] = PrivateAttr()
    _exit_code: int = PrivateAttr(default=ExitCode.SUCCESS)

    def model_post_init(self, __context: Any) -> None:
        # Ensure _data is a fresh dict for each instance
        self._data = {
            "status": "success",
            "warnings": [],
            "errors": [],
        }

    def success(self, message: str, **data: Any) -> None:
        """Output a success message with optional data."""
        if self.json_mode:
            self._data.update(data)
        else:
            self.console.print(f"[green]✓[/green] {message}")

    def warning(
        self,
        message: str,
        *,
        attribute: str | None = None,
        category: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Output a warning message."""
        if self.json_mode:
            warning_obj: dict[str, Any] = {"message": message}
            if attribute:
                warning_obj["attribute"] = attribute
            if category:
                warning_obj["category"] = category
            if suggestion:
                warning_obj["suggestion"] = suggestion
            self._data["warnings"].append(warning_obj)
        else:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
            if suggestion:
                self.console.print(f"  [dim]→ {suggestion}[/dim]")

    def error(
        self,
        message: str,
        *,
        attribute: str | None = None,
        category: str | None = None,
        suggestion: str | None = None,
        exit_code: int = ExitCode.VALIDATION_ERROR,
    ) -> None:
        """Output an error message and set exit code."""
        self._exit_code = exit_code
        self._data["status"] = "error"

        if self.json_mode:
            error_obj: dict[str, Any] = {"message": message}
            if attribute:
                error_obj["attribute"] = attribute
            if category:
                error_obj["category"] = category
            if suggestion:
                error_obj["suggestion"] = suggestion
            self._data["errors"].append(error_obj)
        else:
            self.console.print(f"[red]✗[/red] {message}")
            if suggestion:
                self.console.print(f"  [dim]→ {suggestion}[/dim]")

    def text(self, message: str) -> None:
        """Output plain text (human mode only)."""
        if not self.json_mode:
            self.console.print(message)

    def blank(self) -> None:
        """Output a blank line (human mode only)."""
        if not self.json_mode:
            self.console.print()

    def header(self, title: str) -> None:
        """Output a section header."""
        if not self.json_mode:
            self.console.print()
            self.console.print("┌" + "─" * 58 + "┐")
            self.console.print("│" + f" {title}".ljust(58) + "│")
            self.console.print("└" + "─" * 58 + "┘")
            self.console.print()

    def table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[str]],
        *,
        data_key: str | None = None,
        styles: list[str] | None = None,
    ) -> None:
        """Output a formatted table.

        Args:
            title: Table title
            columns: Column headers
            rows: Table rows (list of lists)
            data_key: Key to use in JSON output (defaults to snake_case of title)
            styles: Optional Rich styles for each column
        """
        key = data_key or title.lower().replace(" ", "_")

        if self.json_mode:
            self._data[key] = [dict(zip(columns, row)) for row in rows]
        else:
            table = Table(title=title, show_header=True, header_style="bold")
            for i, col in enumerate(columns):
                style = styles[i] if styles and i < len(styles) else None
                justify = "right" if i > 0 else "left"  # Right-align numeric columns
                table.add_column(col, style=style, justify=justify)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)

    def set_data(self, key: str, value: Any) -> None:
        """Set arbitrary data in JSON output."""
        self._data[key] = value

    def needs_clarification(
        self,
        questions: list,
        resume_command: str,
        partial_data: dict | None = None,
    ) -> None:
        """Set needs_clarification status with structured questions.

        Used in agent mode when the CLI needs more information to proceed.
        Returns exit code 2 with JSON containing questions and a resume command.

        Args:
            questions: List of ClarificationQuestion objects
            resume_command: Command template for retrying with --answers
            partial_data: Any partial results to include
        """
        self._exit_code = ExitCode.NEEDS_CLARIFICATION
        self._data["status"] = "needs_clarification"
        self._data["questions"] = [
            {
                "id": q.id,
                "question": q.question,
                "type": q.type,
                "options": q.options,
                "default": q.default,
            }
            for q in questions
        ]
        self._data["resume_command"] = resume_command
        if partial_data:
            self._data["partial"] = partial_data

    def divider(self) -> None:
        """Output a visual divider (human mode only)."""
        if not self.json_mode:
            self.console.print("═" * 60)

    def finish(self) -> int:
        """Finalize output and return exit code.

        In JSON mode, prints the accumulated data as JSON to stdout.
        Returns the exit code that should be passed to sys.exit().
        """
        if self.json_mode:
            # Add exit_code to JSON for programmatic access
            self._data["exit_code"] = self._exit_code
            print(json.dumps(self._data, indent=2, default=str))

        return self._exit_code


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as Xm Ys or Xs."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    return f"{seconds:.0f}s"


def grounding_indicator(level: str) -> str:
    """Get colored grounding indicator."""
    indicators = {
        "strong": "[green]🟢 Strong[/green]",
        "medium": "[yellow]🟡 Medium[/yellow]",
        "low": "[red]🔴 Low[/red]",
    }
    return indicators.get(level, "[dim]Unknown[/dim]")


def get_next_invalid_artifact_path(
    target_path: Path | str,
    *,
    stem: str | None = None,
    extension: str | None = None,
) -> Path:
    """Compute the next versioned invalid-artifact path.

    Examples:
        scenario.v1.yaml -> scenario.v1.invalid.v1.yaml
        scenario.v1.yaml -> scenario.v1.invalid.v2.yaml (if v1 already exists)
        stem="sample", extension=".json" -> sample.invalid.vN.json
    """
    target = Path(target_path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    base_stem = stem if stem is not None else target.stem
    suffix = extension if extension is not None else target.suffix
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"

    pattern = re.compile(
        rf"^{re.escape(base_stem)}\.invalid\.v(?P<version>\d+){re.escape(suffix)}$"
    )
    max_version = 0
    for existing in parent.iterdir():
        if not existing.is_file():
            continue
        match = pattern.match(existing.name)
        if not match:
            continue
        max_version = max(max_version, int(match.group("version")))

    next_version = max_version + 1
    return parent / f"{base_stem}.invalid.v{next_version}{suffix}"


def save_invalid_json_artifact(
    payload: dict[str, Any],
    target_path: Path | str,
    *,
    stem: str | None = None,
    extension: str = ".json",
) -> Path:
    """Save a versioned invalid artifact as JSON and return its path."""
    path = get_next_invalid_artifact_path(target_path, stem=stem, extension=extension)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def format_validation_for_json(result) -> dict[str, Any]:
    """Convert ValidationResult to JSON-serializable dict."""
    return {
        "valid": result.valid,
        "error_count": len(result.errors) if result.errors else 0,
        "warning_count": len(result.warnings) if result.warnings else 0,
        "errors": [
            {
                "location": e.location,
                "category": e.category,
                "message": e.message,
                "modifier_index": e.modifier_index,
                "suggestion": e.suggestion,
                "value": e.value,
            }
            for e in (result.errors or [])
        ],
        "warnings": [
            {
                "location": w.location,
                "category": w.category,
                "message": w.message,
                "modifier_index": w.modifier_index,
                "suggestion": w.suggestion,
                "value": w.value,
            }
            for w in (result.warnings or [])
        ],
    }


def format_sampling_stats_for_json(stats, spec) -> dict[str, Any]:
    """Convert SamplingStats to JSON-serializable dict."""
    result: dict[str, Any] = {}

    # Numeric attributes
    numeric_attrs = [a for a in spec.attributes if a.type in ("int", "float")]
    if numeric_attrs:
        result["numeric_attributes"] = {
            attr.name: {
                "mean": stats.attribute_means.get(attr.name, 0),
                "std": stats.attribute_stds.get(attr.name, 0),
            }
            for attr in numeric_attrs
        }

    # Categorical attributes
    cat_attrs = [a for a in spec.attributes if a.type == "categorical"]
    if cat_attrs:
        result["categorical_attributes"] = {}
        for attr in cat_attrs:
            counts = stats.categorical_counts.get(attr.name, {})
            total = sum(counts.values()) or 1
            result["categorical_attributes"][attr.name] = {
                k: {"count": v, "percentage": v / total} for k, v in counts.items()
            }

    # Boolean attributes
    bool_attrs = [a for a in spec.attributes if a.type == "boolean"]
    if bool_attrs:
        result["boolean_attributes"] = {}
        for attr in bool_attrs:
            counts = stats.boolean_counts.get(attr.name, {True: 0, False: 0})
            total = sum(counts.values()) or 1
            result["boolean_attributes"][attr.name] = {
                "true_count": counts.get(True, 0),
                "false_count": counts.get(False, 0),
                "true_percentage": counts.get(True, 0) / total,
            }

    # Modifier triggers
    triggered_mods = {
        k: v for k, v in stats.modifier_triggers.items() if any(v.values())
    }
    if triggered_mods:
        result["modifier_triggers"] = triggered_mods

    # Constraint violations
    if stats.constraint_violations:
        result["constraint_violations"] = stats.constraint_violations

    if stats.condition_warnings:
        result["condition_warnings"] = stats.condition_warnings

    if stats.reconciliation_counts:
        result["reconciliation_counts"] = stats.reconciliation_counts

    if stats.rule_pack:
        result["rule_pack"] = stats.rule_pack

    return result
