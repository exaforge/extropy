"""CLI utilities for dual-mode output (human-friendly + machine-readable).

This module provides utilities for CLI commands to support both:
- Human mode (default): Rich formatting with colors, tables, progress bars
- Machine mode (--json): Structured JSON output for AI coding tools

Example:
    from .cli_utils import Output, ExitCode, get_output_context

    @app.command()
    def my_command(json_output: bool = typer.Option(False, "--json")):
        out = Output(console, json_mode=json_output)
        out.success("Loaded spec", spec_name="surgeons.yaml", count=500)
        out.table("Attributes", ["Name", "Type"], [["age", "int"], ["income", "float"]])
        return out.finish()
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree


class ExitCode:
    """Standardized exit codes for CLI commands.

    AI tools can check $? and know exactly what failed:
        0 = Success
        1 = Validation error (fix spec/input first)
        2 = File not found
        3 = Sampling error
        4 = Network generation error
        5 = Simulation error
        6 = Scenario error
        10 = User cancelled
    """

    SUCCESS = 0
    VALIDATION_ERROR = 1
    FILE_NOT_FOUND = 2
    SAMPLING_ERROR = 3
    NETWORK_ERROR = 4
    SIMULATION_ERROR = 5
    SCENARIO_ERROR = 6
    USER_CANCELLED = 10


@dataclass
class Output:
    """Dual-mode output handler for CLI commands.

    In human mode: Uses Rich for pretty terminal output with colors and formatting.
    In JSON mode: Collects structured data and outputs JSON at the end.

    Usage:
        out = Output(console, json_mode=False)
        out.success("Loaded spec", count=500)
        out.warning("Some warning")
        out.table("Stats", ["Attr", "Mean"], [["age", "43.2"]])
        exit_code = out.finish()
    """

    console: Console
    json_mode: bool = False
    _data: dict = field(default_factory=lambda: {
        "status": "success",
        "warnings": [],
        "errors": [],
    })
    _exit_code: int = ExitCode.SUCCESS

    def __post_init__(self):
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

    def info(self, message: str, **data: Any) -> None:
        """Output an info message."""
        if self.json_mode:
            self._data.update(data)
        else:
            self.console.print(f"[blue]ℹ[/blue] {message}")

    def warning(self, message: str, *, attribute: str | None = None,
                category: str | None = None, suggestion: str | None = None) -> None:
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

    def error(self, message: str, *, attribute: str | None = None,
              category: str | None = None, suggestion: str | None = None,
              exit_code: int = ExitCode.VALIDATION_ERROR) -> None:
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

    def panel(self, content: str, *, title: str | None = None,
              style: str = "blue") -> None:
        """Output a Rich panel (human mode only)."""
        if not self.json_mode:
            self.console.print(Panel(content, title=title, border_style=style))

    def table(self, title: str, columns: list[str], rows: list[list[str]], *,
              data_key: str | None = None, styles: list[str] | None = None) -> None:
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

    def tree(self, title: str, items: list[tuple[str, list[str]]], *,
             data_key: str | None = None) -> None:
        """Output a tree structure showing dependencies.

        Args:
            title: Tree root label
            items: List of (name, dependencies) tuples
            data_key: Key for JSON output
        """
        key = data_key or "dependencies"

        if self.json_mode:
            self._data[key] = [
                {"name": name, "depends_on": deps}
                for name, deps in items
            ]
        else:
            tree = Tree(title)
            for name, deps in items:
                if deps:
                    branch = tree.add(f"[cyan]{name}[/cyan]")
                    for dep in deps:
                        branch.add(f"[dim]← {dep}[/dim]")
                else:
                    tree.add(f"[green]{name}[/green]")
            self.console.print(tree)

    def progress_summary(self, items: dict[str, Any]) -> None:
        """Add progress/summary data (JSON mode stores it, human mode ignores)."""
        if self.json_mode:
            self._data.update(items)

    def set_data(self, key: str, value: Any) -> None:
        """Set arbitrary data in JSON output."""
        self._data[key] = value

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


def format_validation_for_json(result) -> dict[str, Any]:
    """Convert ValidationResult to JSON-serializable dict."""
    return {
        "valid": result.valid,
        "error_count": len(result.errors) if result.errors else 0,
        "warning_count": len(result.warnings) if result.warnings else 0,
        "errors": [
            {
                "attribute": e.attribute,
                "category": e.category,
                "message": e.message,
                "modifier_index": e.modifier_index,
                "suggestion": e.suggestion,
            }
            for e in (result.errors or [])
        ],
        "warnings": [
            {
                "attribute": w.attribute,
                "category": w.category,
                "message": w.message,
                "modifier_index": w.modifier_index,
                "suggestion": w.suggestion,
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
                k: {"count": v, "percentage": v / total}
                for k, v in counts.items()
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
    triggered_mods = {k: v for k, v in stats.modifier_triggers.items() if any(v.values())}
    if triggered_mods:
        result["modifier_triggers"] = triggered_mods

    # Constraint violations
    if stats.constraint_violations:
        result["constraint_violations"] = stats.constraint_violations

    return result
