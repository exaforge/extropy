"""Core CLI app definition and global state."""

import atexit
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="extropy",
    help="Generate population specs for agent-based simulation.",
    no_args_is_help=True,
)

console = Console()

# Global state for JSON mode (set by callback)
_json_mode = False
_show_cost = False
_study_path: Path | None = None


def get_json_mode() -> bool:
    """Get current JSON mode state."""
    return _json_mode


def get_study_path() -> Path | None:
    """Get explicitly set study path, or None to auto-detect."""
    return _study_path


def get_study_context():
    """Get study context from --study flag or auto-detection.

    Returns:
        StudyContext for the current study

    Raises:
        FileNotFoundError: If no study folder found
    """
    from .study import get_study_context as _get_study_context

    return _get_study_context(_study_path)


def is_agent_mode() -> bool:
    """Check if CLI is in agent mode (from config).

    Agent mode means:
    - JSON output instead of rich terminal formatting
    - Exit codes for structured error handling
    - No interactive prompts (clarifications return exit code 2)
    """
    from ..config import get_config

    return get_config().cli.mode == "agent"


def _version_callback(value: bool) -> None:
    if value:
        from .. import __version__

        print(f"extropy {__version__}")
        raise typer.Exit()


def _print_cost_footer() -> None:
    """Print cost summary footer at CLI exit (if enabled and there are records)."""
    try:
        from ..core.cost.tracker import CostTracker
        from ..core.cost.ledger import record_session

        tracker = CostTracker.get()
        if not tracker.has_records:
            return

        # Persist to ledger
        summary = tracker.summary()
        record_session(summary)

        # Print footer
        line = tracker.summary_line()
        if line:
            console.print(f"\n[dim]Cost: {line}[/dim]")
    except Exception:
        pass  # Never let cost display crash the CLI


@app.callback()
def main_callback(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output machine-readable JSON instead of human-friendly text",
            is_eager=True,
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    cost: Annotated[
        bool,
        typer.Option(
            "--cost",
            help="Show cost summary after command completes",
            is_eager=True,
        ),
    ] = False,
    study: Annotated[
        Path | None,
        typer.Option(
            "--study",
            help="Study folder path (auto-detected from cwd if not specified)",
            is_eager=True,
        ),
    ] = None,
):
    """Extropy: Population simulation engine for agent-based modeling.

    Use --json for machine-readable output suitable for scripting and AI tools.
    Use --cost to show token usage and cost summary after each command.
    Use --study to specify a study folder (otherwise auto-detected from cwd).
    """
    global _json_mode, _show_cost, _study_path
    _json_mode = json_output
    _study_path = study

    # Determine if cost footer should be shown: --cost flag or config setting
    show = cost
    if not show:
        try:
            from ..config import get_config

            show = get_config().show_cost
        except Exception:
            pass

    _show_cost = show
    if _show_cost:
        atexit.register(_print_cost_footer)


# Import commands to register them with the app
from .commands import (  # noqa: E402, F401
    validate,
    extend,
    spec,
    sample,
    network,
    persona,
    scenario,
    simulate,
    estimate,
    results,
    config_cmd,
    inspect,
    query,
    report,
    export,
    chat,
    migrate,
)
