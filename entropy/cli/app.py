"""Core CLI app definition and global state."""

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="entropy",
    help="Generate population specs for agent-based simulation.",
    no_args_is_help=True,
)

console = Console()

# Global state for JSON mode (set by callback)
_json_mode = False


def get_json_mode() -> bool:
    """Get current JSON mode state."""
    return _json_mode


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
):
    """Entropy: Population simulation engine for agent-based modeling.

    Use --json for machine-readable output suitable for scripting and AI tools.
    """
    global _json_mode
    _json_mode = json_output


# Import commands to register them with the app
from .commands import (  # noqa: E402, F401
    validate,
    overlay,
    spec,
    sample,
    network,
    scenario,
    validate_scenario,
    simulate,
    results,
)
