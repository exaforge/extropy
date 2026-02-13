"""CLI package for Extropy.

Supports dual-mode output:
- Human mode (default): Rich formatting with colors, tables, progress bars
- Machine mode (--json): Structured JSON output for AI coding tools

Exit codes:
    0 = Success
    1 = Validation error
    2 = File not found
    3 = Sampling error
    4 = Network error
    5 = Simulation error
    6 = Scenario error
    10 = User cancelled
"""

from .app import app

# Import commands to register them with the app
from . import commands  # noqa: F401

__all__ = ["app"]
