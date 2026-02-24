"""Name generation for synthetic agents.

Generates culturally plausible first + last names based on
gender, ethnicity, and birth decade using Faker locale routing.
Bundled CSV registries are retained as a deterministic fallback.
"""

from .generator import generate_name

__all__ = ["generate_name"]
