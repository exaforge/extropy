"""Name generation for synthetic agents.

Generates culturally plausible first + last names based on
gender, ethnicity, and birth decade. US-only for now.
"""

from .generator import generate_name

__all__ = ["generate_name"]
