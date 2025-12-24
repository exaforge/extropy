"""Sampler module for Entropy population generation.

The sampler is a generic spec interpreter that generates agents from a
PopulationSpec. It doesn't know about surgeons or farmers - it just
executes whatever spec it's given.

Usage:
    from entropy.sampler import sample_population, save_json, save_sqlite

    result = sample_population(spec, count=500, seed=42)
    save_json(result, "agents.json")
    # or
    save_sqlite(result, "agents.db")

Pipeline Position:
    1. entropy spec → base spec (surgeons.yaml)
    2. entropy overlay → merged spec (surgeons_ai.yaml) — optional
    3. entropy sample → agents from whichever spec you provide
"""

from .core import (
    sample_population,
    save_json,
    save_sqlite,
    SamplingError,
    SamplingResult,
    SamplingStats,
)
from .eval_safe import (
    eval_safe,
    eval_formula,
    eval_condition,
    FormulaError,
    ConditionError,
)
from .distributions import sample_distribution, coerce_to_type
from .modifiers import apply_modifiers_and_sample

__all__ = [
    # Core functions
    "sample_population",
    "save_json",
    "save_sqlite",
    # Result types
    "SamplingResult",
    "SamplingStats",
    "SamplingError",
    # Evaluation utilities
    "eval_safe",
    "eval_formula",
    "eval_condition",
    "FormulaError",
    "ConditionError",
    # Lower-level functions (for testing/extension)
    "sample_distribution",
    "coerce_to_type",
    "apply_modifiers_and_sample",
]
