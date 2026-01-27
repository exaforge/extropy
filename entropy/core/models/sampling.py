"""Sampling-related models.

Contains models for sampling results and statistics.
"""

from typing import Any

from pydantic import BaseModel, Field


class SamplingStats(BaseModel):
    """Statistics collected during sampling."""

    # Attribute-level stats
    attribute_means: dict[str, float] = Field(default_factory=dict)
    attribute_stds: dict[str, float] = Field(default_factory=dict)
    categorical_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    boolean_counts: dict[str, dict[bool, int]] = Field(default_factory=dict)

    # Modifier trigger counts: attr_name -> modifier_index -> count
    modifier_triggers: dict[str, dict[int, int]] = Field(default_factory=dict)

    # Constraint violations (expression constraints)
    constraint_violations: dict[str, int] = Field(default_factory=dict)

    # Condition evaluation warnings
    condition_warnings: list[str] = Field(default_factory=list)


class SamplingResult(BaseModel):
    """Result of sampling a population."""

    agents: list[dict[str, Any]]
    meta: dict[str, Any]
    stats: SamplingStats
