"""Persona generation and rendering for agent embodiment.

This package handles:
1. PersonaConfig generation (LLM-driven, one-time per population)
2. Population statistics computation (mean/std for relative positioning)
3. Persona rendering (template-based, per-agent, no LLM)
"""

from .config import (
    AttributeTreatment,
    TreatmentType,
    RelativeLabels,
    AttributeGroup,
    BooleanPhrasing,
    CategoricalPhrasing,
    RelativePhrasing,
    ConcretePhrasing,
    AttributePhrasing,
    PopulationStats,
    PersonaConfig,
)
from .generator import generate_persona_config, PersonaConfigError
from .renderer import render_persona, render_persona_section, preview_persona
from .stats import compute_population_stats
from .validator import validate_persona_config

__all__ = [
    # Config models
    "AttributeTreatment",
    "TreatmentType",
    "RelativeLabels",
    "AttributeGroup",
    "BooleanPhrasing",
    "CategoricalPhrasing",
    "RelativePhrasing",
    "ConcretePhrasing",
    "AttributePhrasing",
    "PopulationStats",
    "PersonaConfig",
    # Generator
    "generate_persona_config",
    "PersonaConfigError",
    # Renderer
    "render_persona",
    "render_persona_section",
    "preview_persona",
    # Stats
    "compute_population_stats",
    # Validator
    "validate_persona_config",
]
