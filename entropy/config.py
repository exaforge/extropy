"""Configuration management for Entropy.

Includes:
- Settings: Environment-based configuration
- ModelProfile: Per-task model configuration
- PhaseProfiles: Phase-specific model defaults
"""

from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Model Profile Configuration
# =============================================================================


class ModelProfile(BaseModel):
    """Configuration for a specific workflow task.

    Defines which models to use for different types of LLM calls,
    along with reasoning effort level.

    Attributes:
        fast: Model for simple_call (fast, no reasoning)
        reasoning: Model for reasoning_call (with reasoning, no web search)
        research: Model for agentic_research (with reasoning + web search)
        reasoning_effort: Effort level for reasoning models
    """

    fast: str = Field(
        default="openai/gpt-4o-mini",
        description="Model for simple_call",
    )
    reasoning: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model for reasoning_call",
    )
    research: str = Field(
        default="openai/gpt-4o",
        description="Model for agentic_research",
    )
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Reasoning effort level",
    )


class PhaseProfiles(BaseModel):
    """Model profiles organized by workflow phase.

    Allows different model configurations for each phase of the
    Entropy workflow.

    Attributes:
        population: Profile for population creation (Phase 1)
        scenario: Profile for scenario injection (Phase 2)
        simulation: Profile for simulation (Phase 3)
    """

    population: ModelProfile = Field(default_factory=ModelProfile)
    scenario: ModelProfile = Field(default_factory=ModelProfile)
    simulation: ModelProfile = Field(default_factory=ModelProfile)


# =============================================================================
# Application Settings
# =============================================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Supports hierarchical model configuration:
    1. Global defaults (DEFAULT_*_MODEL)
    2. Phase defaults (POPULATION_*_MODEL, SCENARIO_*_MODEL, SIMULATION_*_MODEL)
    3. Task overrides (passed directly to LLM functions)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter (primary LLM provider)
    openrouter_api_key: str = ""

    # Legacy OpenAI (for backward compatibility)
    openai_api_key: str = ""

    # Global model defaults
    default_fast_model: str = "openai/gpt-4o-mini"
    default_reasoning_model: str = "anthropic/claude-sonnet-4"
    default_research_model: str = "openai/gpt-4o"

    # Population phase overrides
    population_fast_model: str = ""
    population_reasoning_model: str = ""
    population_research_model: str = ""

    # Scenario phase overrides
    scenario_fast_model: str = ""
    scenario_reasoning_model: str = ""
    scenario_research_model: str = ""

    # Simulation phase overrides
    simulation_fast_model: str = ""
    simulation_reasoning_model: str = ""
    simulation_research_model: str = ""

    # LM Studio (Phase 3 local models)
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "llama-3.2-3b"

    # Database
    db_path: str = "./storage/entropy.db"

    # Defaults
    default_population_size: int = 1000

    @property
    def db_path_resolved(self) -> Path:
        """Resolve database path."""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """Cache directory for research results."""
        path = Path("./data/cache")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_phase_profiles(self) -> PhaseProfiles:
        """Build PhaseProfiles from environment settings.

        Phase-specific settings override global defaults.

        Returns:
            PhaseProfiles with all phases configured
        """
        return PhaseProfiles(
            population=ModelProfile(
                fast=self.population_fast_model or self.default_fast_model,
                reasoning=self.population_reasoning_model or self.default_reasoning_model,
                research=self.population_research_model or self.default_research_model,
            ),
            scenario=ModelProfile(
                fast=self.scenario_fast_model or self.default_fast_model,
                reasoning=self.scenario_reasoning_model or self.default_reasoning_model,
                research=self.scenario_research_model or self.default_research_model,
            ),
            simulation=ModelProfile(
                fast=self.simulation_fast_model or self.default_fast_model,
                reasoning=self.simulation_reasoning_model or self.default_reasoning_model,
                research=self.simulation_research_model or self.default_research_model,
            ),
        )

    def get_default_profile(self) -> ModelProfile:
        """Get the default model profile (global defaults).

        Returns:
            ModelProfile with global default models
        """
        return ModelProfile(
            fast=self.default_fast_model,
            reasoning=self.default_reasoning_model,
            research=self.default_research_model,
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_profile(phase: str | None = None) -> ModelProfile:
    """Get model profile for a specific phase or global default.

    Args:
        phase: Phase name ("population", "scenario", "simulation") or None

    Returns:
        ModelProfile for the specified phase or global default
    """
    settings = get_settings()

    if phase is None:
        return settings.get_default_profile()

    profiles = settings.get_phase_profiles()
    phase_map = {
        "population": profiles.population,
        "scenario": profiles.scenario,
        "simulation": profiles.simulation,
    }

    return phase_map.get(phase, settings.get_default_profile())
