"""Configuration management for Extropy.

Two-tier config system:
- models: fast/strong model strings for pipeline phases 1-2
- simulation: fast/strong model strings for phase 3 (agent reasoning)

Model strings use "provider/model" format (e.g., "openai/gpt-5-mini").

Config resolution order (highest priority first):
1. Programmatic (ExtropyConfig constructed in code)
2. Environment variables (MODELS_FAST, MODELS_STRONG, etc.)
3. Config file (~/.config/extropy/config.json, managed by `extropy config`)
4. Hardcoded defaults

API keys are ALWAYS from env vars — never stored in config file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Config file location
# =============================================================================

CONFIG_DIR = Path.home() / ".config" / "extropy"
CONFIG_FILE = CONFIG_DIR / "config.json"


# =============================================================================
# Model string parsing
# =============================================================================


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parse a "provider/model" string into (provider, model) tuple.

    Examples:
        "openai/gpt-5-mini" → ("openai", "gpt-5-mini")
        "anthropic/claude-sonnet-4.5" → ("anthropic", "claude-sonnet-4.5")
        "openrouter/anthropic/claude-sonnet-4.5" → ("openrouter", "anthropic/claude-sonnet-4.5")

    Raises:
        ValueError: If the string doesn't contain a '/' separator.
    """
    if "/" not in model_string:
        raise ValueError(
            f"Invalid model string: {model_string!r}. "
            f"Expected format: 'provider/model' (e.g., 'openai/gpt-5-mini')"
        )
    provider, _, model = model_string.partition("/")
    if not provider or not model:
        raise ValueError(
            f"Invalid model string: {model_string!r}. "
            f"Both provider and model must be non-empty."
        )
    return provider, model


# =============================================================================
# Two-tier config models
# =============================================================================


class ModelsConfig(BaseModel):
    """Pipeline model configuration (phases 1-2).

    Uses "provider/model" format strings.
    - fast: used for simple_call (cheap, fast tasks)
    - strong: used for reasoning_call, agentic_research (complex tasks)
    """

    model_config = ConfigDict(populate_by_name=True)

    fast: str = "openai/gpt-5-mini"
    strong: str = "openai/gpt-5"


class SimulationConfig(BaseModel):
    """Simulation model + tuning configuration (phase 3).

    Uses "provider/model" format strings.
    - fast: used for Pass 2 (classification/routine)
    - strong: used for Pass 1 (pivotal/role-play reasoning)
    """

    model_config = ConfigDict(populate_by_name=True)

    fast: str = ""  # empty = same as models.fast
    strong: str = ""  # empty = same as models.strong
    max_concurrent: int = 50
    rate_tier: int | None = None
    rpm_override: int | None = None
    tpm_override: int | None = None


class CustomProviderConfig(BaseModel):
    """Config for a custom OpenAI-compatible provider endpoint."""

    base_url: str = ""
    api_key_env: str = ""


class DefaultsConfig(BaseModel):
    """Non-zone default settings."""

    population_size: int = 1000
    db_path: str = "./storage/extropy.db"
    show_cost: bool = False  # Show cost footer after every CLI command


# =============================================================================
# Main config class
# =============================================================================


class ExtropyConfig(BaseModel):
    """Top-level extropy configuration.

    Construct programmatically for package use, or load from config file for CLI use.

    Examples:
        # Package use — no files needed
        config = ExtropyConfig(
            models=ModelsConfig(fast="openai/gpt-5-mini", strong="anthropic/claude-sonnet-4.5"),
        )

        # CLI use — loads from ~/.config/extropy/config.json
        config = ExtropyConfig.load()

        # Override just simulation
        config = ExtropyConfig.load()
        config.simulation.strong = "openrouter/anthropic/claude-sonnet-4.5"
    """

    model_config = ConfigDict(populate_by_name=True)

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    providers: dict[str, CustomProviderConfig] = Field(default_factory=dict)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    @classmethod
    def load(cls) -> "ExtropyConfig":
        """Load config from file + env vars.

        Priority: env var values > config.json values > defaults.
        """
        config = cls()

        # Load from config file if it exists
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                _apply_dict(config, data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load config from %s: %s", CONFIG_FILE, exc)

        # Env var overrides
        if val := os.environ.get("MODELS_FAST"):
            config.models.fast = val
        if val := os.environ.get("MODELS_STRONG"):
            config.models.strong = val
        if val := os.environ.get("SIMULATION_FAST"):
            config.simulation.fast = val
        if val := os.environ.get("SIMULATION_STRONG"):
            config.simulation.strong = val
        if val := os.environ.get("SIMULATION_MAX_CONCURRENT"):
            try:
                config.simulation.max_concurrent = int(val)
            except ValueError:
                logger.warning("Invalid SIMULATION_MAX_CONCURRENT=%r, ignoring", val)
        if val := os.environ.get("SIMULATION_RATE_TIER"):
            try:
                config.simulation.rate_tier = int(val)
            except ValueError:
                logger.warning("Invalid SIMULATION_RATE_TIER=%r, ignoring", val)
        if val := os.environ.get("SIMULATION_RPM_OVERRIDE"):
            try:
                config.simulation.rpm_override = int(val)
            except ValueError:
                logger.warning("Invalid SIMULATION_RPM_OVERRIDE=%r, ignoring", val)
        if val := os.environ.get("SIMULATION_TPM_OVERRIDE"):
            try:
                config.simulation.tpm_override = int(val)
            except ValueError:
                logger.warning("Invalid SIMULATION_TPM_OVERRIDE=%r, ignoring", val)

        return config

    def save(self) -> None:
        """Save config to ~/.config/extropy/config.json."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "models": self.models.model_dump(),
            "simulation": self.simulation.model_dump(),
        }
        if self.providers:
            data["providers"] = {
                name: cfg.model_dump() for name, cfg in self.providers.items()
            }
        if self.defaults != DefaultsConfig():
            data["defaults"] = self.defaults.model_dump()
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for display."""
        result = {
            "models": self.models.model_dump(),
            "simulation": self.simulation.model_dump(),
            "defaults": self.defaults.model_dump(),
        }
        if self.providers:
            result["providers"] = {
                name: cfg.model_dump() for name, cfg in self.providers.items()
            }
        return result

    # ── Convenience resolution methods ──

    def resolve_pipeline_fast(self) -> str:
        """Resolve the fast model string for pipeline use."""
        return self.models.fast

    def resolve_pipeline_strong(self) -> str:
        """Resolve the strong model string for pipeline use."""
        return self.models.strong

    def resolve_sim_strong(self) -> str:
        """Resolve the strong model string for simulation."""
        return self.simulation.strong or self.models.strong

    def resolve_sim_fast(self) -> str:
        """Resolve the fast model string for simulation."""
        return self.simulation.fast or self.models.fast

    # ── Backward compat properties ──

    @property
    def cache_dir(self) -> Path:
        """Cache directory for research results."""
        path = Path("./data/cache")
        path.mkdir(parents=True, exist_ok=True)
        return path


# =============================================================================
# Config dict application
# =============================================================================


def _apply_dict(config: ExtropyConfig, data: dict) -> None:
    """Apply a dict of values onto an ExtropyConfig (v2 format)."""
    if "models" in data and isinstance(data["models"], dict):
        for k, v in data["models"].items():
            if hasattr(config.models, k):
                setattr(config.models, k, v)
    if "simulation" in data and isinstance(data["simulation"], dict):
        for k, v in data["simulation"].items():
            if hasattr(config.simulation, k):
                setattr(config.simulation, k, v)
    if "providers" in data and isinstance(data["providers"], dict):
        for name, provider_data in data["providers"].items():
            if isinstance(provider_data, dict):
                config.providers[name] = CustomProviderConfig(
                    base_url=provider_data.get("base_url", ""),
                    api_key_env=provider_data.get("api_key_env", ""),
                )
    if "defaults" in data and isinstance(data["defaults"], dict):
        for k, v in data["defaults"].items():
            if hasattr(config.defaults, k):
                if k == "population_size":
                    v = int(v)
                elif k == "show_cost":
                    v = bool(v)
                setattr(config.defaults, k, v)


# =============================================================================
# API key resolution
# =============================================================================

_dotenv_loaded = False


def _ensure_dotenv() -> None:
    """Load .env file into os.environ if not already loaded."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        _dotenv_loaded = True
        try:
            from dotenv import find_dotenv, load_dotenv

            dotenv_path = find_dotenv(usecwd=True)
            if dotenv_path:
                load_dotenv(dotenv_path=dotenv_path, override=False)
            else:
                load_dotenv(override=False)
        except ImportError:
            pass
        except Exception:
            pass


def get_api_key_for_provider(
    provider_name: str,
    custom_providers: dict[str, CustomProviderConfig] | None = None,
) -> str:
    """Get API key for a provider.

    Resolution order:
    1. Custom provider api_key_env override
    2. Convention: {PROVIDER_UPPER}_API_KEY

    Special cases:
    - "anthropic" → ANTHROPIC_API_KEY
    - "azure" → AZURE_OPENAI_API_KEY

    Returns empty string if not found.
    """
    _ensure_dotenv()

    # Check custom provider override first
    if custom_providers and provider_name in custom_providers:
        custom = custom_providers[provider_name]
        if custom.api_key_env:
            return os.environ.get(custom.api_key_env, "")

    # Convention: {PROVIDER}_API_KEY
    # Special cases for backward compat
    key_map = {
        "azure": "AZURE_OPENAI_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
    }
    env_var = key_map.get(provider_name, f"{provider_name.upper()}_API_KEY")
    return os.environ.get(env_var, "")


# =============================================================================
# Global config singleton
# =============================================================================

_config: ExtropyConfig | None = None


def get_config() -> ExtropyConfig:
    """Get the global ExtropyConfig instance.

    First call loads from file + env vars. Subsequent calls return cached instance.
    Use configure() to replace the global config programmatically.
    """
    global _config
    if _config is None:
        _config = ExtropyConfig.load()
    return _config


def configure(config: ExtropyConfig) -> None:
    """Set the global ExtropyConfig programmatically.

    Use this when extropy is used as a package:
        from extropy.config import configure, ExtropyConfig, ModelsConfig
        configure(ExtropyConfig(models=ModelsConfig(fast="openai/gpt-5-mini")))
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config (forces reload on next get_config())."""
    global _config
    _config = None
