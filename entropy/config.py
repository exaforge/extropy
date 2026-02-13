"""Configuration management for Entropy.

Two-zone config system:
- pipeline: provider + models for phases 1-2 (spec, extend, sample, network, persona, scenario)
- simulation: provider + model for phase 3 (agent reasoning)

Config resolution order (highest priority first):
1. Programmatic (EntropyConfig constructed in code)
2. Environment variables (PIPELINE_PROVIDER, SIMULATION_MODEL, etc.)
3. Config file (~/.config/entropy/config.json, managed by `entropy config`)
4. Hardcoded defaults

API keys are ALWAYS from env vars — never stored in config file.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# =============================================================================
# Config file location
# =============================================================================

CONFIG_DIR = Path.home() / ".config" / "entropy"
CONFIG_FILE = CONFIG_DIR / "config.json"


# =============================================================================
# Two-zone config dataclasses
# =============================================================================


@dataclass
class PipelineConfig:
    """Config for phases 1-2: spec, extend, sample, network, persona, scenario."""

    provider: str = "openai"
    model_simple: str = ""  # empty = provider default
    model_reasoning: str = ""  # empty = provider default
    model_research: str = ""  # empty = provider default


@dataclass
class SimZoneConfig:
    """Config for phase 3: agent reasoning during simulation."""

    provider: str = "openai"
    model: str = ""  # empty = provider default
    pivotal_model: str = ""  # model for pivotal reasoning (default: same as model)
    routine_model: str = (
        ""  # cheap model for classification (default: provider cheap tier)
    )
    max_concurrent: int = 50
    rate_tier: int | None = None  # rate limit tier (1-4, None = Tier 1)
    rpm_override: int | None = None  # override RPM limit
    tpm_override: int | None = None  # override TPM limit
    api_format: str = (
        ""  # empty = auto (responses for openai, chat_completions for azure)
    )


@dataclass
class EntropyConfig:
    """Top-level entropy configuration.

    Construct programmatically for package use, or load from config file for CLI use.

    Examples:
        # Package use — no files needed
        config = EntropyConfig(
            pipeline=PipelineConfig(provider="claude"),
            simulation=SimZoneConfig(provider="openai", model="gpt-5-mini"),
        )

        # CLI use — loads from ~/.config/entropy/config.json
        config = EntropyConfig.load()

        # Override just simulation
        config = EntropyConfig.load()
        config.simulation.model = "gpt-5-nano"
    """

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    simulation: SimZoneConfig = field(default_factory=SimZoneConfig)

    # Non-zone settings
    db_path: str = "./storage/entropy.db"
    default_population_size: int = 1000

    @classmethod
    def load(cls) -> "EntropyConfig":
        """Load config from file + env vars.

        Priority: env var values > config.json values > defaults.
        """
        config = cls()

        # Layer 1: Load from config file if it exists
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                _apply_dict(config, data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load config from %s: %s", CONFIG_FILE, exc)

        # Layer 2: Env var overrides
        if provider := os.environ.get("LLM_PROVIDER"):
            # Legacy: single provider applied to both zones
            config.pipeline.provider = provider
            config.simulation.provider = provider
        if val := os.environ.get("PIPELINE_PROVIDER"):
            config.pipeline.provider = val
        if val := os.environ.get("SIMULATION_PROVIDER"):
            config.simulation.provider = val
        if val := os.environ.get("MODEL_SIMPLE"):
            config.pipeline.model_simple = val
        if val := os.environ.get("MODEL_REASONING"):
            config.pipeline.model_reasoning = val
        if val := os.environ.get("MODEL_RESEARCH"):
            config.pipeline.model_research = val
        if val := os.environ.get("SIMULATION_MODEL"):
            config.simulation.model = val
        if val := os.environ.get("SIMULATION_PIVOTAL_MODEL"):
            config.simulation.pivotal_model = val
        if val := os.environ.get("SIMULATION_ROUTINE_MODEL"):
            config.simulation.routine_model = val
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
        if val := os.environ.get("SIMULATION_API_FORMAT"):
            config.simulation.api_format = val
        if val := os.environ.get("DB_PATH"):
            config.db_path = val
        if val := os.environ.get("DEFAULT_POPULATION_SIZE"):
            try:
                config.default_population_size = int(val)
            except ValueError:
                logger.warning("Invalid DEFAULT_POPULATION_SIZE=%r, ignoring", val)

        return config

    def save(self) -> None:
        """Save config to ~/.config/entropy/config.json."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Don't persist non-zone settings that are better as env vars
        data.pop("db_path", None)
        data.pop("default_population_size", None)
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for display."""
        return asdict(self)

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


def _apply_dict(config: EntropyConfig, data: dict) -> None:
    """Apply a dict of values onto an EntropyConfig."""
    if "pipeline" in data and isinstance(data["pipeline"], dict):
        for k, v in data["pipeline"].items():
            if hasattr(config.pipeline, k):
                setattr(config.pipeline, k, v)
    if "simulation" in data and isinstance(data["simulation"], dict):
        for k, v in data["simulation"].items():
            if hasattr(config.simulation, k):
                setattr(config.simulation, k, v)
    if "db_path" in data:
        config.db_path = data["db_path"]
    if "default_population_size" in data:
        config.default_population_size = int(data["default_population_size"])


# =============================================================================
# API key resolution (env vars + .env file)
# =============================================================================

_dotenv_loaded = False


def _ensure_dotenv() -> None:
    """Load .env file into os.environ if not already loaded."""
    global _dotenv_loaded
    if not _dotenv_loaded:
        _dotenv_loaded = True
        try:
            from dotenv import find_dotenv, load_dotenv

            # Resolve from current working directory first so CLI commands run
            # from study repos consistently pick up that repo's `.env`.
            dotenv_path = find_dotenv(usecwd=True)
            if dotenv_path:
                load_dotenv(dotenv_path=dotenv_path, override=False)
            else:
                # Fallback for environments where no discoverable .env exists.
                load_dotenv(override=False)
        except ImportError:
            pass  # python-dotenv not installed, skip
        except Exception:
            # Keep config loading resilient even if dotenv discovery has runtime issues.
            pass


def get_api_key(provider: str) -> str:
    """Get API key for a provider from environment variables or .env file.

    Supports:
        - openai: OPENAI_API_KEY
        - claude: ANTHROPIC_API_KEY
        - azure_openai: AZURE_OPENAI_API_KEY

    Returns empty string if not found (providers will raise on missing keys).
    """
    _ensure_dotenv()
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY", "")
    elif provider == "claude":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    elif provider == "azure_openai":
        return os.environ.get("AZURE_OPENAI_API_KEY", "")
    return ""


def get_azure_config(provider: str) -> dict[str, str]:
    """Get Azure-specific configuration from environment variables.

    Args:
        provider: 'azure_openai'

    Returns:
        Dict of Azure config values (endpoint, api_version, deployment).
    """
    _ensure_dotenv()
    if provider == "azure_openai":
        return {
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2025-03-01-preview"
            ),
            "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        }
    return {}


# =============================================================================
# Global config singleton
# =============================================================================

_config: EntropyConfig | None = None


def get_config() -> EntropyConfig:
    """Get the global EntropyConfig instance.

    First call loads from file + env vars. Subsequent calls return cached instance.
    Use configure() to replace the global config programmatically.
    """
    global _config
    if _config is None:
        _config = EntropyConfig.load()
    return _config


def configure(config: EntropyConfig) -> None:
    """Set the global EntropyConfig programmatically.

    Use this when entropy is used as a package:
        from entropy.config import configure, EntropyConfig, PipelineConfig
        configure(EntropyConfig(pipeline=PipelineConfig(provider="claude")))
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config (forces reload on next get_config())."""
    global _config
    _config = None
