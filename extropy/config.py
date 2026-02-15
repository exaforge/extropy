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
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


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
# New two-tier config dataclasses
# =============================================================================


@dataclass
class ModelsConfig:
    """Pipeline model configuration (phases 1-2).

    Uses "provider/model" format strings.
    - fast: used for simple_call (cheap, fast tasks)
    - strong: used for reasoning_call, agentic_research (complex tasks)
    """

    fast: str = "openai/gpt-5-mini"
    strong: str = "openai/gpt-5"


@dataclass
class SimulationConfig:
    """Simulation model + tuning configuration (phase 3).

    Uses "provider/model" format strings.
    - fast: used for Pass 2 (classification/routine)
    - strong: used for Pass 1 (pivotal/role-play reasoning)
    """

    fast: str = ""  # empty = same as models.fast
    strong: str = ""  # empty = same as models.strong
    max_concurrent: int = 50
    rate_tier: int | None = None
    rpm_override: int | None = None
    tpm_override: int | None = None


@dataclass
class CustomProviderConfig:
    """Configuration for a custom OpenAI-compatible provider endpoint."""

    base_url: str = ""
    api_key_env: str = ""


@dataclass
class DefaultsConfig:
    """Non-zone default settings."""

    population_size: int = 1000
    db_path: str = "./storage/extropy.db"
    show_cost: bool = False  # Show cost footer after every CLI command


# =============================================================================
# Legacy config dataclasses (kept for migration)
# =============================================================================


@dataclass
class PipelineConfig:
    """DEPRECATED: Config for phases 1-2. Use ModelsConfig instead."""

    provider: str = "openai"
    model_simple: str = ""
    model_reasoning: str = ""
    model_research: str = ""


@dataclass
class SimZoneConfig:
    """DEPRECATED: Config for phase 3. Use SimulationConfig instead."""

    provider: str = "openai"
    model: str = ""
    pivotal_model: str = ""
    routine_model: str = ""
    max_concurrent: int = 50
    rate_tier: int | None = None
    rpm_override: int | None = None
    tpm_override: int | None = None
    api_format: str = ""


# =============================================================================
# Main config class
# =============================================================================


@dataclass
class ExtropyConfig:
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

    models: ModelsConfig = field(default_factory=ModelsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    providers: dict[str, CustomProviderConfig] = field(default_factory=dict)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)

    @classmethod
    def load(cls) -> "ExtropyConfig":
        """Load config from file + env vars.

        Priority: env var values > config.json values > defaults.
        Auto-migrates v1 config format if detected.
        """
        config = cls()

        # Layer 1: Load from config file if it exists
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)

                # Auto-migrate v1 config
                if _is_v1_config(data):
                    warnings.warn(
                        "Detected legacy config format. Migrating to v2. "
                        "Run `extropy config show` to verify, then `extropy config set` to update.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    data = _migrate_v1_to_v2(data)

                _apply_dict(config, data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load config from %s: %s", CONFIG_FILE, exc)

        # Layer 2: Env var overrides (new format)
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
        if val := os.environ.get("DB_PATH"):
            config.defaults.db_path = val
        if val := os.environ.get("DEFAULT_POPULATION_SIZE"):
            try:
                config.defaults.population_size = int(val)
            except ValueError:
                logger.warning("Invalid DEFAULT_POPULATION_SIZE=%r, ignoring", val)

        # Layer 3: Legacy env var overrides (emit deprecation warnings)
        _apply_legacy_env_vars(config)

        return config

    def save(self) -> None:
        """Save config to ~/.config/extropy/config.json."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "models": asdict(self.models),
            "simulation": asdict(self.simulation),
        }
        if self.providers:
            data["providers"] = {
                name: asdict(cfg) for name, cfg in self.providers.items()
            }
        if self.defaults != DefaultsConfig():
            data["defaults"] = asdict(self.defaults)
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for display."""
        result = {
            "models": asdict(self.models),
            "simulation": asdict(self.simulation),
            "defaults": asdict(self.defaults),
        }
        if self.providers:
            result["providers"] = {
                name: asdict(cfg) for name, cfg in self.providers.items()
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
    def db_path(self) -> str:
        return self.defaults.db_path

    @db_path.setter
    def db_path(self, value: str) -> None:
        self.defaults.db_path = value

    @property
    def default_population_size(self) -> int:
        return self.defaults.population_size

    @default_population_size.setter
    def default_population_size(self, value: int) -> None:
        self.defaults.population_size = value

    @property
    def db_path_resolved(self) -> Path:
        """Resolve database path."""
        path = Path(self.defaults.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

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
                setattr(config.defaults, k, v)
    # Backward compat: top-level db_path / default_population_size
    if "db_path" in data:
        config.defaults.db_path = data["db_path"]
    if "default_population_size" in data:
        config.defaults.population_size = int(data["default_population_size"])


# =============================================================================
# V1 → V2 migration
# =============================================================================

# Provider name mapping for migration
_PROVIDER_CANONICAL = {
    "openai": "openai",
    "claude": "anthropic",
    "anthropic": "anthropic",
    "azure_openai": "azure",
}

# Default model names per old provider
_V1_PROVIDER_DEFAULTS = {
    "openai": {"fast": "gpt-5-mini", "strong": "gpt-5"},
    "claude": {
        "fast": "claude-haiku-4-5-20251001",
        "strong": "claude-sonnet-4-5-20250929",
    },
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "strong": "claude-sonnet-4-5-20250929",
    },
    "azure_openai": {"fast": "gpt-5-mini", "strong": "gpt-5"},
}


def _is_v1_config(data: dict) -> bool:
    """Detect if config data is in v1 format (has 'pipeline' key)."""
    return "pipeline" in data and "models" not in data


def _migrate_v1_to_v2(data: dict) -> dict:
    """Convert v1 config format to v2.

    v1 format:
        {"pipeline": {"provider": "openai", "model_simple": "...", ...},
         "simulation": {"provider": "openai", "model": "...", ...}}

    v2 format:
        {"models": {"fast": "openai/gpt-5-mini", "strong": "openai/gpt-5"},
         "simulation": {"fast": "...", "strong": "...", ...}}
    """
    result: dict[str, Any] = {}

    # Migrate pipeline → models
    pipeline = data.get("pipeline", {})
    old_provider = pipeline.get("provider", "openai")
    canonical = _PROVIDER_CANONICAL.get(old_provider, old_provider)
    defaults = _V1_PROVIDER_DEFAULTS.get(old_provider, _V1_PROVIDER_DEFAULTS["openai"])

    fast_model = pipeline.get("model_simple") or defaults["fast"]
    strong_model = pipeline.get("model_reasoning") or defaults["strong"]

    result["models"] = {
        "fast": f"{canonical}/{fast_model}",
        "strong": f"{canonical}/{strong_model}",
    }

    # Migrate simulation
    sim = data.get("simulation", {})
    sim_provider = sim.get("provider", "openai")
    sim_canonical = _PROVIDER_CANONICAL.get(sim_provider, sim_provider)
    sim_result: dict[str, Any] = {}

    # Map model/pivotal_model → strong, routine_model → fast
    pivotal = sim.get("pivotal_model") or sim.get("model") or ""
    routine = sim.get("routine_model") or ""

    if pivotal:
        sim_result["strong"] = f"{sim_canonical}/{pivotal}"
    if routine:
        sim_result["fast"] = f"{sim_canonical}/{routine}"

    for k in ("max_concurrent", "rate_tier", "rpm_override", "tpm_override"):
        if k in sim and sim[k] is not None:
            sim_result[k] = sim[k]

    result["simulation"] = sim_result

    # Carry forward non-zone settings
    if "db_path" in data:
        result.setdefault("defaults", {})["db_path"] = data["db_path"]
    if "default_population_size" in data:
        result.setdefault("defaults", {})["population_size"] = data[
            "default_population_size"
        ]

    return result


# =============================================================================
# Legacy env var handling
# =============================================================================

_LEGACY_ENV_WARNED: set[str] = set()


def _warn_legacy_env(name: str, replacement: str) -> None:
    """Emit a one-time deprecation warning for a legacy env var."""
    if name not in _LEGACY_ENV_WARNED:
        _LEGACY_ENV_WARNED.add(name)
        warnings.warn(
            f"Environment variable {name} is deprecated. Use {replacement} instead.",
            DeprecationWarning,
            stacklevel=4,
        )


def _apply_legacy_env_vars(config: ExtropyConfig) -> None:
    """Apply legacy env vars with deprecation warnings."""
    # LLM_PROVIDER → both zones
    if val := os.environ.get("LLM_PROVIDER"):
        _warn_legacy_env("LLM_PROVIDER", "MODELS_FAST / MODELS_STRONG")
        canonical = _PROVIDER_CANONICAL.get(val, val)
        defaults = _V1_PROVIDER_DEFAULTS.get(val, _V1_PROVIDER_DEFAULTS["openai"])
        # Only override if no new-format env vars set
        if not os.environ.get("MODELS_FAST"):
            config.models.fast = f"{canonical}/{defaults['fast']}"
        if not os.environ.get("MODELS_STRONG"):
            config.models.strong = f"{canonical}/{defaults['strong']}"

    if val := os.environ.get("PIPELINE_PROVIDER"):
        _warn_legacy_env("PIPELINE_PROVIDER", "MODELS_FAST / MODELS_STRONG")
        canonical = _PROVIDER_CANONICAL.get(val, val)
        defaults = _V1_PROVIDER_DEFAULTS.get(val, _V1_PROVIDER_DEFAULTS["openai"])
        if not os.environ.get("MODELS_FAST"):
            config.models.fast = f"{canonical}/{defaults['fast']}"
        if not os.environ.get("MODELS_STRONG"):
            config.models.strong = f"{canonical}/{defaults['strong']}"

    if val := os.environ.get("SIMULATION_PROVIDER"):
        _warn_legacy_env("SIMULATION_PROVIDER", "SIMULATION_FAST / SIMULATION_STRONG")
        canonical = _PROVIDER_CANONICAL.get(val, val)
        defaults = _V1_PROVIDER_DEFAULTS.get(val, _V1_PROVIDER_DEFAULTS["openai"])
        if not os.environ.get("SIMULATION_FAST"):
            config.simulation.fast = f"{canonical}/{defaults['fast']}"
        if not os.environ.get("SIMULATION_STRONG"):
            config.simulation.strong = f"{canonical}/{defaults['strong']}"

    if val := os.environ.get("MODEL_SIMPLE"):
        _warn_legacy_env("MODEL_SIMPLE", "MODELS_FAST")
        if not os.environ.get("MODELS_FAST"):
            provider, _ = parse_model_string(config.models.fast)
            config.models.fast = f"{provider}/{val}"

    if val := os.environ.get("MODEL_REASONING"):
        _warn_legacy_env("MODEL_REASONING", "MODELS_STRONG")
        if not os.environ.get("MODELS_STRONG"):
            provider, _ = parse_model_string(config.models.strong)
            config.models.strong = f"{provider}/{val}"

    if val := os.environ.get("SIMULATION_MODEL"):
        _warn_legacy_env("SIMULATION_MODEL", "SIMULATION_STRONG")
        if not os.environ.get("SIMULATION_STRONG"):
            # Resolve provider from sim strong or models strong
            base = config.simulation.strong or config.models.strong
            provider, _ = parse_model_string(base)
            config.simulation.strong = f"{provider}/{val}"

    if val := os.environ.get("SIMULATION_PIVOTAL_MODEL"):
        _warn_legacy_env("SIMULATION_PIVOTAL_MODEL", "SIMULATION_STRONG")
        if not os.environ.get("SIMULATION_STRONG"):
            base = config.simulation.strong or config.models.strong
            provider, _ = parse_model_string(base)
            config.simulation.strong = f"{provider}/{val}"

    if val := os.environ.get("SIMULATION_ROUTINE_MODEL"):
        _warn_legacy_env("SIMULATION_ROUTINE_MODEL", "SIMULATION_FAST")
        if not os.environ.get("SIMULATION_FAST"):
            base = config.simulation.fast or config.models.fast
            provider, _ = parse_model_string(base)
            config.simulation.fast = f"{provider}/{val}"

    # SIMULATION_API_FORMAT — no direct replacement, just warn
    if os.environ.get("SIMULATION_API_FORMAT"):
        _warn_legacy_env(
            "SIMULATION_API_FORMAT",
            "provider-based routing (api_format is now automatic)",
        )


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


def get_api_key(provider: str) -> str:
    """DEPRECATED: Get API key for a provider. Use get_api_key_for_provider instead.

    Kept for backward compatibility.
    """
    # Map old provider names
    mapping = {
        "claude": "anthropic",
        "azure_openai": "azure",
    }
    canonical = mapping.get(provider, provider)
    return get_api_key_for_provider(canonical)


def get_azure_config(provider: str) -> dict[str, str]:
    """DEPRECATED: Get Azure-specific configuration.

    Azure is now handled as an OpenAI-compatible provider.
    """
    _ensure_dotenv()
    if provider in ("azure_openai", "azure"):
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
