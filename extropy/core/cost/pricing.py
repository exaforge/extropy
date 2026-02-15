"""Model pricing resolution for cost estimation and tracking.

Three-tier pricing resolution:
1. OpenRouter API (free, no auth, covers 200+ models) → cached locally
2. Local cache file (~/.config/extropy/pricing_cache.json) with 24h TTL
3. Hardcoded fallback table for offline/known models

Provides per-model input/output pricing (USD per million tokens)
and provider default model resolution without needing API keys.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Cache location and TTL
_CACHE_DIR = Path.home() / ".config" / "extropy"
_CACHE_FILE = _CACHE_DIR / "pricing_cache.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


class ModelPricing(BaseModel, frozen=True):
    """Pricing for a single model (USD per million tokens)."""

    input_per_mtok: float
    output_per_mtok: float


# ── Hardcoded fallback (Tier 3) ──────────────────────────────────────────────

# Known model pricing (USD per million tokens)
# Sources: OpenAI and Anthropic pricing pages as of 2025
FALLBACK_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-5": ModelPricing(input_per_mtok=2.50, output_per_mtok=10.00),
    "gpt-5-mini": ModelPricing(input_per_mtok=0.30, output_per_mtok=1.50),
    "gpt-5-nano": ModelPricing(input_per_mtok=0.10, output_per_mtok=0.40),
    "gpt-5.2": ModelPricing(input_per_mtok=2.50, output_per_mtok=10.00),
    # Azure-hosted models
    "DeepSeek-V3.2": ModelPricing(input_per_mtok=0.80, output_per_mtok=2.00),
    "Kimi-K2.5": ModelPricing(input_per_mtok=1.00, output_per_mtok=4.00),
    # Claude
    "claude-sonnet-4-5-20250929": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00
    ),
    "claude-sonnet-4-5-20250514": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00
    ),
    "claude-sonnet-4.5": ModelPricing(input_per_mtok=3.00, output_per_mtok=15.00),
    "claude-sonnet-4": ModelPricing(input_per_mtok=3.00, output_per_mtok=15.00),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_per_mtok=0.80, output_per_mtok=4.00
    ),
    "claude-haiku-4.5": ModelPricing(input_per_mtok=0.80, output_per_mtok=4.00),
    "claude-haiku-4": ModelPricing(input_per_mtok=0.80, output_per_mtok=4.00),
    # DeepSeek (direct API)
    "deepseek-chat": ModelPricing(input_per_mtok=0.14, output_per_mtok=0.28),
    "deepseek-reasoner": ModelPricing(input_per_mtok=0.55, output_per_mtok=2.19),
}

# Provider default models — 2-tier (fast/strong)
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "fast": "gpt-5-mini",
        "strong": "gpt-5",
    },
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "strong": "claude-sonnet-4-5-20250929",
    },
    "azure": {
        "fast": "gpt-5-mini",
        "strong": "gpt-5",
    },
    "openrouter": {
        "fast": "openai/gpt-5-mini",
        "strong": "openai/gpt-5",
    },
    "deepseek": {
        "fast": "deepseek-chat",
        "strong": "deepseek-reasoner",
    },
    "together": {
        "fast": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "strong": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    },
    "groq": {
        "fast": "llama-3.3-70b-versatile",
        "strong": "llama-3.3-70b-versatile",
    },
    # Legacy aliases
    "claude": {
        "fast": "claude-haiku-4-5-20251001",
        "strong": "claude-sonnet-4-5-20250929",
    },
    "azure_openai": {
        "fast": "gpt-5-mini",
        "strong": "gpt-5",
    },
}


# ── In-memory cache ──────────────────────────────────────────────────────────

_memory_cache: dict[str, ModelPricing] = {}
_memory_cache_loaded: bool = False


# ── Tier 1: OpenRouter API ───────────────────────────────────────────────────


def _fetch_openrouter_pricing() -> dict[str, ModelPricing] | None:
    """Fetch pricing from OpenRouter API (no auth required).

    Returns:
        Dict of model_id → ModelPricing, or None if fetch failed.
    """
    try:
        import urllib.request
        import urllib.error

        url = "https://openrouter.ai/api/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "extropy"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        result: dict[str, ModelPricing] = {}
        for model in data.get("data", []):
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})

            # OpenRouter returns pricing as string USD per token (not per MTok)
            prompt_price = pricing.get("prompt")
            completion_price = pricing.get("completion")

            if prompt_price is not None and completion_price is not None:
                try:
                    input_per_tok = float(prompt_price)
                    output_per_tok = float(completion_price)
                except (ValueError, TypeError):
                    continue

                # Skip free/zero-cost models
                if input_per_tok == 0 and output_per_tok == 0:
                    continue

                # Convert per-token to per-million-tokens
                result[model_id] = ModelPricing(
                    input_per_mtok=input_per_tok * 1_000_000,
                    output_per_mtok=output_per_tok * 1_000_000,
                )

        if result:
            logger.debug(f"Fetched pricing for {len(result)} models from OpenRouter")
            return result

    except Exception as e:
        logger.debug(f"Failed to fetch OpenRouter pricing: {e}")

    return None


# ── Tier 2: Local cache file ─────────────────────────────────────────────────


def _load_cache() -> dict[str, ModelPricing] | None:
    """Load pricing from local cache file if it exists and is fresh.

    Returns:
        Dict of model_id → ModelPricing, or None if cache is stale/missing.
    """
    if not _CACHE_FILE.exists():
        return None

    try:
        with open(_CACHE_FILE) as f:
            data = json.load(f)

        # Check TTL
        cached_at = data.get("cached_at", 0)
        if time.time() - cached_at > _CACHE_TTL_SECONDS:
            logger.debug("Pricing cache expired")
            return None

        result: dict[str, ModelPricing] = {}
        for model_id, pricing in data.get("models", {}).items():
            result[model_id] = ModelPricing(
                input_per_mtok=pricing["input_per_mtok"],
                output_per_mtok=pricing["output_per_mtok"],
            )

        logger.debug(f"Loaded {len(result)} models from pricing cache")
        return result

    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.debug(f"Failed to load pricing cache: {e}")
        return None


def _save_cache(pricing: dict[str, ModelPricing]) -> None:
    """Save pricing to local cache file."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "cached_at": time.time(),
            "models": {model_id: p.model_dump() for model_id, p in pricing.items()},
        }
        with open(_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved pricing cache with {len(pricing)} models")
    except OSError as e:
        logger.debug(f"Failed to save pricing cache: {e}")


# ── Resolution logic ─────────────────────────────────────────────────────────


def _ensure_cache_loaded() -> None:
    """Lazily load the pricing cache into memory (once per process)."""
    global _memory_cache, _memory_cache_loaded
    if _memory_cache_loaded:
        return

    # Try local cache first (fast, no network)
    cached = _load_cache()
    if cached:
        _memory_cache = cached
        _memory_cache_loaded = True
        return

    # Try OpenRouter API
    fetched = _fetch_openrouter_pricing()
    if fetched:
        _memory_cache = fetched
        _save_cache(fetched)
        _memory_cache_loaded = True
        return

    # No dynamic pricing available — will fall through to hardcoded
    _memory_cache_loaded = True


def _normalize_model_id(model: str) -> list[str]:
    """Generate candidate lookup keys for a model name.

    Handles the mapping between bare model names (used by providers)
    and OpenRouter-style IDs (provider/model).

    Args:
        model: Model name (e.g., "gpt-5-mini" or "openai/gpt-5-mini")

    Returns:
        List of candidate keys to try, in priority order.
    """
    candidates = [model]

    # If it's already a provider/model format, also try the bare model name
    if "/" in model:
        bare = model.rsplit("/", 1)[-1]
        candidates.append(bare)
    else:
        # Try common provider prefixes for bare model names
        if model.startswith("gpt-"):
            candidates.append(f"openai/{model}")
        elif model.startswith("claude-"):
            candidates.append(f"anthropic/{model}")
        elif model.startswith("deepseek-"):
            candidates.append(f"deepseek/{model}")
        elif model.startswith("llama-") or model.startswith("meta-llama/"):
            candidates.append(f"meta-llama/{model}")

    return candidates


def get_pricing(model: str) -> ModelPricing | None:
    """Get pricing for a model using three-tier resolution.

    Resolution order:
    1. OpenRouter API cache (refreshed every 24h)
    2. Local cache file
    3. Hardcoded fallback table

    Args:
        model: Model name (bare like "gpt-5-mini" or qualified like "openai/gpt-5-mini")

    Returns:
        ModelPricing or None if no pricing found.
    """
    _ensure_cache_loaded()

    candidates = _normalize_model_id(model)

    # Try dynamic cache first
    for candidate in candidates:
        if candidate in _memory_cache:
            return _memory_cache[candidate]

    # Fall back to hardcoded
    for candidate in candidates:
        if candidate in FALLBACK_PRICING:
            return FALLBACK_PRICING[candidate]

    return None


def resolve_default_model(provider: str, tier: str = "strong") -> str:
    """Resolve default model name for a provider without instantiating it.

    Args:
        provider: Provider name ('openai', 'anthropic', etc.)
        tier: 'fast' or 'strong' (also accepts legacy 'simple'/'reasoning')

    Returns:
        Model name string
    """
    # Map legacy tier names
    tier_map = {"simple": "fast", "reasoning": "strong"}
    tier = tier_map.get(tier, tier)

    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    return defaults.get(tier, defaults["strong"])


def refresh_pricing() -> bool:
    """Force-refresh pricing from OpenRouter API.

    Returns:
        True if refresh succeeded.
    """
    global _memory_cache, _memory_cache_loaded

    fetched = _fetch_openrouter_pricing()
    if fetched:
        _memory_cache = fetched
        _memory_cache_loaded = True
        _save_cache(fetched)
        return True
    return False


def get_cache_info() -> dict[str, Any]:
    """Get info about the pricing cache state (for diagnostics)."""
    info: dict[str, Any] = {
        "cache_file": str(_CACHE_FILE),
        "cache_exists": _CACHE_FILE.exists(),
        "memory_loaded": _memory_cache_loaded,
        "memory_models": len(_memory_cache),
        "fallback_models": len(FALLBACK_PRICING),
    }

    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE) as f:
                data = json.load(f)
            cached_at = data.get("cached_at", 0)
            age_hours = (time.time() - cached_at) / 3600
            info["cache_age_hours"] = round(age_hours, 1)
            info["cache_fresh"] = age_hours < (_CACHE_TTL_SECONDS / 3600)
            info["cached_models"] = len(data.get("models", {}))
        except (json.JSONDecodeError, OSError):
            info["cache_corrupt"] = True

    return info
