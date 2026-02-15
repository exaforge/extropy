"""LLM Provider registry and factory.

Provides:
- BUILTIN_PROVIDERS: Registry of known provider names → factory info
- get_provider(): Create a provider instance from a provider name
- get_pipeline_provider() / get_simulation_provider(): Zone-based provider access

The simulation provider is cached so its async client can be reused
across batch calls and closed cleanly before the event loop shuts down.
"""

import os

from .base import LLMProvider
from ...config import (
    get_config,
    get_api_key_for_provider,
    parse_model_string,
    CustomProviderConfig,
)


# =============================================================================
# Provider Registry
# =============================================================================

# Each entry: (module, class_name, default_kwargs)
# Lazy-imported to avoid loading all SDKs at startup.
_BUILTIN_REGISTRY: dict[str, dict] = {
    "openai": {
        "module": ".openai",
        "class": "OpenAIProvider",
    },
    "anthropic": {
        "module": ".anthropic",
        "class": "AnthropicProvider",
    },
    "openrouter": {
        "module": ".openai_compat",
        "class": "OpenAICompatProvider",
        "kwargs": {
            "base_url": "https://openrouter.ai/api/v1",
            "supports_search": True,
            "provider_label": "openrouter",
            "default_fast": "openai/gpt-5-mini",
            "default_strong": "openai/gpt-5",
        },
    },
    "azure": {
        "module": ".azure",
        "class": "AzureProvider",
    },
    "deepseek": {
        "module": ".openai_compat",
        "class": "OpenAICompatProvider",
        "kwargs": {
            "base_url": "https://api.deepseek.com/v1",
            "supports_search": False,
            "provider_label": "deepseek",
            "default_fast": "deepseek-chat",
            "default_strong": "deepseek-reasoner",
        },
    },
    "together": {
        "module": ".openai_compat",
        "class": "OpenAICompatProvider",
        "kwargs": {
            "base_url": "https://api.together.xyz/v1",
            "supports_search": False,
            "provider_label": "together",
            "default_fast": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "default_strong": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        },
    },
    "groq": {
        "module": ".openai_compat",
        "class": "OpenAICompatProvider",
        "kwargs": {
            "base_url": "https://api.groq.com/openai/v1",
            "supports_search": False,
            "provider_label": "groq",
            "default_fast": "llama-3.3-70b-versatile",
            "default_strong": "llama-3.3-70b-versatile",
        },
    },
}


def get_provider(
    provider_name: str,
    custom_providers: dict[str, CustomProviderConfig] | None = None,
) -> LLMProvider:
    """Create a provider instance by name.

    Checks custom providers first, then built-in registry.

    Args:
        provider_name: Provider name (e.g., "openai", "anthropic", "openrouter")
        custom_providers: Optional custom provider configs from ExtropyConfig

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider is unknown
    """
    api_key = get_api_key_for_provider(provider_name, custom_providers)

    # Check custom providers first
    if custom_providers and provider_name in custom_providers:
        from .openai_compat import OpenAICompatProvider

        custom = custom_providers[provider_name]
        return OpenAICompatProvider(
            api_key=api_key,
            base_url=custom.base_url,
            supports_search=False,
            provider_label=provider_name,
        )

    # Check built-in registry
    if provider_name not in _BUILTIN_REGISTRY:
        available = sorted(
            set(list(_BUILTIN_REGISTRY.keys()) + list((custom_providers or {}).keys()))
        )
        raise ValueError(
            f"Unknown LLM provider: {provider_name!r}. "
            f"Available: {', '.join(available)}"
        )

    entry = _BUILTIN_REGISTRY[provider_name]

    # Special case: Azure needs endpoint from env
    if provider_name == "azure":
        endpoint = os.environ.get("AZURE_ENDPOINT") or os.environ.get(
            "AZURE_OPENAI_ENDPOINT", ""
        )
        entry = dict(entry)
        entry["kwargs"] = {"endpoint": endpoint}

    # Lazy import
    import importlib

    module = importlib.import_module(entry["module"], package=__package__)
    cls = getattr(module, entry["class"])

    kwargs = dict(entry.get("kwargs", {}))
    kwargs["api_key"] = api_key

    return cls(**kwargs)


# =============================================================================
# Zone-based provider access (backward compat)
# =============================================================================

# Cached providers — reused across calls for connection reuse
_cached_providers: dict[str, LLMProvider] = {}


def _get_or_create_provider(provider_name: str, cache_key: str = "") -> LLMProvider:
    """Get or create a cached provider instance."""
    key = cache_key or provider_name
    if key not in _cached_providers:
        config = get_config()
        _cached_providers[key] = get_provider(provider_name, config.providers)
    return _cached_providers[key]


def get_pipeline_provider() -> LLMProvider:
    """Get the provider for pipeline phases (spec, extend, persona, scenario).

    Uses the provider from models.fast (pipeline calls use both fast and strong,
    but the provider is determined by the fast model string).
    """
    config = get_config()
    provider, _ = parse_model_string(config.models.fast)
    return _get_or_create_provider(provider, f"pipeline:{provider}")


def get_simulation_provider() -> LLMProvider:
    """Get the cached provider for simulation phase (agent reasoning).

    Uses the provider from the resolved simulation strong model.
    """
    config = get_config()
    strong_model = config.resolve_sim_strong()
    provider, _ = parse_model_string(strong_model)
    return _get_or_create_provider(provider, f"simulation:{provider}")


async def close_simulation_provider() -> None:
    """Close cached providers' async clients.

    Call this before the event loop shuts down to cleanly release
    HTTP connections and avoid 'Event loop is closed' errors.
    """
    for key, provider in list(_cached_providers.items()):
        await provider.close_async()
    _cached_providers.clear()


def reset_provider_cache() -> None:
    """Reset the provider cache (for testing)."""
    _cached_providers.clear()


# Legacy factory (kept for backward compat in tests)
def _create_provider(provider_name: str) -> LLMProvider:
    """DEPRECATED: Use get_provider() instead."""
    # Map old names
    name_map = {"claude": "anthropic", "azure_openai": "azure"}
    canonical = name_map.get(provider_name, provider_name)
    config = get_config()
    return get_provider(canonical, config.providers)


__all__ = [
    "LLMProvider",
    "get_provider",
    "get_pipeline_provider",
    "get_simulation_provider",
    "close_simulation_provider",
    "reset_provider_cache",
    "parse_model_string",
]
