"""LLM Provider factory.

Provides two-zone provider routing:
- Pipeline provider: used for phases 1-2 (spec, extend, persona, scenario)
- Simulation provider: used for phase 3 (agent reasoning)

The simulation provider is cached so its async client can be reused
across batch calls and closed cleanly before the event loop shuts down.
"""

from .base import LLMProvider
from ...config import get_config, get_api_key, get_azure_config


# Cached simulation provider — reused across batch calls so the async
# client isn't re-created per request, and can be closed cleanly.
_simulation_provider: LLMProvider | None = None


def _create_provider(provider_name: str) -> LLMProvider:
    """Create a provider instance by name."""
    api_key = get_api_key(provider_name)

    if provider_name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key)
    elif provider_name == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(api_key=api_key)
    elif provider_name == "azure_openai":
        from .openai import OpenAIProvider

        azure_cfg = get_azure_config(provider_name)
        if not azure_cfg.get("azure_endpoint"):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT not found. Set it as an environment variable.\n"
                "  export AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/"
            )
        # Resolve api_format: config value > auto-default (chat_completions for Azure)
        config = get_config()
        api_format = config.simulation.api_format or "chat_completions"
        return OpenAIProvider(
            api_key=api_key,
            azure_endpoint=azure_cfg["azure_endpoint"],
            api_version=azure_cfg.get("api_version", "2025-03-01-preview"),
            azure_deployment=azure_cfg.get("azure_deployment", ""),
            api_format=api_format,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Valid options: 'openai', 'claude', 'azure_openai'"
        )


def get_pipeline_provider() -> LLMProvider:
    """Get the provider for pipeline phases (spec, extend, persona, scenario)."""
    config = get_config()
    return _create_provider(config.pipeline.provider)


def get_simulation_provider() -> LLMProvider:
    """Get the cached provider for simulation phase (agent reasoning).

    Caches the provider so the underlying async HTTP client is reused
    across all calls in a batch, avoiding orphaned connections.
    """
    global _simulation_provider
    config = get_config()
    provider_name = config.simulation.provider

    if _simulation_provider is None:
        _simulation_provider = _create_provider(provider_name)

    return _simulation_provider


async def close_simulation_provider() -> None:
    """Close the cached simulation provider's async client.

    Call this before the event loop shuts down to cleanly release
    HTTP connections and avoid 'Event loop is closed' errors.
    """
    global _simulation_provider
    if _simulation_provider is not None:
        await _simulation_provider.close_async()
        _simulation_provider = None


__all__ = [
    "LLMProvider",
    "get_pipeline_provider",
    "get_simulation_provider",
    "close_simulation_provider",
]
