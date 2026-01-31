"""LLM Provider factory.

Provides two-zone provider routing:
- Pipeline provider: used for phases 1-2 (spec, extend, persona, scenario)
- Simulation provider: used for phase 3 (agent reasoning)
"""

from .base import LLMProvider
from ...config import get_config, get_api_key


def _create_provider(provider_name: str) -> LLMProvider:
    """Create a provider instance by name."""
    api_key = get_api_key(provider_name)

    if provider_name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key)
    elif provider_name == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(api_key=api_key)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. Valid options: 'openai', 'claude'"
        )


def get_pipeline_provider() -> LLMProvider:
    """Get the provider for pipeline phases (spec, extend, persona, scenario)."""
    config = get_config()
    return _create_provider(config.pipeline.provider)


def get_simulation_provider() -> LLMProvider:
    """Get the provider for simulation phase (agent reasoning)."""
    config = get_config()
    return _create_provider(config.simulation.provider)



__all__ = [
    "LLMProvider",
    "get_pipeline_provider",
    "get_simulation_provider",
]
