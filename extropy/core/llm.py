"""LLM clients for Extropy - Facade Layer.

This module provides a unified interface to LLM providers with two-tier routing:
- fast: simple_call → uses models.fast (cheap, fast tasks)
- strong: reasoning_call, agentic_research → uses models.strong (complex tasks)
- simulation: simple_call_async → uses simulation.strong/fast

Model strings use "provider/model" format. The provider is extracted to route
to the correct backend; the model name is passed through.

Configure via `extropy config` CLI or programmatically via extropy.config.configure().
"""

from .providers import get_provider, get_simulation_provider
from .providers.base import TokenUsage, ValidatorCallback, RetryCallback
from ..config import get_config, parse_model_string


__all__ = [
    "simple_call",
    "simple_call_async",
    "reasoning_call",
    "agentic_research",
    "TokenUsage",
    "ValidatorCallback",
    "RetryCallback",
]


def _resolve_provider_and_model(
    model_string: str,
) -> tuple:
    """Resolve a "provider/model" string to (provider_instance, model_name)."""
    config = get_config()
    provider_name, model_name = parse_model_string(model_string)
    provider = get_provider(provider_name, config.providers)
    return provider, model_name


def simple_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str | None = None,
    log: bool = True,
    max_tokens: int | None = None,
) -> dict:
    """Simple LLM call with structured output, no reasoning, no web search.

    Uses the FAST tier (config.models.fast).

    Use for fast, cheap tasks:
    - Context sufficiency checks
    - Simple classification
    - Validation
    """
    config = get_config()
    model_string = model or config.resolve_pipeline_fast()
    provider, model_name = _resolve_provider_and_model(model_string)
    return provider.simple_call(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=model_name,
        log=log,
        max_tokens=max_tokens,
    )


async def simple_call_async(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[dict, TokenUsage]:
    """Async version of simple_call for concurrent API requests.

    Used for batch agent reasoning during simulation.
    Model is passed explicitly from simulation caller (provider/model format).
    Returns (structured_data, token_usage) tuple.
    """
    config = get_config()
    model_string = model or config.resolve_sim_strong()
    _, model_name = parse_model_string(model_string)
    provider = get_simulation_provider(model_string)
    return await provider.simple_call_async(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=model_name,
        max_tokens=max_tokens,
    )


def reasoning_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str | None = None,
    reasoning_effort: str = "low",
    log: bool = True,
    previous_errors: str | None = None,
    validator: ValidatorCallback | None = None,
    max_retries: int = 2,
    on_retry: RetryCallback | None = None,
) -> dict:
    """LLM call with reasoning and structured output, but NO web search.

    Uses the STRONG tier (config.models.strong).

    Use for tasks that require reasoning but not external data:
    - Attribute selection/categorization
    - Schema design
    - Logical analysis
    """
    config = get_config()
    model_string = model or config.resolve_pipeline_strong()
    provider, model_name = _resolve_provider_and_model(model_string)
    return provider.reasoning_call(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=model_name,
        reasoning_effort=reasoning_effort,
        log=log,
        previous_errors=previous_errors,
        validator=validator,
        max_retries=max_retries,
        on_retry=on_retry,
    )


def agentic_research(
    prompt: str,
    response_schema: dict,
    schema_name: str = "research_data",
    model: str | None = None,
    reasoning_effort: str = "low",
    log: bool = True,
    previous_errors: str | None = None,
    validator: ValidatorCallback | None = None,
    max_retries: int = 2,
    on_retry: RetryCallback | None = None,
) -> tuple[dict, list[str]]:
    """Perform agentic research with web search and structured output.

    Uses the STRONG tier (config.models.strong).
    Web search is a provider capability, not a tier distinction.
    """
    config = get_config()
    model_string = model or config.resolve_pipeline_strong()
    provider, model_name = _resolve_provider_and_model(model_string)
    return provider.agentic_research(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=model_name,
        reasoning_effort=reasoning_effort,
        log=log,
        previous_errors=previous_errors,
        validator=validator,
        max_retries=max_retries,
        on_retry=on_retry,
    )
