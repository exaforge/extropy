"""LLM clients for Entropy - Facade Layer.

This module provides a unified interface to LLM providers with two-zone routing:
- Pipeline (sync calls): simple_call, reasoning_call, agentic_research
  → Uses the pipeline provider (configured for phases 1-2)
- Simulation (async calls): simple_call_async
  → Uses the simulation provider (configured for phase 3)

Configure via `entropy config` CLI or programmatically via entropy.config.configure().

Each function supports retry with error feedback via the `previous_errors` parameter.
When validation fails, pass the error message back to let the LLM self-correct.
"""

from .providers import get_pipeline_provider, get_simulation_provider
from .providers.base import ValidatorCallback, RetryCallback
from ..config import get_config


__all__ = [
    "simple_call",
    "simple_call_async",
    "reasoning_call",
    "agentic_research",
    "ValidatorCallback",
    "RetryCallback",
]


def _get_pipeline_model_override(tier: str) -> str | None:
    """Get pipeline model override from config if configured."""
    config = get_config()
    pipeline = config.pipeline
    if tier == "simple" and pipeline.model_simple:
        return pipeline.model_simple
    elif tier == "reasoning" and pipeline.model_reasoning:
        return pipeline.model_reasoning
    elif tier == "research" and pipeline.model_research:
        return pipeline.model_research
    return None


def _get_simulation_model_override() -> str | None:
    """Get simulation model override from config if configured."""
    config = get_config()
    if config.simulation.model:
        return config.simulation.model
    return None


def simple_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str | None = None,
    log: bool = True,
    max_tokens: int | None = None,
) -> dict:
    """Simple LLM call with structured output, no reasoning, no web search.

    Routed through the PIPELINE provider.

    Use for fast, cheap tasks:
    - Context sufficiency checks
    - Simple classification
    - Validation
    """
    provider = get_pipeline_provider()
    effective_model = model or _get_pipeline_model_override("simple")
    return provider.simple_call(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=effective_model,
        log=log,
        max_tokens=max_tokens,
    )


async def simple_call_async(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Async version of simple_call for concurrent API requests.

    Routed through the SIMULATION provider.

    Used for batch agent reasoning during simulation.
    """
    provider = get_simulation_provider()
    effective_model = model or _get_simulation_model_override()
    return await provider.simple_call_async(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=effective_model,
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

    Routed through the PIPELINE provider.

    Use for tasks that require reasoning but not external data:
    - Attribute selection/categorization
    - Schema design
    - Logical analysis
    """
    provider = get_pipeline_provider()
    effective_model = model or _get_pipeline_model_override("reasoning")
    return provider.reasoning_call(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=effective_model,
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

    Routed through the PIPELINE provider.

    The model will:
    1. Decide what to search for
    2. Search the web (possibly multiple times)
    3. Reason about the results
    4. Return structured data matching the schema
    """
    provider = get_pipeline_provider()
    effective_model = model or _get_pipeline_model_override("research")
    return provider.agentic_research(
        prompt=prompt,
        response_schema=response_schema,
        schema_name=schema_name,
        model=effective_model,
        reasoning_effort=reasoning_effort,
        log=log,
        previous_errors=previous_errors,
        validator=validator,
        max_retries=max_retries,
        on_retry=on_retry,
    )
