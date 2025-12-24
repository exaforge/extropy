"""LLM clients for Entropy - OpenRouter-based with Model Profiles.

Three main functions:
- simple_call: Fast, no reasoning, no web search
- reasoning_call: With reasoning, no web search
- agentic_research: With reasoning AND web search

All functions support:
- ModelProfile for consistent phase configuration
- Explicit model override for fine-grained control
- Backward-compatible defaults
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import get_settings, ModelProfile, get_profile
from .providers import get_openrouter_client


def _get_logs_dir() -> Path:
    """Get logs directory, create if needed."""
    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _log_request_response(
    function_name: str,
    request: dict,
    response: Any,
    sources: list[str] | None = None,
) -> None:
    """Log full request and response to a JSON file."""
    logs_dir = _get_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{timestamp}_{function_name}.json"

    # Convert response to dict if possible
    response_dict = None
    if hasattr(response, "model_dump"):
        response_dict = response.model_dump()
    elif hasattr(response, "__dict__"):
        response_dict = str(response)
    else:
        response_dict = str(response)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "function": function_name,
        "request": request,
        "response": response_dict,
        "sources_extracted": sources or [],
    }

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"  [Logged to {log_file}]")


def _resolve_model(
    model: str | None,
    profile: ModelProfile | None,
    call_type: str,
    default: str,
) -> str:
    """Resolve which model to use based on precedence.

    Priority order:
    1. Explicit model parameter
    2. Profile-based model
    3. Default from global settings

    Args:
        model: Explicitly provided model string
        profile: ModelProfile for this task
        call_type: Type of call ("fast", "reasoning", "research")
        default: Default model if nothing else specified

    Returns:
        Model identifier string
    """
    if model:
        return model

    if profile:
        model_map = {
            "fast": profile.fast,
            "reasoning": profile.reasoning,
            "research": profile.research,
        }
        return model_map.get(call_type, default)

    return default


def _build_json_schema_response_format(
    schema: dict,
    schema_name: str,
) -> dict[str, Any]:
    """Build response_format for JSON schema structured output.

    Args:
        schema: JSON schema dictionary
        schema_name: Name for the schema

    Returns:
        response_format dictionary
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def simple_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str | None = None,
    profile: ModelProfile | None = None,
    log: bool = True,
) -> dict:
    """
    Simple LLM call with structured output, no reasoning, no web search.

    Use for fast, cheap tasks:
    - Context sufficiency checks
    - Simple classification
    - Validation

    Args:
        prompt: The prompt
        response_schema: JSON schema for the response
        schema_name: Name for the schema
        model: Explicit model override (e.g., "openai/gpt-4o-mini")
        profile: ModelProfile to use (uses profile.fast if no explicit model)
        log: Whether to log request/response to file

    Returns:
        Structured data matching the schema

    Example:
        # Using explicit model (backward compatible)
        simple_call(prompt, schema, model="openai/gpt-4o-mini")

        # Using profile
        simple_call(prompt, schema, profile=profiles.population)
    """
    settings = get_settings()
    resolved_model = _resolve_model(
        model=model,
        profile=profile,
        call_type="fast",
        default=settings.default_fast_model,
    )

    client = get_openrouter_client()

    messages = [{"role": "user", "content": prompt}]
    response_format = _build_json_schema_response_format(response_schema, schema_name)

    request_params = {
        "model": resolved_model,
        "messages": messages,
        "response_format": response_format,
    }

    response = client.chat_completion(**request_params)

    # Extract structured data from response
    structured_data = None
    if response.choices and response.choices[0].message.content:
        content = response.choices[0].message.content
        structured_data = json.loads(content)

    if log:
        _log_request_response(
            function_name="simple_call",
            request={**request_params, "resolved_model": resolved_model},
            response=response,
        )

    return structured_data or {}


def reasoning_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str | None = None,
    profile: ModelProfile | None = None,
    reasoning_effort: str | None = None,
    log: bool = True,
) -> dict:
    """
    LLM call with reasoning and structured output, but NO web search.

    Use this for tasks that require reasoning but not external data:
    - Attribute selection/categorization
    - Schema design
    - Logical analysis

    Args:
        prompt: What to reason about
        response_schema: JSON schema for the response
        schema_name: Name for the schema
        model: Explicit model override (e.g., "anthropic/claude-sonnet-4")
        profile: ModelProfile to use (uses profile.reasoning if no explicit model)
        reasoning_effort: Override reasoning effort ("low", "medium", "high")
        log: Whether to log request/response to file

    Returns:
        Structured data matching the schema

    Example:
        # Using explicit model (backward compatible)
        reasoning_call(prompt, schema, model="anthropic/claude-sonnet-4")

        # Using profile
        reasoning_call(prompt, schema, profile=profiles.population)
    """
    settings = get_settings()
    resolved_model = _resolve_model(
        model=model,
        profile=profile,
        call_type="reasoning",
        default=settings.default_reasoning_model,
    )

    # Resolve reasoning effort
    effort = reasoning_effort or (profile.reasoning_effort if profile else "low")

    client = get_openrouter_client()

    # Build system message to encourage structured reasoning
    system_message = (
        "You are an expert analyst. Think through this problem step by step, "
        "then provide your answer in the requested JSON format."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    response_format = _build_json_schema_response_format(response_schema, schema_name)

    request_params: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "response_format": response_format,
    }

    # Add provider-specific reasoning parameters
    reasoning_params = client.get_reasoning_params(resolved_model, effort)
    request_params.update(reasoning_params)

    response = client.chat_completion(**request_params)

    # Extract structured data from response
    structured_data = None
    if response.choices and response.choices[0].message.content:
        content = response.choices[0].message.content
        structured_data = json.loads(content)

    if log:
        _log_request_response(
            function_name="reasoning_call",
            request={
                **request_params,
                "resolved_model": resolved_model,
                "reasoning_effort": effort,
            },
            response=response,
        )

    return structured_data or {}


def agentic_research(
    prompt: str,
    response_schema: dict,
    schema_name: str = "research_data",
    model: str | None = None,
    profile: ModelProfile | None = None,
    reasoning_effort: str | None = None,
    log: bool = True,
) -> tuple[dict, list[str]]:
    """
    Perform agentic research with web search and structured output.

    The model will:
    1. Decide what to search for
    2. Search the web (possibly multiple times)
    3. Reason about the results
    4. Return structured data matching the schema

    Note: Web search is currently best supported by OpenAI models.
    For other providers, consider using a research model or
    providing context directly.

    Args:
        prompt: What to research
        response_schema: JSON schema for the response
        schema_name: Name for the schema
        model: Explicit model override (e.g., "openai/gpt-4o")
        profile: ModelProfile to use (uses profile.research if no explicit model)
        reasoning_effort: Override reasoning effort ("low", "medium", "high")
        log: Whether to log request/response to file

    Returns:
        Tuple of (structured_data, source_urls)

    Example:
        # Using explicit model (backward compatible)
        data, sources = agentic_research(prompt, schema, model="openai/gpt-4o")

        # Using profile
        data, sources = agentic_research(prompt, schema, profile=profiles.population)
    """
    settings = get_settings()
    resolved_model = _resolve_model(
        model=model,
        profile=profile,
        call_type="research",
        default=settings.default_research_model,
    )

    # Resolve reasoning effort
    effort = reasoning_effort or (profile.reasoning_effort if profile else "low")

    client = get_openrouter_client()

    # Check if model supports web search
    if not client.supports_web_search(resolved_model):
        # For models without web search, we do a reasoning call
        # and return empty sources
        # Future: Could integrate alternative search tools
        result = reasoning_call(
            prompt=prompt,
            response_schema=response_schema,
            schema_name=schema_name,
            model=resolved_model,
            reasoning_effort=effort,
            log=log,
        )
        return result, []

    # For OpenAI models with web search support, we use the responses API
    # via the underlying OpenAI client
    from openai import OpenAI

    # Create a direct OpenAI client for responses API
    openai_client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    request_params = {
        "model": resolved_model,
        "input": prompt,
        "tools": [{"type": "web_search_preview"}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            }
        },
    }

    # Try responses API first (for web search support)
    try:
        response = openai_client.responses.create(**request_params)

        # Extract structured data and sources
        structured_data = None
        sources = []

        for item in response.output:
            # Check for web search results - sources are here
            if hasattr(item, "type") and item.type == "web_search_call":
                if hasattr(item, "action") and item.action:
                    if hasattr(item.action, "sources") and item.action.sources:
                        for source in item.action.sources:
                            if isinstance(source, dict):
                                if "url" in source:
                                    sources.append(source["url"])
                            elif hasattr(source, "url"):
                                sources.append(source.url)

            # Check message content
            if hasattr(item, "type") and item.type == "message":
                for content_item in item.content:
                    if hasattr(content_item, "type") and content_item.type == "output_text":
                        if hasattr(content_item, "text"):
                            structured_data = json.loads(content_item.text)
                        if hasattr(content_item, "annotations") and content_item.annotations:
                            for annotation in content_item.annotations:
                                if hasattr(annotation, "type") and annotation.type == "url_citation":
                                    if hasattr(annotation, "url"):
                                        sources.append(annotation.url)

        if log:
            _log_request_response(
                function_name="agentic_research",
                request={
                    **request_params,
                    "resolved_model": resolved_model,
                    "reasoning_effort": effort,
                },
                response=response,
                sources=list(set(sources)),
            )

        return structured_data or {}, list(set(sources))

    except Exception as e:
        # Fall back to regular chat completion if responses API fails
        if log:
            print(f"  [Responses API failed, falling back to chat: {e}]")

        result = reasoning_call(
            prompt=f"Research the following and provide accurate information:\n\n{prompt}",
            response_schema=response_schema,
            schema_name=schema_name,
            model=resolved_model,
            reasoning_effort=effort,
            log=log,
        )
        return result, []


# =============================================================================
# Backward Compatibility
# =============================================================================


def get_openai_client():
    """Legacy function for backward compatibility.

    Returns the OpenRouter client's underlying OpenAI client.

    Deprecated:
        Use get_openrouter_client() instead.
    """
    return get_openrouter_client().client
