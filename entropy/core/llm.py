"""LLM clients for Entropy - Architect Layer.

Three main functions:
- simple_call: Fast, no reasoning, no web search (gpt-5-mini)
- reasoning_call: With reasoning, no web search (gpt-5)
- agentic_research: With reasoning AND web search (gpt-5)

Each function supports retry with error feedback via the `previous_errors` parameter.
When validation fails, pass the error message back to let the LLM self-correct.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from ..config import get_settings


# Type for validation callbacks: takes response data, returns (is_valid, error_message)
ValidatorCallback = Callable[[dict], tuple[bool, str]]

# Type for retry notification callbacks: (attempt, max_retries, short_error_summary)
RetryCallback = Callable[[int, int, str], None]


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
    # Use mode='json' to avoid Pydantic warnings on complex SDK types
    response_dict = None
    if hasattr(response, "model_dump"):
        try:
            response_dict = response.model_dump(mode="json", warnings=False)
        except Exception:
            # Fallback if model_dump fails
            response_dict = str(response)
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


def get_openai_client() -> OpenAI:
    """Get OpenAI client."""
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def simple_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str = "gpt-5-mini",
    log: bool = True,
    max_tokens: int | None = None,
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
        model: Model to use (gpt-5-mini recommended)
        log: Whether to log request/response to file
        max_tokens: Maximum output tokens (None = unlimited, use schema to guide length)

    Returns:
        Structured data matching the schema
    """
    import time
    import logging

    logger = logging.getLogger(__name__)

    client = get_openai_client()

    request_params = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            }
        },
    }

    if max_tokens is not None:
        request_params["max_output_tokens"] = max_tokens

    # Heavy logging - before API call
    logger.info(f"[LLM] simple_call starting - model={model}, schema={schema_name}, max_tokens={max_tokens}")
    logger.info(f"[LLM] prompt length: {len(prompt)} chars")
    logger.debug(f"[LLM] FULL REQUEST PARAMS: {json.dumps(request_params, indent=2, default=str)}")

    api_start = time.time()
    response = client.responses.create(**request_params)
    api_elapsed = time.time() - api_start

    logger.info(f"[LLM] API response received in {api_elapsed:.2f}s")

    # Extract structured data
    structured_data = None
    for item in response.output:
        if hasattr(item, "type") and item.type == "message":
            for content_item in item.content:
                if hasattr(content_item, "type") and content_item.type == "output_text":
                    if hasattr(content_item, "text"):
                        structured_data = json.loads(content_item.text)

    logger.info(f"[LLM] Extracted data: {json.dumps(structured_data) if structured_data else 'None'}")

    # Log usage stats if available
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        logger.info(
            f"[LLM] Token usage - input: {getattr(usage, 'input_tokens', 'N/A')}, "
            f"output: {getattr(usage, 'output_tokens', 'N/A')}, "
            f"total: {getattr(usage, 'total_tokens', 'N/A')}"
        )

    if log:
        _log_request_response(
            function_name="simple_call",
            request=request_params,
            response=response,
        )

    return structured_data or {}


async def simple_call_async(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str = "gpt-5-mini",
    max_tokens: int | None = None,
) -> dict:
    """
    Async version of simple_call for concurrent API requests.

    Args:
        prompt: The prompt
        response_schema: JSON schema for the response
        schema_name: Name for the schema
        model: Model to use
        max_tokens: Maximum output tokens (None = unlimited)

    Returns:
        Structured data matching the schema
    """
    import logging
    from openai import AsyncOpenAI

    logger = logging.getLogger(__name__)
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    request_params = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            }
        },
    }

    if max_tokens is not None:
        request_params["max_output_tokens"] = max_tokens

    response = await client.responses.create(**request_params)

    # Extract structured data
    structured_data = None
    for item in response.output:
        if hasattr(item, "type") and item.type == "message":
            for content_item in item.content:
                if hasattr(content_item, "type") and content_item.type == "output_text":
                    if hasattr(content_item, "text"):
                        structured_data = json.loads(content_item.text)

    return structured_data or {}


def reasoning_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    log: bool = True,
    previous_errors: str | None = None,
    validator: ValidatorCallback | None = None,
    max_retries: int = 2,
    on_retry: RetryCallback | None = None,
) -> dict:
    """
    GPT-5 with reasoning and structured output, but NO web search.

    Use this for tasks that require reasoning but not external data:
    - Attribute selection/categorization
    - Schema design
    - Logical analysis

    Args:
        prompt: What to reason about
        response_schema: JSON schema for the response
        schema_name: Name for the schema
        model: Model to use (gpt-5, gpt-5.1, etc.)
        reasoning_effort: "low", "medium", or "high"
        log: Whether to log request/response to file
        previous_errors: Error feedback from failed validation to prepend to prompt
        validator: Optional callback to validate response; if invalid, will retry
        max_retries: Maximum number of retry attempts if validation fails
        on_retry: Optional callback called when validation fails and retry begins

    Returns:
        Structured data matching the schema
    """
    client = get_openai_client()

    # Prepend previous errors if provided
    effective_prompt = prompt
    if previous_errors:
        effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

    attempts = 0
    last_error_summary = ""

    while attempts <= max_retries:
        request_params = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "input": [{"role": "user", "content": effective_prompt}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                }
            },
        }

        response = client.responses.create(**request_params)

        # Extract structured data
        structured_data = None
        for item in response.output:
            if hasattr(item, "type") and item.type == "message":
                for content_item in item.content:
                    if (
                        hasattr(content_item, "type")
                        and content_item.type == "output_text"
                    ):
                        if hasattr(content_item, "text"):
                            structured_data = json.loads(content_item.text)

        if log:
            _log_request_response(
                function_name="reasoning_call",
                request=request_params,
                response=response,
            )

        result = structured_data or {}

        # If no validator, return immediately
        if validator is None:
            return result

        # Validate the response
        is_valid, error_msg = validator(result)

        if is_valid:
            return result

        # Validation failed - prepare for retry
        attempts += 1

        # Extract meaningful error summary (skip header lines)
        last_error_summary = _extract_error_summary(error_msg)

        if attempts <= max_retries:
            # Notify via callback if provided
            if on_retry:
                on_retry(attempts, max_retries, last_error_summary)
            # Prepend error feedback for next attempt
            effective_prompt = f"{error_msg}\n\n---\n\n{prompt}"

    # All retries exhausted - notify one final time
    if on_retry:
        on_retry(max_retries + 1, max_retries, f"EXHAUSTED: {last_error_summary}")
    return result


def _extract_error_summary(error_msg: str) -> str:
    """Extract a concise error summary from validation error message."""
    if not error_msg:
        return "validation error"

    lines = error_msg.strip().split("\n")

    # Skip header lines that start with "##" or are empty
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("---"):
            # Found a meaningful line - extract key info
            # Look for "ERROR in X:" or "Problem:" patterns
            if "ERROR in" in line:
                return line[:60]
            elif "Problem:" in line:
                return line.replace("Problem:", "").strip()[:60]
            elif line:
                return line[:60]

    return "validation error"


def agentic_research(
    prompt: str,
    response_schema: dict,
    schema_name: str = "research_data",
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    log: bool = True,
    previous_errors: str | None = None,
    validator: ValidatorCallback | None = None,
    max_retries: int = 2,
    on_retry: RetryCallback | None = None,
) -> tuple[dict, list[str]]:
    """
    Perform agentic research with web search and structured output.

    The model will:
    1. Decide what to search for
    2. Search the web (possibly multiple times)
    3. Reason about the results
    4. Return structured data matching the schema

    Args:
        prompt: What to research
        response_schema: JSON schema for the response
        schema_name: Name for the schema
        model: Model to use (gpt-5, gpt-5.1, etc.)
        reasoning_effort: "low", "medium", or "high"
        log: Whether to log request/response to file
        previous_errors: Error feedback from failed validation to prepend to prompt
        validator: Optional callback to validate response; if invalid, will retry
        max_retries: Maximum number of retry attempts if validation fails
        on_retry: Optional callback called when validation fails and retry begins

    Returns:
        Tuple of (structured_data, source_urls)
    """
    client = get_openai_client()

    # Prepend previous errors if provided
    effective_prompt = prompt
    if previous_errors:
        effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

    attempts = 0
    last_error_summary = ""
    all_sources: list[str] = []

    while attempts <= max_retries:
        request_params = {
            "model": model,
            "input": effective_prompt,
            "tools": [{"type": "web_search"}],
            "reasoning": {"effort": reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                }
            },
            "include": ["web_search_call.action.sources"],
        }

        response = client.responses.create(**request_params)

        # Extract structured data and sources
        structured_data = None
        sources: list[str] = []

        for item in response.output:
            # Check for web search results - sources are here
            if hasattr(item, "type") and item.type == "web_search_call":
                if hasattr(item, "action") and item.action:
                    if hasattr(item.action, "sources") and item.action.sources:
                        for source in item.action.sources:
                            # Handle both dict and object access
                            if isinstance(source, dict):
                                if "url" in source:
                                    sources.append(source["url"])
                            elif hasattr(source, "url"):
                                sources.append(source.url)

            # Check message content
            if hasattr(item, "type") and item.type == "message":
                for content_item in item.content:
                    if (
                        hasattr(content_item, "type")
                        and content_item.type == "output_text"
                    ):
                        if hasattr(content_item, "text"):
                            structured_data = json.loads(content_item.text)
                        if (
                            hasattr(content_item, "annotations")
                            and content_item.annotations
                        ):
                            for annotation in content_item.annotations:
                                if (
                                    hasattr(annotation, "type")
                                    and annotation.type == "url_citation"
                                ):
                                    if hasattr(annotation, "url"):
                                        sources.append(annotation.url)

        # Accumulate sources across retries
        all_sources.extend(sources)

        if log:
            _log_request_response(
                function_name="agentic_research",
                request=request_params,
                response=response,
                sources=list(set(sources)),
            )

        result = structured_data or {}

        # If no validator, return immediately
        if validator is None:
            return result, list(set(all_sources))

        # Validate the response
        is_valid, error_msg = validator(result)

        if is_valid:
            return result, list(set(all_sources))

        # Validation failed - prepare for retry
        attempts += 1
        # Extract meaningful error summary (skip header lines)
        last_error_summary = _extract_error_summary(error_msg)

        if attempts <= max_retries:
            # Notify via callback if provided
            if on_retry:
                on_retry(attempts, max_retries, last_error_summary)
            # Prepend error feedback for next attempt
            effective_prompt = f"{error_msg}\n\n---\n\n{prompt}"

    # All retries exhausted - notify one final time
    if on_retry:
        on_retry(max_retries + 1, max_retries, f"EXHAUSTED: {last_error_summary}")
    return result, list(set(all_sources))
