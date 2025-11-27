"""LLM clients for Entropy - Architect Layer.

Three main functions:
- simple_call: Fast, no reasoning, no web search (gpt-5-mini)
- reasoning_call: With reasoning, no web search (gpt-5)
- agentic_research: With reasoning AND web search (gpt-5)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import get_settings


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
    
    Returns:
        Structured data matching the schema
    """
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
    
    response = client.responses.create(**request_params)
    
    # Extract structured data
    structured_data = None
    for item in response.output:
        if hasattr(item, "type") and item.type == "message":
            for content_item in item.content:
                if hasattr(content_item, "type") and content_item.type == "output_text":
                    if hasattr(content_item, "text"):
                        structured_data = json.loads(content_item.text)
    
    if log:
        _log_request_response(
            function_name="simple_call",
            request=request_params,
            response=response,
        )
    
    return structured_data or {}


def reasoning_call(
    prompt: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    log: bool = True,
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
    
    Returns:
        Structured data matching the schema
    """
    client = get_openai_client()
    
    request_params = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "input": [{"role": "user", "content": prompt}],
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
                if hasattr(content_item, "type") and content_item.type == "output_text":
                    if hasattr(content_item, "text"):
                        structured_data = json.loads(content_item.text)
    
    if log:
        _log_request_response(
            function_name="reasoning_call",
            request=request_params,
            response=response,
        )
    
    return structured_data or {}


def agentic_research(
    prompt: str,
    response_schema: dict,
    schema_name: str = "research_data",
    model: str = "gpt-5",
    reasoning_effort: str = "low",
    log: bool = True,
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
    
    Returns:
        Tuple of (structured_data, source_urls)
    """
    client = get_openai_client()
    
    request_params = {
        "model": model,
        "input": prompt,
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
    sources = []
    
    for item in response.output:
        # Check for web search results - sources are here
        if hasattr(item, "type") and item.type == "web_search_call":
            if hasattr(item, "action") and item.action:
                if hasattr(item.action, "sources") and item.action.sources:
                    for source in item.action.sources:
                        if hasattr(source, "url"):
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
            request=request_params,
            response=response,
            sources=list(set(sources)),
        )
    
    return structured_data or {}, list(set(sources))
