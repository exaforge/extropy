"""LLM clients for Entropy (OpenAI + LM Studio)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

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


def get_lmstudio_client() -> OpenAI:
    """Get LM Studio client (OpenAI-compatible API)."""
    settings = get_settings()
    return OpenAI(
        base_url=settings.lmstudio_base_url,
        api_key="lm-studio",  # LM Studio doesn't require a real key
    )


def agentic_research(
    prompt: str,
    response_schema: dict,
    schema_name: str = "research_data",
    model: str = "gpt-5",
    reasoning_effort: str = "medium",
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
        "include": ["web_search_call.action.sources"],  # Get all sources, not just citations
    }
    
    response = client.responses.create(**request_params)
    
    # Extract structured data and sources
    structured_data = None
    sources = []
    
    for item in response.output:
        # Check for web search results - sources are here
        if hasattr(item, "type") and item.type == "web_search_call":
            # Extract sources from web_search_call.action.sources
            if hasattr(item, "action") and item.action:
                if hasattr(item.action, "sources") and item.action.sources:
                    for source in item.action.sources:
                        if hasattr(source, "url"):
                            sources.append(source.url)
        
        # Check message content
        if hasattr(item, "type") and item.type == "message":
            for content_item in item.content:
                # Get the structured JSON output
                if hasattr(content_item, "type") and content_item.type == "output_text":
                    if hasattr(content_item, "text"):
                        structured_data = json.loads(content_item.text)
                    # Get source URLs from annotations (inline citations)
                    if hasattr(content_item, "annotations") and content_item.annotations:
                        for annotation in content_item.annotations:
                            if hasattr(annotation, "type") and annotation.type == "url_citation":
                                if hasattr(annotation, "url"):
                                    sources.append(annotation.url)
    
    # Log request and response
    if log:
        _log_request_response(
            function_name="agentic_research",
            request=request_params,
            response=response,
            sources=list(set(sources)),
        )
    
    return structured_data or {}, list(set(sources))


def validate_situation_attributes(
    attributes: list[dict],
    context_description: str,
    model: str = "gpt-5-mini",
    log: bool = True,
) -> list[dict]:
    """
    Validate situation attributes and filter out aggregate/population-level stats.
    
    Uses gpt-5-mini for fast, cheap validation without web search.
    
    Args:
        attributes: List of attribute dicts with name, field_type, description
        context_description: What population this is for (e.g., "Netflix subscribers")
        model: Model to use (gpt-5-instant recommended)
        log: Whether to log request/response
    
    Returns:
        Filtered list of attributes (only per_agent ones)
    """
    if not attributes:
        return []
    
    client = get_openai_client()
    
    # Build a simple prompt for classification
    attr_list = "\n".join(
        f"- {attr['name']}: {attr['description']}"
        for attr in attributes
    )
    
    prompt = f"""You are classifying attributes for a simulation of individual {context_description}.

For each attribute below, determine if it is:
- "per_agent": Varies between individual people (e.g., their tenure, their plan, their satisfaction)
- "aggregate": A population-level statistic that doesn't vary per person (e.g., market churn rate, industry share)

Attributes to classify:
{attr_list}

Return ONLY the names of attributes that are "per_agent" (can vary between individuals)."""

    # Schema for structured output
    validation_schema = {
        "type": "object",
        "properties": {
            "per_agent_attributes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of attributes that vary per individual",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of classification decisions",
            },
        },
        "required": ["per_agent_attributes", "reasoning"],
        "additionalProperties": False,
    }
    
    request_params = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "attribute_validation",
                "strict": True,
                "schema": validation_schema,
            }
        },
    }
    
    response = client.responses.create(**request_params)
    
    # Extract result
    result = None
    for item in response.output:
        if hasattr(item, "type") and item.type == "message":
            for content_item in item.content:
                if hasattr(content_item, "type") and content_item.type == "output_text":
                    if hasattr(content_item, "text"):
                        result = json.loads(content_item.text)
    
    if log:
        _log_request_response(
            function_name="validate_situation_attributes",
            request=request_params,
            response=response,
        )
    
    if not result:
        return attributes  # Return all if validation failed
    
    # Filter to only per_agent attributes
    per_agent_names = set(result.get("per_agent_attributes", []))
    filtered = [attr for attr in attributes if attr["name"] in per_agent_names]
    
    return filtered


def chat_completion(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    response_format: type[BaseModel] | None = None,
) -> str | BaseModel:
    """
    Call OpenAI chat completion (legacy, for simple non-agentic tasks).
    
    If response_format is provided, uses structured output and returns parsed model.
    """
    client = get_openai_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if response_format:
        response = client.beta.chat.completions.parse(
            **kwargs,
            response_format=response_format,
        )
        return response.choices[0].message.parsed
    else:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def batch_generate(
    prompts: list[str],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> list[str]:
    """
    Generate responses for multiple prompts.
    
    For efficiency, this batches prompts into groups and processes them together.
    """
    client = get_openai_client()
    results = []

    # Process in batches of 50
    batch_size = 50

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        # Create a single prompt with all items
        combined_prompt = "\n\n---\n\n".join(
            f"[ITEM {j + 1}]\n{prompt}" for j, prompt in enumerate(batch)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Process each item below and return responses in the same order, separated by '---':\n\n{combined_prompt}",
            },
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )

        content = response.choices[0].message.content or ""

        # Parse responses
        batch_results = [r.strip() for r in content.split("---")]

        # Handle case where response format is different
        if len(batch_results) != len(batch):
            # Fall back to individual processing
            for prompt in batch:
                individual_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                results.append(individual_response.choices[0].message.content or "")
        else:
            results.extend(batch_results)

    return results
