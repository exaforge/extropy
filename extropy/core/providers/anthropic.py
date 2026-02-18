"""Anthropic (Claude) LLM Provider implementation.

Uses the tool use pattern for reliable structured output:
instead of asking Claude to output JSON in text, we define a tool
with the response schema. Claude "calls" the tool, returning structured
data guaranteed to match the schema.
"""

import logging
import random
import time

import anthropic

from .base import LLMProvider, TokenUsage, ValidatorCallback, RetryCallback
from .logging import log_request_response, extract_error_summary

_TRANSIENT_ANTHROPIC_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
)
_MAX_API_RETRIES = 3


logger = logging.getLogger(__name__)


def _clean_schema_for_tool(schema: dict) -> dict:
    """Clean a JSON schema for use as a tool input_schema.

    Anthropic structured outputs support additionalProperties: false but NOT
    schema-valued additionalProperties (e.g. {"type": "number"}).

    This function:
    - Keeps additionalProperties: false (valid and useful)
    - Strips additionalProperties when it's a dict/schema (not supported)
    - Logs a warning when stripping schema-valued additionalProperties
    """
    cleaned = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            if value is False:
                # Keep additionalProperties: false - it's valid
                cleaned[key] = value
            elif isinstance(value, dict):
                # Schema-valued additionalProperties not supported - strip with warning
                logger.warning(
                    "Stripping schema-valued additionalProperties from tool schema "
                    "(not supported by Anthropic structured outputs)"
                )
            # Skip other truthy values (True, etc.)
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema_for_tool(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_schema_for_tool(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _make_structured_tool(schema_name: str, response_schema: dict) -> dict:
    """Create a tool definition that forces structured output."""
    return {
        "name": schema_name,
        "description": (
            "Return your response as structured data. "
            "You MUST call this tool with your complete response."
        ),
        "input_schema": _clean_schema_for_tool(response_schema),
    }


def _extract_tool_input(response) -> dict | None:
    """Extract tool_use input from a Claude response."""
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return None


def _extract_usage(response) -> TokenUsage:
    """Extract token usage from an Anthropic API response."""
    if not hasattr(response, "usage") or response.usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
    )


def _stream_to_message(client, **kwargs):
    """Use streaming to avoid 10-minute timeout, collect final message."""
    with client.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM provider.

    Uses the tool use pattern for structured output — Claude "calls" a tool
    with the response data, guaranteeing valid JSON matching the schema.
    """

    provider_name = "anthropic"

    def __init__(self, api_key: str = "", *, base_url: str = "") -> None:
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set it via:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "Get your key from: https://console.anthropic.com/settings/keys"
            )
        super().__init__(api_key)
        self._base_url = base_url

    def _with_retry(self, fn, max_retries: int = _MAX_API_RETRIES):
        """Retry a sync API call on transient errors with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except _TRANSIENT_ANTHROPIC_ERRORS as e:
                if attempt == max_retries:
                    raise
                wait = (2**attempt) + random.random()
                logger.warning(
                    f"[Claude] Transient error (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {wait:.1f}s"
                )
                time.sleep(wait)

    async def _with_retry_async(self, fn, max_retries: int = _MAX_API_RETRIES):
        """Retry an async API call on transient errors with exponential backoff."""
        import asyncio

        for attempt in range(max_retries + 1):
            try:
                return await fn()
            except _TRANSIENT_ANTHROPIC_ERRORS as e:
                if attempt == max_retries:
                    raise
                wait = (2**attempt) + random.random()
                logger.warning(
                    f"[Claude] Transient error (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)

    @property
    def default_fast_model(self) -> str:
        return "claude-haiku-4-5-20251001"

    @property
    def default_strong_model(self) -> str:
        return "claude-sonnet-4-6"

    def _get_client(self) -> anthropic.Anthropic:
        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return anthropic.Anthropic(**kwargs)

    def _get_async_client(self) -> anthropic.AsyncAnthropic:
        if self._cached_async_client is None:
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._cached_async_client = anthropic.AsyncAnthropic(**kwargs)
        return self._cached_async_client

    def simple_call(
        self,
        prompt: str,
        response_schema: dict,
        schema_name: str = "response",
        model: str | None = None,
        log: bool = True,
        max_tokens: int | None = None,
    ) -> dict:
        model = model or self.default_simple_model
        client = self._get_client()
        tool = _make_structured_tool(schema_name, response_schema)

        # Acquire rate limit capacity before making the call
        self._acquire_rate_limit(prompt, model, max_output=max_tokens or 4096)

        logger.info(
            f"[Claude] simple_call starting - model={model}, schema={schema_name}"
        )

        response = self._with_retry(
            lambda: client.messages.create(
                model=model,
                max_tokens=max_tokens or 4096,
                tools=[tool],
                tool_choice={"type": "tool", "name": schema_name},
                messages=[{"role": "user", "content": prompt}],
            )
        )

        structured_data = _extract_tool_input(response)

        # Record token usage
        usage = _extract_usage(response)
        self._record_usage(model, usage, call_type="simple")

        if log:
            log_request_response(
                function_name="simple_call",
                request={"model": model, "prompt_length": len(prompt)},
                response=response,
                provider="claude",
            )

        return structured_data or {}

    async def simple_call_async(
        self,
        prompt: str,
        response_schema: dict,
        schema_name: str = "response",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict, TokenUsage]:
        model = model or self.default_simple_model
        client = self._get_async_client()
        tool = _make_structured_tool(schema_name, response_schema)

        response = await self._with_retry_async(
            lambda: client.messages.create(
                model=model,
                max_tokens=max_tokens or 4096,
                tools=[tool],
                tool_choice={"type": "tool", "name": schema_name},
                messages=[{"role": "user", "content": prompt}],
            )
        )

        # Extract and record token usage
        usage = _extract_usage(response)
        self._record_usage(model, usage, call_type="async")

        return _extract_tool_input(response) or {}, usage

    def reasoning_call(
        self,
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
        """Claude reasoning call with tool-based structured output."""
        model = model or self.default_reasoning_model
        client = self._get_client()
        tool = _make_structured_tool(schema_name, response_schema)

        effective_prompt = prompt
        if previous_errors:
            effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

        def _call(ep: str) -> dict:
            # Acquire rate limit capacity before each API call
            self._acquire_rate_limit(ep, model, max_output=65536)

            # Use streaming to bypass 10-minute timeout for long operations
            response = self._with_retry(
                lambda: _stream_to_message(
                    client,
                    model=model,
                    max_tokens=65536,  # Max allowed for Claude (64K)
                    tools=[tool],
                    tool_choice={"type": "tool", "name": schema_name},
                    messages=[{"role": "user", "content": ep}],
                )
            )
            structured_data = _extract_tool_input(response)

            # Record token usage
            ru = _extract_usage(response)
            self._record_usage(model, ru, call_type="reasoning")

            if log:
                log_request_response(
                    function_name="reasoning_call",
                    request={"model": model, "prompt_length": len(ep)},
                    response=response,
                    provider="claude",
                )
            return structured_data or {}

        return self._retry_with_validation(
            call_fn=_call,
            prompt=prompt,
            validator=validator,
            max_retries=max_retries,
            on_retry=on_retry,
            extract_error_summary_fn=extract_error_summary,
            initial_prompt=effective_prompt if previous_errors else None,
        )

    def agentic_research(
        self,
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
        """Claude agentic research with web search + tool-based structured output.

        Uses web_search tool for research and a structured output tool for the response.
        Claude first searches, then calls the output tool with results.
        """
        model = model or self.default_research_model
        client = self._get_client()
        output_tool = _make_structured_tool(schema_name, response_schema)

        effective_prompt = prompt
        if previous_errors:
            effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

        all_sources: list[str] = []

        def _call(ep: str) -> dict:
            research_prompt = (
                f"{ep}\n\n"
                f"After researching, call the '{schema_name}' tool with your structured findings."
            )

            # Acquire rate limit capacity before each API call
            self._acquire_rate_limit(research_prompt, model, max_output=65536)

            logger.info(f"[Claude] agentic_research - model={model}")

            # Use streaming to bypass 10-minute timeout for long operations
            response = self._with_retry(
                lambda: _stream_to_message(
                    client,
                    model=model,
                    max_tokens=65536,  # Max allowed for Claude Sonnet (64K)
                    tools=[
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": 5,
                        },
                        output_tool,
                    ],
                    messages=[{"role": "user", "content": research_prompt}],
                )
            )

            structured_data = None
            sources: list[str] = []

            for block in response.content:
                if block.type == "web_search_tool_result":
                    if hasattr(block, "content") and block.content:
                        for res in block.content:
                            if hasattr(res, "url"):
                                sources.append(res.url)

                if block.type == "tool_use" and block.name == schema_name:
                    structured_data = block.input

                if block.type == "text":
                    if hasattr(block, "citations") and block.citations:
                        for citation in block.citations:
                            if hasattr(citation, "url"):
                                sources.append(citation.url)

            all_sources.extend(sources)
            logger.info(f"[Claude] Web search completed, found {len(sources)} sources")

            # Record token usage
            ru = _extract_usage(response)
            self._record_usage(model, ru, call_type="agentic_research")

            if log:
                log_request_response(
                    function_name="agentic_research",
                    request={"model": model, "prompt_length": len(research_prompt)},
                    response=response,
                    provider="claude",
                    sources=list(set(sources)),
                )

            return structured_data or {}

        result = self._retry_with_validation(
            call_fn=_call,
            prompt=prompt,
            validator=validator,
            max_retries=max_retries,
            on_retry=on_retry,
            extract_error_summary_fn=extract_error_summary,
            initial_prompt=effective_prompt if previous_errors else None,
        )

        return result, list(set(all_sources))


# Backward compat alias
ClaudeProvider = AnthropicProvider
