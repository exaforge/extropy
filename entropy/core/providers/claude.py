"""Claude (Anthropic) LLM Provider implementation.

Uses the tool use pattern for reliable structured output:
instead of asking Claude to output JSON in text, we define a tool
with the response schema. Claude "calls" the tool, returning structured
data guaranteed to match the schema.
"""

import logging

import anthropic

from .base import LLMProvider, ValidatorCallback, RetryCallback
from .logging import log_request_response, extract_error_summary


logger = logging.getLogger(__name__)


def _clean_schema_for_tool(schema: dict) -> dict:
    """Clean a JSON schema for use as a tool input_schema.

    Removes fields that aren't valid in tool input schemas
    (like 'additionalProperties' in nested objects that Claude
    doesn't support in tool definitions).
    """
    cleaned = {}
    for key, value in schema.items():
        if key == "additionalProperties":
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


class ClaudeProvider(LLMProvider):
    """Claude (Anthropic) LLM provider.

    Uses the tool use pattern for structured output — Claude "calls" a tool
    with the response data, guaranteeing valid JSON matching the schema.

    Supports both API key (sk-ant-api...) and OAuth access token authentication.
    """

    def __init__(self, api_key: str = "") -> None:
        if not api_key:
            raise ValueError(
                "Anthropic credentials not found. Set one of:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...       # API key\n"
                "  export ANTHROPIC_ACCESS_TOKEN=...         # OAuth token\n"
                "Get your key from: https://console.anthropic.com/settings/keys"
            )
        super().__init__(api_key)

    @property
    def default_simple_model(self) -> str:
        return "claude-haiku-4-5-20251001"

    @property
    def default_reasoning_model(self) -> str:
        return "claude-sonnet-4-5-20250929"

    @property
    def default_research_model(self) -> str:
        return "claude-sonnet-4-5-20250929"

    def _is_oauth_token(self) -> bool:
        """Check if the credential is an OAuth access token (not an API key).

        OAuth tokens start with 'sk-ant-oat' while regular API keys start
        with 'sk-ant-api' or similar.
        """
        return "oat" in self._api_key[:15]

    def _get_client(self) -> anthropic.Anthropic:
        if self._is_oauth_token():
            return anthropic.Anthropic(auth_token=self._api_key)
        return anthropic.Anthropic(api_key=self._api_key)

    def _get_async_client(self) -> anthropic.AsyncAnthropic:
        if self._is_oauth_token():
            return anthropic.AsyncAnthropic(auth_token=self._api_key)
        return anthropic.AsyncAnthropic(api_key=self._api_key)

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

        logger.info(
            f"[Claude] simple_call starting - model={model}, schema={schema_name}"
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens or 4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": schema_name},
            messages=[{"role": "user", "content": prompt}],
        )

        structured_data = _extract_tool_input(response)

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
    ) -> dict:
        model = model or self.default_simple_model
        client = self._get_async_client()
        tool = _make_structured_tool(schema_name, response_schema)

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens or 4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": schema_name},
            messages=[{"role": "user", "content": prompt}],
        )

        return _extract_tool_input(response) or {}

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

        attempts = 0
        last_error_summary = ""

        while attempts <= max_retries:
            response = client.messages.create(
                model=model,
                max_tokens=16384,
                tools=[tool],
                tool_choice={"type": "tool", "name": schema_name},
                messages=[{"role": "user", "content": effective_prompt}],
            )

            structured_data = _extract_tool_input(response)

            if log:
                log_request_response(
                    function_name="reasoning_call",
                    request={"model": model, "prompt_length": len(effective_prompt)},
                    response=response,
                    provider="claude",
                )

            result = structured_data or {}

            if validator is None:
                return result

            is_valid, error_msg = validator(result)
            if is_valid:
                return result

            attempts += 1
            last_error_summary = extract_error_summary(error_msg)

            if attempts <= max_retries:
                if on_retry:
                    on_retry(attempts, max_retries, last_error_summary)
                effective_prompt = f"{error_msg}\n\n---\n\n{prompt}"

        if on_retry:
            on_retry(max_retries + 1, max_retries, f"EXHAUSTED: {last_error_summary}")
        return result

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

        attempts = 0
        last_error_summary = ""
        all_sources: list[str] = []

        while attempts <= max_retries:
            research_prompt = (
                f"{effective_prompt}\n\n"
                f"After researching, call the '{schema_name}' tool with your structured findings."
            )

            logger.info(f"[Claude] agentic_research - model={model}")

            response = client.messages.create(
                model=model,
                max_tokens=16384,
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

            # Extract structured data and sources
            structured_data = None
            sources: list[str] = []

            for block in response.content:
                if block.type == "web_search_tool_result":
                    if hasattr(block, "content") and block.content:
                        for result in block.content:
                            if hasattr(result, "url"):
                                sources.append(result.url)

                if block.type == "tool_use" and block.name == schema_name:
                    structured_data = block.input

                if block.type == "text":
                    if hasattr(block, "citations") and block.citations:
                        for citation in block.citations:
                            if hasattr(citation, "url"):
                                sources.append(citation.url)

            all_sources.extend(sources)

            logger.info(f"[Claude] Web search completed, found {len(sources)} sources")

            if log:
                log_request_response(
                    function_name="agentic_research",
                    request={"model": model, "prompt_length": len(research_prompt)},
                    response=response,
                    provider="claude",
                    sources=list(set(sources)),
                )

            result = structured_data or {}

            if validator is None:
                return result, list(set(all_sources))

            is_valid, error_msg = validator(result)
            if is_valid:
                return result, list(set(all_sources))

            attempts += 1
            last_error_summary = extract_error_summary(error_msg)

            if attempts <= max_retries:
                if on_retry:
                    on_retry(attempts, max_retries, last_error_summary)
                effective_prompt = f"{error_msg}\n\n---\n\n{prompt}"

        if on_retry:
            on_retry(max_retries + 1, max_retries, f"EXHAUSTED: {last_error_summary}")
        return result, list(set(all_sources))
