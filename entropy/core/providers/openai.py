"""OpenAI LLM Provider implementation."""

import json
import logging
import random
import time

import openai
from openai import OpenAI, AsyncOpenAI

from .base import LLMProvider, TokenUsage, ValidatorCallback, RetryCallback
from .logging import log_request_response, extract_error_summary

_TRANSIENT_OPENAI_ERRORS = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)
_MAX_API_RETRIES = 3


logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider supporting both Responses API and Chat Completions API.

    Supports both standard OpenAI and Azure OpenAI endpoints.
    When azure_endpoint is provided, uses AzureOpenAI/AsyncAzureOpenAI clients.

    The api_format parameter controls which API is used for simple_call/simple_call_async:
    - "responses" (default): Uses the Responses API (client.responses.create)
    - "chat_completions": Uses the Chat Completions API (client.chat.completions.create)

    Azure-hosted models (e.g. DeepSeek-V3.2, Kimi-K2.5) only support Chat Completions,
    so Azure providers default to chat_completions when created via the factory.
    """

    provider_name = "openai"

    def __init__(
        self,
        api_key: str = "",
        *,
        azure_endpoint: str = "",
        api_version: str = "",
        azure_deployment: str = "",
        api_format: str = "responses",
    ) -> None:
        self._is_azure = bool(azure_endpoint)
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._azure_deployment = azure_deployment
        self._api_format = api_format

        if not api_key:
            if self._is_azure:
                raise ValueError(
                    "AZURE_OPENAI_API_KEY not found. Set it as an environment variable.\n"
                    "  export AZURE_OPENAI_API_KEY=<your-subscription-key>"
                )
            else:
                raise ValueError(
                    "OPENAI_API_KEY not found. Set it as an environment variable.\n"
                    "  export OPENAI_API_KEY=sk-..."
                )
        super().__init__(api_key)

        if self._is_azure:
            self.provider_name = "azure_openai"

    def _resolve_model(self, model: str | None, default: str) -> str:
        """Resolve model name, using Azure deployment as fallback when applicable."""
        if model:
            return model
        if self._is_azure and self._azure_deployment:
            return self._azure_deployment
        return default

    @staticmethod
    def _extract_output_text(response) -> str | None:
        """Extract the output text content from an OpenAI Responses API response.

        Traverses response.output looking for a message with output_text content.

        Returns:
            Raw text string, or None if not found.
        """
        for item in response.output:
            if hasattr(item, "type") and item.type == "message":
                for content_item in item.content:
                    if (
                        hasattr(content_item, "type")
                        and content_item.type == "output_text"
                    ):
                        if hasattr(content_item, "text"):
                            return content_item.text
        return None

    @staticmethod
    def _extract_chat_completions_text(response) -> str | None:
        """Extract text content from a Chat Completions API response.

        Returns:
            Raw text string, or None if not found.
        """
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content
        return None

    def _build_responses_params(
        self,
        model: str,
        prompt: str,
        schema: dict,
        schema_name: str,
        max_tokens: int | None,
    ) -> dict:
        """Build request parameters for the Responses API."""
        params = {
            "model": model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        return params

    def _build_chat_completions_params(
        self,
        model: str,
        prompt: str,
        schema: dict,
        schema_name: str,
        max_tokens: int | None,
    ) -> dict:
        """Build request parameters for the Chat Completions API."""
        params: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params

    def _with_retry(self, fn, max_retries: int = _MAX_API_RETRIES):
        """Retry a sync API call on transient errors with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except _TRANSIENT_OPENAI_ERRORS as e:
                if attempt == max_retries:
                    raise
                wait = (2**attempt) + random.random()
                logger.warning(
                    f"[OpenAI] Transient error (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {wait:.1f}s"
                )
                time.sleep(wait)

    async def _with_retry_async(self, fn, max_retries: int = _MAX_API_RETRIES):
        """Retry an async API call on transient errors with exponential backoff."""
        import asyncio

        for attempt in range(max_retries + 1):
            try:
                return await fn()
            except _TRANSIENT_OPENAI_ERRORS as e:
                if attempt == max_retries:
                    raise
                wait = (2**attempt) + random.random()
                logger.warning(
                    f"[OpenAI] Transient error (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)

    @property
    def default_simple_model(self) -> str:
        return "gpt-5-mini"

    @property
    def default_reasoning_model(self) -> str:
        return "gpt-5"

    @property
    def default_research_model(self) -> str:
        return "gpt-5"

    def _get_client(self) -> OpenAI:
        if self._is_azure:
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return OpenAI(api_key=self._api_key)

    def _get_async_client(self) -> AsyncOpenAI:
        if self._cached_async_client is None:
            if self._is_azure:
                from openai import AsyncAzureOpenAI

                self._cached_async_client = AsyncAzureOpenAI(
                    api_key=self._api_key,
                    azure_endpoint=self._azure_endpoint,
                    api_version=self._api_version,
                )
            else:
                self._cached_async_client = AsyncOpenAI(api_key=self._api_key)
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
        model = self._resolve_model(model, self.default_simple_model)
        client = self._get_client()

        # Acquire rate limit capacity before making the call
        self._acquire_rate_limit(prompt, model, max_output=max_tokens or 4096)

        use_chat = self._api_format == "chat_completions"

        if use_chat:
            request_params = self._build_chat_completions_params(
                model, prompt, response_schema, schema_name, max_tokens
            )
        else:
            request_params = self._build_responses_params(
                model, prompt, response_schema, schema_name, max_tokens
            )

        logger.info(f"[LLM] simple_call starting - model={model}, schema={schema_name}")
        logger.info(f"[LLM] prompt length: {len(prompt)} chars")

        api_start = time.time()
        if use_chat:
            response = self._with_retry(
                lambda: client.chat.completions.create(**request_params)
            )
        else:
            response = self._with_retry(
                lambda: client.responses.create(**request_params)
            )
        api_elapsed = time.time() - api_start

        logger.info(f"[LLM] API response received in {api_elapsed:.2f}s")

        # Extract structured data
        if use_chat:
            raw_text = self._extract_chat_completions_text(response)
        else:
            raw_text = self._extract_output_text(response)
        structured_data = json.loads(raw_text) if raw_text else None

        if log:
            log_request_response(
                function_name="simple_call",
                request=request_params,
                response=response,
                provider="openai",
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
        model = self._resolve_model(model, self.default_simple_model)
        client = self._get_async_client()

        use_chat = self._api_format == "chat_completions"

        if use_chat:
            request_params = self._build_chat_completions_params(
                model, prompt, response_schema, schema_name, max_tokens
            )
        else:
            request_params = self._build_responses_params(
                model, prompt, response_schema, schema_name, max_tokens
            )

        if use_chat:
            response = await self._with_retry_async(
                lambda: client.chat.completions.create(**request_params)
            )
        else:
            response = await self._with_retry_async(
                lambda: client.responses.create(**request_params)
            )

        # Extract structured data
        if use_chat:
            raw_text = self._extract_chat_completions_text(response)
        else:
            raw_text = self._extract_output_text(response)
        structured_data = json.loads(raw_text) if raw_text else None

        # Extract token usage
        usage = TokenUsage()
        if hasattr(response, "usage") and response.usage is not None:
            if use_chat:
                usage = TokenUsage(
                    input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                )
            else:
                usage = TokenUsage(
                    input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                )

        return structured_data or {}, usage

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
        model = self._resolve_model(model, self.default_reasoning_model)
        client = self._get_client()

        effective_prompt = prompt
        if previous_errors:
            effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

        def _call(ep: str) -> dict:
            # Acquire rate limit capacity before each API call
            self._acquire_rate_limit(ep, model, max_output=16384)

            request_params = {
                "model": model,
                "reasoning": {"effort": reasoning_effort},
                "input": [{"role": "user", "content": ep}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": response_schema,
                    }
                },
            }
            response = self._with_retry(
                lambda: client.responses.create(**request_params)
            )
            raw_text = self._extract_output_text(response)
            structured_data = json.loads(raw_text) if raw_text else None
            if log:
                log_request_response(
                    function_name="reasoning_call",
                    request=request_params,
                    response=response,
                    provider="openai",
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
        model = self._resolve_model(model, self.default_research_model)
        client = self._get_client()

        effective_prompt = prompt
        if previous_errors:
            effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

        all_sources: list[str] = []

        def _call(ep: str) -> dict:
            # Acquire rate limit capacity before each API call
            self._acquire_rate_limit(ep, model, max_output=16384)

            request_params = {
                "model": model,
                "input": ep,
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

            response = self._with_retry(
                lambda: client.responses.create(**request_params)
            )

            raw_text = self._extract_output_text(response)
            structured_data = json.loads(raw_text) if raw_text else None
            sources: list[str] = []

            for item in response.output:
                if hasattr(item, "type") and item.type == "web_search_call":
                    if hasattr(item, "action") and item.action:
                        if hasattr(item.action, "sources") and item.action.sources:
                            for source in item.action.sources:
                                if isinstance(source, dict):
                                    if "url" in source:
                                        sources.append(source["url"])
                                elif hasattr(source, "url"):
                                    sources.append(source.url)

                if hasattr(item, "type") and item.type == "message":
                    for content_item in item.content:
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

            all_sources.extend(sources)

            if log:
                log_request_response(
                    function_name="agentic_research",
                    request=request_params,
                    response=response,
                    provider="openai",
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
