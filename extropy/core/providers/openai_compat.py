"""OpenAI-compatible LLM Provider for third-party endpoints.

Supports any provider that implements the OpenAI Chat Completions API:
- OpenRouter, DeepSeek, Together, Groq, Azure OpenAI, etc.

Uses `openai.OpenAI(base_url=...)` with Chat Completions API for all calls.
Supports `json_schema` response format for structured output.
For agentic_research, appends `:online` to model name if provider supports search,
and parses `url_citation` annotations for sources.
"""

import json
import logging
import random
import time

import openai
from openai import OpenAI, AsyncOpenAI

from .base import LLMProvider, TokenUsage, ValidatorCallback, RetryCallback
from .logging import log_request_response, extract_error_summary

_TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)
_MAX_API_RETRIES = 3

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    """OpenAI-compatible provider for third-party endpoints.

    Uses the Chat Completions API with json_schema response format.
    """

    def __init__(
        self,
        api_key: str = "",
        *,
        base_url: str = "",
        supports_search: bool = False,
        provider_label: str = "openai_compat",
        default_fast: str = "gpt-5-mini",
        default_strong: str = "gpt-5",
    ) -> None:
        if not api_key:
            raise ValueError(
                f"API key not found for {provider_label}. "
                f"Set it as an environment variable."
            )
        super().__init__(api_key)
        self._base_url = base_url
        self._supports_search = supports_search
        self.provider_name = provider_label
        self._default_fast = default_fast
        self._default_strong = default_strong

    @property
    def default_fast_model(self) -> str:
        return self._default_fast

    @property
    def default_strong_model(self) -> str:
        return self._default_strong

    def _get_client(self) -> OpenAI:
        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def _get_async_client(self) -> AsyncOpenAI:
        if self._cached_async_client is None:
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._cached_async_client = AsyncOpenAI(**kwargs)
        return self._cached_async_client

    def _build_params(
        self,
        model: str,
        prompt: str,
        schema: dict,
        schema_name: str,
        max_tokens: int | None,
    ) -> dict:
        """Build Chat Completions API request parameters."""
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

    @staticmethod
    def _extract_text(response) -> str | None:
        """Extract text from Chat Completions response."""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content
        return None

    @staticmethod
    def _extract_sources(response) -> list[str]:
        """Extract citation URLs from response annotations."""
        sources: list[str] = []
        if not response.choices:
            return sources
        message = response.choices[0].message
        if hasattr(message, "annotations") and message.annotations:
            for annotation in message.annotations:
                if hasattr(annotation, "type") and annotation.type == "url_citation":
                    if hasattr(annotation, "url"):
                        sources.append(annotation.url)
        return sources

    def _with_retry(self, fn, max_retries: int = _MAX_API_RETRIES):
        """Retry on transient errors with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except _TRANSIENT_ERRORS as e:
                if attempt == max_retries:
                    raise
                wait = (2**attempt) + random.random()
                lbl = self.provider_name
                att = f"{attempt + 1}/{max_retries + 1}"
                logger.warning(
                    f"[{lbl}] Transient error ({att}): "
                    f"{type(e).__name__}: {e}. "
                    f"Retrying in {wait:.1f}s"
                )
                time.sleep(wait)

    async def _with_retry_async(self, fn, max_retries: int = _MAX_API_RETRIES):
        """Async retry on transient errors."""
        import asyncio

        for attempt in range(max_retries + 1):
            try:
                return await fn()
            except _TRANSIENT_ERRORS as e:
                if attempt == max_retries:
                    raise
                wait = (2**attempt) + random.random()
                lbl = self.provider_name
                att = f"{attempt + 1}/{max_retries + 1}"
                logger.warning(
                    f"[{lbl}] Transient error ({att}): "
                    f"{type(e).__name__}: {e}. "
                    f"Retrying in {wait:.1f}s"
                )
                await asyncio.sleep(wait)

    def simple_call(
        self,
        prompt: str,
        response_schema: dict,
        schema_name: str = "response",
        model: str | None = None,
        log: bool = True,
        max_tokens: int | None = None,
    ) -> dict:
        model = model or self.default_fast_model
        client = self._get_client()

        self._acquire_rate_limit(prompt, model, max_output=max_tokens or 4096)

        params = self._build_params(
            model,
            prompt,
            response_schema,
            schema_name,
            max_tokens,
        )
        lbl = self.provider_name
        logger.info(f"[{lbl}] simple_call model={model} schema={schema_name}")

        api_start = time.time()
        response = self._with_retry(lambda: client.chat.completions.create(**params))
        api_elapsed = time.time() - api_start
        logger.info(f"[{self.provider_name}] API response in {api_elapsed:.2f}s")

        raw_text = self._extract_text(response)
        structured_data = json.loads(raw_text) if raw_text else None

        if log:
            log_request_response(
                function_name="simple_call",
                request=params,
                response=response,
                provider=self.provider_name,
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
        model = model or self.default_fast_model
        client = self._get_async_client()

        params = self._build_params(
            model,
            prompt,
            response_schema,
            schema_name,
            max_tokens,
        )

        response = await self._with_retry_async(
            lambda: client.chat.completions.create(**params)
        )

        raw_text = self._extract_text(response)
        structured_data = json.loads(raw_text) if raw_text else None

        usage = TokenUsage()
        if hasattr(response, "usage") and response.usage is not None:
            usage = TokenUsage(
                input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
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
        model = model or self.default_strong_model
        client = self._get_client()

        effective_prompt = prompt
        if previous_errors:
            effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

        def _call(ep: str) -> dict:
            self._acquire_rate_limit(ep, model, max_output=16384)
            params = self._build_params(model, ep, response_schema, schema_name, None)
            response = self._with_retry(
                lambda: client.chat.completions.create(**params)
            )
            raw_text = self._extract_text(response)
            structured_data = json.loads(raw_text) if raw_text else None
            if log:
                log_request_response(
                    function_name="reasoning_call",
                    request=params,
                    response=response,
                    provider=self.provider_name,
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
        model = model or self.default_strong_model
        client = self._get_client()

        effective_prompt = prompt
        if previous_errors:
            effective_prompt = f"{previous_errors}\n\n---\n\n{prompt}"

        # For providers that support search, append :online suffix
        search_model = f"{model}:online" if self._supports_search else model

        all_sources: list[str] = []

        def _call(ep: str) -> dict:
            self._acquire_rate_limit(ep, model, max_output=16384)
            params = self._build_params(
                search_model, ep, response_schema, schema_name, None
            )
            response = self._with_retry(
                lambda: client.chat.completions.create(**params)
            )
            raw_text = self._extract_text(response)
            structured_data = json.loads(raw_text) if raw_text else None
            sources = self._extract_sources(response)
            all_sources.extend(sources)

            if log:
                log_request_response(
                    function_name="agentic_research",
                    request=params,
                    response=response,
                    provider=self.provider_name,
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
