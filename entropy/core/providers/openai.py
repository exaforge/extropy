"""OpenAI LLM Provider implementation."""

import json
import logging
import time

from openai import OpenAI, AsyncOpenAI

from .base import LLMProvider, ValidatorCallback, RetryCallback
from .logging import log_request_response, extract_error_summary


logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using the Responses API."""

    def __init__(self, api_key: str = "") -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it as an environment variable.\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        super().__init__(api_key)

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
        return OpenAI(api_key=self._api_key)

    def _get_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self._api_key)

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

        logger.info(f"[LLM] simple_call starting - model={model}, schema={schema_name}")
        logger.info(f"[LLM] prompt length: {len(prompt)} chars")

        api_start = time.time()
        response = client.responses.create(**request_params)
        api_elapsed = time.time() - api_start

        logger.info(f"[LLM] API response received in {api_elapsed:.2f}s")

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
    ) -> dict:
        model = model or self.default_simple_model
        client = self._get_async_client()

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
                    if (
                        hasattr(content_item, "type")
                        and content_item.type == "output_text"
                    ):
                        if hasattr(content_item, "text"):
                            structured_data = json.loads(content_item.text)

        return structured_data or {}

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
        model = model or self.default_reasoning_model
        client = self._get_client()

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
                log_request_response(
                    function_name="reasoning_call",
                    request=request_params,
                    response=response,
                    provider="openai",
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
        model = model or self.default_research_model
        client = self._get_client()

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

            all_sources.extend(sources)

            if log:
                log_request_response(
                    function_name="agentic_research",
                    request=request_params,
                    response=response,
                    provider="openai",
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
