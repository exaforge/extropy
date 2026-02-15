"""Azure AI Foundry provider — unified gateway for Claude + OpenAI-compat models.

Routes requests based on model family:
- claude-* models → Anthropic SDK via AnthropicProvider(base_url=<endpoint>/anthropic/)
- Everything else → OpenAI SDK via OpenAICompatProvider(base_url=<endpoint>/openai/v1/)

Both paths share the same API key and endpoint.
"""

import logging

from .base import LLMProvider, TokenUsage, ValidatorCallback, RetryCallback

logger = logging.getLogger(__name__)


class UnsupportedCapabilityError(ValueError):
    """Raised when a model doesn't support the requested capability."""


_BACKEND_CAPABILITIES = {
    "anthropic": {
        "simple_call",
        "simple_call_async",
        "reasoning_call",
        "agentic_research",
    },
    "openai": {"simple_call", "simple_call_async"},
}


def _detect_backend(model: str) -> str:
    """Detect which backend SDK to use based on model name."""
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


class AzureProvider(LLMProvider):
    """Azure AI Foundry provider that delegates to Anthropic or OpenAI sub-providers."""

    provider_name = "azure"

    def __init__(self, api_key: str, endpoint: str) -> None:
        if not api_key:
            raise ValueError(
                "Azure API key not found. Set it via:\n"
                "  export AZURE_API_KEY=<key>\n"
                "Or (legacy): export AZURE_OPENAI_API_KEY=<key>"
            )
        if not endpoint:
            raise ValueError(
                "Azure endpoint not found. Set it via:\n"
                "  export AZURE_ENDPOINT=https://<resource>.services.ai.azure.com/\n"
                "Or (legacy): export AZURE_OPENAI_ENDPOINT=<url>"
            )
        super().__init__(api_key)
        self._endpoint = endpoint.rstrip("/")
        self._openai_sub: LLMProvider | None = None
        self._anthropic_sub: LLMProvider | None = None

    @property
    def default_fast_model(self) -> str:
        return "gpt-5-mini"

    @property
    def default_strong_model(self) -> str:
        return "gpt-5"

    def _get_openai_sub(self) -> LLMProvider:
        if self._openai_sub is None:
            from .openai_compat import OpenAICompatProvider

            self._openai_sub = OpenAICompatProvider(
                api_key=self._api_key,
                base_url=f"{self._endpoint}/openai/v1/",
                provider_label="azure",
            )
        return self._openai_sub

    def _get_anthropic_sub(self) -> LLMProvider:
        if self._anthropic_sub is None:
            from .anthropic import AnthropicProvider

            self._anthropic_sub = AnthropicProvider(
                api_key=self._api_key,
                base_url=f"{self._endpoint}/anthropic/",
            )
        return self._anthropic_sub

    def _get_sub(self, model: str) -> LLMProvider:
        backend = _detect_backend(model)
        if backend == "anthropic":
            return self._get_anthropic_sub()
        return self._get_openai_sub()

    def _check_capability(self, model: str, capability: str) -> None:
        """Raise UnsupportedCapabilityError if the model's backend can't do this."""
        backend = _detect_backend(model)
        if capability not in _BACKEND_CAPABILITIES.get(backend, set()):
            raise UnsupportedCapabilityError(
                f"azure/{model} does not support {capability}.\n"
                f"Use a model that supports this feature, e.g.:\n"
                f"  - azure/claude-sonnet-4-5\n"
                f"  - openai/gpt-5\n"
                f"  - anthropic/claude-sonnet-4-5"
            )

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
        self._check_capability(model, "simple_call")
        return self._get_sub(model).simple_call(
            prompt=prompt,
            response_schema=response_schema,
            schema_name=schema_name,
            model=model,
            log=log,
            max_tokens=max_tokens,
        )

    async def simple_call_async(
        self,
        prompt: str,
        response_schema: dict,
        schema_name: str = "response",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict, TokenUsage]:
        model = model or self.default_fast_model
        self._check_capability(model, "simple_call_async")
        return await self._get_sub(model).simple_call_async(
            prompt=prompt,
            response_schema=response_schema,
            schema_name=schema_name,
            model=model,
            max_tokens=max_tokens,
        )

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
        self._check_capability(model, "reasoning_call")
        return self._get_sub(model).reasoning_call(
            prompt=prompt,
            response_schema=response_schema,
            schema_name=schema_name,
            model=model,
            reasoning_effort=reasoning_effort,
            log=log,
            previous_errors=previous_errors,
            validator=validator,
            max_retries=max_retries,
            on_retry=on_retry,
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
        self._check_capability(model, "agentic_research")
        return self._get_sub(model).agentic_research(
            prompt=prompt,
            response_schema=response_schema,
            schema_name=schema_name,
            model=model,
            reasoning_effort=reasoning_effort,
            log=log,
            previous_errors=previous_errors,
            validator=validator,
            max_retries=max_retries,
            on_retry=on_retry,
        )

    async def close_async(self) -> None:
        """Close both sub-providers' async clients."""
        if self._openai_sub is not None:
            await self._openai_sub.close_async()
        if self._anthropic_sub is not None:
            await self._anthropic_sub.close_async()
        self._cached_async_client = None
