"""OpenRouter client wrapper with provider-agnostic interface.

Provides a unified interface for accessing various LLM providers through
OpenRouter's API, which is compatible with the OpenAI client library.
"""

from functools import lru_cache
from typing import Any

from openai import OpenAI

from ..config import get_settings


class OpenRouterClient:
    """OpenRouter API client wrapper.

    Uses the OpenAI client library with OpenRouter's base URL.
    Handles provider-specific quirks and structured output differences.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, uses settings.
        """
        settings = get_settings()
        self._api_key = api_key or settings.openrouter_api_key

        if not self._api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY in .env"
            )

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self.BASE_URL,
        )

    @property
    def client(self) -> OpenAI:
        """Get the underlying OpenAI client."""
        return self._client

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion.

        Args:
            messages: Chat messages
            model: Model identifier (e.g., "openai/gpt-4o-mini")
            response_format: Optional response format specification
            tools: Optional tool definitions
            **kwargs: Additional parameters passed to the API

        Returns:
            Chat completion response
        """
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools

        return self._client.chat.completions.create(**params)

    def is_openai_model(self, model: str) -> bool:
        """Check if a model is an OpenAI model.

        Args:
            model: Model identifier

        Returns:
            True if this is an OpenAI model
        """
        return model.startswith("openai/")

    def is_anthropic_model(self, model: str) -> bool:
        """Check if a model is an Anthropic model.

        Args:
            model: Model identifier

        Returns:
            True if this is an Anthropic model
        """
        return model.startswith("anthropic/")

    def supports_structured_output(self, model: str) -> bool:
        """Check if a model supports JSON schema structured output.

        Some models support response_format with JSON schema,
        others may need prompting or other approaches.

        Args:
            model: Model identifier

        Returns:
            True if model supports structured JSON output
        """
        # Most modern models support JSON mode at minimum
        # OpenAI models support full JSON schema
        # Anthropic models support structured output via their API
        supported_prefixes = (
            "openai/gpt-4",
            "openai/gpt-3.5",
            "anthropic/claude",
            "google/gemini",
            "meta-llama/llama-3",
        )
        return any(model.startswith(prefix) for prefix in supported_prefixes)

    def supports_web_search(self, model: str) -> bool:
        """Check if a model supports web search tools.

        Currently, only OpenAI models via OpenRouter support
        the web_search tool that the Responses API uses.
        For other providers, we may need alternative approaches.

        Args:
            model: Model identifier

        Returns:
            True if model supports web search
        """
        # OpenAI models support web search through OpenRouter
        return model.startswith("openai/")

    def get_reasoning_params(self, model: str, effort: str) -> dict[str, Any]:
        """Get provider-specific reasoning parameters.

        Different providers have different ways of enabling reasoning/thinking.
        This method returns the appropriate parameters for each provider.

        Args:
            model: Model identifier
            effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            Dictionary of provider-specific parameters for reasoning
        """
        # OpenAI models (o1, etc.) use reasoning parameter
        if model.startswith("openai/o"):
            return {"reasoning": {"effort": effort}}

        # For other models, we might use different parameters
        # or include reasoning instructions in the prompt
        return {}


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenRouterClient:
    """Get cached OpenRouter client instance.

    Returns:
        Cached OpenRouterClient instance
    """
    return OpenRouterClient()
