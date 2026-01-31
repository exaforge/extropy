"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Callable


# Type for validation callbacks: takes response data, returns (is_valid, error_message)
ValidatorCallback = Callable[[dict], tuple[bool, str]]

# Type for retry notification callbacks: (attempt, max_retries, short_error_summary)
RetryCallback = Callable[[int, int, str], None]


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement these methods with the same signatures
    to ensure drop-in compatibility.

    Args:
        api_key: API key or access token for the provider.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    @abstractmethod
    def default_simple_model(self) -> str:
        """Default model for simple_call (fast, cheap)."""
        ...

    @property
    @abstractmethod
    def default_reasoning_model(self) -> str:
        """Default model for reasoning_call (balanced)."""
        ...

    @property
    @abstractmethod
    def default_research_model(self) -> str:
        """Default model for agentic_research (with web search)."""
        ...

    @abstractmethod
    def simple_call(
        self,
        prompt: str,
        response_schema: dict,
        schema_name: str = "response",
        model: str | None = None,
        log: bool = True,
        max_tokens: int | None = None,
    ) -> dict:
        """Simple LLM call with structured output, no reasoning, no web search."""
        ...

    @abstractmethod
    async def simple_call_async(
        self,
        prompt: str,
        response_schema: dict,
        schema_name: str = "response",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Async version of simple_call for concurrent API requests."""
        ...

    @abstractmethod
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
        """LLM call with reasoning and structured output, but NO web search."""
        ...

    @abstractmethod
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
        """Perform agentic research with web search and structured output."""
        ...
