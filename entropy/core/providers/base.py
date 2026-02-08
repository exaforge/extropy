"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..rate_limiter import RateLimiter


@dataclass
class TokenUsage:
    """Token usage from a single LLM API call."""

    input_tokens: int = 0
    output_tokens: int = 0


# Type for validation callbacks: takes response data, returns (is_valid, error_message)
ValidatorCallback = Callable[[dict], tuple[bool, str]]

# Type for retry notification callbacks: (attempt, max_retries, short_error_summary)
RetryCallback = Callable[[int, int, str], None]


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough heuristic: ~4 chars per token)."""
    return max(100, len(text) // 4)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement these methods with the same signatures
    to ensure drop-in compatibility.

    Args:
        api_key: API key or access token for the provider.
    """

    # Provider name for rate limiter initialization (override in subclasses)
    provider_name: str = "unknown"

    # Set to True to disable rate limiting (for tests)
    _disable_rate_limiting: bool = False

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._cached_async_client = None
        self._rate_limiter: "RateLimiter | None" = None

    def _ensure_rate_limiter(self, model: str = "") -> "RateLimiter":
        """Lazily initialize rate limiter for this provider.

        Uses Tier 1 limits by default. Called before each API request.
        """
        # Defensive check for tests that use __new__ without __init__
        if not hasattr(self, "_rate_limiter") or self._rate_limiter is None:
            from ..rate_limiter import RateLimiter

            self._rate_limiter = RateLimiter.for_provider(
                provider=self.provider_name,
                model=model,
                tier=1,  # Default to Tier 1 (most restrictive)
            )
        return self._rate_limiter

    def _acquire_rate_limit(
        self, prompt: str, model: str = "", max_output: int = 1000
    ) -> float:
        """Acquire rate limit capacity before making an API call.

        Args:
            prompt: The prompt text (used to estimate input tokens)
            model: Model name (for rate limiter initialization)
            max_output: Estimated max output tokens

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        # Skip rate limiting if disabled (e.g., in tests)
        if getattr(self, "_disable_rate_limiting", False):
            return 0.0

        limiter = self._ensure_rate_limiter(model)
        estimated_input = estimate_tokens(prompt)
        return limiter.acquire_sync(
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=max_output,
        )

    async def close_async(self) -> None:
        """Close the cached async client to release connections cleanly.

        Must be called before the event loop shuts down to avoid
        'Event loop is closed' errors from orphaned httpx connections.
        """
        if self._cached_async_client is not None:
            await self._cached_async_client.close()
            self._cached_async_client = None

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
    ) -> tuple[dict, TokenUsage]:
        """Async version of simple_call for concurrent API requests."""
        ...

    def _retry_with_validation(
        self,
        call_fn,
        prompt: str,
        validator: ValidatorCallback | None,
        max_retries: int,
        on_retry: RetryCallback | None,
        extract_error_summary_fn,
        initial_prompt: str | None = None,
    ) -> dict:
        """Shared validation-retry loop for reasoning_call and agentic_research.

        Args:
            call_fn: Callable(effective_prompt) -> result_dict.
                     Called each attempt with the (possibly error-prepended) prompt.
            prompt: Base prompt text used as the suffix on validation retries.
            validator: Optional validator callback.
            max_retries: Max validation retries.
            on_retry: Optional retry notification callback.
            extract_error_summary_fn: Function to shorten error messages.
            initial_prompt: If provided, used for the first call instead of prompt.
                This allows previous_errors to be included on the first attempt
                without persisting them across validation retries.

        Returns:
            Validated result dict (or last attempt if retries exhausted).
        """
        effective_prompt = initial_prompt if initial_prompt is not None else prompt
        attempts = 0
        last_error_summary = ""
        result = {}

        while attempts <= max_retries:
            result = call_fn(effective_prompt)

            if validator is None:
                return result

            is_valid, error_msg = validator(result)
            if is_valid:
                return result

            attempts += 1
            last_error_summary = extract_error_summary_fn(error_msg)

            if attempts <= max_retries:
                if on_retry:
                    on_retry(attempts, max_retries, last_error_summary)
                effective_prompt = f"{error_msg}\n\n---\n\n{prompt}"

        if on_retry:
            on_retry(max_retries + 1, max_retries, f"EXHAUSTED: {last_error_summary}")
        return result

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
