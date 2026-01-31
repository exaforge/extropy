"""Token bucket rate limiter for LLM API calls.

Provides provider-aware rate limiting that sits between the simulation
engine and LLM providers. Dual bucket (RPM + TPM) with auto-pacing.

Usage:
    limiter = RateLimiter.for_provider("openai", "gpt-5")
    await limiter.acquire(estimated_tokens=800)
    # ... make API call ...
    limiter.update_from_headers(response_headers)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .rate_limits import get_limits

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Tokens refill continuously at `refill_rate` per second,
    up to `capacity`. Each acquire() consumes tokens.
    """

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_acquire(self, amount: float = 1.0) -> float:
        """Try to acquire tokens. Returns wait time in seconds if insufficient.

        Returns:
            0.0 if acquired, or positive float = seconds to wait
        """
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return 0.0
        # Calculate wait time
        deficit = amount - self.tokens
        return deficit / self.refill_rate if self.refill_rate > 0 else 60.0

    def update_capacity(self, new_capacity: float) -> None:
        """Update bucket capacity (e.g., from response headers)."""
        self._refill()
        self.capacity = new_capacity
        self.refill_rate = new_capacity / 60.0  # per-minute limits
        self.tokens = min(self.tokens, new_capacity)


class RateLimiter:
    """Provider-aware rate limiter with dual RPM + TPM buckets.

    Tracks both requests-per-minute and tokens-per-minute simultaneously,
    blocking on whichever is tighter.
    """

    def __init__(
        self,
        rpm: int,
        tpm: int,
        provider: str = "",
        model: str = "",
    ):
        """Initialize rate limiter with explicit limits.

        Args:
            rpm: Requests per minute limit
            tpm: Tokens per minute limit (total for OpenAI, output for Anthropic)
            provider: Provider name (for logging)
            model: Model name (for logging)
        """
        self.provider = provider
        self.model = model
        self.rpm = rpm
        self.tpm = tpm

        # Create buckets
        self.rpm_bucket = TokenBucket(
            capacity=float(rpm),
            refill_rate=rpm / 60.0,
        )
        self.tpm_bucket = TokenBucket(
            capacity=float(tpm),
            refill_rate=tpm / 60.0,
        )

        # Track stats
        self.total_acquired = 0
        self.total_wait_time = 0.0

        logger.info(
            f"[RATE_LIMIT] Initialized for {provider}/{model}: "
            f"RPM={rpm}, TPM={tpm}, max_concurrent≈{self.max_safe_concurrent}"
        )

    @property
    def max_safe_concurrent(self) -> int:
        """Calculate max safe concurrent requests.

        Based on RPM and TPM limits with average call characteristics.
        """
        avg_call_duration = 5.0  # seconds
        avg_tokens_per_call = 800  # input + output estimate

        rpm_concurrent = self.rpm / 60.0 * avg_call_duration
        tpm_concurrent = self.tpm / avg_tokens_per_call

        return max(1, int(min(rpm_concurrent, tpm_concurrent)))

    async def acquire(self, estimated_tokens: int = 800) -> float:
        """Wait until we have capacity, then consume.

        Args:
            estimated_tokens: Estimated total tokens for the request

        Returns:
            Actual wait time in seconds (0 if no wait needed)
        """
        total_wait = 0.0

        while True:
            # Check both buckets
            rpm_wait = self.rpm_bucket.try_acquire(1.0)
            tpm_wait = self.tpm_bucket.try_acquire(float(estimated_tokens))

            if rpm_wait == 0.0 and tpm_wait == 0.0:
                # Both acquired successfully
                self.total_acquired += 1
                self.total_wait_time += total_wait
                return total_wait

            # Need to wait — release what we acquired and sleep
            if rpm_wait == 0.0:
                # RPM was acquired but TPM wasn't — give back the RPM token
                self.rpm_bucket.tokens += 1.0
            if tpm_wait == 0.0:
                # TPM was acquired but RPM wasn't — give back TPM tokens
                self.tpm_bucket.tokens += float(estimated_tokens)

            wait_time = max(rpm_wait, tpm_wait)
            # Cap single wait to 30 seconds to stay responsive
            wait_time = min(wait_time, 30.0)
            total_wait += wait_time

            if total_wait > 0.5:  # Only log if significant wait
                logger.debug(
                    f"[RATE_LIMIT] Waiting {wait_time:.1f}s "
                    f"(rpm_wait={rpm_wait:.1f}s, tpm_wait={tpm_wait:.1f}s)"
                )

            await asyncio.sleep(wait_time)

    def update_from_headers(self, headers: dict[str, str] | None) -> None:
        """Adjust limits based on API response headers.

        Parses both Anthropic and OpenAI rate limit headers.

        Args:
            headers: Response headers dict (or None to skip)
        """
        if not headers:
            return

        # Anthropic headers
        remaining_requests = headers.get("anthropic-ratelimit-requests-remaining")
        remaining_tokens = headers.get(
            "anthropic-ratelimit-output-tokens-remaining",
            headers.get("anthropic-ratelimit-input-tokens-remaining"),
        )

        # OpenAI headers
        if remaining_requests is None:
            remaining_requests = headers.get("x-ratelimit-remaining-requests")
        if remaining_tokens is None:
            remaining_tokens = headers.get("x-ratelimit-remaining-tokens")

        # Retry-after (both providers)
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                wait = float(retry_after)
                logger.warning(f"[RATE_LIMIT] Server requested retry-after={wait}s")
                # Drain both buckets to force waiting
                self.rpm_bucket.tokens = 0
                self.tpm_bucket.tokens = 0
            except ValueError:
                pass

    @classmethod
    def for_provider(
        cls,
        provider: str,
        model: str = "",
        tier: int | None = None,
        rpm_override: int | None = None,
        tpm_override: int | None = None,
    ) -> "RateLimiter":
        """Factory with sensible defaults per provider/model.

        Args:
            provider: Provider name ('openai', 'claude', 'anthropic')
            model: Model name
            tier: Tier number (1-4, None = Tier 1)
            rpm_override: Override RPM limit
            tpm_override: Override TPM limit

        Returns:
            Configured RateLimiter instance
        """
        limits = get_limits(provider, model, tier)

        rpm = rpm_override or limits.get("rpm", 50)
        # For TPM: OpenAI uses 'tpm', Anthropic uses 'otpm' (output tokens per minute)
        tpm = tpm_override or limits.get("tpm", limits.get("otpm", 100_000))

        return cls(
            rpm=rpm,
            tpm=tpm,
            provider=provider,
            model=model,
        )

    def stats(self) -> dict:
        """Return rate limiter statistics."""
        return {
            "provider": self.provider,
            "model": self.model,
            "rpm_limit": self.rpm,
            "tpm_limit": self.tpm,
            "max_safe_concurrent": self.max_safe_concurrent,
            "total_acquired": self.total_acquired,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
        }
