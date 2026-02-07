"""Token bucket rate limiter for LLM API calls.

Provides provider-aware rate limiting that sits between the simulation
engine and LLM providers. Supports:
- Dual bucket (RPM + TPM) for OpenAI
- Triple bucket (RPM + ITPM + OTPM) for Anthropic
- DualRateLimiter for separate pivotal/routine model pacing

Usage:
    limiter = RateLimiter.for_provider("openai", "gpt-5", tier=1)
    await limiter.acquire(estimated_input_tokens=600, estimated_output_tokens=200)

    # For simulation with two models:
    dual = DualRateLimiter.create(
        provider="openai",
        pivotal_model="gpt-5",
        routine_model="gpt-5-mini",
        tier=1,
    )
    await dual.pivotal.acquire(...)
    await dual.routine.acquire(...)
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
    """Provider-aware rate limiter with RPM + token buckets.

    For OpenAI: dual bucket (RPM + TPM).
    For Anthropic: triple bucket (RPM + ITPM + OTPM).
    """

    def __init__(
        self,
        rpm: int,
        tpm: int = 0,
        itpm: int = 0,
        otpm: int = 0,
        provider: str = "",
        model: str = "",
    ):
        """Initialize rate limiter with explicit limits.

        For OpenAI, pass tpm. For Anthropic, pass itpm and otpm.
        If only tpm is provided, it's used as a combined token limit.

        Args:
            rpm: Requests per minute limit
            tpm: Tokens per minute (combined, for OpenAI)
            itpm: Input tokens per minute (for Anthropic)
            otpm: Output tokens per minute (for Anthropic)
            provider: Provider name (for logging)
            model: Model name (for logging)
        """
        self.provider = provider
        self.model = model
        self.rpm = rpm
        self.tpm = tpm
        self.itpm = itpm
        self.otpm = otpm
        self._has_split_tokens = itpm > 0 and otpm > 0

        # RPM bucket
        self.rpm_bucket = TokenBucket(
            capacity=float(rpm),
            refill_rate=rpm / 60.0,
        )

        if self._has_split_tokens:
            # Anthropic: separate input and output token buckets
            self.itpm_bucket = TokenBucket(
                capacity=float(itpm),
                refill_rate=itpm / 60.0,
            )
            self.otpm_bucket = TokenBucket(
                capacity=float(otpm),
                refill_rate=otpm / 60.0,
            )
            self.tpm_bucket = None
        else:
            # OpenAI: single combined token bucket
            effective_tpm = tpm or (itpm + otpm) or 100_000
            self.tpm_bucket = TokenBucket(
                capacity=float(effective_tpm),
                refill_rate=effective_tpm / 60.0,
            )
            self.itpm_bucket = None
            self.otpm_bucket = None

        # Track stats
        self.total_acquired = 0
        self.total_wait_time = 0.0

        if self._has_split_tokens:
            logger.info(
                f"[RATE_LIMIT] Initialized for {provider}/{model}: "
                f"RPM={rpm}, ITPM={itpm}, OTPM={otpm}, "
                f"max_concurrent≈{self.max_safe_concurrent}"
            )
        else:
            logger.info(
                f"[RATE_LIMIT] Initialized for {provider}/{model}: "
                f"RPM={rpm}, TPM={tpm or itpm + otpm}, "
                f"max_concurrent≈{self.max_safe_concurrent}"
            )

    @property
    def max_safe_concurrent(self) -> int:
        """Calculate max safe concurrent requests.

        Sized to match what the API can absorb in a 2-second window,
        since asyncio.gather fires all permitted tasks near-simultaneously.
        With 500 RPM, that's ~8 RPM/60*2 ≈ 16 requests per 2s burst.
        """
        burst_window = 2.0  # seconds — tasks fire within this window
        avg_tokens_per_call = 800  # input + output estimate

        rpm_concurrent = self.rpm / 60.0 * burst_window

        if self._has_split_tokens:
            tpm_concurrent = self.itpm / avg_tokens_per_call * (burst_window / 60.0)
        else:
            effective_tpm = self.tpm or 100_000
            tpm_concurrent = effective_tpm / avg_tokens_per_call * (burst_window / 60.0)

        return max(1, int(min(rpm_concurrent, tpm_concurrent)))

    async def acquire(
        self,
        estimated_input_tokens: int = 600,
        estimated_output_tokens: int = 200,
    ) -> float:
        """Wait until we have capacity, then consume.

        Args:
            estimated_input_tokens: Estimated input tokens for the request
            estimated_output_tokens: Estimated output tokens for the request

        Returns:
            Actual wait time in seconds (0 if no wait needed)
        """
        total_wait = 0.0

        # Cap estimates to bucket capacity to avoid infinite loops
        # (can't acquire more than capacity, so just use capacity as upper bound)
        if self._has_split_tokens:
            estimated_input_tokens = min(
                estimated_input_tokens, int(self.itpm_bucket.capacity * 0.9)
            )
            estimated_output_tokens = min(
                estimated_output_tokens, int(self.otpm_bucket.capacity * 0.9)
            )
        else:
            max_total = int(self.tpm_bucket.capacity * 0.9)
            if estimated_input_tokens + estimated_output_tokens > max_total:
                # Scale down proportionally
                ratio = max_total / (estimated_input_tokens + estimated_output_tokens)
                estimated_input_tokens = int(estimated_input_tokens * ratio)
                estimated_output_tokens = int(estimated_output_tokens * ratio)

        estimated_total = estimated_input_tokens + estimated_output_tokens

        while True:
            # Check RPM bucket
            rpm_wait = self.rpm_bucket.try_acquire(1.0)

            if self._has_split_tokens:
                # Anthropic: check both input and output token buckets
                itpm_wait = self.itpm_bucket.try_acquire(float(estimated_input_tokens))
                otpm_wait = self.otpm_bucket.try_acquire(float(estimated_output_tokens))

                if rpm_wait == 0.0 and itpm_wait == 0.0 and otpm_wait == 0.0:
                    self.total_acquired += 1
                    self.total_wait_time += total_wait
                    return total_wait

                # Release what was acquired
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens += 1.0
                if itpm_wait == 0.0:
                    self.itpm_bucket.tokens += float(estimated_input_tokens)
                if otpm_wait == 0.0:
                    self.otpm_bucket.tokens += float(estimated_output_tokens)

                wait_time = max(rpm_wait, itpm_wait, otpm_wait)
            else:
                # OpenAI: check combined token bucket
                tpm_wait = self.tpm_bucket.try_acquire(float(estimated_total))

                if rpm_wait == 0.0 and tpm_wait == 0.0:
                    self.total_acquired += 1
                    self.total_wait_time += total_wait
                    return total_wait

                # Release what was acquired
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens += 1.0
                if tpm_wait == 0.0:
                    self.tpm_bucket.tokens += float(estimated_total)

                wait_time = max(rpm_wait, tpm_wait)

            # Cap single wait to 30 seconds to stay responsive
            wait_time = min(wait_time, 30.0)
            total_wait += wait_time

            if total_wait > 0.5:  # Only log if significant wait
                logger.debug(
                    f"[RATE_LIMIT] {self.provider}/{self.model} waiting {wait_time:.1f}s "
                    f"(total_wait={total_wait:.1f}s)"
                )

            await asyncio.sleep(wait_time)

    def acquire_sync(
        self,
        estimated_input_tokens: int = 600,
        estimated_output_tokens: int = 200,
    ) -> float:
        """Blocking wait until we have capacity, then consume.

        Sync version of acquire() for use in pipeline (non-async) code paths.

        Args:
            estimated_input_tokens: Estimated input tokens for the request
            estimated_output_tokens: Estimated output tokens for the request

        Returns:
            Actual wait time in seconds (0 if no wait needed)
        """
        total_wait = 0.0

        # Cap estimates to bucket capacity to avoid infinite loops
        # (can't acquire more than capacity, so just use capacity as upper bound)
        if self._has_split_tokens:
            estimated_input_tokens = min(
                estimated_input_tokens, int(self.itpm_bucket.capacity * 0.9)
            )
            estimated_output_tokens = min(
                estimated_output_tokens, int(self.otpm_bucket.capacity * 0.9)
            )
        else:
            max_total = int(self.tpm_bucket.capacity * 0.9)
            if estimated_input_tokens + estimated_output_tokens > max_total:
                # Scale down proportionally
                ratio = max_total / (estimated_input_tokens + estimated_output_tokens)
                estimated_input_tokens = int(estimated_input_tokens * ratio)
                estimated_output_tokens = int(estimated_output_tokens * ratio)

        estimated_total = estimated_input_tokens + estimated_output_tokens

        while True:
            # Check RPM bucket
            rpm_wait = self.rpm_bucket.try_acquire(1.0)

            if self._has_split_tokens:
                # Anthropic: check both input and output token buckets
                itpm_wait = self.itpm_bucket.try_acquire(float(estimated_input_tokens))
                otpm_wait = self.otpm_bucket.try_acquire(float(estimated_output_tokens))

                if rpm_wait == 0.0 and itpm_wait == 0.0 and otpm_wait == 0.0:
                    self.total_acquired += 1
                    self.total_wait_time += total_wait
                    return total_wait

                # Release what was acquired
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens += 1.0
                if itpm_wait == 0.0:
                    self.itpm_bucket.tokens += float(estimated_input_tokens)
                if otpm_wait == 0.0:
                    self.otpm_bucket.tokens += float(estimated_output_tokens)

                wait_time = max(rpm_wait, itpm_wait, otpm_wait)
            else:
                # OpenAI: check combined token bucket
                tpm_wait = self.tpm_bucket.try_acquire(float(estimated_total))

                if rpm_wait == 0.0 and tpm_wait == 0.0:
                    self.total_acquired += 1
                    self.total_wait_time += total_wait
                    return total_wait

                # Release what was acquired
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens += 1.0
                if tpm_wait == 0.0:
                    self.tpm_bucket.tokens += float(estimated_total)

                wait_time = max(rpm_wait, tpm_wait)

            # Cap single wait to 30 seconds to stay responsive
            wait_time = min(wait_time, 30.0)
            total_wait += wait_time

            if total_wait > 0.5:  # Only log if significant wait
                logger.info(
                    f"[RATE_LIMIT] {self.provider}/{self.model} waiting {wait_time:.1f}s "
                    f"(total_wait={total_wait:.1f}s)"
                )

            time.sleep(wait_time)

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
                # Drain all buckets to force waiting
                self.rpm_bucket.tokens = 0
                if self.tpm_bucket:
                    self.tpm_bucket.tokens = 0
                if self.itpm_bucket:
                    self.itpm_bucket.tokens = 0
                if self.otpm_bucket:
                    self.otpm_bucket.tokens = 0
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
            tpm_override: Override TPM limit (applies to tpm or itpm+otpm)

        Returns:
            Configured RateLimiter instance
        """
        limits = get_limits(provider, model, tier)

        rpm = rpm_override or limits.get("rpm", 50)

        # Check if provider uses split tokens (Anthropic)
        has_itpm = "itpm" in limits
        if has_itpm and not tpm_override:
            return cls(
                rpm=rpm,
                itpm=limits["itpm"],
                otpm=limits.get("otpm", 8_000),
                provider=provider,
                model=model,
            )
        else:
            # OpenAI or override — use combined TPM
            tpm = tpm_override or limits.get("tpm", limits.get("otpm", 100_000))
            return cls(
                rpm=rpm,
                tpm=tpm,
                provider=provider,
                model=model,
            )

    def stats(self) -> dict:
        """Return rate limiter statistics."""
        result = {
            "provider": self.provider,
            "model": self.model,
            "rpm_limit": self.rpm,
            "max_safe_concurrent": self.max_safe_concurrent,
            "total_acquired": self.total_acquired,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
        }
        if self._has_split_tokens:
            result["itpm_limit"] = self.itpm
            result["otpm_limit"] = self.otpm
        else:
            result["tpm_limit"] = self.tpm
        return result


class DualRateLimiter:
    """Manages separate rate limiters for pivotal (Pass 1) and routine (Pass 2) models.

    When pivotal and routine models are the same, uses a single shared limiter.
    When they differ, uses independent limiters since API limits are per-model.
    """

    def __init__(
        self,
        pivotal: RateLimiter,
        routine: RateLimiter,
    ):
        self.pivotal = pivotal
        self.routine = routine

    @classmethod
    def create(
        cls,
        provider: str,
        pivotal_model: str = "",
        routine_model: str = "",
        tier: int | None = None,
        rpm_override: int | None = None,
        tpm_override: int | None = None,
    ) -> "DualRateLimiter":
        """Create dual rate limiter for two-pass reasoning.

        If both models are the same (or routine is empty), a single
        shared limiter is used for both passes.

        Args:
            provider: Provider name
            pivotal_model: Model for Pass 1 (role-play reasoning)
            routine_model: Model for Pass 2 (classification)
            tier: Rate limit tier (1-4)
            rpm_override: Override RPM (applies to pivotal limiter)
            tpm_override: Override TPM (applies to pivotal limiter)

        Returns:
            DualRateLimiter instance
        """
        pivotal_limiter = RateLimiter.for_provider(
            provider=provider,
            model=pivotal_model,
            tier=tier,
            rpm_override=rpm_override,
            tpm_override=tpm_override,
        )

        # If routine model is the same as pivotal (or not specified), share the limiter
        effective_routine = routine_model or pivotal_model
        if effective_routine == pivotal_model or not effective_routine:
            return cls(pivotal=pivotal_limiter, routine=pivotal_limiter)

        # Different models — create separate limiter for routine
        # Overrides apply to both (on Azure, limits are per-resource not per-model)
        routine_limiter = RateLimiter.for_provider(
            provider=provider,
            model=effective_routine,
            tier=tier,
            rpm_override=rpm_override,
            tpm_override=tpm_override,
        )

        return cls(pivotal=pivotal_limiter, routine=routine_limiter)

    @property
    def max_safe_concurrent(self) -> int:
        """Return the more conservative of the two limiters' concurrency."""
        return min(self.pivotal.max_safe_concurrent, self.routine.max_safe_concurrent)

    def stats(self) -> dict:
        """Return combined stats from both limiters."""
        result = {"pivotal": self.pivotal.stats()}
        if self.routine is not self.pivotal:
            result["routine"] = self.routine.stats()
        return result
