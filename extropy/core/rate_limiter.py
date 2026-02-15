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
        """Max in-flight requests that RPM can sustain.

        With stagger pacing (1 launch per 60/rpm seconds), the number of
        calls in-flight at steady state = rpm/60 * avg_latency.
        We don't know latency upfront, so use RPM-derived ceiling:
        allow up to rpm/2 in-flight (30s worth of launches).

        Also capped by OS file descriptor limit — each concurrent request
        needs ~2 fds (socket + TLS), plus ~50 for Python runtime overhead.

        TPM is already enforced by token buckets during acquire().
        """
        import resource

        rpm_derived = self.rpm // 2
        try:
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        except (ValueError, OSError):
            soft_limit = 256  # conservative fallback
        # Each connection uses ~2 fds; reserve 100 for runtime overhead
        fd_derived = max(1, (soft_limit - 100) // 2)
        return max(1, min(rpm_derived, fd_derived))

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

                # Release what was acquired (capped to capacity)
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens = min(
                        self.rpm_bucket.capacity, self.rpm_bucket.tokens + 1.0
                    )
                if itpm_wait == 0.0:
                    self.itpm_bucket.tokens = min(
                        self.itpm_bucket.capacity,
                        self.itpm_bucket.tokens + float(estimated_input_tokens),
                    )
                if otpm_wait == 0.0:
                    self.otpm_bucket.tokens = min(
                        self.otpm_bucket.capacity,
                        self.otpm_bucket.tokens + float(estimated_output_tokens),
                    )

                wait_time = max(rpm_wait, itpm_wait, otpm_wait)
            else:
                # OpenAI: check combined token bucket
                tpm_wait = self.tpm_bucket.try_acquire(float(estimated_total))

                if rpm_wait == 0.0 and tpm_wait == 0.0:
                    self.total_acquired += 1
                    self.total_wait_time += total_wait
                    return total_wait

                # Release what was acquired (capped to capacity)
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens = min(
                        self.rpm_bucket.capacity, self.rpm_bucket.tokens + 1.0
                    )
                if tpm_wait == 0.0:
                    self.tpm_bucket.tokens = min(
                        self.tpm_bucket.capacity,
                        self.tpm_bucket.tokens + float(estimated_total),
                    )

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

                # Release what was acquired (capped to capacity)
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens = min(
                        self.rpm_bucket.capacity, self.rpm_bucket.tokens + 1.0
                    )
                if itpm_wait == 0.0:
                    self.itpm_bucket.tokens = min(
                        self.itpm_bucket.capacity,
                        self.itpm_bucket.tokens + float(estimated_input_tokens),
                    )
                if otpm_wait == 0.0:
                    self.otpm_bucket.tokens = min(
                        self.otpm_bucket.capacity,
                        self.otpm_bucket.tokens + float(estimated_output_tokens),
                    )

                wait_time = max(rpm_wait, itpm_wait, otpm_wait)
            else:
                # OpenAI: check combined token bucket
                tpm_wait = self.tpm_bucket.try_acquire(float(estimated_total))

                if rpm_wait == 0.0 and tpm_wait == 0.0:
                    self.total_acquired += 1
                    self.total_wait_time += total_wait
                    return total_wait

                # Release what was acquired (capped to capacity)
                if rpm_wait == 0.0:
                    self.rpm_bucket.tokens = min(
                        self.rpm_bucket.capacity, self.rpm_bucket.tokens + 1.0
                    )
                if tpm_wait == 0.0:
                    self.tpm_bucket.tokens = min(
                        self.tpm_bucket.capacity,
                        self.tpm_bucket.tokens + float(estimated_total),
                    )

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
    """Manages separate rate limiters for strong (Pass 1) and fast (Pass 2) models.

    When strong and fast models are the same, uses a single shared limiter.
    When they differ, uses independent limiters since API limits are per-model.
    Supports mixed providers (e.g., strong=anthropic, fast=openai).
    """

    def __init__(
        self,
        pivotal: RateLimiter,
        routine: RateLimiter,
    ):
        self.pivotal = pivotal
        self.routine = routine
        # Aliases for new naming convention
        self.strong = pivotal
        self.fast = routine

    @classmethod
    def create(
        cls,
        provider: str = "",
        pivotal_model: str = "",
        routine_model: str = "",
        tier: int | None = None,
        rpm_override: int | None = None,
        tpm_override: int | None = None,
        *,
        strong_model_string: str = "",
        fast_model_string: str = "",
    ) -> "DualRateLimiter":
        """Create dual rate limiter for two-pass reasoning.

        Accepts either:
        - Legacy: provider + pivotal_model + routine_model (single provider)
        - New: strong_model_string + fast_model_string (provider/model format, mixed providers)

        Args:
            provider: Provider name (legacy, used if model strings not provided)
            pivotal_model: Model for Pass 1 (legacy)
            routine_model: Model for Pass 2 (legacy)
            tier: Rate limit tier (1-4)
            rpm_override: Override RPM
            tpm_override: Override TPM
            strong_model_string: "provider/model" for strong/pivotal (new)
            fast_model_string: "provider/model" for fast/routine (new)

        Returns:
            DualRateLimiter instance
        """
        # Resolve strong limiter
        if strong_model_string and "/" in strong_model_string:
            from ..config import parse_model_string

            strong_provider, strong_model = parse_model_string(strong_model_string)
        else:
            strong_provider = provider
            strong_model = pivotal_model

        pivotal_limiter = RateLimiter.for_provider(
            provider=strong_provider,
            model=strong_model,
            tier=tier,
            rpm_override=rpm_override,
            tpm_override=tpm_override,
        )

        # Resolve fast limiter
        if fast_model_string and "/" in fast_model_string:
            from ..config import parse_model_string

            fast_provider, fast_model = parse_model_string(fast_model_string)
        else:
            fast_provider = provider
            fast_model = routine_model

        # If same provider+model, share the limiter
        effective_fast_model = fast_model or strong_model
        if fast_provider == strong_provider and effective_fast_model == strong_model:
            return cls(pivotal=pivotal_limiter, routine=pivotal_limiter)

        if not effective_fast_model and not fast_provider:
            return cls(pivotal=pivotal_limiter, routine=pivotal_limiter)

        routine_limiter = RateLimiter.for_provider(
            provider=fast_provider or strong_provider,
            model=effective_fast_model,
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
