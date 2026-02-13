"""Tests for the token bucket rate limiter."""

from unittest.mock import patch

import pytest

from extropy.core.rate_limiter import TokenBucket, RateLimiter


class TestTokenBucket:
    """Tests for the TokenBucket primitive."""

    def test_initial_capacity(self):
        bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
        assert bucket.tokens == 100.0

    def test_acquire_within_capacity(self):
        bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
        wait = bucket.try_acquire(50.0)
        assert wait == 0.0
        assert bucket.tokens == pytest.approx(50.0, abs=1.0)

    def test_acquire_exceeds_capacity(self):
        bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
        # Drain the bucket
        bucket.try_acquire(100.0)
        wait = bucket.try_acquire(50.0)
        assert wait > 0.0

    def test_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        start = 1000.0

        with patch("time.monotonic", return_value=start):
            bucket = TokenBucket(capacity=100.0, refill_rate=10.0)
            bucket.try_acquire(100.0)  # drain

        # Advance 5 seconds → refill 50 tokens
        with patch("time.monotonic", return_value=start + 5.0):
            wait = bucket.try_acquire(50.0)
            assert wait == 0.0

    def test_refill_capped_at_capacity(self):
        """Refill never exceeds capacity."""
        start = 1000.0

        with patch("time.monotonic", return_value=start):
            bucket = TokenBucket(capacity=100.0, refill_rate=10.0)

        # Advance a long time
        with patch("time.monotonic", return_value=start + 1000.0):
            bucket._refill()
            assert bucket.tokens == 100.0

    def test_update_capacity(self):
        start = 1000.0
        with patch("time.monotonic", return_value=start):
            bucket = TokenBucket(capacity=100.0, refill_rate=10.0)

        with patch("time.monotonic", return_value=start + 0.1):
            bucket.update_capacity(200.0)
            assert bucket.capacity == 200.0
            assert bucket.refill_rate == pytest.approx(200.0 / 60.0)


class TestRateLimiter:
    """Tests for the RateLimiter (dual/triple bucket)."""

    def test_max_safe_concurrent_openai(self):
        limiter = RateLimiter(rpm=500, tpm=500_000, provider="openai", model="test")
        concurrent = limiter.max_safe_concurrent
        assert concurrent >= 1
        # With 500 RPM, burst window of 2s: ~16
        assert concurrent <= 500

    def test_max_safe_concurrent_anthropic(self):
        limiter = RateLimiter(
            rpm=50, itpm=30_000, otpm=8_000, provider="anthropic", model="test"
        )
        concurrent = limiter.max_safe_concurrent
        assert concurrent >= 1

    def test_stats(self):
        limiter = RateLimiter(rpm=100, tpm=50_000, provider="openai", model="test")
        stats = limiter.stats()
        assert stats["provider"] == "openai"
        assert stats["model"] == "test"
        assert stats["rpm_limit"] == 100
        assert stats["tpm_limit"] == 50_000
        assert stats["total_acquired"] == 0

    def test_stats_anthropic_split_tokens(self):
        limiter = RateLimiter(
            rpm=50, itpm=30_000, otpm=8_000, provider="anthropic", model="test"
        )
        stats = limiter.stats()
        assert "itpm_limit" in stats
        assert "otpm_limit" in stats
        assert "tpm_limit" not in stats

    def test_for_provider_openai(self):
        limiter = RateLimiter.for_provider("openai", "gpt-5-mini", tier=1)
        assert limiter.rpm == 500
        assert limiter.tpm == 500_000

    def test_for_provider_anthropic(self):
        limiter = RateLimiter.for_provider("anthropic", "default", tier=1)
        assert limiter.rpm == 50

    def test_for_provider_unknown(self):
        limiter = RateLimiter.for_provider("unknown_provider")
        # Should use conservative defaults
        assert limiter.rpm == 50

    def test_rpm_override(self):
        limiter = RateLimiter.for_provider("openai", tier=1, rpm_override=999)
        assert limiter.rpm == 999

    def test_tpm_override(self):
        limiter = RateLimiter.for_provider("openai", tier=1, tpm_override=999_999)
        assert limiter.tpm == 999_999
