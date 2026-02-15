"""Tests for resource auto-tuning and runtime guardrails."""

import asyncio

from extropy.core.rate_limiter import RateLimiter
from extropy.utils.resource_governor import ResourceGovernor


def test_downshift_int_respects_minimum():
    assert ResourceGovernor.downshift_int(100, factor=0.5, minimum=1) == 50
    assert ResourceGovernor.downshift_int(2, factor=0.1, minimum=4) == 4


def test_memory_pressure_ratio_uses_budget(monkeypatch):
    governor = ResourceGovernor(resource_mode="auto", max_memory_gb=8.0)
    monkeypatch.setattr(
        governor,
        "_detect_total_memory_gb",
        lambda: 8.0,
    )
    monkeypatch.setattr(
        governor,
        "_current_process_memory_gb",
        lambda: 3.2,
    )

    # Budget is 80% of capped memory => 6.4 GB, so ratio should be 0.5.
    assert governor.memory_pressure_ratio() == 0.5


def test_max_safe_concurrent_scales_with_rpm():
    """max_safe_concurrent should be rpm // 2."""
    rl = RateLimiter(provider="test", model="m", rpm=100, tpm=100_000)
    assert rl.max_safe_concurrent == 50

    rl2 = RateLimiter(provider="test", model="m", rpm=1000, tpm=1_000_000)
    assert rl2.max_safe_concurrent == 500

    rl3 = RateLimiter(provider="test", model="m", rpm=1, tpm=10_000)
    assert rl3.max_safe_concurrent == 1


def test_semaphore_caps_concurrency():
    """Verify semaphore limits in-flight tasks to target_concurrency."""
    max_concurrent = 0
    current_concurrent = 0

    async def fake_work(sem):
        nonlocal max_concurrent, current_concurrent
        async with sem:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            current_concurrent -= 1

    async def run():
        sem = asyncio.Semaphore(5)
        tasks = [asyncio.create_task(fake_work(sem)) for _ in range(20)]
        await asyncio.gather(*tasks)

    asyncio.run(run())
    assert max_concurrent == 5
