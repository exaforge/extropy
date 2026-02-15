"""Tests for resource auto-tuning and runtime guardrails."""

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
