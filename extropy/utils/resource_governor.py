"""Resource auto-tuning helpers for CPU/memory constrained environments."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_count: int
    total_memory_gb: float
    memory_budget_gb: float


class ResourceGovernor:
    """Computes safe worker/chunk recommendations from local machine resources."""

    def __init__(
        self,
        resource_mode: str = "auto",
        safe_auto_workers: bool = True,
        max_memory_gb: float | None = None,
    ):
        self.resource_mode = resource_mode
        self.safe_auto_workers = safe_auto_workers
        self.max_memory_gb = max_memory_gb

    @staticmethod
    def _detect_total_memory_gb() -> float:
        # Linux and many Unix systems
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if page_size > 0 and phys_pages > 0:
                return (page_size * phys_pages) / (1024**3)
        except (ValueError, OSError, AttributeError):
            pass

        # macOS fallback
        if platform.system().lower() == "darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                return int(out.strip()) / (1024**3)
            except Exception:
                pass

        # Conservative fallback
        return 8.0

    def snapshot(self) -> ResourceSnapshot:
        cpu_count = max(1, os.cpu_count() or 1)
        total_mem = self._detect_total_memory_gb()
        capped = min(total_mem, self.max_memory_gb) if self.max_memory_gb else total_mem
        budget = max(1.0, capped * 0.80)
        return ResourceSnapshot(
            cpu_count=cpu_count,
            total_memory_gb=round(total_mem, 2),
            memory_budget_gb=round(budget, 2),
        )

    def recommend_workers(
        self,
        requested_workers: int,
        memory_per_worker_gb: float,
    ) -> int:
        requested_workers = max(1, int(requested_workers))
        if self.resource_mode != "auto":
            return requested_workers

        snap = self.snapshot()
        cpu_cap = max(1, snap.cpu_count - 1) if self.safe_auto_workers else snap.cpu_count
        mem_cap = max(1, int(snap.memory_budget_gb / max(0.1, memory_per_worker_gb)))

        if self.safe_auto_workers:
            cpu_cap = min(cpu_cap, 8)

        return max(1, min(requested_workers, cpu_cap, mem_cap))

    def recommend_chunk_size(
        self,
        requested_chunk_size: int,
        min_chunk_size: int = 8,
        max_chunk_size: int = 4096,
    ) -> int:
        requested_chunk_size = max(min_chunk_size, int(requested_chunk_size))
        if self.resource_mode != "auto":
            return min(max_chunk_size, requested_chunk_size)

        snap = self.snapshot()
        if snap.memory_budget_gb <= 4:
            tuned = min(requested_chunk_size, 32)
        elif snap.memory_budget_gb <= 8:
            tuned = min(requested_chunk_size, 64)
        elif snap.memory_budget_gb <= 16:
            tuned = min(requested_chunk_size, 128)
        else:
            tuned = requested_chunk_size

        return max(min_chunk_size, min(max_chunk_size, tuned))
