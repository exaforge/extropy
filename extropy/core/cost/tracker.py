"""Session-scoped cost accumulator.

Automatically records token usage from every LLM provider call within
a CLI session. Providers push usage via CostTracker.record(); the CLI
reads the totals at exit via CostTracker.summary().

Thread-safe — simulation calls record() from async workers concurrently.
"""

import logging
import threading
import time
from typing import Any

from pydantic import BaseModel

from ..providers.base import TokenUsage
from .pricing import get_pricing

logger = logging.getLogger(__name__)


class CallRecord(BaseModel):
    """A single LLM API call's token usage."""

    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float
    call_type: str = ""  # "simple", "reasoning", "agentic_research", "async"


class ModelUsage(BaseModel):
    """Accumulated usage for a single model."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class CostTracker:
    """Session-scoped cost accumulator.

    Singleton per process. Providers auto-record into this after each call.
    The CLI reads summary/cost at session end.

    Thread-safe: uses a lock for mutation since simulation workers
    call record() concurrently.

    Note: This is not a Pydantic model because it manages mutable state
    with thread locks and singleton lifecycle — patterns that don't fit
    Pydantic's immutable-validation model.
    """

    _instance: "CostTracker | None" = None
    _lock_cls = threading.Lock()  # Class-level lock for singleton creation

    def __init__(self) -> None:
        self._records: list[CallRecord] = []
        self._by_model: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._command: str = ""  # Set by CLI (e.g., "spec", "simulate")
        self._scenario: str = ""  # Set by CLI for ledger tagging

    @classmethod
    def get(cls) -> "CostTracker":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock_cls:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing or new session)."""
        with cls._lock_cls:
            cls._instance = None

    def set_context(self, command: str = "", scenario: str = "") -> None:
        """Set session context for ledger tagging.

        Args:
            command: CLI command name (e.g., "spec", "simulate")
            scenario: Scenario/population name for identification
        """
        self._command = command
        self._scenario = scenario

    def record(
        self,
        model: str,
        usage: TokenUsage,
        call_type: str = "",
    ) -> None:
        """Record token usage from a single LLM API call.

        Called automatically by provider base class after each call.

        Args:
            model: Model name used for the call
            usage: Token usage from the API response
            call_type: Type of call ("simple", "reasoning", etc.)
        """
        if usage.input_tokens == 0 and usage.output_tokens == 0:
            return

        record = CallRecord(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            timestamp=time.time(),
            call_type=call_type,
        )

        with self._lock:
            self._records.append(record)

            if model not in self._by_model:
                self._by_model[model] = ModelUsage()

            mu = self._by_model[model]
            mu.calls += 1
            mu.input_tokens += usage.input_tokens
            mu.output_tokens += usage.output_tokens

    @property
    def total_calls(self) -> int:
        """Total number of LLM calls recorded."""
        with self._lock:
            return sum(mu.calls for mu in self._by_model.values())

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all models."""
        with self._lock:
            return sum(mu.input_tokens for mu in self._by_model.values())

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all models."""
        with self._lock:
            return sum(mu.output_tokens for mu in self._by_model.values())

    def total_cost(self) -> float | None:
        """Compute total USD cost from recorded usage.

        Returns:
            Total cost in USD, or None if no pricing available for any model.
        """
        with self._lock:
            total = 0.0
            has_any_pricing = False

            for model, mu in self._by_model.items():
                pricing = get_pricing(model)
                if pricing:
                    has_any_pricing = True
                    total += (
                        mu.input_tokens * pricing.input_per_mtok
                        + mu.output_tokens * pricing.output_per_mtok
                    ) / 1_000_000

            return total if has_any_pricing else None

    def cost_by_model(self) -> dict[str, dict[str, Any]]:
        """Get cost breakdown by model.

        Returns:
            Dict of model → {calls, input_tokens, output_tokens, cost}
        """
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for model, mu in self._by_model.items():
                pricing = get_pricing(model)
                cost = None
                if pricing:
                    cost = (
                        mu.input_tokens * pricing.input_per_mtok
                        + mu.output_tokens * pricing.output_per_mtok
                    ) / 1_000_000

                result[model] = {
                    "calls": mu.calls,
                    "input_tokens": mu.input_tokens,
                    "output_tokens": mu.output_tokens,
                    "cost": cost,
                }
            return result

    def summary(self) -> dict[str, Any]:
        """Full session summary for export/display.

        Returns:
            Dict with total and per-model breakdowns.
        """
        with self._lock:
            by_model = {}
            total_cost = 0.0
            has_pricing = False

            for model, mu in self._by_model.items():
                pricing = get_pricing(model)
                model_cost = None
                if pricing:
                    has_pricing = True
                    model_cost = (
                        mu.input_tokens * pricing.input_per_mtok
                        + mu.output_tokens * pricing.output_per_mtok
                    ) / 1_000_000
                    total_cost += model_cost

                by_model[model] = {
                    "calls": mu.calls,
                    "input_tokens": mu.input_tokens,
                    "output_tokens": mu.output_tokens,
                    "cost": round(model_cost, 4) if model_cost is not None else None,
                }

            total_in = sum(mu.input_tokens for mu in self._by_model.values())
            total_out = sum(mu.output_tokens for mu in self._by_model.values())

            return {
                "command": self._command,
                "scenario": self._scenario,
                "total_calls": sum(mu.calls for mu in self._by_model.values()),
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_cost": round(total_cost, 4) if has_pricing else None,
                "by_model": by_model,
                "elapsed_seconds": round(time.time() - self._started_at, 1),
            }

    def summary_line(self) -> str | None:
        """One-line cost summary for CLI footer.

        Returns:
            Formatted string like "$0.38 · openai/gpt-5 · 8 calls · 87k in / 12k out",
            or None if no calls were recorded.
        """
        with self._lock:
            total_calls = sum(mu.calls for mu in self._by_model.values())
            if total_calls == 0:
                return None

            total_in = sum(mu.input_tokens for mu in self._by_model.values())
            total_out = sum(mu.output_tokens for mu in self._by_model.values())
            models = list(self._by_model.keys())

        cost = self.total_cost()

        parts = []

        # Cost
        if cost is not None:
            parts.append(f"${cost:.2f}")
        else:
            parts.append("cost unknown")

        # Model(s)
        if len(models) == 1:
            parts.append(models[0])
        elif len(models) > 1:
            parts.append(f"{len(models)} models")

        # Call count
        parts.append(f"{total_calls} call{'s' if total_calls != 1 else ''}")

        # Token counts
        parts.append(f"{_format_tokens(total_in)} in / {_format_tokens(total_out)} out")

        return " · ".join(parts)

    @property
    def has_records(self) -> bool:
        """Whether any calls have been recorded."""
        with self._lock:
            return len(self._records) > 0


def _format_tokens(n: int) -> str:
    """Format token count for display (e.g., 87k, 1.5M)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)
