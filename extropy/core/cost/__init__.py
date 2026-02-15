"""Cost tracking, pricing resolution, and persistent ledger.

This package provides:
- CostTracker: Session-scoped accumulator (auto-records from providers)
- Pricing: Three-tier model pricing resolution (OpenRouter → cache → fallback)
- Ledger: Persistent cost history (~/.config/extropy/cost_ledger.db)
"""

from .tracker import CostTracker, CallRecord, ModelUsage
from .pricing import (
    ModelPricing,
    get_pricing,
    resolve_default_model,
    refresh_pricing,
    get_cache_info,
    FALLBACK_PRICING,
    PROVIDER_DEFAULTS,
)
from .ledger import CostEntry, record_session, query_entries, query_totals

__all__ = [
    # Tracker
    "CostTracker",
    "CallRecord",
    "ModelUsage",
    # Pricing
    "ModelPricing",
    "get_pricing",
    "resolve_default_model",
    "refresh_pricing",
    "get_cache_info",
    "FALLBACK_PRICING",
    "PROVIDER_DEFAULTS",
    # Ledger
    "CostEntry",
    "record_session",
    "query_entries",
    "query_totals",
]
