"""Default rate limit profiles per provider/model/tier.

These are Tier 1 (lowest) defaults. Higher tiers are multiples.
The rate limiter uses these as starting points and self-corrects
from response headers.
"""

from typing import Any


# Provider rate limit profiles
# Each model entry has: rpm (requests per minute), and either
# tpm (tokens per minute, OpenAI) or itpm+otpm (Anthropic input/output tokens)
RATE_LIMIT_PROFILES: dict[str, dict[str, Any]] = {
    "anthropic": {
        "default": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-sonnet-4-5-20250514": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-sonnet-4.5": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-sonnet-4": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-haiku-4.5": {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
        "claude-haiku-4": {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
        "tiers": {
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 800_000, "otpm": 160_000},
            4: {"rpm": 4_000, "itpm": 2_000_000, "otpm": 400_000},
        },
    },
    "openai": {
        "default": {"rpm": 500, "tpm": 500_000},
        "gpt-5": {"rpm": 500, "tpm": 500_000},
        "gpt-5-mini": {"rpm": 500, "tpm": 500_000},
        "gpt-5.2": {"rpm": 500, "tpm": 500_000},
        "tiers": {
            2: {"rpm": 5_000, "tpm": 1_000_000},
            3: {"rpm": 5_000, "tpm": 2_000_000},
            4: {"rpm": 10_000, "tpm": 4_000_000},
        },
    },
    # Map "claude" provider name to anthropic profiles
    "claude": {
        "default": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-sonnet-4-5-20250514": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-sonnet-4.5": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-sonnet-4": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-haiku-4.5": {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
        "claude-haiku-4": {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
        "tiers": {
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 800_000, "otpm": 160_000},
            4: {"rpm": 4_000, "itpm": 2_000_000, "otpm": 400_000},
        },
    },
}


def get_limits(
    provider: str,
    model: str = "",
    tier: int | None = None,
) -> dict[str, int]:
    """Get rate limits for a provider/model/tier combination.

    Args:
        provider: Provider name ('openai', 'claude', 'anthropic')
        model: Model name (falls back to provider default if not found)
        tier: Tier number (1-4, None = Tier 1)

    Returns:
        Dict with rpm and tpm (or itpm+otpm) limits
    """
    provider_key = provider.lower()
    if provider_key not in RATE_LIMIT_PROFILES:
        # Unknown provider — use conservative defaults
        return {"rpm": 50, "tpm": 100_000}

    profile = RATE_LIMIT_PROFILES[provider_key]

    # Get base limits for model (or default)
    base = dict(profile.get(model, profile["default"]))

    # Apply tier multipliers if specified
    if tier and tier > 1:
        tiers = profile.get("tiers", {})
        if tier in tiers:
            base = dict(tiers[tier])

    return base
