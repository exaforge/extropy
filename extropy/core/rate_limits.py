"""Default rate limit profiles per provider/model/tier.

Profiles are structured as provider → model → tier → limits.
The rate limiter uses these as starting points.

Sources:
- OpenAI: https://platform.openai.com/docs/guides/rate-limits
  (Updated Sep 2025 with doubled gpt-5/gpt-5-mini limits)
- Anthropic: https://docs.anthropic.com/en/api/rate-limits
"""

# Provider rate limit profiles
# Structure: provider → model → tier → {rpm, tpm} or {rpm, itpm, otpm}
# Tier 1 is the default (lowest). Tiers 2-4 require higher spend thresholds.
RATE_LIMIT_PROFILES: dict[str, dict[str, dict[int, dict[str, int]]]] = {
    "openai": {
        "default": {
            1: {"rpm": 500, "tpm": 500_000},
            2: {"rpm": 5_000, "tpm": 1_000_000},
            3: {"rpm": 5_000, "tpm": 2_000_000},
            4: {"rpm": 10_000, "tpm": 4_000_000},
        },
        "gpt-5": {
            1: {"rpm": 500, "tpm": 500_000},
            2: {"rpm": 5_000, "tpm": 1_000_000},
            3: {"rpm": 5_000, "tpm": 2_000_000},
            4: {"rpm": 10_000, "tpm": 4_000_000},
        },
        "gpt-5-mini": {
            1: {"rpm": 500, "tpm": 500_000},
            2: {"rpm": 5_000, "tpm": 2_000_000},
            3: {"rpm": 5_000, "tpm": 4_000_000},
            4: {"rpm": 10_000, "tpm": 10_000_000},
        },
        "gpt-5.1": {
            1: {"rpm": 500, "tpm": 500_000},
            2: {"rpm": 5_000, "tpm": 1_000_000},
            3: {"rpm": 5_000, "tpm": 2_000_000},
            4: {"rpm": 10_000, "tpm": 4_000_000},
        },
        "gpt-5.2": {
            1: {"rpm": 500, "tpm": 500_000},
            2: {"rpm": 5_000, "tpm": 1_000_000},
            3: {"rpm": 5_000, "tpm": 2_000_000},
            4: {"rpm": 10_000, "tpm": 4_000_000},
        },
    },
    "anthropic": {
        "default": {
            1: {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 800_000, "otpm": 160_000},
            4: {"rpm": 4_000, "itpm": 2_000_000, "otpm": 400_000},
        },
        "claude-sonnet-4-5-20250514": {
            1: {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 800_000, "otpm": 160_000},
            4: {"rpm": 4_000, "itpm": 2_000_000, "otpm": 400_000},
        },
        "claude-sonnet-4.5": {
            1: {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 800_000, "otpm": 160_000},
            4: {"rpm": 4_000, "itpm": 2_000_000, "otpm": 400_000},
        },
        "claude-sonnet-4": {
            1: {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 800_000, "otpm": 160_000},
            4: {"rpm": 4_000, "itpm": 2_000_000, "otpm": 400_000},
        },
        "claude-haiku-4.5": {
            1: {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 1_000_000, "otpm": 200_000},
            4: {"rpm": 4_000, "itpm": 4_000_000, "otpm": 800_000},
        },
        "claude-haiku-4": {
            1: {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 1_000_000, "otpm": 200_000},
            4: {"rpm": 4_000, "itpm": 4_000_000, "otpm": 800_000},
        },
    },
}

# Provider aliases — map alternate names to canonical profiles
RATE_LIMIT_PROFILES["claude"] = RATE_LIMIT_PROFILES["anthropic"]
RATE_LIMIT_PROFILES["azure_openai"] = RATE_LIMIT_PROFILES["openai"]
RATE_LIMIT_PROFILES["azure"] = RATE_LIMIT_PROFILES["openai"]

# Third-party providers — conservative defaults
# These providers typically have per-key limits; adjust via rate_tier/rpm_override.
_THIRD_PARTY_DEFAULT = {
    "default": {
        1: {"rpm": 60, "tpm": 100_000},
        2: {"rpm": 200, "tpm": 500_000},
        3: {"rpm": 500, "tpm": 1_000_000},
        4: {"rpm": 1_000, "tpm": 2_000_000},
    },
}
RATE_LIMIT_PROFILES["openrouter"] = _THIRD_PARTY_DEFAULT
RATE_LIMIT_PROFILES["deepseek"] = _THIRD_PARTY_DEFAULT
RATE_LIMIT_PROFILES["together"] = _THIRD_PARTY_DEFAULT
RATE_LIMIT_PROFILES["groq"] = {
    "default": {
        1: {"rpm": 30, "tpm": 15_000},
        2: {"rpm": 60, "tpm": 50_000},
        3: {"rpm": 200, "tpm": 100_000},
        4: {"rpm": 500, "tpm": 500_000},
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
    effective_tier = tier if tier and tier >= 1 else 1

    provider_key = provider.lower()
    if provider_key not in RATE_LIMIT_PROFILES:
        # Unknown provider — use conservative defaults
        return {"rpm": 50, "tpm": 100_000}

    provider_profiles = RATE_LIMIT_PROFILES[provider_key]

    # Look up model-specific profile, fall back to default
    model_profile = provider_profiles.get(model, provider_profiles["default"])

    # Look up tier, fall back to tier 1
    limits = model_profile.get(effective_tier, model_profile.get(1, {}))

    return dict(limits)
