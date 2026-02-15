"""Model pricing data for cost estimation.

Provides per-model input/output pricing (USD per million tokens)
and provider default model resolution without needing API keys.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model (USD per million tokens)."""

    input_per_mtok: float
    output_per_mtok: float


# Known model pricing (USD per million tokens)
# Sources: OpenAI and Anthropic pricing pages as of 2025
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-5": ModelPricing(input_per_mtok=2.50, output_per_mtok=10.00),
    "gpt-5-mini": ModelPricing(input_per_mtok=0.30, output_per_mtok=1.50),
    "gpt-5-nano": ModelPricing(input_per_mtok=0.10, output_per_mtok=0.40),
    "gpt-5.2": ModelPricing(input_per_mtok=2.50, output_per_mtok=10.00),
    # Azure-hosted models
    "DeepSeek-V3.2": ModelPricing(input_per_mtok=0.80, output_per_mtok=2.00),
    "Kimi-K2.5": ModelPricing(input_per_mtok=1.00, output_per_mtok=4.00),
    # Claude
    "claude-sonnet-4-5-20250929": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00
    ),
    "claude-sonnet-4-5-20250514": ModelPricing(
        input_per_mtok=3.00, output_per_mtok=15.00
    ),
    "claude-sonnet-4.5": ModelPricing(input_per_mtok=3.00, output_per_mtok=15.00),
    "claude-sonnet-4": ModelPricing(input_per_mtok=3.00, output_per_mtok=15.00),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_per_mtok=0.80, output_per_mtok=4.00
    ),
    "claude-haiku-4.5": ModelPricing(input_per_mtok=0.80, output_per_mtok=4.00),
    "claude-haiku-4": ModelPricing(input_per_mtok=0.80, output_per_mtok=4.00),
    # DeepSeek (direct API)
    "deepseek-chat": ModelPricing(input_per_mtok=0.14, output_per_mtok=0.28),
    "deepseek-reasoner": ModelPricing(input_per_mtok=0.55, output_per_mtok=2.19),
}

# Provider default models — 2-tier (fast/strong)
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "fast": "gpt-5-mini",
        "strong": "gpt-5",
    },
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "strong": "claude-sonnet-4-5-20250929",
    },
    "azure": {
        "fast": "gpt-5-mini",
        "strong": "gpt-5",
    },
    "openrouter": {
        "fast": "openai/gpt-5-mini",
        "strong": "openai/gpt-5",
    },
    "deepseek": {
        "fast": "deepseek-chat",
        "strong": "deepseek-reasoner",
    },
    "together": {
        "fast": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "strong": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    },
    "groq": {
        "fast": "llama-3.3-70b-versatile",
        "strong": "llama-3.3-70b-versatile",
    },
    # Legacy aliases
    "claude": {
        "fast": "claude-haiku-4-5-20251001",
        "strong": "claude-sonnet-4-5-20250929",
    },
    "azure_openai": {
        "fast": "gpt-5-mini",
        "strong": "gpt-5",
    },
}


def get_pricing(model: str) -> ModelPricing | None:
    """Get pricing for a model, or None if unknown."""
    return MODEL_PRICING.get(model)


def resolve_default_model(provider: str, tier: str = "strong") -> str:
    """Resolve default model name for a provider without instantiating it.

    Args:
        provider: Provider name ('openai', 'anthropic', etc.)
        tier: 'fast' or 'strong' (also accepts legacy 'simple'/'reasoning')

    Returns:
        Model name string
    """
    # Map legacy tier names
    tier_map = {"simple": "fast", "reasoning": "strong"}
    tier = tier_map.get(tier, tier)

    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    return defaults.get(tier, defaults["strong"])
