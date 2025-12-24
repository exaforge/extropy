"""Core infrastructure for Entropy.

This package contains shared infrastructure used across all phases:
- llm: LLM function wrappers (simple_call, reasoning_call, agentic_research)
- providers: OpenRouter client and provider utilities
- models: All Pydantic models organized by domain

Note: LLM functions are not eagerly imported to avoid requiring openai
dependency for model imports. Use:
    from entropy.core.llm import simple_call
    from entropy.core.providers import get_openrouter_client
"""

# Don't eagerly import llm/providers to allow core.models to work without openai
__all__ = [
    "llm",
    "models",
    "providers",
]
