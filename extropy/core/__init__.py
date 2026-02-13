"""Core infrastructure for Extropy.

This package contains shared infrastructure used across all phases:
- llm: OpenAI API wrappers for LLM calls
- models: All Pydantic models organized by domain

Note: LLM functions are not eagerly imported to avoid requiring openai
dependency for model imports. Use:
    from extropy.core.llm import simple_call
"""

# Don't eagerly import llm to allow core.models to work without openai
__all__ = [
    "llm",
    "models",
]
