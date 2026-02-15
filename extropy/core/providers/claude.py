"""DEPRECATED: Use extropy.core.providers.anthropic instead.

This module re-exports AnthropicProvider as ClaudeProvider for backward compatibility.
"""

from .anthropic import AnthropicProvider, ClaudeProvider  # noqa: F401

__all__ = ["ClaudeProvider", "AnthropicProvider"]
