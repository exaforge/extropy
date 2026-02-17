import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

from extropy.core import llm
from extropy.core.providers.base import TokenUsage


def test_simple_call_async_uses_cached_simulation_provider_default_model(monkeypatch):
    provider = MagicMock()
    provider.simple_call_async = AsyncMock(
        return_value=({"ok": True}, TokenUsage(input_tokens=11, output_tokens=7))
    )
    config = SimpleNamespace(resolve_sim_strong=lambda: "openai/gpt-5", providers={})
    get_sim_provider = Mock(return_value=provider)

    monkeypatch.setattr(llm, "get_config", lambda: config)
    monkeypatch.setattr(llm, "get_simulation_provider", get_sim_provider)
    monkeypatch.setattr(
        llm,
        "get_provider",
        Mock(side_effect=AssertionError("simple_call_async should use simulation cache")),
    )

    result, usage = asyncio.run(
        llm.simple_call_async(
            prompt="hello",
            response_schema={"type": "object"},
            schema_name="response",
        )
    )

    assert result == {"ok": True}
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    get_sim_provider.assert_called_once_with("openai/gpt-5")
    provider.simple_call_async.assert_awaited_once()
    assert provider.simple_call_async.await_args.kwargs["model"] == "gpt-5"


def test_simple_call_async_uses_cached_simulation_provider_for_explicit_model(monkeypatch):
    provider = MagicMock()
    provider.simple_call_async = AsyncMock(return_value=({"ok": True}, TokenUsage()))
    config = SimpleNamespace(resolve_sim_strong=lambda: "openai/gpt-5", providers={})
    get_sim_provider = Mock(return_value=provider)

    monkeypatch.setattr(llm, "get_config", lambda: config)
    monkeypatch.setattr(llm, "get_simulation_provider", get_sim_provider)
    monkeypatch.setattr(
        llm,
        "get_provider",
        Mock(side_effect=AssertionError("explicit async model should still use simulation cache")),
    )

    asyncio.run(
        llm.simple_call_async(
            prompt="hello",
            response_schema={"type": "object"},
            schema_name="response",
            model="anthropic/claude-sonnet-4-5",
        )
    )

    get_sim_provider.assert_called_once_with("anthropic/claude-sonnet-4-5")
    provider.simple_call_async.assert_awaited_once()
    assert (
        provider.simple_call_async.await_args.kwargs["model"]
        == "claude-sonnet-4-5"
    )


def test_simple_call_sync_path_still_uses_regular_provider_factory(monkeypatch):
    provider = MagicMock()
    provider.simple_call.return_value = {"ok": True}
    config = SimpleNamespace(resolve_pipeline_fast=lambda: "openai/gpt-5-mini", providers={})
    get_provider = Mock(return_value=provider)

    monkeypatch.setattr(llm, "get_config", lambda: config)
    monkeypatch.setattr(llm, "get_provider", get_provider)
    monkeypatch.setattr(
        llm,
        "get_simulation_provider",
        Mock(side_effect=AssertionError("sync calls should not use simulation provider cache")),
    )

    result = llm.simple_call(
        prompt="hello",
        response_schema={"type": "object"},
        schema_name="response",
        model="openai/gpt-5-mini",
    )

    assert result == {"ok": True}
    get_provider.assert_called_once_with("openai", config.providers)
    provider.simple_call.assert_called_once()
    assert provider.simple_call.call_args.kwargs["model"] == "gpt-5-mini"
