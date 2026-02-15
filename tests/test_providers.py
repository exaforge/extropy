"""Provider tests with mocked HTTP clients.

Tests response extraction, retry on transient errors,
validation-retry exhaustion, and source URL extraction.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from extropy.core.providers.base import LLMProvider, TokenUsage
from extropy.core.providers.openai import OpenAIProvider
from extropy.core.providers.claude import ClaudeProvider


# Disable rate limiting for all provider tests (avoid waits with mocked clients)
OpenAIProvider._disable_rate_limiting = True
ClaudeProvider._disable_rate_limiting = True


def _make_openai_provider(**overrides):
    """Create an OpenAIProvider via __new__ with all required attrs set.

    Bypasses __init__ (no API key validation) for unit testing with mocked clients.
    """
    provider = OpenAIProvider.__new__(OpenAIProvider)
    defaults = {
        "_api_key": "test-key",
        "_is_azure": False,
        "_azure_endpoint": "",
        "_api_version": "",
        "_azure_deployment": "",
        "_api_format": "responses",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(provider, k, v)
    return provider


def _make_claude_provider(**overrides):
    """Create a ClaudeProvider via __new__ with all required attrs set."""
    provider = ClaudeProvider.__new__(ClaudeProvider)
    defaults = {"_api_key": "test-key"}
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(provider, k, v)
    return provider


# =============================================================================
# Mock response factories
# =============================================================================


def _make_openai_response(
    text: str = '{"key": "value"}',
    input_tokens: int = 100,
    output_tokens: int = 50,
):
    """Create a mock OpenAI Responses API response."""
    content_item = MagicMock()
    content_item.type = "output_text"
    content_item.text = text
    content_item.annotations = []

    message = MagicMock()
    message.type = "message"
    message.content = [content_item]

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.output = [message]
    response.usage = usage

    return response


def _make_claude_response(
    tool_input: dict | None = None,
    input_tokens: int = 80,
    output_tokens: int = 40,
):
    """Create a mock Claude Messages API response with tool_use block."""
    blocks = []

    if tool_input is not None:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "response"
        tool_block.input = tool_input
        blocks.append(tool_block)

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.content = blocks
    response.usage = usage

    return response


# =============================================================================
# OpenAI Provider Tests
# =============================================================================


class TestOpenAIExtractOutputText:
    """Test the _extract_output_text helper."""

    def test_extracts_text(self):
        provider = _make_openai_provider()
        response = _make_openai_response('{"hello": "world"}')
        text = provider._extract_output_text(response)
        assert text == '{"hello": "world"}'

    def test_returns_none_on_empty(self):
        provider = _make_openai_provider()
        response = MagicMock()
        response.output = []
        text = provider._extract_output_text(response)
        assert text is None


class TestOpenAISimpleCall:
    """Test OpenAI simple_call with mocked client."""

    @patch.object(OpenAIProvider, "_get_client")
    def test_returns_parsed_json(self, mock_get_client):
        provider = _make_openai_provider()

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _make_openai_response(
            '{"result": "ok"}'
        )
        mock_get_client.return_value = mock_client

        result = provider.simple_call(
            prompt="test prompt",
            response_schema={"type": "object", "properties": {}},
            log=False,
        )

        assert result == {"result": "ok"}

    @patch.object(OpenAIProvider, "_get_client")
    def test_returns_empty_dict_on_no_output(self, mock_get_client):
        provider = _make_openai_provider()

        empty_response = MagicMock()
        empty_response.output = []
        mock_client = MagicMock()
        mock_client.responses.create.return_value = empty_response
        mock_get_client.return_value = mock_client

        result = provider.simple_call(
            prompt="test",
            response_schema={"type": "object", "properties": {}},
            log=False,
        )
        assert result == {}


class TestOpenAIRetry:
    """Test OpenAI transient error retry."""

    def test_with_retry_succeeds_after_failure(self):
        import openai

        provider = _make_openai_provider()

        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise openai.APIConnectionError(request=MagicMock())
            return "success"

        with patch("time.sleep"):  # don't actually wait
            result = provider._with_retry(flaky_fn, max_retries=3)

        assert result == "success"
        assert call_count == 3

    def test_with_retry_exhausted(self):
        import openai

        provider = _make_openai_provider()

        def always_fails():
            raise openai.InternalServerError(
                message="server error",
                response=MagicMock(status_code=500, headers={}),
                body=None,
            )

        with patch("time.sleep"):
            with pytest.raises(openai.InternalServerError):
                provider._with_retry(always_fails, max_retries=2)


class TestOpenAIValidationRetry:
    """Test validation-retry loop for OpenAI reasoning_call."""

    @patch.object(OpenAIProvider, "_get_client")
    def test_validation_retry_succeeds(self, mock_get_client):
        """Test that a failing validation retries and eventually succeeds."""
        provider = _make_openai_provider()

        call_count = 0

        def create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_openai_response('{"status": "bad"}')
            return _make_openai_response('{"status": "good"}')

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = create_side_effect
        mock_get_client.return_value = mock_client

        def validator(data):
            if data.get("status") == "good":
                return True, ""
            return False, "PREVIOUS ATTEMPT FAILED: status was bad"

        result = provider.reasoning_call(
            prompt="test",
            response_schema={"type": "object", "properties": {}},
            validator=validator,
            max_retries=2,
            log=False,
        )

        assert result["status"] == "good"

    @patch.object(OpenAIProvider, "_get_client")
    def test_validation_retry_exhausted(self, mock_get_client):
        """Test that exhausting retries returns the last result."""
        provider = _make_openai_provider()

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _make_openai_response(
            '{"status": "always_bad"}'
        )
        mock_get_client.return_value = mock_client

        retry_calls = []

        def on_retry(attempt, max_retries, summary):
            retry_calls.append((attempt, summary))

        def validator(data):
            return False, "Still bad"

        result = provider.reasoning_call(
            prompt="test",
            response_schema={"type": "object", "properties": {}},
            validator=validator,
            max_retries=1,
            on_retry=on_retry,
            log=False,
        )

        assert result["status"] == "always_bad"
        # on_retry should have been called for attempt 1 and exhaustion
        assert len(retry_calls) == 2
        assert "EXHAUSTED" in retry_calls[-1][1]


# =============================================================================
# Claude Provider Tests
# =============================================================================


class TestClaudeSimpleCall:
    """Test Claude simple_call with mocked client."""

    @patch.object(ClaudeProvider, "_get_client")
    def test_returns_tool_input(self, mock_get_client):
        provider = _make_claude_provider()

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_claude_response(
            {"result": "ok"}
        )
        mock_get_client.return_value = mock_client

        result = provider.simple_call(
            prompt="test",
            response_schema={"type": "object", "properties": {}},
            log=False,
        )

        assert result == {"result": "ok"}


class TestClaudeRetry:
    """Test Claude transient error retry."""

    def test_with_retry_succeeds_after_failure(self):
        import anthropic

        provider = _make_claude_provider()

        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise anthropic.APIConnectionError(request=MagicMock())
            return "success"

        with patch("time.sleep"):
            result = provider._with_retry(flaky_fn, max_retries=3)

        assert result == "success"
        assert call_count == 2


class TestClaudeAgenticResearch:
    """Test Claude agentic_research source extraction."""

    @patch.object(ClaudeProvider, "_get_client")
    def test_extracts_sources(self, mock_get_client):
        provider = _make_claude_provider()

        # Build response with web search results and tool_use
        search_result = MagicMock()
        search_result.url = "https://example.com/source1"

        search_block = MagicMock()
        search_block.type = "web_search_tool_result"
        search_block.content = [search_result]

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "research_data"
        tool_block.input = {"finding": "something"}

        text_block = MagicMock()
        text_block.type = "text"
        text_block.citations = []

        response = MagicMock()
        response.content = [search_block, tool_block, text_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        mock_get_client.return_value = mock_client

        result, sources = provider.agentic_research(
            prompt="research something",
            response_schema={"type": "object", "properties": {}},
            log=False,
        )

        assert result == {"finding": "something"}
        assert "https://example.com/source1" in sources


# =============================================================================
# Base Provider validation-retry Tests
# =============================================================================


class TestBaseRetryWithValidation:
    """Test the shared _retry_with_validation method on the base class."""

    def test_no_validator_returns_immediately(self):
        """With no validator, first result is returned."""

        class ConcreteProvider(LLMProvider):
            default_fast_model = "test"
            default_strong_model = "test"

            def simple_call(self, *a, **kw):
                return {}

            async def simple_call_async(self, *a, **kw):
                return {}, TokenUsage()

            def reasoning_call(self, *a, **kw):
                return {}

            def agentic_research(self, *a, **kw):
                return {}, []

        provider = ConcreteProvider.__new__(ConcreteProvider)

        call_count = 0

        def call_fn(prompt):
            nonlocal call_count
            call_count += 1
            return {"data": call_count}

        result = provider._retry_with_validation(
            call_fn=call_fn,
            prompt="test",
            validator=None,
            max_retries=3,
            on_retry=None,
            extract_error_summary_fn=lambda x: x[:50],
        )

        assert result == {"data": 1}
        assert call_count == 1

    def test_initial_prompt_used_on_first_call(self):
        """When initial_prompt is provided, it should be used for the first call."""

        class ConcreteProvider(LLMProvider):
            default_fast_model = "test"
            default_strong_model = "test"

            def simple_call(self, *a, **kw):
                return {}

            async def simple_call_async(self, *a, **kw):
                return {}, TokenUsage()

            def reasoning_call(self, *a, **kw):
                return {}

            def agentic_research(self, *a, **kw):
                return {}, []

        provider = ConcreteProvider.__new__(ConcreteProvider)

        prompts_received = []

        def call_fn(prompt):
            prompts_received.append(prompt)
            return {"data": "ok"}

        result = provider._retry_with_validation(
            call_fn=call_fn,
            prompt="base prompt",
            initial_prompt="PREVIOUS ERRORS: failed\n\n---\n\nbase prompt",
            validator=None,
            max_retries=3,
            on_retry=None,
            extract_error_summary_fn=lambda x: x[:50],
        )

        assert result == {"data": "ok"}
        assert len(prompts_received) == 1
        assert prompts_received[0] == "PREVIOUS ERRORS: failed\n\n---\n\nbase prompt"

    def test_validation_retries_use_base_prompt_not_initial(self):
        """Validation retries should use prompt, not initial_prompt."""

        class ConcreteProvider(LLMProvider):
            default_fast_model = "test"
            default_strong_model = "test"

            def simple_call(self, *a, **kw):
                return {}

            async def simple_call_async(self, *a, **kw):
                return {}, TokenUsage()

            def reasoning_call(self, *a, **kw):
                return {}

            def agentic_research(self, *a, **kw):
                return {}, []

        provider = ConcreteProvider.__new__(ConcreteProvider)

        call_count = 0
        prompts_received = []

        def call_fn(prompt):
            nonlocal call_count
            call_count += 1
            prompts_received.append(prompt)
            return {"status": f"attempt_{call_count}"}

        def validator(data):
            # Fail on first two attempts, succeed on third
            if "attempt_3" in data.get("status", ""):
                return True, ""
            return False, "VALIDATION FAILED: bad status"

        result = provider._retry_with_validation(
            call_fn=call_fn,
            prompt="base prompt",
            initial_prompt="PREVIOUS: error\n\n---\n\nbase prompt",
            validator=validator,
            max_retries=3,
            on_retry=None,
            extract_error_summary_fn=lambda x: x[:50],
        )

        # Should have made 3 calls
        assert call_count == 3
        assert result == {"status": "attempt_3"}

        # First call uses initial_prompt
        assert prompts_received[0] == "PREVIOUS: error\n\n---\n\nbase prompt"

        # Retry calls should use base prompt with new validation errors
        # (NOT initial_prompt)
        assert "VALIDATION FAILED" in prompts_received[1]
        assert "base prompt" in prompts_received[1]
        # Should NOT contain the original "PREVIOUS: error"
        assert "PREVIOUS: error" not in prompts_received[1]

        assert "VALIDATION FAILED" in prompts_received[2]
        assert "base prompt" in prompts_received[2]
        assert "PREVIOUS: error" not in prompts_received[2]

    def test_validator_succeeds_on_first_attempt_with_initial_prompt(self):
        """When validator passes on first try with initial_prompt, no retries occur."""

        class ConcreteProvider(LLMProvider):
            default_fast_model = "test"
            default_strong_model = "test"

            def simple_call(self, *a, **kw):
                return {}

            async def simple_call_async(self, *a, **kw):
                return {}, TokenUsage()

            def reasoning_call(self, *a, **kw):
                return {}

            def agentic_research(self, *a, **kw):
                return {}, []

        provider = ConcreteProvider.__new__(ConcreteProvider)

        call_count = 0

        def call_fn(prompt):
            nonlocal call_count
            call_count += 1
            return {"status": "good"}

        def validator(data):
            return data.get("status") == "good", ""

        result = provider._retry_with_validation(
            call_fn=call_fn,
            prompt="base prompt",
            initial_prompt="WITH CONTEXT: base prompt",
            validator=validator,
            max_retries=3,
            on_retry=None,
            extract_error_summary_fn=lambda x: x[:50],
        )

        # Should only have called once
        assert call_count == 1
        assert result == {"status": "good"}

    def test_on_retry_callback_invoked_correctly(self):
        """Test that on_retry callback is invoked with correct parameters."""

        class ConcreteProvider(LLMProvider):
            default_fast_model = "test"
            default_strong_model = "test"

            def simple_call(self, *a, **kw):
                return {}

            async def simple_call_async(self, *a, **kw):
                return {}, TokenUsage()

            def reasoning_call(self, *a, **kw):
                return {}

            def agentic_research(self, *a, **kw):
                return {}, []

        provider = ConcreteProvider.__new__(ConcreteProvider)

        call_count = 0
        retry_invocations = []

        def call_fn(prompt):
            nonlocal call_count
            call_count += 1
            return {"attempt": call_count}

        def validator(data):
            # Always fail
            return False, f"Error on attempt {data['attempt']}"

        def on_retry(attempt, max_retries, summary):
            retry_invocations.append((attempt, max_retries, summary))

        provider._retry_with_validation(
            call_fn=call_fn,
            prompt="base prompt",
            validator=validator,
            max_retries=2,
            on_retry=on_retry,
            extract_error_summary_fn=lambda x: x[:50],
        )

        # Should have tried 3 times (initial + 2 retries)
        assert call_count == 3

        # on_retry should have been called for each retry + exhaustion
        assert len(retry_invocations) == 3

        # Check first retry
        assert retry_invocations[0][0] == 1  # attempt 1
        assert retry_invocations[0][1] == 2  # max_retries
        assert "Error on attempt 1" in retry_invocations[0][2]

        # Check second retry
        assert retry_invocations[1][0] == 2
        assert retry_invocations[1][1] == 2

        # Check exhaustion notification
        assert retry_invocations[2][0] == 3  # max_retries + 1
        assert retry_invocations[2][1] == 2
        assert "EXHAUSTED" in retry_invocations[2][2]

    def test_no_initial_prompt_defaults_to_prompt(self):
        """When initial_prompt is None, prompt is used for first call."""

        class ConcreteProvider(LLMProvider):
            default_fast_model = "test"
            default_strong_model = "test"

            def simple_call(self, *a, **kw):
                return {}

            async def simple_call_async(self, *a, **kw):
                return {}, TokenUsage()

            def reasoning_call(self, *a, **kw):
                return {}

            def agentic_research(self, *a, **kw):
                return {}, []

        provider = ConcreteProvider.__new__(ConcreteProvider)

        prompts_received = []

        def call_fn(prompt):
            prompts_received.append(prompt)
            return {"data": "ok"}

        provider._retry_with_validation(
            call_fn=call_fn,
            prompt="base prompt",
            initial_prompt=None,
            validator=None,
            max_retries=3,
            on_retry=None,
            extract_error_summary_fn=lambda x: x[:50],
        )

        assert len(prompts_received) == 1
        assert prompts_received[0] == "base prompt"


# =============================================================================
# Azure AI Foundry Provider Tests
# =============================================================================


class TestAzureProvider:
    """Test Azure AI Foundry delegating provider."""

    def test_construction(self):
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com/",
        )
        assert provider._endpoint == "https://my-resource.services.ai.azure.com"
        assert provider._openai_sub is None
        assert provider._anthropic_sub is None

    def test_missing_api_key_raises(self):
        from extropy.core.providers.azure import AzureProvider

        with pytest.raises(ValueError, match="AZURE_API_KEY"):
            AzureProvider(api_key="", endpoint="https://example.com")

    def test_missing_endpoint_raises(self):
        from extropy.core.providers.azure import AzureProvider

        with pytest.raises(ValueError, match="AZURE_ENDPOINT"):
            AzureProvider(api_key="test-key", endpoint="")

    def test_claude_model_routes_to_anthropic(self):
        from extropy.core.providers.azure import _detect_backend

        assert _detect_backend("claude-sonnet-4-5") == "anthropic"
        assert _detect_backend("claude-haiku-4-5") == "anthropic"

    def test_non_claude_model_routes_to_openai(self):
        from extropy.core.providers.azure import _detect_backend

        assert _detect_backend("gpt-5-mini") == "openai"
        assert _detect_backend("DeepSeek-V3.2") == "openai"
        assert _detect_backend("Kimi-K2.5") == "openai"

    def test_simple_call_delegates_to_openai_sub(self):
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        mock_sub = MagicMock()
        mock_sub.simple_call.return_value = {"result": "ok"}
        provider._openai_sub = mock_sub

        result = provider.simple_call(
            prompt="test",
            response_schema={"type": "object"},
            model="gpt-5-mini",
            log=False,
        )

        assert result == {"result": "ok"}
        mock_sub.simple_call.assert_called_once()

    def test_simple_call_delegates_to_anthropic_sub(self):
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        mock_sub = MagicMock()
        mock_sub.simple_call.return_value = {"result": "claude_ok"}
        provider._anthropic_sub = mock_sub

        result = provider.simple_call(
            prompt="test",
            response_schema={"type": "object"},
            model="claude-sonnet-4-5",
            log=False,
        )

        assert result == {"result": "claude_ok"}
        mock_sub.simple_call.assert_called_once()

    def test_reasoning_call_delegates_to_openai_sub(self):
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        mock_sub = MagicMock()
        mock_sub.reasoning_call.return_value = {"result": "reasoned_gpt"}
        provider._openai_sub = mock_sub

        result = provider.reasoning_call(
            prompt="test",
            response_schema={"type": "object"},
            model="DeepSeek-V3.2",
        )

        assert result == {"result": "reasoned_gpt"}
        mock_sub.reasoning_call.assert_called_once()

    def test_agentic_research_delegates_to_openai_sub(self):
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        mock_sub = MagicMock()
        mock_sub.agentic_research.return_value = (
            {"finding": "data"},
            ["https://src.com"],
        )
        provider._openai_sub = mock_sub

        result, sources = provider.agentic_research(
            prompt="test",
            response_schema={"type": "object"},
            model="Kimi-K2.5",
        )

        assert result == {"finding": "data"}
        assert sources == ["https://src.com"]
        mock_sub.agentic_research.assert_called_once()

    def test_reasoning_call_delegates_to_anthropic_sub(self):
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        mock_sub = MagicMock()
        mock_sub.reasoning_call.return_value = {"result": "reasoned"}
        provider._anthropic_sub = mock_sub

        result = provider.reasoning_call(
            prompt="test",
            response_schema={"type": "object"},
            model="claude-sonnet-4-5",
        )

        assert result == {"result": "reasoned"}

    def test_lazy_sub_provider_creation(self):
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        # Sub-providers should not be created until needed
        assert provider._openai_sub is None
        assert provider._anthropic_sub is None

        # Access openai sub
        openai_sub = provider._get_openai_sub()
        assert openai_sub is not None
        assert provider._anthropic_sub is None  # still lazy

        # Access anthropic sub
        anthropic_sub = provider._get_anthropic_sub()
        assert anthropic_sub is not None

    def test_sub_provider_base_urls(self):
        from extropy.core.providers.azure import AzureProvider
        from extropy.core.providers.openai_compat import OpenAICompatProvider
        from extropy.core.providers.anthropic import AnthropicProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        openai_sub = provider._get_openai_sub()
        assert isinstance(openai_sub, OpenAICompatProvider)
        assert (
            openai_sub._base_url
            == "https://my-resource.services.ai.azure.com/openai/v1/"
        )

        anthropic_sub = provider._get_anthropic_sub()
        assert isinstance(anthropic_sub, AnthropicProvider)
        assert (
            anthropic_sub._base_url
            == "https://my-resource.services.ai.azure.com/anthropic/"
        )

    def test_close_async_closes_both_subs(self):
        import asyncio
        from extropy.core.providers.azure import AzureProvider

        provider = AzureProvider(
            api_key="test-key",
            endpoint="https://my-resource.services.ai.azure.com",
        )

        mock_openai = MagicMock()
        mock_openai.close_async = AsyncMock()
        mock_anthropic = MagicMock()
        mock_anthropic.close_async = AsyncMock()

        provider._openai_sub = mock_openai
        provider._anthropic_sub = mock_anthropic

        asyncio.run(provider.close_async())

        mock_openai.close_async.assert_called_once()
        mock_anthropic.close_async.assert_called_once()


# =============================================================================
# Provider Factory Azure Tests
# =============================================================================


class TestProviderFactoryAzure:
    """Test provider factory with Azure provider creation."""

    @patch.dict(
        "os.environ",
        {
            "AZURE_API_KEY": "test-azure-key",
            "AZURE_ENDPOINT": "https://my-resource.services.ai.azure.com",
        },
    )
    def test_create_azure_provider(self):
        from extropy.core.providers import get_provider
        from extropy.core.providers.azure import AzureProvider

        provider = get_provider("azure")
        assert isinstance(provider, AzureProvider)

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-azure-key",
            "AZURE_OPENAI_ENDPOINT": "https://my-resource.services.ai.azure.com",
        },
    )
    def test_create_azure_provider_legacy_env_vars(self):
        """Legacy AZURE_OPENAI_* env vars still work."""
        from extropy.core.providers import get_provider
        from extropy.core.providers.azure import AzureProvider

        provider = get_provider("azure")
        assert isinstance(provider, AzureProvider)

    @patch.dict(
        "os.environ",
        {
            "AZURE_API_KEY": "test-azure-key",
            "AZURE_ENDPOINT": "",
            "AZURE_OPENAI_ENDPOINT": "",
        },
    )
    def test_azure_missing_endpoint_raises(self):
        from extropy.core.providers import get_provider

        with pytest.raises(ValueError, match="AZURE_ENDPOINT"):
            get_provider("azure")


# =============================================================================
# Chat Completions API Tests
# =============================================================================


def _make_chat_completions_response(
    text: str = '{"key": "value"}',
    prompt_tokens: int = 120,
    completion_tokens: int = 60,
):
    """Create a mock Chat Completions API response."""
    message = MagicMock()
    message.content = text

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage

    return response


class TestOpenAIChatCompletions:
    """Test Chat Completions API codepath."""

    def test_extract_chat_completions_text(self):
        provider = _make_openai_provider()
        response = _make_chat_completions_response('{"hello": "world"}')
        text = provider._extract_chat_completions_text(response)
        assert text == '{"hello": "world"}'

    def test_extract_chat_completions_text_empty(self):
        provider = _make_openai_provider()
        response = MagicMock()
        response.choices = []
        text = provider._extract_chat_completions_text(response)
        assert text is None

    def test_extract_chat_completions_text_none_content(self):
        provider = _make_openai_provider()
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        text = provider._extract_chat_completions_text(response)
        assert text is None

    @patch.object(OpenAIProvider, "_get_client")
    def test_simple_call_chat_completions(self, mock_get_client):
        """Chat Completions format uses client.chat.completions.create."""
        provider = _make_openai_provider(_api_format="chat_completions")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            _make_chat_completions_response('{"result": "chat_ok"}')
        )
        mock_get_client.return_value = mock_client

        result = provider.simple_call(
            prompt="test prompt",
            response_schema={"type": "object", "properties": {}},
            log=False,
        )

        assert result == {"result": "chat_ok"}
        mock_client.chat.completions.create.assert_called_once()
        # Responses API should NOT have been called
        mock_client.responses.create.assert_not_called()

    @patch.object(OpenAIProvider, "_get_async_client")
    def test_simple_call_async_chat_completions(self, mock_get_async_client):
        """Async Chat Completions format uses client.chat.completions.create."""
        import asyncio

        provider = _make_openai_provider(_api_format="chat_completions")

        mock_client = MagicMock()
        mock_response = _make_chat_completions_response('{"result": "async_chat_ok"}')
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client.responses.create = AsyncMock()
        mock_get_async_client.return_value = mock_client

        result, usage = asyncio.run(
            provider.simple_call_async(
                prompt="test prompt",
                response_schema={"type": "object", "properties": {}},
            )
        )

        assert result == {"result": "async_chat_ok"}
        assert usage.input_tokens == 120
        assert usage.output_tokens == 60
        mock_client.chat.completions.create.assert_called_once()
        mock_client.responses.create.assert_not_called()

    @patch.object(OpenAIProvider, "_get_client")
    def test_responses_format_unchanged(self, mock_get_client):
        """Default (responses) format still uses client.responses.create."""
        provider = _make_openai_provider(_api_format="responses")

        mock_client = MagicMock()
        mock_client.responses.create.return_value = _make_openai_response(
            '{"result": "responses_ok"}'
        )
        mock_get_client.return_value = mock_client

        result = provider.simple_call(
            prompt="test prompt",
            response_schema={"type": "object", "properties": {}},
            log=False,
        )

        assert result == {"result": "responses_ok"}
        mock_client.responses.create.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()

    def test_build_chat_completions_params(self):
        """Verify Chat Completions param structure."""
        provider = _make_openai_provider()
        params = provider._build_chat_completions_params(
            model="DeepSeek-V3.2",
            prompt="test prompt",
            schema={"type": "object", "properties": {}},
            schema_name="response",
            max_tokens=1000,
        )

        assert params["model"] == "DeepSeek-V3.2"
        assert params["messages"] == [{"role": "user", "content": "test prompt"}]
        assert params["response_format"]["type"] == "json_schema"
        assert params["response_format"]["json_schema"]["name"] == "response"
        assert params["response_format"]["json_schema"]["strict"] is True
        assert params["max_tokens"] == 1000

    def test_build_chat_completions_params_no_max_tokens(self):
        """Without max_tokens, the param should not be present."""
        provider = _make_openai_provider()
        params = provider._build_chat_completions_params(
            model="gpt-5-mini",
            prompt="test",
            schema={"type": "object"},
            schema_name="test",
            max_tokens=None,
        )
        assert "max_tokens" not in params

    def test_build_responses_params(self):
        """Verify Responses API param structure."""
        provider = _make_openai_provider()
        params = provider._build_responses_params(
            model="gpt-5-mini",
            prompt="test prompt",
            schema={"type": "object", "properties": {}},
            schema_name="response",
            max_tokens=2000,
        )

        assert params["model"] == "gpt-5-mini"
        assert params["input"] == "test prompt"
        assert params["text"]["format"]["type"] == "json_schema"
        assert params["text"]["format"]["name"] == "response"
        assert params["max_output_tokens"] == 2000

    def test_build_responses_params_no_max_tokens(self):
        """Without max_tokens, max_output_tokens should not be present."""
        provider = _make_openai_provider()
        params = provider._build_responses_params(
            model="gpt-5-mini",
            prompt="test",
            schema={"type": "object"},
            schema_name="test",
            max_tokens=None,
        )
        assert "max_output_tokens" not in params


# =============================================================================
# Token Usage Extraction Tests
# =============================================================================


class TestOpenAITokenExtraction:
    """Test token usage extraction from OpenAI async responses."""

    @patch.object(OpenAIProvider, "_get_async_client")
    def test_responses_api_extracts_tokens(self, mock_get_async_client):
        """Responses API extracts input_tokens/output_tokens."""
        import asyncio

        provider = _make_openai_provider(_api_format="responses")

        mock_client = MagicMock()
        mock_response = _make_openai_response(
            '{"result": "ok"}', input_tokens=500, output_tokens=200
        )
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        mock_get_async_client.return_value = mock_client

        result, usage = asyncio.run(
            provider.simple_call_async(
                prompt="test", response_schema={"type": "object", "properties": {}}
            )
        )

        assert usage.input_tokens == 500
        assert usage.output_tokens == 200

    @patch.object(OpenAIProvider, "_get_async_client")
    def test_chat_completions_extracts_tokens(self, mock_get_async_client):
        """Chat Completions API extracts prompt_tokens/completion_tokens."""
        import asyncio

        provider = _make_openai_provider(_api_format="chat_completions")

        mock_client = MagicMock()
        mock_response = _make_chat_completions_response(
            '{"result": "ok"}', prompt_tokens=300, completion_tokens=150
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_async_client.return_value = mock_client

        result, usage = asyncio.run(
            provider.simple_call_async(
                prompt="test", response_schema={"type": "object", "properties": {}}
            )
        )

        assert usage.input_tokens == 300
        assert usage.output_tokens == 150

    @patch.object(OpenAIProvider, "_get_async_client")
    def test_missing_usage_returns_zeros(self, mock_get_async_client):
        """When response.usage is None, returns TokenUsage(0, 0)."""
        import asyncio

        provider = _make_openai_provider(_api_format="responses")

        mock_client = MagicMock()
        mock_response = _make_openai_response('{"result": "ok"}')
        mock_response.usage = None
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        mock_get_async_client.return_value = mock_client

        result, usage = asyncio.run(
            provider.simple_call_async(
                prompt="test", response_schema={"type": "object", "properties": {}}
            )
        )

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestClaudeTokenExtraction:
    """Test token usage extraction from Claude async responses."""

    @patch.object(ClaudeProvider, "_get_async_client")
    def test_extracts_tokens(self, mock_get_async_client):
        """Claude extracts input_tokens/output_tokens from response.usage."""
        import asyncio

        provider = _make_claude_provider()

        mock_client = MagicMock()
        mock_response = _make_claude_response(
            {"result": "ok"}, input_tokens=400, output_tokens=180
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_get_async_client.return_value = mock_client

        result, usage = asyncio.run(
            provider.simple_call_async(
                prompt="test", response_schema={"type": "object", "properties": {}}
            )
        )

        assert usage.input_tokens == 400
        assert usage.output_tokens == 180

    @patch.object(ClaudeProvider, "_get_async_client")
    def test_missing_usage_returns_zeros(self, mock_get_async_client):
        """When response.usage is None, returns TokenUsage(0, 0)."""
        import asyncio

        provider = _make_claude_provider()

        mock_client = MagicMock()
        mock_response = _make_claude_response({"result": "ok"})
        mock_response.usage = None
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_get_async_client.return_value = mock_client

        result, usage = asyncio.run(
            provider.simple_call_async(
                prompt="test", response_schema={"type": "object", "properties": {}}
            )
        )

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
