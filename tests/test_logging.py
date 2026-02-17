"""Tests for logging hygiene and redaction."""

import json

import pytest

from extropy.core.providers import logging as provider_logging
from extropy.population.network import metrics as network_metrics


def test_provider_log_request_response_redacts_secrets_and_prompt_content(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(provider_logging, "get_logs_dir", lambda: tmp_path)

    provider_logging.log_request_response(
        function_name="simple_call",
        provider="openai",
        request={
            "model": "gpt-5-mini",
            "api_key": "sk-test-secret",
            "messages": [{"role": "user", "content": "Sensitive persona prompt"}],
            "Authorization": "Bearer abc123",
            "metadata": {"public_note": "ok"},
        },
        response={
            "output_text": "Potentially sensitive response content",
            "usage": {"prompt_tokens": 21, "completion_tokens": 9},
        },
        sources=["https://example.com/source"],
    )

    log_files = list(tmp_path.glob("*_openai_simple_call.json"))
    assert len(log_files) == 1

    payload = json.loads(log_files[0].read_text())
    assert payload["request"]["api_key"] == "[REDACTED_SECRET]"
    assert payload["request"]["Authorization"] == "[REDACTED_SECRET]"
    assert (
        payload["request"]["messages"][0]["content"]
        == "[REDACTED_TEXT length=24]"
    )
    assert payload["response"]["output_text"] == "[REDACTED_TEXT length=38]"
    assert payload["response"]["usage"]["prompt_tokens"] == 21
    assert payload["sources_extracted"] == ["https://example.com/source"]


def test_validate_network_verbose_logs_instead_of_print(capsys, caplog):
    if not network_metrics.HAS_NETWORKX:
        pytest.skip("networkx not installed")

    edges = [
        {"source": "a0", "target": "a1", "weight": 0.8},
        {"source": "a1", "target": "a2", "weight": 0.7},
    ]
    agent_ids = ["a0", "a1", "a2"]

    with caplog.at_level("INFO"):
        is_valid, metrics, warnings = network_metrics.validate_network(
            edges, agent_ids, verbose=True
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert is_valid in (True, False)
    assert metrics.node_count == 3
    assert isinstance(warnings, list)
    assert "Network Validation Report:" in caplog.text
