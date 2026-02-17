"""Shared logging helpers for LLM providers."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SECRET_KEY_MARKERS = ("api_key", "authorization", "token", "secret", "password")
_TOKEN_COUNT_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
}
_TEXT_KEY_MARKERS = (
    "prompt",
    "content",
    "input",
    "output",
    "text",
    "message",
    "reasoning",
    "statement",
    "thought",
    "elaboration",
)


def get_logs_dir() -> Path:
    """Get logs directory, create if needed."""
    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _sanitize_for_logs(value: Any, key_hint: str = "") -> Any:
    """Recursively sanitize payloads before persisting debug logs."""
    key = key_hint.lower()
    if key in _TOKEN_COUNT_KEYS:
        return value
    if any(marker in key for marker in _SECRET_KEY_MARKERS):
        return "[REDACTED_SECRET]"

    if isinstance(value, dict):
        return {
            str(k): _sanitize_for_logs(v, key_hint=str(k))
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_for_logs(item, key_hint=key_hint) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_for_logs(item, key_hint=key_hint) for item in value]

    if isinstance(value, str):
        if any(marker in key for marker in _TEXT_KEY_MARKERS):
            return f"[REDACTED_TEXT length={len(value)}]"
        if len(value) > 200:
            return value[:200] + "...[truncated]"
        return value

    return value


def _serialize_response(response: Any) -> Any:
    """Convert provider response to a serializable, sanitized structure."""
    if isinstance(response, dict):
        return _sanitize_for_logs(response, key_hint="response")

    if hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump(mode="json", warnings=False)
            return _sanitize_for_logs(dumped, key_hint="response")
        except Exception:
            pass

    usage = getattr(response, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = _sanitize_for_logs(
            getattr(usage, "__dict__", str(usage)),
            key_hint="usage",
        )

    summary = {
        "type": type(response).__name__,
    }
    model_name = getattr(response, "model", None)
    if model_name:
        summary["model"] = model_name
    response_id = getattr(response, "id", None)
    if response_id:
        summary["id"] = response_id
    if usage_dict is not None:
        summary["usage"] = usage_dict

    return summary


def log_request_response(
    function_name: str,
    request: dict,
    response: Any,
    provider: str = "",
    sources: list[str] | None = None,
) -> None:
    """Log sanitized request/response metadata to a JSON file."""
    logs_dir = get_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{provider}_" if provider else ""
    log_file = logs_dir / f"{timestamp}_{prefix}{function_name}.json"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "function": function_name,
        "provider": provider,
        "request": _sanitize_for_logs(request, key_hint="request"),
        "response": _serialize_response(response),
        "sources_extracted": sources or [],
    }

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to write provider debug log %s: %s", log_file, exc)


def extract_error_summary(error_msg: str) -> str:
    """Extract a concise error summary from validation error message."""
    if not error_msg:
        return "validation error"

    lines = error_msg.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("---"):
            if "ERROR in" in line:
                return line[:60]
            elif "Problem:" in line:
                return line.replace("Problem:", "").strip()[:60]
            elif line:
                return line[:60]

    return "validation error"
