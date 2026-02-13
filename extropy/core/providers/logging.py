"""Shared logging helpers for LLM providers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def get_logs_dir() -> Path:
    """Get logs directory, create if needed."""
    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def log_request_response(
    function_name: str,
    request: dict,
    response: Any,
    provider: str = "",
    sources: list[str] | None = None,
) -> None:
    """Log full request and response to a JSON file."""
    logs_dir = get_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{provider}_" if provider else ""
    log_file = logs_dir / f"{timestamp}_{prefix}{function_name}.json"

    response_dict = None
    if hasattr(response, "model_dump"):
        try:
            response_dict = response.model_dump(mode="json", warnings=False)
        except Exception:
            response_dict = str(response)
    elif hasattr(response, "__dict__"):
        response_dict = str(response)
    else:
        response_dict = str(response)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "function": function_name,
        "provider": provider,
        "request": request,
        "response": response_dict,
        "sources_extracted": sources or [],
    }

    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)


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
