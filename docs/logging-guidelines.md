# Logging Guidelines

These rules keep logging useful without leaking sensitive data.

## Principles

- Use module loggers: `logger = logging.getLogger(__name__)`.
- Keep user-facing CLI output in `console.print(...)`; use `logger.*` for diagnostics.
- Never log raw prompts, private reasoning text, API keys, auth headers, or tokens.
- Prefer structured metadata in logs (model name, token counts, timings, IDs).
- Use levels consistently:
  - `DEBUG`: high-volume diagnostics safe for local troubleshooting.
  - `INFO`: normal progress and timing.
  - `WARNING`: retries, degraded behavior, recoverable failures.
  - `ERROR`: terminal failures for the current operation.

## Provider Debug Logs

`extropy/core/providers/logging.py` writes sanitized JSON logs:

- Secret fields are replaced with `[REDACTED_SECRET]`.
- Prompt/content-like text fields are replaced with `[REDACTED_TEXT length=N]`.
- Responses are summarized and sanitized before writing.

If you add new request/response fields, ensure they pass through the sanitizer.
