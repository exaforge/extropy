# Concrete Code Fixes

Actionable fixes from the codebase review, organized by priority. Each includes the exact file, what to change, and why.

---

## P0: Performance (blocking at scale)

### Fix 1: Build adjacency list at simulation startup

**File:** `entropy/simulation/engine.py`

In `__init__`, build an adjacency dict from the flat edge list once. Pass it to propagation functions instead of the raw network JSON.

```python
# In SimulationEngine.__init__, after loading network:
self.adjacency: dict[str, list[tuple[str, dict]]] = defaultdict(list)
for edge in self.network.get("edges", []):
    src, tgt = edge["source"], edge["target"]
    self.adjacency[src].append((tgt, edge))
    self.adjacency[tgt].append((src, edge))
```

Then replace every call to `get_neighbors(network, agent_id)` with `self.adjacency[agent_id]`.

**Impact:** Eliminates O(E) scan per neighbor lookup. For 10,000 edges and 100 agents/timestep, saves ~1M comparisons per timestep.

---

### Fix 2: Batch SQLite commits per timestep

**File:** `entropy/simulation/state.py`

Add a transaction context manager and use it in the engine's timestep loop:

```python
@contextmanager
def transaction(self):
    """Batch all writes within this block into a single commit."""
    try:
        yield
    except Exception:
        self.conn.rollback()
        raise
    else:
        self.conn.commit()
```

Remove individual `self.conn.commit()` calls from `record_exposure()`, `update_agent_state()`, `save_memory_entry()`, `log_event()`. Wrap the timestep in `with self.state_manager.transaction():`.

Also: call the existing `batch_update_states()` (line 448) instead of looping `update_agent_state()` one at a time in `engine.py:333-423`.

**Impact:** Reduces per-timestep commits from hundreds to 1. SQLite commit involves fsync -- this is a major I/O win.

---

### Fix 3: Pass engine's agent_map to propagation

**File:** `entropy/simulation/propagation.py:224`

Currently rebuilds `agent_map` every timestep:
```python
agent_map = {a.get("_id", str(i)): a for i, a in enumerate(agents)}
```

Change `propagate_through_network()` to accept `agent_map` as a parameter and pass `self.agent_map` from the engine.

---

## P0: Error Handling (prevents crashes)

### Fix 4: Wrap provider API calls with retry

**Files:** `entropy/core/providers/claude.py`, `entropy/core/providers/openai.py`

Add a retry decorator or wrapper for API calls. Catch provider-specific transient errors:

```python
# claude.py -- wrap each client.messages.create() call
import anthropic

RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
)

def _api_call_with_retry(fn, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RETRYABLE as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
```

Same pattern for openai.py with `openai.APIConnectionError`, `openai.InternalServerError`, `openai.RateLimitError`.

---

### Fix 5: Extract shared retry logic to base class

**File:** `entropy/core/providers/base.py`

Both `claude.py` and `openai.py` have ~70 lines of identical validation-retry logic in `reasoning_call()`. Move to base:

```python
# base.py
def _retry_with_validation(self, call_fn, validator, max_retries, on_retry):
    """Shared retry loop for validation failures."""
    attempts = 0
    last_error = ""
    while attempts <= max_retries:
        result = call_fn(previous_errors=last_error if attempts > 0 else None)
        if validator is None:
            return result
        error = validator(result)
        if error is None:
            return result
        last_error = error
        if on_retry:
            on_retry(attempts, error)
        attempts += 1
    return result  # Return last attempt even if invalid
```

---

## P1: CI & Testing

### Fix 6: Add GitHub Actions test workflow

**File:** `.github/workflows/test.yml` (new)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run ruff check .
      - run: uv run pytest
```

---

### Fix 7: Add rate_limiter tests

**File:** `tests/test_rate_limiter.py` (new)

The rate limiter is pure logic with no external dependencies -- easy to test:
- Token bucket refill behavior
- Dual bucket acquire/release
- Rollback when one bucket fails
- `max_safe_concurrent` calculation
- Provider-specific profile loading

---

### Fix 8: Add CLI smoke tests

**File:** `tests/test_cli.py` (new)

Use Typer's `CliRunner` to test commands handle bad input gracefully:
- `entropy validate` with nonexistent file
- `entropy config show` with no config
- `entropy config set` with invalid key/value
- `entropy sample` with malformed spec

---

## P1: Dependencies

### Fix 9: Remove unused deps, pin versions

**File:** `pyproject.toml`

```diff
- "fastapi>=0.115.0",
- "uvicorn>=0.32.0",
+ "openai>=1.50.0,<2",
+ "anthropic>=0.77.0,<1",
+ "pydantic>=2.9.0,<3",
+ "numpy>=2.0.0,<3",
```

Remove fastapi and uvicorn entirely. Pin major versions on the rest.

---

### Fix 10: Fix .gitignore gaps

**File:** `.gitignore`

Add:
```
*.sqlite
*.db
*.jsonl
__pypackages__/
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

---

## P2: Code Quality

### Fix 11: Extract conviction constants

**File:** `entropy/simulation/engine.py:56-59`

Currently:
```python
_FIRM_CONVICTION = 0.7
_MODERATE_CONVICTION = 0.5
_SHARING_CONVICTION_THRESHOLD = 0.1
```

These duplicate values from `ConvictionLevel` enum. Instead:
```python
from ..core.models.simulation import ConvictionLevel

_FIRM_CONVICTION = ConvictionLevel.FIRM.to_float()
_MODERATE_CONVICTION = ConvictionLevel.MODERATE.to_float()
_SHARING_CONVICTION_THRESHOLD = ConvictionLevel.VERY_UNCERTAIN.to_float()
```

If `to_float()` doesn't exist on the enum, add it.

---

### Fix 12: Extract _extract_output_text in openai.py

**File:** `entropy/core/providers/openai.py`

Replace the 4 copies of nested response traversal with:

```python
def _extract_output_text(self, response) -> str | None:
    """Extract text content from OpenAI response output."""
    for item in response.output:
        if hasattr(item, "type") and item.type == "message":
            for content in item.content:
                if hasattr(content, "type") and content.type == "output_text":
                    if hasattr(content, "text"):
                        return content.text
    return None
```

---

### Fix 13: Break _run_timestep into sub-functions

**File:** `entropy/simulation/engine.py`

Split the ~150-line method into:
- `_apply_seed_exposures(timestep)` -- seed exposure logic
- `_propagate_and_select(timestep)` -- network propagation + agent selection
- `_reason_agents(contexts, old_states)` -- two-pass reasoning
- `_apply_state_updates(results)` -- conviction updates, sharing, flip resistance

---

### Fix 14: Add estimator token constants

**File:** `entropy/simulation/estimator.py:101-110`

Replace:
```python
persona_tokens = 80 + 15 * num_attributes
pass1_input = 250 + persona_tokens + event_tokens + 115
```

With named constants:
```python
# Empirically measured from production prompts (Jan 2026)
PERSONA_BASE_TOKENS = 80        # System prompt overhead for persona section
PERSONA_PER_ATTRIBUTE = 15      # Avg tokens per rendered attribute
PASS1_SYSTEM_OVERHEAD = 250     # System prompt + instructions
PASS1_CONTEXT_OVERHEAD = 115    # Memory trace + peer opinions avg
PASS1_OUTPUT_ESTIMATE = 200     # Avg response (reasoning + public_statement)
PASS2_INPUT_ESTIMATE = 300      # Classification prompt + reasoning text
PASS2_OUTPUT_ESTIMATE = 70      # Position + conviction JSON
```

---

### Fix 15: Unify progress callback type

**File:** `entropy/core/models/` or `entropy/utils/`

Define a single callback protocol:

```python
from typing import Protocol

class ProgressCallback(Protocol):
    def __call__(self, step: str, status: str, count: int | None = None) -> None: ...
```

Use this type in hydrator.py, persona/generator.py, and engine.py.

---

### Fix 16: config_cmd.py int conversion safety

**File:** `entropy/cli/commands/config_cmd.py:168`

```diff
  if field in INT_FIELDS:
-     setattr(target, field, int(value))
+     try:
+         setattr(target, field, int(value))
+     except ValueError:
+         console.print(f"[red]Invalid value for {key}: expected integer, got '{value}'[/red]")
+         raise typer.Exit(1)
```

---

## P3: Test Coverage Expansion

### Fix 17: Mock-based simulation engine tests

Test the engine's orchestration logic with mocked LLM calls. Verify:
- Agents are exposed according to seed rules
- Multi-touch threshold triggers re-reasoning
- Conviction-based flip resistance works
- Stopping conditions are checked
- State is persisted correctly

### Fix 18: Scenario compiler tests

Test the 5-step compilation pipeline with known inputs. Verify:
- Event type detection
- Exposure channel generation
- Outcome configuration
- Simulation parameter auto-config (population size -> timesteps)

### Fix 19: Provider tests with mocked HTTP

Use `unittest.mock.patch` or `responses` library to mock API calls. Verify:
- Structured output extraction
- Source URL extraction from web search results
- Retry on validation failure
- Error propagation on auth failure

---

## Tracking

| Fix | Priority | Effort | Status |
|-----|----------|--------|--------|
| 1. Adjacency list | P0 | Small | TODO |
| 2. Batch SQLite commits | P0 | Small | TODO |
| 3. Pass agent_map | P0 | Trivial | TODO |
| 4. Provider API retry | P0 | Medium | TODO |
| 5. Extract retry to base | P0 | Medium | TODO |
| 6. CI test workflow | P1 | Small | TODO |
| 7. Rate limiter tests | P1 | Medium | TODO |
| 8. CLI smoke tests | P1 | Medium | TODO |
| 9. Pin deps, remove unused | P1 | Trivial | TODO |
| 10. .gitignore gaps | P1 | Trivial | TODO |
| 11. Conviction constants | P2 | Trivial | TODO |
| 12. Extract openai helper | P2 | Small | TODO |
| 13. Break _run_timestep | P2 | Medium | TODO |
| 14. Estimator constants | P2 | Trivial | TODO |
| 15. Unify progress callback | P2 | Small | TODO |
| 16. config_cmd safety | P2 | Trivial | TODO |
| 17. Engine tests (mocked) | P3 | Large | TODO |
| 18. Scenario compiler tests | P3 | Medium | TODO |
| 19. Provider tests (mocked) | P3 | Medium | TODO |
