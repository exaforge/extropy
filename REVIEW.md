# Review

## Scope
- Focused on Python sources under `entropy/` plus CLI utilities and database usage.
- Looked specifically for security risks (eval/SQL), hot fixes, consistency gaps, and quality issues.

## Summary
- 1 P0 security risk related to expression evaluation (`eval`) with user-provided expressions.
- 3 P1 issues around data integrity/perf and hot paths.
- No direct SQL injection paths found in current code.

## Hot Fixes (P0/P1)
1. P0: `eval` is used for user-authored expressions and edge conditions. The current validation only checks syntax, not safety. This allows attribute access and other introspection that can lead to arbitrary code execution if expressions are untrusted. Replace `eval` with an AST-based evaluator that explicitly whitelists node types and disallows attribute/subscript/call except for a strict allowlist.
   Files: `entropy/utils/eval_safe.py:41`, `entropy/population/network/generator.py:24`
2. P1: SQLite foreign keys are declared but never enforced. SQLite disables them by default, so orphaned `exposures`/`memory_traces` rows can accumulate silently. Enable with `PRAGMA foreign_keys=ON` right after connect.
   File: `entropy/simulation/state.py:37`
3. P1: `StateManager.get_agents_to_reason` and `export_final_states` run a per-agent query inside a loop (N+1 pattern). This becomes expensive at large population sizes. Replace with grouped queries or joins to compute counts in one pass.
   Files: `entropy/simulation/state.py:323`, `entropy/simulation/state.py:748`
4. P1: `_apply_rewiring` uses `agent_ids.index(...)` inside the rewiring loop, which is O(n) per iteration and can dominate runtime for large graphs. Precompute an `id_to_idx` map once.
   File: `entropy/population/network/generator.py:542`

## SQL Injection Review
- All SQL statements use parameter binding for dynamic values.
- The only string interpolation is for `ALTER TABLE` with static identifiers in migrations; currently safe as written but should stay static.
- No direct SQL injection vectors found.

## Best Practices / Code Quality
- Expression safety is inconsistent: `utils/expressions.py` validates syntax but does not restrict dangerous AST nodes, while evaluation still uses `eval`. Consolidate to a single, safe AST evaluator so validation and execution match.
  Files: `entropy/utils/expressions.py:1`, `entropy/utils/eval_safe.py:41`, `entropy/population/network/generator.py:24`
- Several broad `except Exception` blocks silently return fallback values, which can mask malformed configs or data corruption. Consider logging warnings or surfacing validation errors to help users diagnose issues.
  Files: `entropy/utils/eval_safe.py:116`, `entropy/population/network/generator.py:43`, `entropy/population/network/metrics.py:87`
- Request/response logging filenames use second-level timestamps and may collide under high throughput. Add milliseconds or a short random suffix to avoid overwrites.
  File: `entropy/core/providers/logging.py:25`
- `StateManager` is never explicitly closed in the simulation engine. A clean close helps ensure all transactions are flushed and file handles released.
  File: `entropy/simulation/engine.py:169`

## Consistency Observations
- Expression evaluation rules differ between population constraints (`eval_safe`) and network edge rules (`_eval_edge_condition`). Centralize evaluation and reuse the same safe evaluator for consistency and safety.
  Files: `entropy/utils/eval_safe.py:41`, `entropy/population/network/generator.py:24`

## Quick Wins
- Add `PRAGMA foreign_keys=ON` immediately after `sqlite3.connect`.
- Cache `id_to_idx` for rewiring.
- Replace `eval` with an AST evaluator and reuse it across modules.
