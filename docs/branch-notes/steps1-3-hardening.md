# Branch Handoff: `codex/steps1-3-hardening`

## Purpose
This branch hardens steps 1-3 of the pipeline (spec/scenario/sample validation and sampler consistency) and related persona consistency checks.

## Commits Included (in order)
1. `969ea9d` - fix(scenario): include extended attributes in validator references (#117)
2. `de2298c` - fix(scenario): validate condition literals against categorical options (#118)
3. `0755fae` - fix(cli): add persona validation path and robust validate type detection (#110)
4. `72af9a4` - fix(sampler): reconcile household-derived attributes after assignment (#114)
5. `696c81c` - refactor(sampler): make partner correlation policy-driven with metadata fallback (#123)
6. `84a4ef3` - feat(validator): detect ambiguous categorical/boolean modifier overlaps (#122)
7. `f85393d` - fix(sampler): surface modifier condition eval failures with strict/permissive modes (#121)
8. `ee7a262` - feat(sample): enforce expression constraints and promoted warning gates without new flags (#119 #120)
9. `b790597` - fix(persona): apply semantic context to avoid unemployed/occupation contradictions (#113)

## Exact Fixes by Area

### Scenario Validation
- Validator now resolves references against both base population attributes and scenario `extended_attributes`.
- Validator now checks condition string literals against known categorical options, preventing case/value drift (e.g., invalid enum labels).
- Timeline exposure rule conditions now get the same attribute/literal validation checks.

### CLI Validation
- `extropy validate` spec-type detection is more robust (population vs scenario vs persona).
- Persona config validation path added with structural checks and context-aware checks when scenario/population can be resolved.

### Sampling / Household Integrity
- Post-sampling reconciliation pass aligns household-derived fields to actual sampled household composition:
  - household size
  - has-children flags
  - children count fields (when present)
  - marital consistency for partnered households/dependent agents
- Sampling stats are recomputed after reconciliation so summary stats reflect final values.

### Partner Correlation Generalization
- Added explicit `partner_correlation_policy` support in population models.
- Correlation algorithm resolution is now policy/metadata driven first, with legacy-name fallback for backward compatibility.
- Added semantic warning when `partner_correlated` attrs lack policy/semantic metadata.

### Semantic Validator Enhancements
- Added overlap analysis for categorical/boolean modifiers (`MODIFIER_OVERLAP`, `MODIFIER_OVERLAP_EXCLUSIVE`).
- Added partner-policy completeness warnings (`PARTNER_POLICY`).

### Sampler Failure Visibility / Strictness
- Modifier condition evaluation failures are now surfaced with strict/permissive behavior:
  - strict mode: fail sampling
  - permissive mode: collect warnings
- `sample` now enforces expression constraints in normal mode (without `--skip-validation`).
- Some semantic warnings are promoted to blocking during strict sampling paths.

### Persona Rendering Consistency
- Added semantic-context phrase override so non-working agents are not rendered with contradictory active-employment phrasing.
- Simulation persona generation now passes semantic metadata to renderer.

## Test Coverage Added/Updated
- `tests/test_scenario_validator.py`
- `tests/test_cli.py`
- `tests/test_household_sampling.py`
- `tests/test_validator.py`
- `tests/test_sampler.py`
- `tests/test_persona_renderer.py`

## Current Known Gaps (not fixed in this branch)
- Spec overlap volume can be high; overlap warnings require spec-side cleanup for deterministic behavior.
- Household type labels can still be semantically mismatched in some edge cases (`couple_with_kids`/`single_parent` labels vs zero dependents after reconciliation).
- Some implausible demographic combinations remain spec-driven (not engine bugs), e.g. age/education/employment combinations where conditional coverage is incomplete.

## Issue Tracking Guidance (keep issues open for merge coordination)
Do **not** close issues until merge + verification in shared integration branch.
Suggested per-issue state update:
- Add comment: "Implemented on `codex/steps1-3-hardening`, pending integration verification".
- Link commit hash(es) above.
- Keep status open with label like `ready-for-merge-test`.

Mapped issues on this branch:
- #110, #113, #114, #117, #118, #119, #120, #121, #122, #123

## Merge Safety Notes
- This branch intentionally increases strictness in sampling validation; expect older specs to fail faster.
- If another branch modifies validator/sampler internals, resolve conflicts by preserving:
  - extended-attr aware scenario validation
  - post-household reconciliation pass
  - strict/permissive condition handling
  - expression constraint enforcement behavior
