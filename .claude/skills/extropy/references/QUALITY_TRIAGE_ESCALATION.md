# Quality, Triage, and Escalation

Use this file to decide what to ask, what to validate, how to debug, and when to escalate.

## 1) Clarify Before Expensive Runs

Ask the fewest high-leverage questions needed to prevent wasted runs.

Priority questions:
1. What concrete decision should this simulation inform?
2. Which outcome/metric is primary?
3. Who is in scope, and which segments must be broken out?
4. What scenario/change and time horizon should be simulated?
5. What constraints apply (cost/runtime/provider/deadline)?
6. What realism contract applies (hard constraints + benchmark priors)?

Optional follow-ups:
- Baseline only vs alternatives/counterfactuals?
- Single run vs confidence sweep?
- Raw outputs only vs recommendations?
- If tradeoffs appear, optimize for realism or speed/cost?

If user is unsure, use practical defaults and state assumptions before execution.

## 2) Gate Profiles

Select one before sample realism checks:
- `generic-all-ages`
- `adults-only`
- `benchmark-calibrated`
- `us-national-all-ages` (example benchmark profile)

If ambiguous, ask once and proceed.

## 3) Schema Map Requirement (for Realism)

Before strict realism checks, define:
1. Required sampled fields
2. Bounded numeric fields + valid ranges
3. Mutually exclusive categorical sets
4. Hard conditional constraints (must always hold)
5. Soft conditional constraints (target rates/tolerances)
6. Optional benchmark priors + tolerance policy

If schema map is incomplete:
- run structural checks
- mark representativeness as uncalibrated
- escalate for missing constraints when decision risk is high

## 4) Stage Gates

Status:
- PASS: continue
- WARN: continue with documented caveat
- FAIL: fix, rerun affected stages

### Gate 1: `spec` (`base.yaml`)

FAIL on any:
1. File missing/empty after command exit
2. `extropy validate <base.yaml>` fails
3. Explicit outcome leakage in pre-event attributes

WARN/FAIL quality:
- critical distributions rely on low-authority sources

### Gate 2: `extend` (`population.yaml`)

FAIL on any:
1. File missing/empty
2. `extropy validate <population.yaml>` fails
3. New attributes violate pre-event intent unless explicitly intended

WARN/FAIL quality:
- modifier stacking causes boundary instability

### Gate 3: `sample` (`agents.json`)

#### 3A) Structural integrity (hard)

FAIL on any:
1. Requested count != generated count
2. Null values in required fields
3. Out-of-range bounded values
4. Hard exclusivity violations
5. Hard conditional constraint violations

General hard rules:
- cohort/eligibility labels must be consistent
- strict impossible combinations must be zero

#### 3B) Realism (schema + profile)

1. Hard-rule violation count
- FAIL if > 0

2. Soft-constraint deviation
- WARN if over soft tolerance
- FAIL if over 2x soft tolerance

3. Marginal prior drift (if priors provided)
- WARN > 3pp
- FAIL > 5pp

4. Conditional prior drift (if conditional priors provided)
- WARN > 5pp
- FAIL > 10pp

5. Distribution support/collapse
- WARN on unexpected support erosion
- FAIL on unexpected collapse of major categories

Example mapping (all-ages household schema; adapt per study):
- adults (`age > 17`) with `employment_status == "not applicable/child"` -> FAIL if > 0
- minors (`age <= 17`) with non-minor marital statuses -> FAIL if > 0
- adult K-12 enrollment leakage -> WARN > 0.2%, FAIL > 1.0%

Do not proceed to `network` unless Gate 3 is PASS or user accepts WARN explicitly.

### Gate 4: `network` (`network.json`)

FAIL on any:
1. Nodes/edges missing or unparseable
2. Orphan edge endpoints
3. Malformed edge fields used by scenario logic

WARN/FAIL quality:
- graph plausibility (near-empty / nearly complete unintentionally)
- large disconnected components unless intended

### Gate 5: `scenario` (`scenario.yaml`)

FAIL on any:
1. File missing or invalid
2. Exposure logic non-executable/empty for intended channels
3. Outcomes not measurable/schema-consistent

WARN/FAIL quality:
- contradictory stop conditions
- ambiguous outcome definitions

### Gate 6: `simulate` (`results/`)

FAIL on any:
1. Missing required artifacts (`meta.json`, `by_timestep.json`, `outcome_distributions.json`, `agent_states.json`)
2. Invalid/no stop reason and no max-timestep completion

WARN/FAIL quality:
- degenerate dynamics (suspiciously flat or broken)
- runtime/cost outside acceptable operating envelope

## 5) Auto-Fix Loop

If any gate FAILs:
1. Apply the smallest upstream fix tied to the failing metric
2. Rerun only dependent downstream stages
3. Re-run the same gate

If the same gate FAILs twice:
- stop autonomous iteration
- escalate with options and recommendation

## 6) Triage Playbook

### A) Command fails immediately
1. Check path existence
2. Check provider/config mismatch (`extropy config show`)
3. Check API key env vars
4. Retry with smallest reproducible command

### B) Validation/spec issues
Run:
```bash
extropy validate <spec_or_scenario>
```
Common causes:
- formula/condition reference errors
- invalid distribution params
- dependency cycles
- scenario references to unknown attributes/edge types

### C) Exposure not spreading
Inspect:
- `seed_exposure.rules` probabilities/conditions
- spread settings (`share_probability`, modifiers, `max_hops`, `decay_per_hop`)
- network connectivity and edge typing

Evidence files:
- `by_timestep.json`
- `timeline.jsonl`

### D) Weird outcome dynamics
Check:
- outcome definitions and classification clarity
- threshold tuning
- conviction/flip-resistance effects

Evidence files:
- `meta.json`
- `agent_states.json`
- `timeline.jsonl`

### E) Cost/latency blowups
1. Compare `extropy estimate` against `meta.json` actuals
2. Reduce population/timesteps/reasoning frequency
3. Move routine/pass-2 to cheaper model
4. Tune rate and chunk settings

Triage output format:
1. Symptom
2. Likely root cause
3. Evidence
4. Minimal fix
5. Re-run command

## 7) Escalation Policy

Escalate to human before further autonomous edits when:
1. Same gate fails twice
2. Fix changes core study assumptions
3. Accuracy vs speed/cost objectives conflict
4. Sensitive policy/ethics framing needs stakeholder decision
5. Required priors/constraints are unavailable but decision quality depends on them

Escalation payload:
1. Current stage
2. Exact blocker
3. Evidence
4. Options (A/B/C) + tradeoffs
5. Recommended option

## 8) Long-Run Waiting Rule

Do not call failure only because:
- file size is unchanged mid-run
- stdout is quiet
- process still active

Failure requires process error exit, user cancellation, or timeout escalation.
