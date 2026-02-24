# Simulate Stage Reference: Runtime Dynamics and Outcome Realization

## Document Intent
This file is the deep reference for the `simulate` stage in Extropy.

It is meant to be reusable team memory for:
- what `simulate` is supposed to do,
- how it should do it,
- why this stage exists in the pipeline,
- what it currently does in code,
- where current behavior diverges from intended quality goals,
- and what runtime backpressure checks this stage should support.

This is not a patch plan. It is a contract + current-state reality map.

## Status Update (2026-02-23)
- Simulation is now scenario-centric in lookup order (`scenario_id` first, legacy IDs as fallback).
- Runtime loop supports DB-backed resume/checkpoint behavior at timestep and chunk levels.
- Timeline events support provenance epochs (`info_epoch`) and forced re-reasoning escalation.
- Timeline fallback exposure no longer reuses full seed rule sets blindly:
  - fallback rules are filtered to representative channel rules,
  - direct timeline exposure is deduplicated per-agent per-event,
  - direct timeline exposure is capped by intensity (`normal|high|extreme`).
- Reasoning supports two modes:
  - default two-pass (role-play + classification),
  - optional merged single-pass (`--merged-pass`).
- Fidelity modes actively control conversation behavior:
  - `low`: no conversations,
  - `medium`: bounded conversations after chunks,
  - `high`: deeper conversations + extra public-text classification.
- Runtime now applies deterministic per-timestep budgets:
  - bounded reasoning candidate set,
  - bounded conversation count,
  - novelty gate before conversation execution.
- Reasoning prompt payload is now trimmed by fidelity (recent exposures + recent memory only).
- Early convergence behavior is now explicit and overrideable (`auto|on|off`).
- Runtime exports include `by_timestep.json`, `meta.json`, and conditional conversation/social artifacts.
- State manager now auto-upgrades legacy simulation tables/columns (including timeline-provenance exposure fields) on startup.

---

## How To Read This Document
Use this order:
1. `Intended Contract` for target behavior.
2. `Current Implementation (As-Is)` for code truth.
3. `Divergences, Mistakes, and Risk Areas` for the quality gap.
4. `Dependency Map` for upstream/downstream coupling.

This file is intentionally verbose so a new teammate can recover context without prior chat history.

---

## Why This Stage Exists
All earlier pipeline stages compile structure:
- who exists (`spec` + `scenario` + `sample`),
- how they are connected (`network`),
- how they speak/think (`persona`).

`simulate` is where those compiled artifacts are executed over time.

Without `simulate`, Extropy has static data but no dynamics:
- no temporal exposure flow,
- no opinion shifts,
- no sharing cascades,
- no conversation-mediated corrections,
- no run-level outcomes.

Design principle:
- **Execute deterministic state transitions around bounded LLM reasoning, then persist full run traceability.**

---

## Pipeline Position
Intended high-level flow:
1. `spec`: base population contract.
2. `scenario`: event + extension contract.
3. `persona`: rendering contract.
4. `sample`: concrete agents.
5. `network`: concrete graph.
6. `simulate`: dynamic execution over timesteps.
7. `results/query/chat`: interpretation and retrieval.

`simulate` is the first stage that mutates dynamic state over time.

---

## Intended Contract

### What `simulate` should do
`simulate` should produce a reproducible run that:
- applies scenario exposure dynamics over timesteps,
- reasons for the right agents at the right moments,
- propagates through network and conversation channels,
- updates public/private agent states coherently,
- persists enough detail for debugging, analysis, and replay.

At minimum it should:
- validate prerequisites (scenario, agents, network, persona),
- run timestep loop with clear phase ordering,
- enforce deterministic gating around LLM calls,
- support bounded resume/recovery behavior,
- output machine-usable artifacts and run metadata.

### What `simulate` should not do
`simulate` should not:
- silently fabricate missing prerequisites,
- mutate upstream schema contracts,
- hide runtime failure causes,
- collapse all behavior to undifferentiated midpoint outputs,
- require manual table surgery for normal resume behavior.

### How `simulate` should do it
Intended runtime model:
1. preflight + config resolution,
2. load scenario/population/agents/network/persona,
3. initialize run scope + state store,
4. for each timestep: exposures -> reasoning -> conversation -> summary,
5. evaluate stopping conditions,
6. finalize aggregates and export artifacts.

### Why this contract matters
Simulation quality is where trust is won or lost.
Weak runtime controls produce plausible-looking but analytically unstable outputs.

`simulate` is therefore not just “run models”; it is the reliability envelope for behavioral dynamics.

---

## Conceptual Model Of Runtime Dynamics

### Exposure stack
Per timestep exposure can come from:
1. seed rules,
2. timeline events (if scheduled at this timestep),
3. network propagation from sharing agents.

### Reasoning triggers
Agents reason when one of these holds:
- aware and never reasoned,
- forced by newer `info_epoch` timeline provenance,
- multi-touch threshold reached (unique source count) and not committed.

### Public/private state split
Runtime distinguishes:
- public-facing expression (`public_position/sentiment/conviction`),
- private behavior state (`private_*`).

High-conviction guardrails resist abrupt flips unless confidence/support criteria are met.

### Conversation interleaving
For medium/high fidelity, conversation executes after reasoning chunks and can override provisional reasoning states.

### Stop semantics
Run can stop via:
- max timesteps,
- explicit stop conditions,
- optional convergence/quiescence auto-stop depending on `allow_early_convergence` and future timeline events.

---

## Intended Ownership Boundary

### `simulate` should own
- runtime state transitions,
- timestep orchestration,
- reasoning/conversation execution policy,
- stop-condition evaluation,
- run persistence + exports.

### `scenario` should own
- event/timeline definitions,
- exposure/spread/outcome schema,
- simulation horizon and stop condition intent.

### `sample` + `network` should own
- initial node/edge data quality and coherence.

### `results/query/chat` should own
- post-run interpretation and retrieval.

---

## Intended Artifact Contract

### Canonical run persistence (DB)
Simulation should persist:
- run lifecycle records (`simulation_runs`, checkpoints, metadata),
- mutable agent state tables,
- exposures, memory traces, timeline events, timestep summaries,
- conversations/social posts when enabled.

### File exports (results dir)
Runtime should export:
- `by_timestep.json`
- `meta.json`
- optional `conversations.json`
- optional `social_posts.json`
- `elaborations.csv`

### Determinism contract
For fixed `(scenario, sampled agents, network, seed, model behavior)`:
- non-LLM state transitions are deterministic,
- LLM calls are bounded by stable schemas/prompts,
- resume should be consistent with prior persisted checkpoints.

---

## Intended Quality Contract

### Realism
Dynamics should look socially plausible:
- propagation and re-reasoning should respond to new information and relationship context,
- conversation effects should be bounded and not chaotic.

### Internal consistency
Public/private state evolution and sharing decisions should stay coherent with conviction, exposures, and scenario rules.

### Downstream support reliability
Outputs should be directly consumable by:
- `results` timeline/summary views,
- `query` exports,
- `chat` grounded context.

### Backpressure intent
Simulation stage should expose clear stop reason, runtime diagnostics, and failure modes that can be triaged without guesswork.

---

## Current Implementation (As-Is)

This section describes actual behavior in code today.

### CLI command flow
`extropy simulate` currently:
1. resolves study + scenario,
2. preflight-checks sampled agents, network edges, persona config,
3. validates `--resume`/`--run-id`, resource mode, and early-convergence flag,
4. resolves effective models/rate settings from CLI > config,
5. starts simulation with optional live progress view,
6. returns summary data and writes outputs.

Primary file:
- `extropy/cli/commands/simulate.py`

### Preconditions enforced by command
Simulation hard-blocks if any are missing:
- scenario,
- sampled agents for scenario,
- network edges for scenario,
- persona config for scenario.

### Early convergence override semantics
`--early-convergence` handling:
- `on`: force `allow_early_convergence=True`,
- `off`: force `allow_early_convergence=False`,
- `auto`: keep scenario value; if null, runtime auto-rule applies.

Auto-rule:
- early convergence is disabled while future timeline events remain,
- enabled once no future timeline events remain.

### Runtime setup (`run_simulation`)
`run_simulation(...)` currently:
1. loads scenario and applies early-convergence override if requested,
2. resolves base population from scenario metadata,
3. resolves canonical study DB and verifies existence,
4. loads agents and network (scenario key first, legacy fallback),
5. creates simulation run record,
6. clears runtime tables for this run when not resuming,
7. loads persona config (CLI path preferred),
8. backfills persona population stats when empty,
9. builds `SimulationRunConfig` and rate limiter,
10. executes engine and updates run status.

Primary file:
- `extropy/simulation/engine.py`

### Timestep execution order
Each timestep does:
1. mark timestep checkpoint started,
2. apply exposures:
   - seed,
   - timeline event (if this timestep has one),
   - network propagation,
3. select and cap reasoning candidates, then reason in chunks,
4. interleave conversations after chunks (medium/high fidelity, novelty + budget gated),
5. record social posts from sharers,
6. apply conviction decay to non-reasoned agents,
7. compute and save timestep summary,
8. clear timestep checkpoint.

### Timeline behavior
Timeline events only apply when `timeline_event.timestep == current_timestep`.
Non-event timesteps do not inject new world events; prior context remains via state/history.

Timeline exposures carry:
- `info_epoch`,
- optional `re_reasoning_intensity` (`normal|high|extreme`).

When a timeline event does not provide explicit `exposure_rules`, runtime fallback is now bounded:
- prefers rules authored for current timestep (else earliest authored seed step),
- keeps representative rule(s) per channel rather than all overlapping seed rules,
- deduplicates direct event exposure per agent,
- applies an intensity-based direct-exposure cap to prevent saturation spikes.

### Reasoning behavior
Default reasoning mode is two-pass:
1. Pass 1 role-play response (free text + sentiment/conviction/share/action fields).
2. Pass 2 outcome classification for categorical/boolean/float outcomes.

Merged mode (`--merged-pass`) combines both in one schema/call.

Important current behavior:
- open-ended outcomes are not classified in Pass 2; they remain in free-text reasoning.
- high fidelity adds separate public-text classification pass for public position.
- reasoning context is trimmed to recent exposure/memory windows by fidelity to bound token growth.

Primary file:
- `extropy/simulation/reasoning.py`

### Agent selection for reasoning
`get_agents_to_reason(...)` includes:
- aware + never reasoned agents,
- forced re-reason agents with newer timeline info epoch,
- explicit forced IDs from timeline direct exposure paths,
- multi-touch candidates with enough unique source exposures since last reasoning and not committed.

Committed agents are protected from routine multi-touch re-reasoning unless forced by timeline policy.

Additional runtime gating now applies before calls:
- explicit forced IDs are no longer expanded to all aware agents under `extreme`,
- forced expansion uses a bounded high-salience subset (directly impacted + nearby connected + sharers),
- per-timestep reasoning budget caps total LLM calls.

### Conversation behavior by fidelity
- `low`: no conversations executed.
- `medium`: conversations enabled; per-agent cap 1 per timestep; 4-message conversations.
- `high`: per-agent cap 2 per timestep; 6-message conversations.

Conversation outcomes can override provisional reasoning state.

Conversation execution now also requires:
- per-timestep global conversation budget not exhausted,
- novelty gate pass (material shift/uncertainty/share signal from fresh reasoning).

Primary file:
- `extropy/simulation/conversation.py`

### Propagation behavior
Network propagation:
- only from current sharers (`will_share=True`),
- one-shot per source-target-position with reshare on position change,
- share probability from base spread probability + `share_modifiers` + hop decay + soft saturation.

Modifier context includes edge fields:
- `edge_type`
- `edge_weight`

Primary file:
- `extropy/simulation/propagation.py`

### Stop condition behavior
Stop evaluation order:
1. `max_timesteps`,
2. explicit `stop_conditions` expressions,
3. convergence/quiescence auto-stop if early convergence is allowed.

Convergence auto-stop currently uses stable position distribution window checks.
Quiescence auto-stop triggers on three timesteps with zero agents reasoned.

Primary file:
- `extropy/simulation/stopping.py`

### Resume and checkpoint behavior
Resume model:
- requires `--run-id`,
- engine resumes from `checkpoint_timestep` if present, else `last_completed + 1`,
- chunk checkpoints skip already-completed chunks within a timestep.

When not resuming:
- runtime tables for that run ID are reset before execution.

### Output artifacts
Simulation exports:
- `by_timestep.json` (timeline aggregates),
- `meta.json` (models, seed, cost, conversation stats, etc.),
- `conversations.json` (if conversations happened),
- `social_posts.json` (if posts exist),
- `elaborations.csv`.

---

## What Current Implementation Already Gets Right

### Strong persistence and resume scaffolding
Run metadata, timestep checkpoints, and chunk checkpoints make long runs recoverable.

### Clear phase ordering
Exposure -> reasoning -> conversation -> summary sequencing is explicit and traceable.

### Fidelity-aware behavior controls
Conversation workload and prompt richness scale with fidelity mode instead of one-size-fits-all.

### Scenario-coupled temporal logic
Timeline event epochs and early-convergence auto rules reduce premature stopping in evolving scenarios.

---

## Divergences, Mistakes, and Risk Areas

### 1) `interaction` config is effectively informational
`scenario.interaction` fields are not currently consumed by simulation runtime behavior.

### 2) Some scenario metadata remains non-operative in runtime control
Fields such as channel `reach` are not runtime control inputs in simulation propagation.

### 3) Open-ended outcomes are not structurally extracted
Open-ended outcomes live in reasoning text, not in structured Pass 2 outcome payloads.

### 4) Output retention semantics are split
`retention_lite` suppresses raw reasoning payload storage even if scenario outcome config requests full reasoning capture.

### 5) Resume usability requires explicit run-id discipline
`--resume` requires `--run-id`; incorrect run-id usage can lead to user confusion about expected continuation behavior.

### 6) Prompt/context size grows with memory and history
Prompt growth is now bounded by fidelity-window trims, but high-fidelity runs still carry materially higher token/cost load than low fidelity.

### 7) Legacy fallback paths still exist
Simulation still includes legacy lookup fallbacks (`population_id`, `network_id`) which can mask schema drift in older studies.

---

## Dependency Map (What `simulate` Must Support)

### Upstream dependencies
Needs valid outputs from:
- `scenario` (`scenario.vN.yaml`),
- `persona` (`persona.vN.yaml`),
- `sample` (agents in `study.db`),
- `network` (edges in `study.db`).

### Downstream consumers
Feeds:
- `results` (summary/timeline/segment/agent views),
- `query` (states/summary/sql exports),
- `chat` (agent context over runs),
- external DS workflows (`elaborations.csv`, JSON exports).

---

## Working Invariants Draft (Simulate-Oriented)

### Preconditions
1. Scenario, persona, agents, and network must all exist before run starts.
2. Run metadata must be persisted with a stable run id before first timestep mutation.

### Runtime invariants
1. Timestep checkpoint must be set before phase execution and cleared after completion.
2. Agent reasoning selection must be deterministic given state + thresholds + forced IDs.
3. Conversation overrides must be applied after reasoning chunk updates, not before.
4. Stop reason must be explicit when run ends before max timesteps.

### Persistence invariants
1. Failed runs must be marked `failed` in `simulation_runs` with reason.
2. Completed/stopped runs must have final status and persisted summary artifacts.

---

## Glossary (Working Terms)
- **Two-pass reasoning**: role-play generation + separate outcome classification.
- **Merged pass**: single-call schema combining role-play and classification.
- **Info epoch**: timeline provenance marker attached to exposures.
- **Committed agent**: agent with firm conviction protected from routine re-reasoning.
- **Quiescence**: no-agent-reasoned condition for early stop.
- **Retention lite**: mode that drops full raw reasoning payloads to reduce storage volume.

---

## Practical Readout
Current simulate stage is best understood as:
1. a checkpointed runtime orchestrator over scenario dynamics,
2. with bounded LLM reasoning loops plus deterministic state logic,
3. with meaningful fidelity controls and timeline-aware stopping,
4. but still carrying some schema/runtime gaps where metadata is descriptive rather than operative.

For team operation, key framing is:
- simulation quality depends as much on upstream spec/scenario/sample/network quality as on runtime prompts,
- run status, stop reason, and artifact completeness are first-line reliability signals.

---

## Final Statement
`simulate` should convert static pipeline artifacts into temporally coherent behavioral dynamics while preserving clear runtime observability.

Current implementation already has strong execution scaffolding.
The main remaining challenge is contract coherence:
- align descriptive scenario fields with actual runtime control,
- keep resume semantics predictable,
- and maintain realism without exploding token/cost overhead.
