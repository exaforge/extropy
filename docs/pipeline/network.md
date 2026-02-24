# Network Stage Reference: Social Graph Generation and Topology Gating

## Document Intent
This file is the deep reference for the `network` stage in Extropy.

It is meant to be reusable team memory for:
- what `network` is supposed to do,
- how it should do it,
- why this stage exists in the pipeline,
- what it currently does in code,
- where current behavior diverges from intended quality goals,
- and what backpressure checks this stage should support.

This is not a patch list. It is a contract + current-state reality map.

## Status Update (2026-02-23)
- Network generation now runs an adaptive multi-stage calibration loop with explicit topology gate bounds.
- Topology gate evaluation now uses final graph metrics after rewiring, with pre-rewire fallback if rewiring degrades an otherwise gate-valid graph.
- Config generation is LLM-based by default, with deterministic config-source priority and automatic scenario-local config discovery.
- Structural attribute role mapping is now deterministic-first with targeted LLM tie-break only for ambiguous roles.
- Structural edge generation supports partner/household/coworker/neighbor/congregation/school-parent channels with deterministic per-channel quotas and caps.
- Checkpoint/resume for similarity and calibration progress is DB-backed (`study.db`), with signature compatibility checks.
- Strict topology-gate failures can be quarantined under a suffixed network id.
- Network command supports quality profiles (`fast|balanced|strict`) and candidate modes (`exact|blocked`) with resource auto-tuning.

---

## How To Read This Document
Use this order:
1. `Intended Contract` for target behavior.
2. `Current Implementation (As-Is)` for actual code behavior.
3. `Divergences, Mistakes, and Risk Areas` for the quality gap.
4. `Dependency Map` for downstream coupling (`simulate`, `results`, `query`).

This file is intentionally verbose so a new teammate can recover context without prior chat history.

---

## Why This Stage Exists
`sample` produces agents. `network` produces relationships.

Without `network`, simulation has isolated individuals and cannot represent:
- exposure through social ties,
- peer influence asymmetry,
- neighborhood/workplace/religious clustering,
- bridge ties across communities,
- conversation and propagation pathways.

Design principle:
- **Use sampled attributes to instantiate realistic social structure, then gate topology quality before simulation.**

---

## Pipeline Position
Intended high-level flow:
1. `spec`: compile reusable base population mechanics.
2. `scenario`: compile event context + extension attributes.
3. `persona`: compile language rendering contract.
4. `sample`: instantiate concrete agents.
5. `network`: generate social graph over sampled agents.
6. `simulate`: propagate and reason over that graph.

`network` is the structural bridge from agent rows to social dynamics.

---

## Intended Contract

### What `network` should do
`network` should generate a scenario-scoped social graph that is:
- realistic enough for downstream diffusion and conversation,
- topology-checked against explicit bounds,
- reproducible under seed and config,
- inspectable via run metadata and calibration traces.

At minimum it should:
- resolve or generate a network config from known inputs,
- compute relationship probabilities from attribute similarity and role structure,
- add deterministic structural ties where semantically justified,
- calibrate graph shape toward target degree/clustering/modularity/connectivity,
- persist graph + diagnostics to canonical study storage.

### What `network` should not do
`network` should not:
- mutate sampled agent attributes,
- invent scenario-level semantics not present in upstream artifacts,
- silently skip topology quality checks while claiming success,
- depend on hidden magic defaults without exposing metadata.

### How `network` should do it
Intended execution model:
1. Load sampled agents for scenario.
2. Resolve network config (file or generated).
3. Compute sparse similarity candidates.
4. Build structural edges and similarity edges.
5. Calibrate network topology to targets.
6. Apply controlled rewiring.
7. Evaluate topology gate and persist outcomes.

### Why this contract matters
If network quality is weak:
- spread realism collapses,
- conversation targets become noisy,
- identity clustering effects are distorted,
- simulation output can look plausible but be structurally wrong.

`network` is therefore a quality gate, not just an edge sampler.

---

## Conceptual Model Of Network Generation

### Dual-source edge construction
Current design combines:
1. **Similarity-driven edges** (homophily + degree effects).
2. **Structural edges** (partner/household/workplace/neighborhood/congregation/school-parent).

This avoids two extremes:
- pure random/flat graphs,
- over-hardcoded deterministic graphs.

### Calibration-first topology shaping
Graph generation is not single-pass final.
It calibrates intra/inter-community scaling with restarts and iterations, then evaluates gate bounds.

### Controlled rewiring as optimization pass
Rewiring is applied after calibration for small-world properties, but should not be allowed to invalidate a graph that already passed the gate.

### Bounded quality semantics
Quality is represented via explicit bounds and deltas:
- average degree range,
- minimum clustering,
- modularity range,
- minimum largest connected component ratio,
- minimum edge floor.

---

## Intended Ownership Boundary

### `sample` should own
- concrete agent attribute values,
- household/partner/dependent realization,
- semantic validity of agent records.

### `network` should own
- edge existence, edge type, edge weight, influence asymmetry,
- structural vs similarity composition,
- topology quality gating + diagnostics.

### `scenario` should own
- scenario semantics and extension attributes,
- optional intent that may influence network config generation.

### `simulate` should own
- runtime propagation and reasoning behavior over the saved graph.

---

## Intended Artifact Contract

### Primary runtime outputs
`network` should emit:
- edge list (source, target, type, weights, influence weights),
- network generation metadata (seed, candidate mode, quality diagnostics),
- optional computed network/node metrics.

### Canonical persistence contract
Current DB entities used by network stage:
- `network_runs`
- `network_edges`
- `network_metrics` (optional)
- `network_generation_status`
- `network_calibration_runs`
- `network_calibration_iterations`
- `network_similarity_jobs`
- `network_similarity_chunks`
- `network_similarity_pairs`

### Determinism contract
For fixed `(agents, config, seed)`:
- candidate selection and calibration are seeded,
- structural edge generation is deterministic under seed,
- final topology still depends on configured calibration limits and stage escalation.

---

## Intended Quality Contract

### Realism
Graph should reflect plausible social structure:
- in-group clustering where relevant,
- bridge ties across groups,
- role-weighted hubs without pathological concentration.

### Internal consistency
Edges should be coherent with agent data:
- structural ties grounded in household/role fields,
- edge type rules referencing existing attributes,
- influence weights consistent with configured factors.

### Downstream support reliability
Output should support:
- propagation rules using edge properties,
- conversation selection over realistic neighbor sets,
- scenario-specific simulation at target scale.

### Backpressure intent
Network stage should expose machine-readable go/no-go signals before simulation.

---

## Current Implementation (As-Is)

This section describes actual behavior in code today.

### CLI command flow
`extropy network` currently performs:
1. resolve study + scenario,
2. pre-flight check that sampled agents exist for scenario,
3. load scenario spec and base population reference,
4. merge base + extension attributes for config-generation context,
5. resolve config source,
6. apply CLI overrides + quality/resource defaults,
7. generate network (with or without metrics),
8. evaluate quality gate status,
9. persist network result (canonical/quarantine behavior),
10. optionally export JSON.

Primary file:
- `extropy/cli/commands/network.py`

### Config source resolution order
Config resolution is deterministic:
1. explicit `--network-config`
2. auto-detected latest `scenario/<name>/*.network-config.yaml`
3. LLM generation (default `--generate-config`)
4. empty config fallback (flat network warning)

Generated configs are auto-saved by default to scenario directory unless disabled with hidden flag.

### LLM config generation behavior
`generate_network_config(...)` runs:
1. core config call (weights, multipliers, edge rules, influence, avg degree),
2. structural-role mapping call only when deterministic resolver is ambiguous.

Prompt context includes:
- population description + geography,
- up to first 50 merged attributes,
- up to first 5 sampled agents as examples.

Primary file:
- `extropy/population/network/config_generator.py`

### Structural-role mapping behavior
Role slots:
- `household_id`, `partner_id`, `age`, `sector`, `region`, `urbanicity`, `religion`, `dependents`

Behavior:
- deterministic scoring picks high-confidence roles directly,
- ambiguous roles use constrained LLM tie-break against candidate list,
- runtime key resolution uses config-selected role first, then minimal canonical fallback keys.

Primary files:
- `extropy/population/network/config_generator.py`
- `extropy/population/network/generator.py`

### Similarity and candidate behavior
Supports candidate modes:
- `exact`: all-pairs candidate space
- `blocked`: block-pruned candidate space with deterministic bridge quota

Blocked mode:
- auto-selects blocking attributes from highest weighted exact/within_n attributes,
- filters over-fragmented blockers by cardinality ratio,
- expands candidate pools when coverage is sparse.

Similarity computation:
- sparse retention threshold via `similarity_store_threshold`,
- serial or process-parallel execution,
- optional checkpoint/resume via study DB similarity job tables.

Primary files:
- `extropy/population/network/similarity.py`
- `extropy/population/network/generator.py`

### Calibration ladder behavior
Generation runs staged candidate ladders:
- exact mode: single stage,
- blocked mode: staged escalation (`blocked`, `blocked-expanded`, and for non-fast profiles `hybrid-dense`).

Per stage:
- compute similarities + coverage diagnostics,
- assign communities,
- create structural edges with budget headroom,
- calibrate intra/inter community scaling across restarts and iterations,
- score against gate bounds.

Best stage/iteration is retained.

### Structural edge behavior
Deterministic structural channels currently implemented:
- mandatory: `partner`, `household`
- optional budgeted: `coworker`, `neighbor`, `congregation`, `school_parent`

Channel behavior includes local caps and quotas.

### Rewiring and final gate behavior
After choosing best calibrated graph:
- rewiring pass runs (skips protected structural pairs),
- topology gate evaluated pre- and post-rewire,
- if rewiring breaks a previously passing graph, pre-rewire graph is restored.

Quality metadata records:
- gate acceptance,
- best/final metrics,
- bounds,
- deltas,
- stage summaries.

### Metrics behavior
Without `--no-metrics`, command computes:
- network metrics (avg degree, clustering, path length, modularity, etc.),
- node metrics (PageRank, betweenness, cluster id, echo chamber score).

Metrics require `networkx`.

Primary file:
- `extropy/population/network/metrics.py`

### Save behavior and strict gate handling
Canonical network id defaults to scenario name.

Strict failure condition in command:
- topology gate is strict,
- quality accepted is false,
- agent count >= 50.

Then:
- with quarantine allowed: saves under suffixed network id and exits non-zero,
- without quarantine: exits non-zero.

Persistence helper sets `meta.outcome` as:
- `accepted`,
- `accepted_with_warnings`,
- `rejected`,
- `rejected_quarantined`.

Primary file:
- `extropy/cli/commands/network.py`

### Runtime coupling with simulation
Simulation loader resolves network by scenario name first, then scenario meta fallback id.
Propagation/conversation scoring uses edge type and edge weight in runtime logic.

Primary files:
- `extropy/simulation/engine.py`
- `extropy/simulation/propagation.py`

---

## What Current Implementation Already Gets Right

### Strong topology-aware generation core
Adaptive calibration + bounds-based gating is materially stronger than single-pass random graph synthesis.

### Checkpoint/resume observability
Long runs expose status and calibration traces in DB tables, supporting inspectability and partial recovery.

### Structural + similarity blend
Combining household/partner structure with similarity-based edges gives better realism than either source alone.

### Rewire safety fallback
Pre-rewire fallback avoids accidental quality regression from late optimization passes.

---

## Divergences, Mistakes, and Risk Areas

### 1) Config generation evidence is thin
LLM config prompt uses at most first 5 agents and truncated attribute list context.
This can underrepresent long-tail population structure for large/narrow populations.

### 2) Scenario intent bridge is weak
Network config generation is primarily population-driven.
Scenario dynamic intent (channels/spread pressure) is not a first-class input to config generation.

### 3) Structural channel set is fixed
Structural edge channels are currently fixed templates.
This is practical, but can miss context-specific relationship modes for some populations.

### 4) Structural semantics still rely on key conventions
Role mapping improved, but fallback still depends on canonical key names when config roles are missing/unpopulated.

### 5) Strict-failure save semantics are operationally subtle
Command-level strict failure handling and persistence ordering create edge cases that can be surprising in workflows.
Team should treat strict-fail artifacts carefully when deciding what is canonical.

### 6) Advanced tuning knobs are mostly hidden
Important controls exist (`topology_gate`, calibration budgets, quarantine toggles) but are hidden flags.
This can reduce discoverability during debugging.

### 7) Networkx dependency bifurcation
Core generation does not require full metrics, but metric paths do.
Tooling/CI environments without `networkx` can see behavior differences under `--no-metrics` vs default.

---

## Dependency Map (What `network` Must Support)

### `simulate`
Needs:
- robust edge set for scenario,
- meaningful edge types and influence weights,
- stable network_id resolution path.

### `results` and `query`
Need:
- persisted network edges and metadata,
- calibration/status records for diagnostics,
- optional network metrics for analysis.

### `chat` and agent-level introspection
Indirectly depend on simulation dynamics that are shaped by network structure.

---

## Working Invariants Draft (Network-Oriented)

### Core invariants
1. Scenario must have sampled agents before network stage runs.
2. Every saved edge must reference valid source/target agent ids.
3. Undirected edge uniqueness must be preserved (no duplicate pair rows).
4. Structural protected edges must not be rewired.

### Config invariants
1. Attribute references in weights/multipliers/rules/factors must exist in merged schema.
2. `within_n` match type must carry ordinal level mapping.
3. Candidate mode and topology gate enums must be valid.

### Quality invariants
1. Gate bounds and deltas must be present in saved quality metadata.
2. Strict gate failures should never silently report success.
3. Final accepted graph should correspond to the same metric set used for pass/fail.

---

## Glossary (Working Terms)
- **Candidate mode**: strategy for similarity pair search (`exact` vs `blocked`).
- **Blocked mode**: candidate pruning using shared block attributes plus deterministic global bridging.
- **Structural edges**: deterministic relation edges from known household/workplace/neighborhood-style roles.
- **Calibration ladder**: staged escalation process to improve similarity coverage and topology fit.
- **Topology gate**: pass/fail check against configured graph metric bounds.
- **Quarantine network**: strict-failed artifact saved under suffixed `network_id` for inspection.

---

## Practical Readout
Current network stage is best understood as:
1. a calibrated graph generator (not one-shot),
2. with mixed deterministic + model-driven configuration,
3. with strong quality metadata and run traceability,
4. but still carrying some fixed structural assumptions and intent-bridge gaps.

For team operation, key framing is:
- generation quality is mostly controlled by config quality + candidate coverage + calibration budgets,
- strict gate outcomes should be treated as hard backpressure before simulation.

---

## Final Statement
`network` should convert sampled populations into topology-credible social structure and expose clear go/no-go signals for simulation.

Current implementation already has substantial quality machinery.
The main remaining challenge is contract coherence:
- stronger scenario-intent coupling,
- clearer canonical vs quarantined persistence semantics,
- and predictable structural-role grounding across diverse populations.
