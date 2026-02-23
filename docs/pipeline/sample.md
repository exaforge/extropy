# Sample Stage Reference: Agent Instantiation and Household Realization

## Document Intent
This file is the deep reference for the `sample` stage in Extropy.

It is meant to be reusable team memory for:
- what `sample` is supposed to do,
- how it should do it,
- why this stage exists in the pipeline,
- what it currently does in code,
- where the current behavior diverges from intended quality goals,
- and what backpressure checks this stage should ultimately support.

This is not a patch plan. It is a contract + current-state reality map.

## Status Update (2026-02-23)
- Merged sampling order now recomputes topologically across base + extension attributes.
- Post-household reconciliation now deterministically aligns:
  - marital/partner coherence
  - household size with realized household composition
  - household surnames across members, partner NPCs, and dependent name fields
- Reconciliation counters are recorded in sampling stats.
- Modifier condition-evaluation failures are now surfaced into `condition_warnings`.
- Deterministic rule-pack gate is applied post-sampling with impossible/implausible thresholds.
- Strict mode (`--strict-gates`) promotes high-risk warnings and condition-eval failures to hard blocks.
- Fail-hard sample paths now emit versioned `sample.invalid.vN.json` diagnostics.

---

## How To Read This Document
Use this order:
1. `Intended Contract` for target behavior.
2. `Current Implementation (As-Is)` for actual behavior.
3. `Divergences, Mistakes, and Risk Areas` for the quality gap.
4. `Dependency Map` for downstream coupling (`network`, `simulate`, etc.).

This file is intentionally verbose so a new teammate can recover context without prior chat history.

---

## Why This Stage Exists
`spec` and `scenario` define generative rules. `sample` executes those rules.

Without sampling, Extropy has only declarations, not a concrete population.
Downstream systems need concrete agents:
- `network` needs actual nodes with attributes,
- `simulate` needs actual individuals with persona context,
- `results/query/chat` need concrete rows tied to a scenario run.

Design principle:
- **Compile semantics upstream, instantiate at scale in `sample`.**

---

## Pipeline Position
Intended high-level flow:
1. `spec`: compile reusable base population mechanics.
2. `scenario`: compile scenario extension + event dynamics.
3. `persona`: compile rendering contract.
4. `sample`: instantiate concrete agents from merged population contract.
5. `network`: build social graph on sampled agents.
6. `simulate`: run behavioral dynamics.

`sample` is the boundary where abstract distributions become actual synthetic people.

---

## Intended Contract

### What `sample` should do
`sample` should produce a concrete, scenario-specific population realization that is:
- statistically aligned with merged spec intent,
- internally consistent at agent and household levels,
- deterministic under seed,
- usable by downstream stages without semantic guessing.

At a minimum it should:
- merge base + scenario extension semantics correctly,
- realize each agent in dependency-safe order,
- apply modifiers/constraints consistently,
- realize household structure when activated,
- assign realistic names with deterministic policy,
- persist agents + household records in canonical study storage,
- emit machine-usable diagnostics (stats + violations).

### What `sample` should not do
`sample` should not:
- invent new schema semantics not defined upstream,
- silently reinterpret contradictory fields,
- depend on free-form LLM generation per agent,
- hide major consistency failures behind soft warnings,
- mutate scenario design intent at runtime.

### How `sample` should do it
Intended execution model:
1. Load scenario + referenced base population.
2. Build merged population contract with dependency-safe sampling order.
3. Validate merged contract before execution.
4. Sample agents under deterministic RNG (seeded).
5. Apply household realization policy if scenario activates it.
6. Run post-sampling consistency checks.
7. Persist canonical outputs and diagnostics.

### Why this contract matters
If sampling is weak, the entire pipeline degrades:
- network structure becomes fake or brittle,
- persona realism is disconnected from actual agents,
- simulation reasoning quality collapses,
- results become analytically untrustworthy.

`sample` is therefore a quality gate, not just a utility command.

---

## Conceptual Model Of Sampling

### Merged-spec execution
`sample` should execute a merged generative program:
- base attributes (reusable population identity),
- plus scenario extension attributes (event-context identity/behavior fields).

### Strategy semantics
Each attribute should be executed by strategy:
- `independent`: sample from distribution directly.
- `derived`: compute from formula deterministically.
- `conditional`: sample base distribution then apply triggered modifiers.

### Household semantics
Household behavior should activate only when scenario/base semantics request it.
When active, attribute scopes should control sampling logic:
- `individual`: sampled per person.
- `household`: shared within household.
- `partner_correlated`: correlated partner realization.

### NPC vs full-agent semantics
Scenario household scope policy should control who becomes full simulation agents:
- `primary_only`: one primary adult as full agent, partner/dependents as NPC context.
- `couples`: both adult partners are full agents, dependents mostly NPC.
- `all`: include adults + eligible dependents as full agents.

### Constraint semantics
`sample` should separate:
- hard numeric clamping,
- impossible contradictions (should block),
- implausible but possible combinations (warn/measure).

### Outlier semantics
`sample` should preserve realistic tails without breaking coherence:
- do not over-flatten to means,
- do not explode into nonsense combinations,
- expose measurable outlier behavior for backpressure.

---

## Intended Artifact Contract

### Primary runtime output
`sample` should emit:
- concrete agent records (`N` agents requested),
- run metadata (seed, count, timestamp, mode signals),
- sampling diagnostics/statistics,
- optional household table for household mode.

### Canonical persistence contract
Expected canonical DB artifacts:
- `sample_runs`: run metadata,
- `agents`: per-agent full attribute JSON tied to scenario,
- `households` (if active): household type/adult/dependent/shared-attribute records.

### Determinism contract
For fixed `(spec, scenario, seed, n)`:
- sampled outputs should be reproducible,
- IDs should be stable format,
- diagnostic summaries should be reproducible up to deterministic ordering rules.

---

## Intended Quality Contract

### Realism
Agents should look like plausible members of the target population and scenario context.

### Internal consistency
Cross-attribute logic should hold:
- dependency-safe realization,
- constraint-respecting values,
- coherent household/person relationships.

### Downstream support reliability
Output should be directly consumable by:
- `network` structural edge logic,
- `persona` rendering assumptions,
- `simulate` contact/NPC logic,
- `results/query/chat` retrieval semantics.

### Backpressure intent
Sampling should expose machine-checkable quality signals for go/no-go decisions, not only human-readable summaries.

---

## Current Implementation (As-Is)

This section describes actual behavior in code today.

### CLI command flow
`extropy sample` currently performs:
1. resolve study folder and scenario,
2. pre-flight require persona config exists,
3. load scenario YAML,
4. load base population referenced by `scenario.meta.base_population`/`population_spec`,
5. merge `extended_attributes` onto base attributes,
6. build merged sampling order with deterministic topological sort over merged dependencies,
7. validate merged spec via `validate_spec` (unless `--skip-validation`),
8. optional strict pre-sample gate (`--strict-gates`) that promotes high-risk warnings,
9. call `sample_population(...)` with scenario `household_config` + `agent_focus_mode` + `sampling_semantic_roles`,
10. apply post-sampling deterministic rule-pack gate (impossible/implausible),
11. save result to `study.db`.

Primary file:
- `extropy/cli/commands/sample.py`

### Merge behavior in command
Current merge is:
- `merged_attributes = base_attributes + extended_attributes`
- `merged_sampling_order = topological_sort(merged_dependencies)`

If merged dependencies are cyclic, command fails before sampling and writes versioned invalid JSON diagnostics.

### Validation gate behavior
Validation uses population validator (structural errors + semantic warnings):
- structural failures block sampling unless `--skip-validation`.
- warnings do not block.

Strict mode behavior (`--strict-gates`):
- pre-sample: promotes warning categories `CONDITION_VALUE` and `MODIFIER_OVERLAP` to hard block,
- post-sample: fails if `condition_warnings` are present.

Primary files:
- `extropy/population/validator/spec.py`
- `extropy/population/validator/structural.py`
- `extropy/population/validator/semantic.py`

### Sampling engine dispatch
Sampler chooses mode by schema inspection:
- household mode if any attribute has `scope == "household"`.
- independent mode otherwise.

Primary file:
- `extropy/population/sampler/core.py`

### Strategy execution behavior
Per-attribute execution:
- `derived`: `eval_formula(...)`
- `independent`: `sample_distribution(...)`
- `conditional`: `apply_modifiers_and_sample(...)` if modifiers exist, else direct distribution sample.

Distribution types implemented:
- `normal`, `lognormal`, `uniform`, `beta`, `categorical`, `boolean`

Formula-based distribution params supported for numeric distributions (`*_formula` fields).

Primary files:
- `extropy/population/sampler/core.py`
- `extropy/population/sampler/distributions.py`
- `extropy/population/sampler/modifiers.py`
- `extropy/utils/eval_safe.py`

### Modifier behavior (actual)
- Numeric distributions: all matching modifiers stack (`multiply` product + `add` sum), then reclamp to min/max bounds.
- Categorical: one deterministic winner is selected via precedence logic (subset/specificity/order), then that rule's `weight_overrides` apply.
- Boolean: one deterministic winner is selected; apply `probability_override` or winner multiply/add transform, then clamp `[0,1]`.

Modifier condition evaluation failures are logged and surfaced into `stats.condition_warnings` (capped to avoid unbounded growth).

### Type coercion behavior
Sampler coerces sampled values to declared `attr.type`:
- int coercion strips non-digits for string numerics like `"6+"`.
- float coercion strips non-numeric chars.
- boolean coercion maps strings like `"true"`, `"1"`, `"yes"`.
- categorical coerces to string.

### Constraint behavior
Current runtime constraint handling is split:
- hard numeric bounds (`hard_min`/`hard_max`, plus legacy `min`/`max`) are applied as clamping.
- expression constraints are checked post-sampling and counted in `stats.constraint_violations`.
- `spec_expression` constraints are not run against agents (treated as spec-level only).

Expression check exceptions are swallowed for the affected constraint evaluation.
CLI rule-pack treats non-zero expression-constraint violations as impossible and fail-hard.

### Household sampling behavior
When household mode is active:
- primary adult is sampled first,
- household type is sampled from `HouseholdConfig` age-bracket weights,
- partner realization depends on `agent_focus_mode`:
  - `primary_only`: partner is NPC profile in `partner_npc`, `partner_id=None`.
  - `couples`: partner is full agent with `partner_id` reciprocal link.
  - `all`: partner is full agent; eligible dependents may be promoted to full agents.
- dependents are generated from household type and age constraints.
- in `all`, dependents above `min_agent_age` can be promoted to full agents; others remain NPC dependent data.

Sampler trims overflow agents if final household overshoots `target_n`.

Primary files:
- `extropy/population/sampler/core.py`
- `extropy/population/sampler/households.py`

### Minor-dependent normalization behavior
Promoted dependent agents (minors) undergo normalization using attribute `semantic_type`:
- education/employment/occupation/income coerced to age-appropriate forms.

This is deterministic rule-based normalization in sampler core.

### Partner correlation behavior
For correlated partner fields, runtime uses policy resolution:
- explicit scenario semantic-role override (`sampling_semantic_roles.partner_correlation_roles`) if present,
- otherwise semantic/identity/default policy resolution (`gaussian_offset`, `same_group_rate`, `same_country_rate`, `same_value_probability`),
- then deterministic fallback rates from `HouseholdConfig`.

### Name generation behavior
Names are generated during sampling, not in `spec`/`scenario` writing.

Current runtime path:
- sampler calls `generate_name(...)` per primary/partner/dependent generation path,
- uses Faker-first locale routing from country/region hints,
- falls back to bundled CSV first/last name frequencies if Faker locale generation fails,
- gender/ethnicity/age-derived birth decade drive selection.

Primary files:
- `extropy/population/sampler/core.py`
- `extropy/population/names/generator.py`

### Stats and diagnostics behavior
Sampler collects:
- numeric means/std,
- categorical counts,
- boolean counts,
- modifier trigger counts,
- expression constraint violation counts,
- condition-evaluation warnings,
- household reconciliation counters.

CLI post-processing adds rule-pack summary (`pass/warn/fail`) into sampling stats for machine-readable backpressure.

Primary file:
- `extropy/core/models/sampling.py`

### Persistence behavior
After sampling:
- saves base population YAML text into `population_specs` under `population_id=scenario_name` (backward-compat path),
- saves sampled agents into `agents` with `scenario_id=scenario_name`,
- saves run metadata into `sample_runs`,
- saves households (if present) into `households`.

Existing records are replaced per scenario scope (`DELETE FROM agents WHERE scenario_id = ?`) before insert.

Primary files:
- `extropy/cli/commands/sample.py`
- `extropy/storage/study_db.py`

---

## Current Runtime Coupling With Downstream Stages

### `network` coupling
`network` requires sampled agents for scenario ID and consumes household fields (`household_id`, `partner_id`, `dependents`) for structural edges.

Primary files:
- `extropy/cli/commands/network.py`
- `extropy/population/network/generator.py`

### `persona` coupling
`sample` command enforces persona pre-flight existence, even though sampler itself does not read persona config.

Primary file:
- `extropy/cli/commands/sample.py`

### `simulate` coupling
Simulation requires sampled agents and loads households from DB.
It also consumes household-adjacent agent fields for contacts/NPC conversation context.

Primary files:
- `extropy/cli/commands/simulate.py`
- `extropy/simulation/engine.py`
- `extropy/simulation/conversation.py`

### `results/query/chat` coupling
All downstream query/results/chat behavior depends on quality and consistency of the concrete sampled agent rows persisted in `study.db`.

---

## What Current Implementation Already Gets Right

### Strong deterministic execution core
Given a seed and fixed inputs, sampler behavior is reproducible and non-LLM at runtime.

### Broad distribution and modifier support
Sampler supports rich distribution types, formula parameters, and conditional modifiers.

### Household-aware realization exists end-to-end
Household sampling, partner correlation, dependent generation, and NPC/full-agent modes are implemented in core runtime.

### Expression constraints are measured
Agent-level expression violations are counted and exposed in stats (not silently ignored entirely).

### Canonical persistence path is in place
`sample` writes to a unified study database with scenario-scoped retrieval patterns used by downstream commands.

---

## Divergences, Mistakes, and Risk Areas

### 1) Household activation tied only to `scope="household"`
Sampler switches to household mode only if at least one household-scoped attribute exists.

Risk:
- scenario may provide household config/focus mode but not trigger household mode if scope annotations are missing or inconsistent.

### 2) Persona is required as pre-flight for `sample`
Current command requires persona config before sampling even though sampler core does not depend on persona content.

Risk:
- operational coupling can block sampling workflows unrelated to sampler logic.

### 3) Name field shape mismatch risk in NPC partner usage
Sampler writes NPC partner fields with `first_name`/`last_name` style semantics, while parts of simulation conversation/contact logic look for `partner_npc.name` in several paths.

Risk:
- household NPC addressing and partner reference resolution can degrade.

### 4) Partner key mismatch risk in simulation contact paths
Sampler uses `partner_id`; parts of simulation contact/conversation logic also check `partner_agent_id`.

Risk:
- some partner-as-agent references may not resolve uniformly.

### 5) Expression constraint failures are counted post-hoc, not repaired
Expression constraints are checked after sampling and can fail-hard via rule-pack.

Risk:
- impossible violations fail via rule-pack gate, but sampler does not attempt automated repair/resample before failing.

### 6) Type coercion can mask upstream categorical/schema drift
Aggressive coercion (e.g., stripping symbols from numeric-like strings) keeps sampling alive.

Risk:
- upstream naming/schema mismatches may appear to "work" while distorting semantics.

### 7) Persistence identity uses backward-compat population keying
Sample saves population spec under `population_id=scenario_name` backward-compat behavior.

Risk:
- conceptual blending of base population identity and scenario identity in storage layer.

---

## How It Can Be Made Better (Direction, Not Patch Plan)

### Promote merged-order contract as first-class artifact
Merged dependency-safe order is now explicit and enforced at sample runtime.
Remaining improvement is surfacing richer diagnostics when dependency-quality is weak upstream.

### Strengthen mismatch diagnostics
Expose dedicated machine-readable warnings for token/key mismatches and condition failures.

### Tighten household/NPC schema contract
Keep partner/dependent key conventions consistent across sampler, persona renderer, and conversation resolution.

### Separate structural validity from behavioral acceptance
Use staged backpressure where severe inconsistency blocks downstream progression.

### Keep runtime deterministic but more observable
Retain non-LLM deterministic sampling while expanding run diagnostics quality.

---

## Dependency Map (What `sample` Must Support)

### `network`
Requires:
- stable `agent_id`s,
- coherent household/partner fields,
- realistic attribute distributions for similarity + structural edge logic.

### `simulate`
Requires:
- complete agent rows,
- household/NPC context,
- scenario-scoped agent retrieval consistency.

### `persona` runtime rendering
Requires:
- attribute-value consistency,
- household context fields for household section rendering.

### `results`, `query`, `chat`
Require:
- canonical DB persistence,
- deterministic scenario scoping,
- minimal schema surprises in agent JSON.

---

## Working Invariants Draft (Sample-Oriented)

### Structural invariants
1. Exactly `N` agents returned for requested `--count`.
2. Every sampled attribute referenced in order exists in merged schema.
3. Every dependency expression name is known or explicitly flagged.

### Household invariants
1. If household mode active, every agent has coherent `household_id`/role semantics.
2. Partner links are reciprocal for partner-agent households.
3. Household-scoped attributes are shared among household members.

### Consistency invariants
1. Hard bounds are respected for numeric attributes.
2. Expression violation rates are measured and thresholded for backpressure.
3. Minor-dependent promoted agents are age-coherent on semantic-type fields.

### Downstream invariants
1. Scenario-scoped agent retrieval returns sampled set used for network/simulation.
2. Household/NPC keys used by simulation are present and shape-consistent.
3. Sampling diagnostics are persisted/exportable for audit.

---

## Glossary (Working Terms)
- **Merged spec**: base attributes + scenario `extended_attributes` used for sampling.
- **Primary adult**: first sampled adult for a household unit.
- **NPC partner/dependent**: household member represented as context, not full agent row.
- **Agent focus mode**: scenario policy for who becomes full agents (`primary_only`, `couples`, `all`).
- **Partner-correlated attribute**: attribute sampled to correlate with partner value.
- **Backpressure**: quality signal that can stop or warn before downstream execution.

---

## Practical Readout
`sample` is not the heaviest compile stage, but it is not trivial.
It is where many independent contracts collide:
- merged dependency execution,
- household realization,
- name semantics,
- consistency enforcement,
- storage contract for the rest of the pipeline.

So it is simpler than `spec + scenario` authoring, but still a high-leverage failure point.

---

## Final Statement
`sample` should be treated as the execution-grade realization gate for Extropy.

Current implementation already has substantial capability and deterministic runtime behavior.
The main gaps are contract coherence at boundaries (especially merged ordering and cross-module household/NPC key consistency), plus stronger machine-readable backpressure signals for quality acceptance.
