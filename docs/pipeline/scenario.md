# Scenario Stage Reference: Scenario Extension and Event Dynamics

## Document Intent
This file is the deep reference for the `scenario` stage in Extropy.

Its job is to make one thing unambiguous for people and agents working on the pipeline:
- what `scenario` should do in principle,
- how that contract should be enforced,
- where the current implementation stands,
- where current behavior diverges from intended behavior,
- and what quality/backpressure signals this stage must expose.

This is not a patch list. It is a stage contract + current-state reality map.

## Status Update (2026-02-23)
- Scenario creation now enforces non-empty `extended_attributes` as a hard contract.
- Scenario validation in create-flow is now real validation (not light-pass placeholder).
- Attribute reference validation uses merged namespace (`base population ∪ scenario extension`).
- Literal-option compatibility checks are enforced for `when` conditions in:
  - `seed_exposure.rules`
  - `spread.share_modifiers`
  - `timeline[*].exposure_rules`
- Sufficiency post-processing now enforces timestep clarity:
  - timeline markers (for example `week 1`, `month 0`) force `evolving`,
  - static scenarios must provide an explicit timestep unit (or ask clarification).
- Household/focus coherence checks are enforced for household-active scenarios.
- Fail-hard scenario paths now emit versioned invalid artifacts (`*.invalid.vN.yaml` and JSON invalid diagnostics where applicable).
- Agent-mode insufficiency now returns structured clarification payloads (exit code 2) instead of falling through generic validation errors.

---

## How To Read This Document
Use this order:
1. `Intended Contract` for design truth.
2. `Current Implementation (As-Is)` for code truth.
3. `Divergences, Mistakes, and Risk Areas` for the gap.
4. `Dependency Map` for how downstream stages consume scenario outputs.

This file is intentionally verbose so a new team member can recover context without prior chat history.

---

## Why This Stage Exists
`spec` gives a reusable population blueprint.
`scenario` should turn that blueprint into a simulation-ready context.

Without scenario compilation, downstream stages would have to infer too much ad hoc:
- which additional attributes matter for this event,
- how information is introduced and spreads,
- whether scenario timing is static or evolving,
- what outcomes should be extracted from reasoning,
- how household/agent scope should be interpreted in this specific run.

Design principle:
- **Compile scenario context once, execute many downstream steps against that contract.**

---

## Pipeline Position
High-level pipeline intent:
1. Build base population encoding (`spec`).
2. Extend that population for a specific event context (`scenario`).
3. Build scenario-grounded rendering and graph context (`persona`, `network`).
4. Instantiate agents, propagate information, reason, and aggregate (`sample`, `simulate`, `results`, `query`, `chat`).

`scenario` is the bridge stage between reusable base population mechanics and concrete event dynamics.

---

## Intended Contract

### What `scenario` should do
`scenario` should output a versioned scenario artifact that contains both:
1. **Scenario-specific population extension** (non-empty).
2. **Scenario dynamics contract** for exposure, spread, timeline, and outcomes.

At minimum, it should:
- discover and encode scenario-specific extension attributes,
- preserve dependency-safe sampling semantics for those attributes,
- define the event and its source properties,
- define how agents hear about the event (channels/rules/timing),
- define interaction and spread behavior assumptions,
- define static vs evolving time semantics,
- define bounded outcome space for extraction,
- define household/agent activation semantics for this scenario,
- provide machine-checkable quality signals (impossible vs implausible).

### What `scenario` should not do
`scenario` should not:
- sample concrete agents,
- generate concrete network edges,
- execute simulation loops,
- rely on hidden downstream defaults for critical semantics,
- silently degrade required fields for convenience.

### How `scenario` should do it
Intended compile pattern:
1. Sufficiency check for event/timeline/outcome clarity.
2. Scenario extension discovery over base population context.
3. Extension hydration and dependency binding.
4. Event/exposure/interaction/timeline/outcome compilation.
5. Contract validation with severity separation.
6. Emit single coherent versioned scenario artifact.

### Why this contract matters
If this stage is weak:
- sampling quality drops (because extension semantics are noisy),
- persona salience loses scenario grounding,
- network generation uses generic heuristics without event intent,
- simulation reasoning becomes brittle and underconstrained,
- result interpretation loses trust.

`scenario` is therefore not just "event text generation." It is a downstream control contract.

---

## Conceptual Model Of Scenario Compilation

### Scenario as downstream control contract
Conceptually, `scenario` is a packet sent to four consumers:
- `sample`,
- `network`,
- `persona`,
- `simulate`.

The packet should tell downstream systems:
- what changed in the world,
- who gets exposed, through what path, at what time,
- what social propagation assumptions apply,
- which dimensions matter for behavioral output,
- what outcomes to classify and track.

### Two sub-domains inside scenario
Scenario naturally contains two sub-systems:
1. **Population extension subsystem** (mini-spec behavior).
2. **Event dynamics subsystem** (timeline/exposure/outcome behavior).

This dual nature is expected. The contract must keep their boundary explicit.

### Strict envelope, loose internals
Design choice for schema stability:
- **Strict envelope**: stable top-level keys must always exist.
- **Loose internals**: optional advanced sub-blocks can be empty when not relevant.

This keeps deterministic parsing without hardcoding scenario-specific internals.

### Required vs optional semantics
From current team direction:
- `extended_attributes` should be required and non-empty.
- Core dynamic blocks (`event`, `seed_exposure`, `spread`, `timeline`, `outcomes`) should be present.
- Advanced optional blocks may be empty when not used.

---

## Intended Base vs Scenario Boundary

### Base population (`spec`) should own
- reusable population mechanics,
- long-horizon demographic and identity distributions,
- generic dependency/distribution logic not tied to one event rollout.

### Scenario should own
- event-context extension attributes,
- event/timeline/exposure/outcome semantics,
- household activation policy for this scenario,
- agent scope mode for this scenario (`primary_only`/`couples`/`all`).

### Downstream stages should own
- `sample`: concrete instantiation with seed + runtime enforcement.
- `network`: concrete graph realization.
- `persona`: rendering config and textual embodiment.
- `simulate`: state evolution and reasoning loop.

---

## Intended Artifact Contract

### Primary artifact
`scenario` should emit versioned YAML under:
- `scenario/<name>/scenario.vN.yaml`

It should serve as the single source of truth for scenario context.

### Envelope contract (intended)
Top-level sections should be stable:
- `meta`
- `event`
- `timeline`
- `seed_exposure`
- `interaction`
- `spread`
- `outcomes`
- `simulation`
- `extended_attributes`
- `household_config`
- `agent_focus_mode`

Optional top-level sections may exist (for example identity framing), but core sections should not be omitted.

### Extension attribute contract (intended)
Each extension attribute should include:
- canonical name/type/category,
- strategy + dependency semantics,
- scope semantics (`individual`/`household`/`partner_correlated`),
- correlation metadata when relevant,
- constraints and grounding info.

### Outcome contract (current team direction)
- bounded outcome set (cap = 8),
- no forced canonical single outcome requirement,
- not every outcome must be directly measurable by a numeric KPI,
- no mandatory misinformation/distortion path requirement.

### Rule-pack contract (intended)
Scenario checks should distinguish:
- **impossible** contradictions -> hard block,
- **implausible** patterns -> warning/penalty signal,
- informational diagnostics -> non-blocking context.

This rule-pack shape should be machine-consumable for backpressure.

---

## Intended Quality Contract

### Realism
Scenario should represent plausible information and response dynamics:
- realistic channels,
- realistic timing,
- realistic subgroup exposure differences.

### Internal consistency
Scenario internals should agree with each other:
- timeline semantics must match simulation config,
- exposure rules should reference known attributes/channels,
- outcomes should align with event scope.

### Downstream support reliability
Downstream stages should not need to infer missing semantics for core behavior.
If they do, the scenario contract is underspecified.

### Determinism where needed
The stage should preserve flexibility without introducing arbitrary ambiguity for required controls.

---

## Intended Backpressure Philosophy (Contract-Level)
Scenario compilation should output enough quality evidence to support go/no-go gating.

Backpressure should be:
- explicit,
- machine-readable,
- tied to severity classes,
- traceable to concrete locations in scenario content.

This is essential for catching functional and behavioral bugs before simulation.

---

## Current Implementation (As-Is)

This section describes actual code behavior now.

### Entrypoint behavior in CLI
`extropy scenario` currently performs:
1. scenario sufficiency check,
2. extension attribute selection (`select_attributes` with base context),
3. extension hydration (`hydrate_attributes`),
4. extension binding (`bind_constraints`),
5. scenario config generation (`create_scenario_spec`),
6. metadata finalization (`meta.name`, `meta.base_population`) + re-validation,
7. save valid artifact or emit versioned invalid artifact and exit non-zero.

Primary orchestration file:
- `extropy/cli/commands/scenario.py`

### Sufficiency behavior in code
Sufficiency model checks:
- event clarity,
- duration/timestep inference,
- static vs evolving inference,
- explicit outcomes presence,
- inferred `agent_focus_mode`.

Current behavior characteristics:
- lenient sufficiency style,
- user clarification loop supported,
- deterministic post-processing layers over LLM output:
  - extracts unit hints from description and inferred duration,
  - promotes explicit timeline markers to evolving mode,
  - blocks static scenarios without explicit timestep unit by issuing a clarification question,
- inferred fields are mostly advisory for messaging except `agent_focus_mode` and unit hints that feed compile defaults.

In agent mode:
- insufficiency returns structured clarification questions with exit code 2 (`needs_clarification`),
- `--use-defaults` attempts a no-prompt re-check by applying question defaults.

Primary file:
- `extropy/scenario/sufficiency.py`

### Mini-spec extension behavior in code
Scenario uses spec-builder primitives directly:
- attribute selector in overlay mode with base attributes as context,
- split hydrator pipeline (independent/derived/conditional/base+modifiers),
- household config hydration (conditional),
- dependency binding with expression-based dependency inference.

Primary files:
- `extropy/population/spec_builder/selector.py`
- `extropy/population/spec_builder/hydrator.py`
- `extropy/population/spec_builder/binder.py`

Important implementation detail:
- extension compilation is strong and feature-rich (close to spec flow), including scope and correlation semantics.

### Scenario dynamics generation behavior in code
`create_scenario_spec` composes:
1. event parse (`parse_scenario`),
2. seed exposure generation (`generate_seed_exposure`),
3. interaction/spread generation (`determine_interaction_model`),
4. timeline+outcomes generation (`generate_timeline_and_outcomes`),
5. identity dimension detection.

Primary files:
- `extropy/scenario/compiler.py`
- `extropy/scenario/parser.py`
- `extropy/scenario/exposure.py`
- `extropy/scenario/interaction.py`
- `extropy/scenario/timeline.py`

### Important current coupling detail
In `create_scenario_spec`, parser/exposure/interaction generation uses a merged population context:
- base population attributes, plus
- scenario `extended_attributes`.

Scenario creation-time generation passes `network_summary=None` (no synthetic placeholder edge types injected at this stage).

### Validation behavior in current new flow
`create_scenario_spec` runs deterministic scenario validation in-flow via:
- `extropy/scenario/validator.py`

`extropy scenario` checks validation errors before save. On fail-hard paths, it writes a versioned `.invalid` artifact and exits non-zero.

Current validator focus areas include:
- merged-namespace attribute reference checks (base + extension),
- expression syntax checks for `when` rules,
- literal compatibility checks against categorical options and boolean literals,
- channel/timestep/outcome consistency checks,
- household + `agent_focus_mode` coherence when household semantics are active.

### Assembly behavior for extension/household fields
`extended_attributes`, `household_config`, and `agent_focus_mode` are passed into `create_scenario_spec` and emitted as part of the assembled `ScenarioSpec` contract.

### Timeline/outcome behavior in code
Timeline generator can produce static or evolving behavior and supports:
- timestep unit,
- max timesteps,
- timeline events with optional re-reasoning intensity.

Outcome generation currently:
- encourages categorical/boolean/open_ended,
- discourages float in prompt,
- supports option friction for categorical outcomes,
- sets `capture_full_reasoning=true`.

Current conversion behavior sets:
- `extraction_instructions=None` by default in generator return path.

### Exposure channel behavior in code
Scenario model supports channel-level `experience_template`.
Runtime reasoning prompt consumes it when present.

Exposure generation schema explicitly requests optional `experience_template`.

### Identity framing behavior in code
Scenario compiler can detect `identity_dimensions`.
Simulation engine consumes those dimensions by matching to attribute `identity_type` and injecting identity relevance text into reasoning context.

This is one of the stronger semantic links between scenario and simulation.

---

## Current Runtime Coupling With Downstream Stages

### Sample coupling
`sample` does:
- load base population from scenario metadata,
- merge `extended_attributes` into attributes list,
- recompute merged sampling order with topological sort over merged dependencies,
- pass scenario `household_config` and `agent_focus_mode` into sampler.

Primary file:
- `extropy/cli/commands/sample.py`

### Household/agent scope behavior in sampler
Sampler supports:
- `primary_only`,
- `couples`,
- `all`.

It uses household config and attribute scope semantics to decide:
- which members are full agents,
- which are NPC partner/dependent context.

Primary files:
- `extropy/population/sampler/core.py`
- `extropy/population/sampler/households.py`

### Persona coupling
`persona` command merges base+extended attributes for config generation.
At runtime, simulation persona rendering uses scenario `decision_relevant_attributes` to prioritize trait salience when available.

Primary files:
- `extropy/cli/commands/persona.py`
- `extropy/simulation/engine.py`
- `extropy/population/persona/renderer.py`

### Network coupling
`network` command merges base+extended attributes when generating network config.
Network config generation itself is population-driven and does not directly consume scenario spread/exposure fields in config synthesis.

Primary files:
- `extropy/cli/commands/network.py`
- `extropy/population/network/config_generator.py`

### Simulation coupling
Simulation consumes scenario directly for:
- event content,
- seed exposure channels/rules,
- spread modifiers,
- timeline events,
- outcomes schema,
- extraction instructions (if present),
- identity dimensions.

Primary files:
- `extropy/simulation/engine.py`
- `extropy/simulation/propagation.py`
- `extropy/simulation/reasoning.py`

---

## What Current Implementation Already Gets Right

### Strong extension modeling capability
Scenario extension currently supports most of the rich attribute semantics that matter:
- strategy types,
- dependency inference,
- scope semantics,
- modifier-based conditional sampling.

### Clear downstream field model
Scenario model surface is expressive and already includes key hooks:
- extension attrs,
- household config,
- agent focus mode,
- identity dimensions,
- decision-relevant attributes,
- experience templates.

### Runtime integration quality in simulation
Simulation integration is substantial and not superficial:
- exposure history shaping,
- timeline-driven propagation,
- outcome schema-driven extraction,
- identity relevance framing.

### CLI lifecycle coherence
Study-folder versioning flow and command sequencing are operationally coherent.

---

## Divergences, Mistakes, and Risk Areas

### 1) Scenario stage currently carries two heavy compilers
It runs both mini-spec extension compilation and event dynamics compilation in one stage.
This is conceptually valid but operationally heavy, raising coupling risk and debugging complexity.

### 2) Contract remains broad/heavy in one stage
Scenario still combines two heavy concerns in one stage:
- scenario extension attribute compilation, and
- event dynamics compilation.
This is valid, but operationally complex.

### 3) Base-vs-extended merge is now explicit in dynamics generation
Parser/exposure/interaction generation uses merged base+extended context in the new flow.  
Remaining risk is quality of extension attributes themselves, not omission from dynamics prompt context.

### 4) Validation strictness has improved, but rule depth can still expand
New-flow scenario creation runs deterministic validation before save.  
Future quality gains are mostly about adding richer rule coverage, not wiring.

### 5) Envelope stability vs practical generation mismatch
Model supports fields like `experience_template` and `extraction_instructions`, but default generation often leaves them null.
This is schema richness without guaranteed population of semantically useful hooks.

### 6) Merged-order correctness now depends on dependency quality
Downstream now recomputes merged order with topological sort.
Primary remaining risk is dependency quality from generated attributes (missing/wrong `depends_on`), not append-only ordering.

### 7) Network intent contract is implicit
Scenario has spread/exposure semantics, but network config generation is primarily population-driven.
The explicit scenario-to-network intent bridge is weak/implicit rather than formalized.

### 8) Outcome policy is prompt-heavy
Outcome quality rules rely strongly on prompt instructions and conversion defaults.
Contract-level invariant enforcement for outcome count/shape/salience is not strongly centralized.

### 9) Complexity debt in ownership clarity
Household semantics are now effectively scenario-owned in practice, but documentation/contracts can still be interpreted ambiguously if not explicit.

---

## How It Can Be Made Better (Direction, Not Patch Plan)

### Make scenario a single atomic contract emission
Avoid conceptual split between "core scenario object" and "later attached core fields."

### Strengthen contract-level validation in authoring path
Bring strict structural + semantic checks earlier in the default `scenario` creation lifecycle.

### Make base+extension context explicit before dynamics generation
Treat extension as first-class compile context for event/exposure/interaction synthesis.

### Codify strict envelope + loose internals
Stabilize top-level keys while preserving contextual flexibility inside optional blocks.

### Promote rule-pack severity model
Separate impossible/implausible checks explicitly and persist those diagnostics for pipeline backpressure.

### Tighten scenario-to-network intent bridge
Clarify and formalize which scenario signals must be consumed by network synthesis.

---

## Downstream Dependency Map (What `scenario` Must Support)

### `sample`
Needs:
- non-empty `extended_attributes`,
- `household_config`,
- `agent_focus_mode`.

### `network`
Needs:
- merged population context (base + extension),
- optional scenario intent signals for spread topology assumptions.

### `persona`
Needs:
- merged attribute context,
- scenario salience hints (`decision_relevant_attributes`) for rendering emphasis.

### `simulate`
Needs:
- event + timeline content,
- exposure channels/rules,
- spread modifiers,
- outcomes schema,
- optional extraction instructions,
- identity dimensions.

### `results`, `query`, `chat`
Depend on scenario-defined outcome schema and timeline semantics for interpretation and querying consistency.

---

## Working Invariants Draft (Scenario-Oriented)
These are contract-level invariants to support future backpressure harnesses.

### Core invariants
1. `extended_attributes` must exist and be non-empty.
2. `event`, `seed_exposure`, `spread`, `outcomes`, and `simulation` must exist.
3. `agent_focus_mode` must be one of `primary_only`, `couples`, `all`.
4. Every exposure rule channel must reference a defined channel.
5. Every expression field must parse and reference known symbols only.

### Outcome invariants
1. Outcome count must be within configured cap (current team direction: <= 8).
2. Categorical outcomes must have valid option sets.
3. Outcome names must be canonicalized and machine-safe.

### Consistency invariants
1. Timeline timesteps must be unique and coherent with simulation horizon.
2. Household-related fields must not conflict with agent focus mode semantics.
3. Scenario metadata must resolve to a valid base population reference in study context.

---

## Glossary (Working Terms)
- **Scenario extension**: Scenario-specific additional attributes over base population.
- **Strict envelope**: Stable required top-level schema keys.
- **Loose internals**: Optional internal subfields that can be empty when irrelevant.
- **Impossible**: Contradictory/invalid condition that should block compile.
- **Implausible**: Unusual but possible condition that should warn, not necessarily block.
- **Agent focus mode**: Household agent scope policy (`primary_only`/`couples`/`all`).
- **Experience template**: First-person phrasing template for how channel exposure is represented in reasoning prompts.

---

## Practical Readout
At present, scenario is best interpreted as:
1. a powerful but heavy dual-compiler stage,
2. with strong extension expressivity,
3. strong runtime simulation integration,
4. and contract-enforcement gaps in authoring path consistency.

For team operation, the most important framing is:
- the stage intent is correct,
- the ownership boundary is mostly right,
- but enforcement and assembly coherence need to be treated as first-class quality concerns.

---

## Final Statement
`scenario` should be the stage that makes simulation context explicit, testable, and reusable.

Current implementation already contains most raw capabilities needed to do this well.
The main remaining challenge is not missing concepts, but contract coherence:
- one atomic scenario contract,
- explicit ownership boundaries,
- and clear backpressure signals that downstream stages can trust.
