# Persona Stage Reference: Persona Compilation and Agent Embodiment

## Document Intent
This file is the deep reference for the `persona` stage in Extropy.

It is designed to be reusable team memory for:
- what `persona` is supposed to accomplish,
- how the stage should operate in theory,
- what the code currently does,
- where current behavior diverges from intended quality goals,
- and what contract-level checks should exist for backpressure.

This is not a patch plan. It is a contract + current-state map.

## Status Update (2026-02-23)
- `extropy validate` now routes persona files (`persona.vN.yaml`) through persona-specific validation.
- Persona validation now checks:
  - merged-attribute treatment coverage
  - group membership completeness/uniqueness
  - boolean/categorical/relative/concrete phrasing completeness
  - intro-template attribute reference validity
- Renderer behavior now includes:
  - mixed relative+concrete fallback for float-like relative attributes
  - strict skip of duplicate “Who I Am” group rendering
  - deterministic boolean phrase sanitation for token-like outputs
- `validate` supports versioned invalid persona artifacts (`persona.vN.invalid.vK.yaml`) and promotes them to canonical path when validation passes.
- Fail-hard persona generation/validation now emits versioned invalid artifacts.

---

## How To Read This Document
Use this sequence:
1. `Intended Contract` for design truth.
2. `Current Implementation (As-Is)` for code truth.
3. `Divergences, Mistakes, and Risk Areas` for gap analysis.
4. `Dependency Map` for downstream implications.

The document is intentionally verbose so someone new can recover context quickly.

---

## Why This Stage Exists
The simulation does not reason directly over raw structured agent rows.
It reasons over natural-language personas.

If persona construction is weak, the simulation degrades even when sampling/network are strong:
- important traits are omitted or buried,
- phrasing is unnatural or token-like,
- household context is lost,
- trait salience for decision-making is unclear,
- LLM reasoning becomes noisy and inconsistent.

`persona` exists to compile one reusable rendering contract:
- generated once for a scenario’s merged attribute set,
- then applied deterministically to every agent at runtime (no per-agent persona LLM call).

Design principle:
- **Compile persona language once, render many agents deterministically.**

---

## Pipeline Position
Pipeline context:
1. `spec` compiles base population mechanics.
2. `scenario` extends population + event dynamics.
3. `persona` compiles rendering semantics for merged attributes.
4. `sample` instantiates agents.
5. `network` builds graph.
6. `simulate` pre-generates and uses personas for reasoning.

In current operational flow, persona is treated as a prerequisite for both `sample` and `simulate`.

---

## Intended Contract

### What `persona` should do
`persona` should emit a versioned persona configuration that defines:
1. intro/narrative framing,
2. attribute treatment mode (concrete vs relative),
3. attribute grouping and ordering,
4. phrasing templates for boolean/categorical/relative/concrete attributes,
5. optional population statistics for relative positioning.

At runtime, this config should produce:
- consistent first-person personas,
- stable, deterministic rendering,
- high coverage of relevant attributes,
- decision-focused salience where scenario requests it.

### What `persona` should not do
`persona` should not:
- run per-agent LLM calls during simulation,
- mutate agent values,
- decide network structure,
- invent scenario dynamics,
- depend on hidden downstream defaults for core rendering semantics.

### How `persona` should do it
Intended compile pattern:
1. ingest merged population attributes (base + scenario extension),
2. classify rendering treatment and groups,
3. generate phrasing maps by type,
4. validate coverage and phrase quality constraints,
5. optionally compute population stats from sampled agents,
6. emit versioned `persona.vN.yaml`.

### Why this contract matters
Persona is the language interface between structured agent states and LLM reasoning.
If this interface is low quality:
- behavioral realism drops,
- contradictions increase,
- outcome extraction quality declines,
- simulation variance becomes harder to trust.

---

## Conceptual Model Of Persona Compilation

### Persona config as rendering compiler
Conceptually, persona stage compiles:
- **input**: attribute schema + optional empirical sampled stats,
- **output**: deterministic rendering program (`PersonaConfig`),
- **runtime**: pure string rendering over each agent dictionary.

### Two rendering layers
Persona rendering has two layers:
1. narrative framing (`intro_template` + household section),
2. structured trait expression (grouped phrases, with optional decision-salience section).

### Treatment semantics
Each attribute is assigned a treatment:
- `concrete`: preserve actual value semantics,
- `relative`: express position relative to population stats (z-score buckets).

### Salience semantics
Scenario can provide `decision_relevant_attributes`.
Runtime renderer should surface these before other groups to improve reasoning focus.

### Determinism semantics
After config generation, per-agent persona rendering should be:
- template-driven,
- non-LLM,
- reproducible from agent state + config.

---

## Intended Ownership Boundary

### `persona` should own
- linguistic embodiment of attributes,
- grouping and ordering for readability + reasoning salience,
- phrase-level normalization for categorical/boolean/null states,
- optional relative-position computation using population stats.

### `scenario` should own
- which attributes matter for decision salience (`decision_relevant_attributes`),
- household activation policy and agent scope semantics upstream.

### `sample` should own
- concrete agent instantiation and household/NPC realization.

### `simulate` should own
- usage of persona text in reasoning pipeline,
- no semantic reinterpretation of persona config contract.

---

## Intended Artifact Contract

### Primary artifact
Persona stage should output:
- `scenario/<name>/persona.vN.yaml`

### Artifact sections (intended stable envelope)
- `population_description`
- `created_at`
- `intro_template`
- `treatments`
- `groups`
- `phrasings`:
  - `boolean`
  - `categorical`
  - `relative`
  - `concrete`
- `population_stats`

### Coverage contract (intended)
At contract level:
1. every attribute should have exactly one treatment,
2. group membership should be complete and non-contradictory,
3. phrasing maps should cover all rendered categorical options,
4. relative phrasings should map each relative attribute to all z-buckets,
5. concrete templates should be syntactically valid for substitution.

### Quality contract for phrasing
Phrasings should:
- be first-person,
- avoid raw enum-token leakage where possible,
- handle explicit null/not-applicable categorical states naturally,
- avoid awkward boolean prose in intro templates.

---

## Intended Quality Contract

### Realism
Persona should sound like a plausible person, not raw table serialization.

### Internal consistency
Rendered sections should not duplicate or contradict one another:
- intro attributes should not be repeated blindly in groups,
- decision-salient attributes should be emphasized once and cleanly.

### Downstream support
Persona output should support simulation reasoning quality directly:
- preserve critical trait signal,
- keep text compact enough for context efficiency,
- keep language clear enough for stable extraction.

### Robustness to token variation
Categorical rendering should be robust to minor token formatting differences (`_`, space, hyphen forms).

---

## Intended Backpressure Philosophy (Contract-Level)
Persona stage should provide machine-checkable evidence that rendering quality is acceptable.

Backpressure classes should distinguish:
- blocking structural failures (coverage missing, parse errors),
- plausibility/readability warnings,
- informational diagnostics.

Without this, low-quality persona configs can silently pass and degrade simulation behavior.

---

## Current Implementation (As-Is)

This section describes what code currently does.

### Persona CLI behavior
`extropy persona` currently:
1. resolves scenario,
2. loads scenario spec + base population,
3. merges scenario `extended_attributes` into base attributes,
4. builds a merged `PopulationSpec`,
5. runs persona config generation,
6. saves versioned persona YAML.

Primary file:
- `extropy/cli/commands/persona.py`

### Merge behavior in persona command
Merge behavior is attribute-list merge:
- base attributes + scenario extended attributes.

Sampling order is inherited from base spec and not recomputed for persona (persona generator does not rely on sampling order for core behavior).

### Generation pipeline behavior
Generator orchestrates 5 LLM-assisted steps:
1. structure (`intro_template`, treatments, groups),
2. boolean phrasings,
3. categorical phrasings,
4. relative phrasings,
5. concrete templates.

Primary file:
- `extropy/population/persona/generator.py`

### Strongest built-in guardrail today
Categorical phrasing generation has explicit validation+retry behavior:
- checks per-attribute option coverage,
- normalizes minor token variants,
- retries up to 3 attempts with targeted feedback.

This is the most explicit quality-enforcement logic in persona generation currently.

### Treatment and grouping behavior in code
Treatment/group assignment is prompt-driven and parsed into:
- `AttributeTreatment`,
- `AttributeGroup`.

Current code runs dedicated cross-field validation (`validate_persona_config`) for:
- exact treatment coverage over merged attributes,
- group membership completeness and uniqueness,
- unknown attribute references in groups,
- intro-template placeholder reference validity.

### Stats behavior in code
Persona command tries to load sampled agents from study DB:
- if agents exist for scenario, compute population stats at generation time,
- else stats remain empty in config initially.

Simulation runtime then backfills stats if persona config exists but has empty stats.

Primary files:
- `extropy/cli/commands/persona.py`
- `extropy/population/persona/stats.py`
- `extropy/simulation/engine.py`

### Renderer behavior in code
Renderer is deterministic and non-LLM.
Key runtime behavior:
- renders intro from `intro_template`,
- renders household section from `partner_npc`/`dependents`/`partner_id`,
- extracts intro attributes and avoids duplicate rendering in groups,
- optionally renders "Most Relevant to This Decision" section using scenario salience list,
- renders remaining groups in configured order.

Primary file:
- `extropy/population/persona/renderer.py`

### Categorical null-state behavior
Renderer supports nuanced categorical null handling:
- `null_options`,
- `null_phrase`,
- fallback phrase,
- token normalization.

This reduces "raw enum token" artifacts in persona text for not-applicable states.

### Relative rendering behavior
Relative phrasing uses z-score buckets from `population_stats`.
If stats are unavailable for an attribute, renderer defaults to the "average" phrase label.

### Simulation integration behavior
Simulation engine pre-generates persona text for all agents at initialization.
It passes scenario `decision_relevant_attributes` into persona generation path for salience.

Primary files:
- `extropy/simulation/engine.py`
- `extropy/simulation/persona.py`

### Legacy fallback behavior still present
Simulation persona module still contains legacy hybrid path:
- narrative + structured characteristics list fallback if no `PersonaConfig`.

In current CLI flow, persona config is usually required, but legacy path remains in runtime code for compatibility.

### CLI prerequisites and ordering behavior
Current CLI enforces persona existence as pre-flight check in:
- `sample`
- `simulate`

So practical command order is strongly constrained around persona generation.

---

## Current Runtime Coupling With Other Stages

### Coupling with `scenario`
Persona command uses scenario to:
- resolve merged attribute surface (via `extended_attributes`),
- indirectly receive salience contract (`decision_relevant_attributes`) consumed at simulation runtime.

### Coupling with `sample`
Even though persona can generate without sampled agents, `sample` currently requires persona config to exist first.
This creates an operational dependency where persona is treated as upstream of sampling.

### Coupling with `simulate`
Simulation requires persona config and uses it to:
- pre-generate per-agent personas,
- improve salience ordering with scenario outcome hints,
- inject text into reasoning prompts.

### Coupling with population models
Renderer formatting behavior can depend on attribute `display_format` metadata from population spec.
Runtime path currently uses loaded population spec for display-format mapping.

---

## What Current Implementation Already Gets Right

### One-time compile, deterministic render model
Core architecture is sound:
- expensive LLM work done once,
- cheap deterministic rendering at runtime.

### Strong typed config surface
Persona config models are structured and explicit, reducing ad hoc rendering logic.

### Practical robustness in categorical phrasing
Token normalization + option coverage retries are meaningful quality controls.

### Household-aware rendering
Renderer has dedicated household section logic and handles partner/NPC/dependent variants.

### Scenario-driven salience support
Runtime path can foreground decision-relevant attributes from scenario outcomes.

---

## Divergences, Mistakes, and Risk Areas

### 1) `--preview` option is currently not functionally wired
CLI exposes `--preview/--no-preview`, but current command flow does not implement a distinct preview gate behavior tied to that flag in generation path.

### 2) Phrase realism remains mostly prompt-driven
Structural validation is strong, but naturalness/readability quality (beyond token-like checks) still depends on prompt outputs.
There is no strict machine score for "human-like narrative quality."

### 3) Uneven generation-time enforcement across phrasing types
Categorical has explicit retry/coverage validation.
Other phrasing classes rely more on one-shot generation and schema shape.

### 4) Validation exists, but generation can still fail late
Coverage issues are caught before save, but they are still discovered post-generation.
This means failed generations can consume LLM calls before deterministic checks reject output.

### 5) Operational ordering rigidity
`sample` requires persona config as a prerequisite even though persona stats can be backfilled later.
This may be intentional for pipeline discipline, but it couples stages more tightly than pure data dependency requires.

### 6) Runtime fallback path complexity
Legacy and new persona paths coexist in simulation module.
Dual paths increase behavior surface area and potential drift.

### 7) Auto persona config fallback path in simulation is legacy-shaped
When explicit persona path is absent, simulation tries `pop_path.with_suffix(".persona.yaml")`.
Current study structure centers scenario-scoped `persona.vN.yaml`, so this fallback is not the primary source-of-truth pattern.

### 8) Display-format mapping scope mismatch risk
Runtime display-format map is derived from loaded population spec path.
If that surface does not fully reflect merged extension metadata, intro formatting for extension fields can degrade.

### 9) Prompt-dependent naturalness
Natural language quality still depends heavily on LLM outputs.
Backpressure for "sounds natural, not templated token noise" is not formalized as explicit quality metrics.

---

## How It Can Be Made Better (Direction, Not Patch Plan)

### Keep contract-level persona validation as hard gate
Current explicit checks (coverage, group uniqueness, phrasing completeness, intro references)
should remain non-optional in create and validate flows.

### Formalize quality bands for phrase realism
Add machine-checkable heuristics for low-quality phrasing patterns (raw token leakage, template artifacts).

### Unify runtime paths
Reduce duality between legacy fallback and new config-driven rendering where possible.

### Clarify stage dependency rationale
Document why sample requires persona pre-flight and what guarantees that provides.

### Strengthen deterministic behavior guarantees
Define and enforce deterministic rendering invariants independent of LLM generation variance.

---

## Downstream Dependency Map (What `persona` Must Support)

### `sample`
Current operational dependency:
- persona config must exist before sampling command proceeds.

### `simulate`
Hard runtime dependency:
- persona config exists,
- persona texts pre-generated for all agents,
- salience-aware rendering available.

### `results` / `query` / `chat`
Indirect dependency:
- quality of persona text influences reasoning trajectories and therefore downstream outputs.

### `scenario`
Provides:
- merged extension context via extended attributes,
- decision salience hints consumed by persona rendering at runtime.

---

## Working Invariants Draft (Persona-Oriented)

### Structural invariants
1. Persona config must parse and include all required top-level sections.
2. Every merged attribute must have exactly one treatment.
3. Every treated attribute must belong to exactly one group.
4. Group references must only contain known attributes.

### Phrasing invariants
1. Boolean phrasing must provide both true/false phrases.
2. Categorical phrasing must cover full option set.
3. Relative phrasing must include all five z-bucket labels.
4. Concrete template must include renderable `{value}` semantics.

### Runtime invariants
1. Intro-rendered attributes should not be duplicated in group rendering.
2. Decision-relevant attributes should be rendered in dedicated salience section when provided.
3. Rendering should not crash on missing or null values.

### Quality invariants
1. Null-state categorical options should render as natural language, not raw enum tokens.
2. Persona text should remain first-person and internally coherent.

---

## Glossary (Working Terms)
- **PersonaConfig**: Compiled rendering contract for first-person personas.
- **Treatment**: Concrete vs relative render mode assignment per attribute.
- **Relative phrasing**: Z-score bucket language compared to population mean/std.
- **Null option**: Categorical value indicating explicit absence/not-applicable state.
- **Decision salience**: Scenario-provided attribute list surfaced first for reasoning focus.

---

## Practical Readout
Current persona stage is directionally strong:
- architecture is right,
- renderer is deterministic,
- simulation integration is substantial.

Main debt is language-quality enforcement completeness:
- structural invariants are now strongly enforced,
- realism/naturalness scoring remains mostly heuristic and prompt-dependent.

Given Extropy goals (realistic population + scenario + persona + network + simulation),
persona is a high-leverage quality gate and should be treated as such in backpressure design.

---

## Final Statement
`persona` is the stage that converts structured agents into reasoning-ready human context.

Today it already provides a capable compile-once/render-many system with meaningful runtime integration.
To fully support functional and behavioral bug detection across the pipeline, the next maturity step is explicit contract enforcement:
- complete coverage invariants,
- phrase quality diagnostics,
- and clearer ownership between persona compilation and simulation consumption.
