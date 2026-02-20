# Spec Stage Reference: Population Creation and Encoding

## Document Intent
This file is the deep reference for the `spec` stage in Extropy.

If a team member or agent forgets prior context, this file should bring them back to working understanding of:
- what `spec` is supposed to accomplish,
- why the design exists,
- how the stage should operate in theory,
- how the current system behaves in practice,
- where the current implementation falls short,
- and what improvement directions are implied by those gaps.

This is not a patch plan and not a code-fix checklist. It is a contract + reality map.

## Status Update (2026-02-20)
- Spec-stage hydration now runs with `include_household=False`.
- Household config ownership is enforced at scenario stage; spec no longer does dead household hydration work.
- Naming is Faker-first at sampling/runtime with deterministic seed and geography routing; CSV remains a safety fallback path.
- Attribute names are canonicalized to snake_case during selection; ambiguous normalization collisions now fail fast.
- Fail-hard spec validation writes versioned artifacts (`*.invalid.vN.yaml`) instead of overwriting a single invalid file.

---

## How To Read This Document
Use this in order:
1. `Intended Contract` for the target design truth.
2. `Current Implementation` for what the code currently does.
3. `Divergences and Mistakes` for the gap between target and reality.
4. `Improvement Directions` for strategy-level correction paths.

This doc is intentionally verbose. It is meant to be reusable team memory.

---

## Why This Stage Exists
Extropy needs to simulate populations at scale with believable heterogeneity.

A naive approach would generate agents directly via LLM per person. That approach breaks down because:
- it is expensive and slow,
- it does not reliably enforce population-level distributions,
- it does not guarantee reproducible structure at scale,
- it cannot be audited easily for relationship consistency between attributes,
- it weakly controls outlier rates and coherence.

The `spec` stage exists to shift intelligence from runtime to compile time:
- infer and encode population mechanics once,
- sample agents cheaply and repeatedly from that encoded blueprint,
- preserve statistical shape, dependency logic, and constraint boundaries.

The design principle is:
- **Compile population semantics once, sample many times.**

---

## Pipeline Position
Within the full Extropy pipeline, `spec` is foundational.

High-level pipeline intent:
1. Create population and population dynamics.
2. Create event/scenario and event dynamics.
3. Build persona and network in scenario-specific context.
4. Simulate behavior and outcomes.

`spec` is the base-population creation stage. It should produce a reusable artifact that later scenario-specific extensions can build on.

If `spec` quality is weak, downstream stages cannot reliably recover.

---

## Intended Contract

### What `spec` should do
`spec` should output a reusable **base population encoding** as versioned YAML that defines how a valid member of the target population is generated.

This encoding should include:
- attribute inventory (who a person can be in this population),
- attribute typing and semantics,
- sampling strategies and distributions,
- cross-attribute dependency structure,
- conditional behavior modifiers,
- consistency constraints,
- support for realistic outliers,
- metadata needed for downstream interpretation.

The artifact should be reusable across scenarios that operate on the same base population.

### What `spec` should not do
`spec` should not:
- generate concrete agents,
- bake in one scenario-specific behavioral rollout,
- run per-agent LLM reasoning,
- collapse population diversity into a single “average” profile,
- treat outliers as pure noise to suppress.

### How `spec` should do it
The intended process is a compile pipeline:
1. Verify intent sufficiency.
2. Select the right attributes (coverage + relevance).
3. Hydrate each attribute with sampling semantics.
4. Resolve dependency graph and compile sampling order.
5. Validate for structural and behavioral quality.
6. Emit versioned base spec artifact.

This process should preserve three priorities (in order from this project context):
1. realism,
2. internal consistency,
3. downstream support reliability.

### Why this contract matters
Without explicit base encoding, all downstream steps become unstable:
- scenario extension becomes ad hoc,
- sampling quality degrades,
- persona and network logic lose grounding,
- simulation outputs become hard to trust.

`spec` is therefore not a convenience stage. It is the system’s causal root.

---

## Conceptual Model Of Population Encoding

### Population as generative blueprint
A population is represented as a constrained generative model:
- each attribute is a variable,
- each variable has a type,
- variables are sampled or computed according to strategy,
- dependencies define valid evaluation order,
- constraints define valid output region,
- modifiers define context-conditional shifts,
- outlier mechanics define tails and rare combinations.

### Attribute classes
At minimum, attribute selection should consider:
- universal attributes (broad demographic anchors),
- population-specific attributes (identity/work/lifestyle specifics),
- personality/behavioral attributes,
- optional context attributes where justified.

### Sampling strategy classes
Core strategy semantics:
- `independent`: sampled directly from a distribution with no upstream dependencies.
- `derived`: deterministic formula output from dependencies; no variance for same input tuple.
- `conditional`: base distribution with dependency-conditioned shifts.

### Scope semantics
Attribute scope shapes household-level instantiation logic later:
- `individual`: value varies per person.
- `household`: value is shared across household members.
- `partner_correlated`: value is correlated between partners.

### Constraints semantics
Constraints should distinguish:
- hard bounds (clamping or strict validity boundaries),
- expression-level validity conditions,
- spec-level validity checks.

### Outlier semantics
The system should treat outliers as first-class:
- some should be explicitly represented,
- others should emerge from conditional distributions,
- impossible combinations should be blocked,
- implausible combinations should be measured, not blindly suppressed.

---

## Intended Base vs Scenario Boundary

### Base population (`spec`) should own
- reusable population mechanics,
- stable demographic/identity/behavioral variable definitions,
- dependency and distribution structure,
- generalized semantic metadata for later interpretation.

### Scenario extension should own
- scenario-specific additional attributes,
- scenario-specific activation policies,
- scenario-contextual household/relationship activation policy,
- scenario-time role policy (who is simulated agent vs NPC vs mixed).

### Sampling should own
- deterministic instantiation from base + extension,
- enforcement of compiled dependency order,
- runtime realization under seeded randomness.

This boundary is critical for reuse and avoiding scope creep.

---

## Intended Artifact Contract

### Primary artifact
`spec` should emit versioned base spec YAML (`population.vN.yaml`) containing:
- metadata,
- grounding summary,
- attribute list,
- compiled sampling order.

### Metadata responsibilities
Metadata should encode:
- source description and geography,
- who the base population focuses on,
- versioning fields,
- optional downstream rendering metadata when available.

### Attribute contract (intended)
Each attribute should fully define:
- identifier and type,
- category,
- descriptive meaning,
- scope semantics,
- semantic/identity/display metadata,
- sampling config,
- grounding quality,
- constraints.

### Sampling-order contract (intended)
Sampling order should be:
- complete (every sampled field appears),
- dependency-safe (dependencies resolved before dependents),
- deterministic for same compiled graph.

---

## Intended Quality Contract

### Realism
The output should produce agents that look like credible members of the population, including plausible variation and nontrivial tails.

### Internal consistency
Attribute combinations should be coherent under known relationships. Contradictory combinations should be blocked or explicitly flagged by severity.

### Downstream support
Spec output must be machine-usable and semantically useful to:
- scenario extension,
- sampler,
- persona rendering,
- network generation,
- simulation reasoning.

### Quality interpretation rule
Structural validity alone is not enough. A spec can be syntactically valid and still behaviorally low-quality.

---

## Intended Backpressure Philosophy (Contract-Level)

`spec` should produce enough measurable evidence so downstream stages can decide whether to proceed.

At contract level, checks should separate:
- `block` class contradictions (impossible/incoherent),
- `warn` class plausibility concerns (rare but possible, suspicious drift),
- informational diagnostics.

Backpressure should be explicit and machine-readable, not only human prompt warnings.

---

## Current Implementation (As-Is)

This section describes what the current codebase does now.

### Entrypoint behavior
`extropy spec` command flow:
1. Resolve study folder and versioned output path.
2. Run sufficiency check.
3. Run attribute selection.
4. Run split hydration pipeline.
5. Run constraint binding and sampling-order compilation.
6. Build `PopulationSpec`.
7. Run spec validation.
8. Save YAML.

### Sufficiency behavior in code
Current sufficiency stage:
- uses fast LLM call,
- infers geography,
- can ask structured clarification questions.

### Selection behavior in code
Current selector:
- uses reasoning LLM call with a strict JSON schema,
- asks model to assign strategy, scope, dependencies, and optional semantic metadata,
- injects Big Five traits when recommended and missing,
- may inject `country` attribute for multi-country geographies.

### Hydration behavior in code
Hydration is split:
- 2a independent distributions via agentic research,
- 2b derived formulas via reasoning call,
- 2c conditional base distributions via agentic research,
- 2d conditional modifiers via agentic research,
- 2e household config via agentic research (conditional trigger),
- 2f name config via agentic research (skips US population heuristic).

Hydration includes fail-fast validation callbacks and retry loops.

### Binding behavior in code
Binding step:
- converts hydrated attributes to final `AttributeSpec`,
- filters unknown dependencies with warnings,
- infers dependencies from formulas and conditions,
- computes topological sort for sampling order.

### Build behavior in code
Spec build:
- creates `SpecMeta`,
- computes grounding summary,
- assembles final `PopulationSpec`.

### Validation behavior in code
Validation currently combines:
- structural checks (errors and some warnings),
- semantic checks (warnings).

Validation catches many shape and compatibility issues, but does not fully guarantee behavioral realism quality.

---

## Current Runtime Coupling With Scenario And Sampling

### Current extension model
Scenario flow currently performs additive extension:
- discovers and hydrates new attributes with base context,
- stores them in scenario as `extended_attributes`.

### Current household ownership in runtime
Practical runtime ownership today:
- base `spec` pipeline can research household config,
- `extropy spec` explicitly ignores/stubs household config for base output,
- scenario stores household config and agent focus mode,
- sampler consumes scenario-level household config and focus mode.

### Current merge model before sampling
`sample` currently:
- merges base + extended attributes,
- appends extended names onto base sampling order if missing,
- samples with scenario-provided household/focus settings.

This is the effective behavior today.

---

## What Current Implementation Already Gets Right

### Architectural direction
- Compile-time modeling + fast runtime sampling is the correct high-level shape.

### Typed modeling surface
- Rich Pydantic models exist for attributes, distributions, constraints, and metadata.

### Dependency-aware compilation
- Topological ordering and cycle detection are implemented.

### Validation scaffolding
- There is a meaningful validation layer both at LLM-response time and pre-sampling.

### Reuse path
- Base specs are versioned and reusable as artifacts.

---

## Divergences, Mistakes, and Risk Areas

### 1) Prompt-heavy contract enforcement
Many intended quality guarantees are represented as prompt instructions rather than strict enforceable contracts.

Risk:
- quality can drift with model behavior while still passing structural checks.

### 2) Silent coercion and fallback behavior
Parsing and binding include permissive fallback logic in some places (defaults, filtered deps).

Risk:
- malformed or weak outputs can degrade silently into “valid enough” artifacts.

### 3) Dependency mutation side effects
Unknown dependencies can be removed with warnings. Inferred dependencies can be auto-added.

Risk:
- resulting compiled behavior differs from model’s original intent without a hard failure.

### 4) Extension/sampling order assumptions
Extension merges are currently append-oriented in CLI flow.

Risk:
- cross-layer dependency semantics can be fragile if extension expectations grow.

### 5) Household ownership ambiguity in build narrative
Build path can hydrate household config during base flow, but final base stage ignores it and runtime ownership shifts to scenario.

Risk:
- conceptual confusion across teams and agents unless boundary is explicit.

### 6) Category/value normalization fragility
Token shape mismatches in categorical ecosystems remain a real risk (typos, underscore/space/case variants).

Risk:
- modifier conditions and downstream interpretation can diverge from intended semantics.

### 7) Outlier contract debt
Outliers can emerge today, but there is no explicit first-class outlier intent contract with measurable acceptance semantics.

Risk:
- too-flat populations or chaotic tails may both pass structural checks.

### 8) Structural-validity bias
Current checks are much stronger on syntax/typing than on distributional realism and joint plausibility at scale.

Risk:
- valid YAML, weak population reality.

---

## How It Can Be Made Better (Direction, Not Patch Plan)

This section captures strategic direction only.

### Make quality explicit and testable
Define and emit explicit quality contract outcomes, not only passive warnings.

### Separate impossibility from implausibility formally
Use severity-aware rule semantics so contradictions are blocked and plausibility drift is visible with calibrated tolerance.

### Elevate outlier intent to explicit contract
Treat outlier representation as a declared quality dimension, not only an emergent property.

### Tighten stage ownership language
Document and enforce base vs scenario vs sampling ownership boundaries clearly.

### Strengthen normalization discipline
Normalize category/token interpretation consistently across selection, hydration, validation, and sampling semantics.

### Upgrade backpressure from “advice” to “gateable evidence”
Produce machine-readable quality evidence suitable for automated stage gating.

---

## Downstream Dependency Map (What `spec` Must Support)

### Scenario
Needs base attribute semantics robust enough to support scenario-specific extension and event relevance targeting.

### Sample
Needs complete, consistent sampling definitions and valid dependency order for deterministic agent generation.

### Persona
Needs interpretable attribute descriptions and formatting metadata to render believable identities.

### Network
Needs coherent attributes with social relevance and reliable categorical structure.

### Simulate
Needs internally coherent agent state primitives that do not collapse under behavioral reasoning.

### Estimate
Needs stable assumptions about population complexity and downstream workload implications.

### Query/Results/Chat
Need consistent field semantics so analytics and conversational explanations remain meaningful.

---

## Glossary (Working Terms)

- **Base population**: reusable population encoding from `spec`.
- **Extended population**: base + scenario-specific attribute layer.
- **Compile-time**: expensive inference stage that writes artifacts.
- **Runtime sampling**: fast generation from compiled artifacts.
- **Outlier**: rare but plausible instance/combination.
- **Impossible**: logically contradictory combination.
- **Implausible**: possible but suspicious/unlikely combination.
- **Backpressure**: quality signal that can block or warn before downstream continuation.

---

## Practical Readout
If you only remember five things from this file:
1. `spec` is a compile stage, not a generation stage.
2. Its job is reusable population semantics, not scenario-specific behavior rollout.
3. Realism + consistency + downstream support are the quality triad.
4. Current implementation has strong structure but weaker explicit behavioral contracts.
5. The main maturity gap is enforceable quality semantics, not missing model surface area.

---

## Final Statement
The current `spec` system is substantial and directionally correct, but it is not yet equivalent to a fully explicit, contract-driven population compiler. It currently behaves as a strong structured generator with partial contract enforcement. To meet the intended standard, quality intent and cross-stage boundaries must be made explicit enough to serve as reliable backpressure for the rest of the pipeline.
