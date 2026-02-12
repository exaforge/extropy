# Entropy Validation Roadmap

Track and validate every phase of the system. Each item is a question we need to answer with evidence before we can trust entropy's output.

---

## Phase 1: Population Generation

### 1.1 Spec Reproducibility
**Question:** How much does `entropy spec` vary across runs for the same description?

- [ ] Run `entropy spec` 5x with identical input, diff the output YAMLs
- [ ] Measure: number of attributes, attribute names, types, sampling strategies
- [ ] Determine acceptable variance (e.g., 80% attribute overlap across runs)
- [ ] If variance is too high: consider temperature control, prompt pinning, or a "lock" step that lets users approve the attribute list before hydration

**Status:** LLM calls are inherently non-deterministic. No seed control exists on `reasoning_call()` or `agentic_research()`. This is by design — but we need to quantify how much the output varies.

### 1.2 Grounding Accuracy
**Question:** Are the real-world distributions sourced by `agentic_research()` actually accurate?

- [ ] Pick 3 completed population specs, extract all `grounding.sources` URLs
- [ ] Manually verify 10-15 sources: do the cited numbers match what's in the spec?
- [ ] Check for dead links, non-authoritative sources (blogs, forums vs. census/journals)
- [ ] Categorize grounding levels: what % of attributes are "strong" vs "medium" vs "low"?
- [ ] Consider: build an `entropy audit` command that reports grounding quality per attribute

**Status:** Infrastructure exists — `GroundingInfo` per attribute stores level/method/source/note. `GroundingSummary` aggregates counts. No automated quality checking yet.

### 1.3 Modifier Bounds Safety
**Question:** Can conditional modifiers push sampled values outside valid ranges?

- [ ] ~~Investigate clamping logic in sampler~~ DONE
- [ ] Write a stress-test: create a spec with aggressive modifiers (10x multiply on a bounded attribute) and verify clamping works
- [ ] Check edge cases: what happens when a modifier pushes a categorical weight below 0? A probability above 1?

**Status:** Solved. `_apply_numeric_modifiers()` re-clamps after every operation. Hard constraints enforced post-coercion. Beta/boolean distributions clamped to [0,1]. Needs a stress-test to confirm edge cases.

### 1.4 Network Generation
**Question:** Does the network produce realistic social structure for arbitrary populations?

- [ ] ~~Explore how network command works~~ DONE
- [ ] ~~Generalize away from domain-specific preset network configs~~ DONE
- [ ] Run LLM-generated network config for 2-3 different populations, inspect the YAML output
- [ ] Validate: do the generated edge type rules make sense? Are influence factors reasonable?
- [ ] Compare metrics (clustering, path length, modularity) between LLM-configured and flat networks

**Status:** Generalized. `NetworkConfig` is now data-driven with LLM generation, YAML I/O, and CLI flags (`-p`, `-c`, `--save-config`). Needs real-world validation runs.

### 1.5 Sampling Quality
**Question:** Does the sampled population actually match the intended distributions?

- [ ] Sample 500+ agents from a completed spec
- [ ] Compare sampled distributions to spec distributions (KS test, chi-squared for categoricals)
- [ ] Check for impossible agents: entry-level + 20yr experience, 16-year-old surgeon, etc.
- [ ] Visualize: age distribution, attribute correlations, conditional modifier effects

---

## Phase 2: Scenario Compilation

### 2.1 Scenario Reproducibility
**Question:** How much does `entropy scenario` vary across runs?

- [ ] Run `entropy scenario` 5x with identical inputs, diff outputs
- [ ] Measure: outcome categories, exposure rules, interaction model, spread config
- [ ] Determine if variance matters — scenario structure is less sensitive than population spec since it's reviewed before simulation

**Status:** Same non-determinism as spec generation. Likely less critical since scenarios are shorter and easier to manually review.

### 2.2 Outcome Design Quality
**Question:** Do LLM-generated outcomes avoid the central tendency trap?

- [ ] Review 3 generated scenario specs for catch-all middle options (the `consider_later` problem)
- [ ] Check if the anti-hedging prompt guidance from Fix 3 (`995bf4a`) actually produces distinct behavioral outcomes
- [ ] Validate: each outcome option should represent a clearly different action, not a spectrum

**Status:** Fix 3 added compiler guidance against catch-all options. Needs empirical validation.

---

## Phase 3: Persona Generation

### 3.1 Persona Quality
**Question:** Do rendered personas produce coherent, plausible people?

- [ ] Generate personas for 2-3 populations, preview 10 agents each with `entropy persona --preview`
- [ ] Check for contradictions: high income + student, retired + 5yr experience, etc.
- [ ] Check trait salience: are decision-relevant attributes properly prioritized?
- [ ] Check relative positioning: does "much more price-sensitive than most" actually correspond to a high z-score?

**Status:** Good tooling. `--preview` and `--agent N` flags exist. Rendering is deterministic (no LLM at simulation time). Contradictions are a sampling issue (Phase 1.5), not persona rendering.

### 3.2 Population Analytics
**Question:** Does the sampled + persona'd population look like the intended real-world population?

- [ ] Export agents JSON + persona config, load into pandas
- [ ] Compute: attribute distributions, correlations, personality trait spreads
- [ ] Compare to spec targets: do the sampled distributions match?
- [ ] Build an `entropy export` command for CSV/Parquet output (agents + rendered personas)

**Status:** Partial. Agents are JSON, results are JSON with segments. No CSV/Parquet export. No built-in analytics. Currently requires custom scripts.

---

## Phase 4: Simulation

### 4.1 Backtesting / Ground Truth
**Question:** Can entropy predict outcomes that match real-world data?

This is the hardest and most important validation. Approaches:

- [ ] **Toy scenarios with known answers:** Design 2-3 simple scenarios where the outcome is obvious (e.g., "free upgrade offered to loyal customers" → near-universal acceptance). Run simulation, check if results match intuition.
- [ ] **Historical scenarios:** Pick 2-3 past events with known outcomes (product launch adoption rates, policy approval polls). Simulate them. Compare.
- [ ] **Sensitivity analysis:** Same scenario, vary one parameter (price, population size, network density). Check if results move in the expected direction.
- [ ] **Convergence testing:** Run same scenario 3x with different seeds. Results should be similar (low variance = stable system).

**Status:** Nothing built. This is the biggest gap and the highest priority validation work.

### 4.2 Central Tendency Bias
**Question:** Did the two-pass reasoning fix actually work?

- [ ] Run the CeraVe scenario (or equivalent) with current prompts
- [ ] Check validation criteria from `todo/fix-central-tendency.md`:
  - `consider_later` < 60%
  - `not_interested` > 0%
  - Conviction std > 20
  - "cautious" in < 40% of reasoning text
  - Sentiment std > 0.35
- [ ] If still failing: implement Fix 4 (behavioral anchoring in persona)

**Status:** Fixes 1-3 implemented (`995bf4a`, `cd3e159`). Fix 4 deferred. Validation checklist defined but never run against the new prompts.

### 4.3 Cost & Scale
**Question:** What does it actually cost to run simulations at various scales?

- [ ] ~~Build cost estimator~~ DONE
- [ ] Run `entropy estimate` for 100, 500, 1000, 5000 agents
- [ ] Validate estimates against actual runs (compare predicted vs actual cost)
- [ ] Establish budget targets: $5-8 for dev runs (100-200 agents), $20-50 for production (500-1000)
- [ ] Test rate limiting: does tier 2 rate limiting prevent 429 errors for 500-agent runs?

**Status:** Estimator is complete and tested. Needs real-run validation to calibrate accuracy.

### 4.4 Experiment Framework
**Question:** Can we systematically compare different configurations?

Experiments to run once infrastructure is ready:

- [ ] **LLM provider comparison:** Same scenario with OpenAI vs Claude for agent reasoning. Compare outcome distributions, reasoning quality, cost.
- [ ] **Model tier comparison:** gpt-5 vs gpt-5-mini for Pass 1. Does the cheaper model produce worse reasoning or just faster?
- [ ] **Persona prompt variants:** Full persona vs minimal persona. Does richness matter?
- [ ] **Network effect:** Same population, flat network vs LLM-configured network. Does social structure change outcomes?
- [ ] **Population size scaling:** 100 vs 500 vs 1000 agents. Do results converge?

Infrastructure needed:
- [ ] Results comparison tool: diff two simulation runs on same scenario
- [ ] Experiment tracking: log run parameters + results in a structured way
- [ ] CSV/Parquet export for statistical analysis in notebooks

**Status:** Config system supports switching providers/models. No experiment tracking, no comparison tooling, no export pipeline.

---

## Infrastructure Gaps

Items not in the original roadmap but needed for production:

### I.1 Checkpoint/Resume
- [ ] If simulation crashes at timestep 47/100, currently lose all work
- [ ] SQLite state already persists — need a resume flag that picks up from last completed timestep
- [ ] Priority: high for any run > 200 agents (where crashes are expensive)

### I.2 Export Pipeline
- [ ] `entropy export results/ --format csv` — agents, outcomes, timeline as CSV/Parquet
- [ ] Enables pandas/notebook analysis without custom JSON parsing
- [ ] Priority: high for any DS/analytics work (Phase 3.2, 4.4)

### I.3 End-to-End Smoke Test
- [ ] Single test that runs the full pipeline: spec → extend → sample → network → persona → scenario → simulate
- [ ] All LLM calls mocked, but validates the data flows correctly between stages
- [ ] Priority: medium — catches integration bugs that unit tests miss

### I.4 Results Comparison
- [ ] `entropy diff results_a/ results_b/` — compare two simulation runs
- [ ] Show: outcome distribution differences, conviction shifts, reasoning divergence
- [ ] Priority: required for experiment framework (4.4)

### I.5 Grounding Audit
- [ ] `entropy audit spec.yaml` — report on source quality, flag weak/missing citations
- [ ] Check for dead links, non-authoritative sources
- [ ] Priority: medium — important for credibility but not blocking

---

## Execution Order

What to tackle first, based on dependencies and impact:

1. **4.1 Backtesting** — toy scenarios + historical validation. Nothing else matters if the system doesn't produce reasonable results.
2. **4.2 Central tendency validation** — run the checklist against current prompts.
3. **1.1 + 2.1 Reproducibility** — quantify variance. Quick to run, informs whether we need determinism controls.
4. **1.2 Grounding audit** — manual spot-check of sources. Quick, high-value.
5. **I.2 Export pipeline** — unblocks all analytics work.
6. **1.5 Sampling quality** — statistical validation of sampled populations.
7. **4.3 Cost calibration** — validate estimator against real runs.
8. **I.1 Checkpoint/resume** — required before production-scale runs.
9. **4.4 Experiment framework** — needs export pipeline + results comparison first.
10. **I.3 E2E smoke test** — catches integration issues, good for CI.
