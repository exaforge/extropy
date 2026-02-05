# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Entropy Is

Entropy is a predictive intelligence framework that simulates how real human populations respond to scenarios. It creates synthetic populations grounded in real-world statistical data, enriches them with LLM-extrapolated psychographic attributes, connects them via social networks, and runs agent-based simulations where each agent reasons individually via LLM calls. The output is not a poll — it's a simulation of emergent collective behavior.

Competitor reference: [Aaru](https://aaru.com) operates in the same space (multi-agent population simulation for predictive intelligence). Entropy differentiates through its grounding pipeline — every attribute distribution is researched from real-world sources with citations, not just LLM-generated.

## Commands

```bash
pip install -e ".[dev]"      # Install with dev deps

# Set API keys (secrets only — in .env or env vars)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Configure providers and models
entropy config set pipeline.provider claude        # Use Claude for population/scenario building
entropy config set simulation.provider openai      # Use OpenAI for agent reasoning
entropy config set simulation.model gpt-5-mini     # Override simulation model
entropy config set simulation.routine_model gpt-5-mini  # Cheap model for Pass 2 classification
entropy config set simulation.rate_tier 2          # Rate limit tier (1-4)
entropy config show                                # View current config

pytest                       # Run all tests
pytest tests/test_sampler.py # Single test file
pytest -k "test_name"        # Single test by name

ruff check .                 # Lint
ruff format .                # Format
```

CLI entry point: `entropy` (defined in `pyproject.toml` → `entropy.cli:app`). Python >=3.11.

## Pipeline (7 sequential commands)

```
entropy spec → entropy extend → entropy sample → entropy network → entropy persona → entropy scenario → entropy simulate
 │
 entropy results
 entropy estimate
```

Each command produces a file consumed by the next. `entropy validate` is a utility runnable at any point. `entropy results` is a viewer for simulation output. `entropy estimate` predicts simulation cost (LLM calls, tokens, USD) without requiring API keys.

## Architecture

Three phases, each mapping to a package under `entropy/`:

### Phase 1: Population Creation (`entropy/population/`)

**The validity pipeline.** This is where predictive accuracy is won or lost.

1. **Sufficiency check** (`spec_builder/sufficiency.py`) — LLM validates the description has enough context (who, how many, where).

2. **Attribute selection** (`spec_builder/selector.py`) — LLM discovers 25-40 attributes across 4 categories: `universal` (age, gender), `population_specific` (specialty, seniority), `context_specific` (scenario-relevant), `personality` (Big Five). Each attribute gets a type (`int`/`float`/`categorical`/`boolean`) and sampling strategy (`independent`/`derived`/`conditional`).

3. **Hydration** (`spec_builder/hydrator.py` → `hydrators/`) — The most important step. Four sub-steps, each using different LLM tiers:
   - **2a: Independent** (`hydrators/independent.py`) — `agentic_research()` with web search finds real-world distributions with source URLs. This is the grounding layer.
   - **2b: Derived** (`hydrators/derived.py`) — `reasoning_call()` specifies deterministic formulas (e.g., `years_experience = age - 26`).
   - **2c: Conditional base** (`hydrators/conditional.py`) — `agentic_research()` finds base distributions for attributes that depend on others.
   - **2d: Conditional modifiers** (`hydrators/conditional.py`) — `reasoning_call()` specifies how attribute values shift based on other attributes. Type-specific: numeric gets `multiply`/`add`, categorical gets `weight_overrides`, boolean gets `probability_override`.

4. **Constraint binding** (`spec_builder/binder.py`) — Topological sort (Kahn's algorithm, `utils/graphs.py`) resolves attribute dependencies into a valid sampling order. Raises `CircularDependencyError` with cycle path.

5. **Sampling** (`sampler/core.py`) — Iterates through `sampling_order`, routing each attribute by strategy. Supports 6 distribution types: normal, lognormal, uniform, beta, categorical, boolean. Hard constraints (min/max) are clamped post-sampling. Formula parameters evaluated via `utils/eval_safe.py` (restricted Python eval, whitelisted builtins only).

6. **Network generation** (`network/`) — Hybrid algorithm: similarity-based edge probability with degree correction, calibrated via binary search to hit target avg_degree, then Watts-Strogatz rewiring (5%) for small-world properties. Edge probability: `base_rate * sigmoid(similarity) * degree_factor_a * degree_factor_b`. All network behavior is data-driven via `NetworkConfig`: attribute weights, edge type rules (evaluated by priority via `_eval_edge_condition()`), influence factors (ordinal/boolean/numeric), and degree multipliers. `NetworkConfig` can be generated from a `PopulationSpec` via LLM (`config_generator.py`), loaded from YAML (`--network-config`), or auto-detected as `{population_stem}.network-config.yaml`. Empty config (no `-p` or `-c`) produces a flat network. `GERMAN_SURGEON_CONFIG` preserves the original hardcoded German surgeon defaults as a reference.

### Phase 2: Scenario Compilation (`entropy/scenario/`)

**Compiler** (`compiler.py`) orchestrates 5 steps: parse event → generate exposure rules → determine interaction model → define outcomes → assemble spec.

- **Event types**: product_launch, policy_change, pricing_change, technology_release, organizational_change, market_event, crisis_event
- **Exposure channels**: broadcast, targeted, organic — with per-timestep rules containing conditions and probabilities
- **Outcomes**: categorical (enum options), boolean, float (with range), open_ended
- Auto-configures simulation parameters based on population size (<500: 50 timesteps, ≤5000: 100, >5000: 168)

### Phase 3: Simulation (`entropy/simulation/`)

**Engine** (`engine.py`) runs per-timestep loop, decomposed into sub-functions:
1. **`_apply_exposures(timestep)`** — Apply seed exposures from scenario rules (`propagation.py`), then propagate through network via conviction-gated sharing (very_uncertain agents don't share). Uses pre-built adjacency list for O(1) neighbor lookups.
2. **`_reason_agents(timestep)`** — Select agents to reason (first exposure OR multi-touch threshold exceeded, default: 3 new exposures since last reasoning), filter out already-processed agents (resume support), split into chunks of `chunk_size` (default 50), run two-pass async LLM reasoning per chunk (`reasoning.py`, rate-limiter-controlled), commit per chunk:
   - **Pass 1 (role-play)**: Agent reasons in first person with no categorical enums. Produces reasoning, public_statement, sentiment, conviction (0-100 integer, bucketed post-hoc), will_share. Memory trace (last 3 reasoning summaries) is fed back for re-reasoning agents.
   - **Pass 2 (classification)**: A cheap model classifies the free-text reasoning into scenario-defined categorical/boolean/float outcomes. Position is extracted here — it is output-only, never used in peer influence.
3. **`_process_reasoning_chunk(timestep, results, old_states)`** — Process a chunk of reasoning results: conviction-based flip resistance (firm+ agents reject flips unless new conviction is moderate+), conviction-gated sharing, state persistence. State updates are batched via `batch_update_states()`.
4. Compute timestep summary, check stopping conditions (`stopping.py`) — Compound conditions like `"exposure_rate > 0.95 and no_state_changes_for > 10"`, convergence detection via sentiment variance.

**Semantic peer influence**: Agents see peers' `public_statement` + sentiment tone, NOT position labels.

**Adjacency list**: Built at engine init from network edges (both directions). Stored as `dict[str, list[tuple[str, dict]]]`. Passed to `propagate_through_network()` and used in `_get_peer_opinions()` for O(1) neighbor lookups instead of O(E) scans.

**Two-pass reasoning rationale**: Single-pass reasoning caused 83% of agents to pick safe middle options (central tendency bias). Splitting role-play from classification fixes this — agents reason naturally in Pass 1, then a cheap model maps to categories in Pass 2.

**Checkpointing** (`state.py`, `engine.py`): Each timestep phase has its own transaction — exposures, each reasoning chunk, and summary are committed separately. The `simulation_metadata` table stores checkpoint state: `mark_timestep_started()` records which timestep is in progress, `mark_timestep_completed()` clears it. On resume (`_get_resume_timestep()`), the engine detects crashed-mid-timestep (checkpoint set) or last-completed-timestep and picks up from there. Already-processed agents within a partial timestep are skipped via `get_agents_already_reasoned_this_timestep()`. Metadata uses immediate commits (not wrapped in `transaction()`). Setup methods (`_create_schema`, `initialize_agents`) also retain their own commits. CLI: `--chunk-size` (default 50).

**Conviction system**: Agents output a 0-100 integer score on a free scale (with descriptive anchors: 0=no idea, 25=leaning, 50=clear opinion, 75=quite sure, 100=certain). `score_to_conviction_float()` buckets it immediately: 0-15→0.1 (very_uncertain), 16-35→0.3 (leaning), 36-60→0.5 (moderate), 61-85→0.7 (firm), 86-100→0.9 (absolute). Agents never see categorical labels or float values — only the 0-100 scale. On re-reasoning, memory traces show the bucketed label (e.g. "you felt *moderate* about this") via `float_to_conviction()`. Engine conviction thresholds reference `CONVICTION_MAP[ConvictionLevel.*]` constants, not hardcoded floats.

**Cost estimation** (`simulation/estimator.py`): `entropy estimate` runs a simplified SIR-like propagation model to predict LLM calls per timestep without making any API calls. Token counts estimated from persona size + scenario content. Pricing from `core/pricing.py` model database. Supports `--verbose` for per-timestep breakdown.

**Rate limiter** (`core/rate_limiter.py`): Token bucket with dual RPM + TPM buckets. Provider-aware defaults from `core/rate_limits.py` (Anthropic/OpenAI, tiers 1-4). Replaces the old hardcoded `Semaphore(50)`. CLI flags: `--rate-tier`, config: `simulation.rate_tier`, `simulation.rpm_override`, `simulation.tpm_override`.

**Persona system** (`population/persona/` + `simulation/persona.py`): The `entropy persona` command generates a `PersonaConfig` via 5-step LLM pipeline (structure → boolean → categorical → relative → concrete phrasings). At simulation time, agents are rendered computationally using this config — no per-agent LLM calls. Relative attributes (personality, attitudes) are positioned against population stats via z-scores ("I'm much more price-sensitive than most people"). Concrete attributes use format specs for proper number/time rendering. **Trait salience**: If `decision_relevant_attributes` is set on `OutcomeConfig`, those attributes are grouped first under "Most Relevant to This Decision" in the persona.

## LLM Integration (`entropy/core/llm.py`)

All LLM calls go through this file — never call providers directly elsewhere. Two-zone routing:

**Pipeline zone** (phases 1-2: spec, extend, persona, scenario) — configured via `entropy config set pipeline.*`:

| Function | Default Model | Tools | Use |
|----------|--------------|-------|-----|
| `simple_call()` | provider default (haiku/gpt-5-mini) | none | Sufficiency checks, simple extractions |
| `reasoning_call()` | provider default (sonnet/gpt-5) | none | Attribute selection, hydration, scenario compilation. Supports validator callback + retry |
| `agentic_research()` | provider default (sonnet/gpt-5) | web_search | Distribution hydration with real-world data. Extracts source URLs |

**Simulation zone** (phase 3: agent reasoning) — configured via `entropy config set simulation.*`:

| Function | Default Model | Tools | Use |
|----------|--------------|-------|-----|
| `simple_call_async()` | provider default | none | Pass 1 role-play reasoning + Pass 2 classification (async) |

Two-pass model routing: Pass 1 uses `simulation.model` (pivotal reasoning). Pass 2 uses `simulation.routine_model` (cheap classification). Both default to provider default if not set. CLI: `--model`, `--pivotal-model`, `--routine-model`. Standard inference only — no thinking/extended models (no o1, o3, extended thinking).

**Provider abstraction** (`entropy/core/providers/`): `LLMProvider` base class with `OpenAIProvider` and `ClaudeProvider` implementations. Factory functions `get_pipeline_provider()` and `get_simulation_provider()` read from `EntropyConfig`. Base class provides `_retry_with_validation()` — shared validation-retry loop used by both providers' `reasoning_call()` and `agentic_research()`. Both providers implement `_with_retry()` / `_with_retry_async()` for transient API errors (APIConnectionError, InternalServerError, RateLimitError) with exponential backoff (`2^attempt + random(0,1)` seconds, max 3 retries).

**Config** (`entropy/config.py`): `EntropyConfig` with `PipelineConfig` and `SimZoneConfig` zones. Resolution order: env vars > config file (`~/.config/entropy/config.json`) > defaults. API keys always from env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). For package use: `from entropy.config import configure, EntropyConfig`.

**Default zones**: Pipeline = Claude (population/scenario building). Simulation = OpenAI (agent reasoning). `SimZoneConfig` fields: `provider`, `model`, `pivotal_model`, `routine_model`, `max_concurrent`, `rate_tier`, `rpm_override`, `tpm_override`.

All calls use structured output (`response_format: json_schema`). Failed validations are fed back as "PREVIOUS ATTEMPT FAILED" prompts for self-correction.

## Data Models (`entropy/core/models/`)

All Pydantic v2. Key hierarchy:

- `population.py`: `PopulationSpec` → `AttributeSpec` → `SamplingConfig` → `Distribution` / `Modifier` / `Constraint`
- `scenario.py`: `ScenarioSpec` → `Event`, `SeedExposure` (channels + rules), `InteractionConfig`, `SpreadConfig`, `OutcomeConfig`
- `simulation.py`: `ConvictionLevel`, `MemoryEntry`, `AgentState` (conviction, public_statement), `PeerOpinion` (public_statement, credibility), `ReasoningContext` (memory_trace), `ReasoningResponse` (conviction, public_statement, reasoning_summary), `SimulationRunConfig` (pivotal_model, routine_model, chunk_size), `TimestepSummary` (average_conviction, sentiment_variance)
- `network.py`: `Edge`, `NodeMetrics`, `NetworkMetrics`
- `validation.py`: `ValidationIssue`, `ValidationResult`

YAML serialization via `to_yaml()`/`from_yaml()` on `PopulationSpec` and `ScenarioSpec`.

## Validation (`entropy/population/validator/`)

Two layers for population specs:
- **Structural** (`structural.py`): ERROR-level — type/modifier compatibility, range violations, distribution params, dependency cycles, condition syntax, formula references, duplicates, strategy consistency
- **Semantic** (`semantic.py`): WARNING-level — no-op detection, modifier stacking, categorical option reference validity

Scenario validation (`entropy/scenario/validator.py`): attribute reference validity, edge type references, probability ranges.

## Key Conventions

- Conditions and formulas use restricted Python syntax via `eval_safe()` — whitelisted builtins only (abs, min, max, round, int, float, str, len, sum, all, any, bool)
- Agent IDs use the `_id` field from agent JSON, falling back to string index
- Network edges are bidirectional (stored as source/target, traversed both ways)
- Exposure credibility: `event_credibility * channel_credibility` for seed, fixed 0.85 for peer
- "Position" = first required categorical outcome (extracted in Pass 2, used for aggregation/output only — never used in peer influence)
- Peer influence is semantic: agents see neighbors' `public_statement` + sentiment tone, not position labels
- Conviction: agents output 0-100 integer, bucketed to 5 float levels (0.1/0.3/0.5/0.7/0.9) via `score_to_conviction_float()`. Agents see only the 0-100 scale; memory traces show categorical labels. Engine thresholds reference `CONVICTION_MAP[ConvictionLevel.*]`
- Memory traces: 3-entry sliding window per agent, fed back into reasoning prompts for re-reasoning
- Progress callbacks use typed `Protocol` classes from `entropy/utils/callbacks.py` (`StepProgressCallback`, `TimestepProgressCallback`, `ItemProgressCallback`, `HydrationProgressCallback`, `NetworkProgressCallback`) — structurally compatible with plain callables via duck typing
- The `persona` command generates detailed persona configs; `extend` still generates a simpler `persona_template` for backwards compatibility
- Simulation auto-detects `{population_stem}.persona.yaml` and uses the new rendering if present
- Network config is data-driven via `NetworkConfig` (YAML-serializable). `GERMAN_SURGEON_CONFIG` is the reference example. CLI: `entropy network -p population.yaml` (LLM-generated), `-c config.yaml` (manual), `--save-config` (export)

## Tests

pytest + pytest-asyncio. Fixtures in `tests/conftest.py` include seeded RNG (`Random(42)`), minimal/complex population specs, and sample agents. 570+ tests across 14+ test files:

- `test_models.py`, `test_network.py`, `test_sampler.py`, `test_scenario.py`, `test_simulation.py`, `test_validator.py` — core logic
- `test_engine.py` — mock-based engine integration (seed exposure, flip resistance, conviction-gated sharing, chunked reasoning, checkpointing, resume logic, metadata lifecycle)
- `test_compiler.py` — scenario compiler pipeline with mocked LLM calls, auto-configuration
- `test_providers.py` — provider response extraction, transient error retry, validation-retry exhaustion, source URL extraction (mocked HTTP)
- `test_rate_limiter.py` — token bucket, dual-bucket rate limiter, `for_provider` factory
- `test_cli.py` — CLI smoke tests (`config show/set`, `validate`, `--version`)
- `test_estimator.py` — cost estimation, pricing lookup, token estimation

CI: `.github/workflows/test.yml` — lint (ruff check + format) and test (pytest, matrix: Python 3.11/3.12/3.13) via `astral-sh/setup-uv@v4`.

## File Formats

- Population/scenario specs: YAML
- Network config: YAML (`NetworkConfig.to_yaml()`/`from_yaml()`) — attribute weights, edge type rules, influence factors, degree multipliers
- Agents: JSON (array of objects with `_id`)
- Network: JSON (`{meta, nodes, edges}`)
- Simulation state: SQLite (tables: agent_states, exposures, memory_traces, timeline, timestep_summaries, shared_to, simulation_metadata)
- Timeline: JSONL (streaming, crash-safe)
- Results: JSON files in output directory (agent_states.json, by_timestep.json, outcome_distributions.json, meta.json)
