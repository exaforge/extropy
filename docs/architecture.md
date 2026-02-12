# Architecture

Entropy has three phases, each mapping to a package under `entropy/`.

---

## Phase 1: Population Creation (`entropy/population/`)

The validity pipeline. This is where predictive accuracy is won or lost.

### 1. Sufficiency Check (`spec_builder/sufficiency.py`)

LLM validates the description has enough context (who, how many, where).

### 2. Attribute Selection (`spec_builder/selector.py`)

LLM discovers 25-40 attributes across 4 categories:
- `universal` — age, gender, income
- `population_specific` — specialty, seniority, commute method
- `context_specific` — scenario-relevant attitudes and behaviors
- `personality` — Big Five traits

Each attribute gets a type (`int`/`float`/`categorical`/`boolean`) and sampling strategy (`independent`/`derived`/`conditional`).

### 3. Hydration (`spec_builder/hydrator.py` -> `hydrators/`)

The most important step. Four sub-steps, each using different LLM tiers:

- **Independent** (`hydrators/independent.py`) — `agentic_research()` with web search finds real-world distributions with source URLs. This is the grounding layer.
- **Derived** (`hydrators/derived.py`) — `reasoning_call()` specifies deterministic formulas (e.g., `years_experience = age - 26`).
- **Conditional base** (`hydrators/conditional.py`) — `agentic_research()` finds base distributions for attributes that depend on others.
- **Conditional modifiers** (`hydrators/conditional.py`) — `reasoning_call()` specifies how attribute values shift based on other attributes. Type-specific: numeric gets `multiply`/`add`, categorical gets `weight_overrides`, boolean gets `probability_override`.

### 4. Constraint Binding (`spec_builder/binder.py`)

Topological sort (Kahn's algorithm, `utils/graphs.py`) resolves attribute dependencies into a valid sampling order. Raises `CircularDependencyError` with cycle path.

### 5. Sampling (`sampler/core.py`)

Iterates through `sampling_order`, routing each attribute by strategy. Supports 6 distribution types: normal, lognormal, uniform, beta, categorical, boolean. Hard constraints (min/max) are clamped post-sampling. Formula parameters evaluated via `utils/eval_safe.py` (restricted Python eval, whitelisted builtins only).

### 6. Network Generation (`network/`)

Hybrid algorithm: similarity-based edge probability with degree correction, calibrated via binary search to hit target avg_degree, then Watts-Strogatz rewiring (5%) for small-world properties.

Edge probability: `base_rate * sigmoid(similarity) * degree_factor_a * degree_factor_b`.

All network behavior is data-driven via `NetworkConfig`: attribute weights for similarity, edge type rules (priority-ordered condition expressions evaluated by `_eval_edge_condition()`), influence factors (ordinal/boolean/numeric), and degree multipliers. Config can be LLM-generated from a `PopulationSpec` (`config_generator.py`), loaded from YAML (`--network-config`), or auto-detected as `{population_stem}.network-config.yaml`. Empty config (no `-p` or `-c`) produces a flat network.

---

## Phase 2: Scenario Compilation (`entropy/scenario/`)

**Compiler** (`compiler.py`) orchestrates 5 steps: parse event -> generate exposure rules -> determine interaction model -> define outcomes -> assemble spec.

- **Event types**: product_launch, policy_change, pricing_change, technology_release, organizational_change, market_event, crisis_event
- **Exposure channels**: broadcast, targeted, organic — with per-timestep rules containing conditions and probabilities
- **Outcomes**: categorical (enum options), boolean, float (with range), open_ended
- Auto-configures simulation parameters based on population size (<500: 50 timesteps, <=5000: 100, >5000: 168)

---

## Phase 3: Simulation (`entropy/simulation/`)

### Engine (`engine.py`)

Per-timestep loop, decomposed into sub-functions:

1. **`_apply_exposures(timestep)`** — Apply seed exposures from scenario rules (`propagation.py`), then propagate through network via conviction-gated sharing (very_uncertain agents don't share). Uses pre-built adjacency list for O(1) neighbor lookups.

2. **`_reason_agents(timestep)`** — Select agents to reason (first exposure OR multi-touch threshold exceeded, default: 3 new exposures since last reasoning), filter out already-processed agents (resume support), split into chunks of `chunk_size` (default 50), run two-pass async LLM reasoning per chunk (`reasoning.py`, rate-limiter-controlled), commit per chunk:
   - **Pass 1** (pivotal model): Agent role-plays in first person with no categorical enums. Produces reasoning, public_statement, sentiment, conviction (0-100 integer, bucketed post-hoc), will_share. Memory trace (last 3 reasoning summaries) is fed back for re-reasoning agents.
   - **Pass 2** (routine model): A cheap model classifies the free-text reasoning into scenario-defined categorical/boolean/float outcomes. Position is extracted here — it is output-only, never used in peer influence.

3. **`_process_reasoning_chunk(timestep, results, old_states)`** — Process a chunk of reasoning results: bounded confidence opinion update (public sentiment/conviction adjusted toward peer averages via `_BOUNDED_CONFIDENCE_RHO`), conviction-based flip resistance (firm+ agents reject flips unless new conviction is moderate+), private opinion tracking (separate private sentiment/conviction with slower `_PRIVATE_ADJUSTMENT_RHO` adjustment), conviction-gated sharing, state persistence. State updates are batched via `batch_update_states()`.

4. **Conviction decay** — Non-reasoning agents experience gradual conviction decay (`_CONVICTION_DECAY_RATE = 0.05`) via `state_manager.apply_conviction_decay()`, preventing stale high-conviction states from persisting indefinitely.

5. Compute timestep summary, check stopping conditions (`stopping.py`) — Compound conditions like `"exposure_rate > 0.95 and no_state_changes_for > 10"`, convergence detection via sentiment variance.

### Two-Pass Reasoning

Single-pass reasoning caused 83% of agents to pick safe middle options (central tendency bias). Splitting role-play from classification fixes this — agents reason naturally in Pass 1, then a cheap model maps to categories in Pass 2.

### Conviction System

Agents output a 0-100 integer score on a free scale (with descriptive anchors: 0=no idea, 25=leaning, 50=clear opinion, 75=quite sure, 100=certain). `score_to_conviction_float()` buckets it immediately:

| Score Range | Float | Level | Meaning |
|-------------|-------|-------|---------|
| 0-15 | 0.1 | `very_uncertain` | Barely formed opinion |
| 16-35 | 0.3 | `leaning` | Tentative position |
| 36-60 | 0.5 | `moderate` | Reasonably confident |
| 61-85 | 0.7 | `firm` | Strong position |
| 86-100 | 0.9 | `absolute` | Unwavering |

Agents never see categorical labels or float values — only the 0-100 scale. On re-reasoning, memory traces show the bucketed label (e.g. "you felt *moderate* about this") via `float_to_conviction()`. Engine conviction thresholds reference `CONVICTION_MAP[ConvictionLevel.*]` constants, not hardcoded floats.

### Semantic Peer Influence

Agents see their neighbors' `public_statement` + sentiment tone, NOT position labels. This means influence is semantic — an agent can be swayed by a compelling argument, not just a categorical stance.

### Memory

Each agent maintains a 3-entry sliding window memory trace. Entries include the timestep, a summary of what they processed, and how it affected their thinking. This gives agents continuity across reasoning rounds without unbounded context growth.

### Persona System

`population/persona/` + `simulation/persona.py`: The `entropy persona` command generates a `PersonaConfig` via 5-step LLM pipeline (structure -> boolean -> categorical -> relative -> concrete phrasings). At simulation time, agents are rendered computationally using this config — no per-agent LLM calls.

Relative attributes (personality, attitudes) are positioned against population stats via z-scores ("I'm much more price-sensitive than most people"). Concrete attributes use format specs for proper number/time rendering.

**Trait salience**: When the scenario defines `decision_relevant_attributes`, those traits are grouped first in the persona under "Most Relevant to This Decision", ensuring the LLM focuses on what matters.

### Checkpointing & Resume (`state.py`, `engine.py`)

Each timestep phase has its own transaction — exposures, each reasoning chunk, and summary are committed separately. The `simulation_metadata` table stores checkpoint state: `mark_timestep_started()` records which timestep is in progress, `mark_timestep_completed()` clears it. On resume (`_get_resume_timestep()` in the engine), the engine detects crashed-mid-timestep (checkpoint set) or last-completed-timestep and picks up from there. Already-processed agents within a partial timestep are skipped via `get_agents_already_reasoned_this_timestep()`. Metadata uses immediate commits (not wrapped in outer transactions). Setup methods (`_create_schema`, `initialize_agents`) also retain their own commits. CLI: `--chunk-size` (default 50).

### Rate Limiting (`core/rate_limiter.py`)

Token bucket rate limiter with dual RPM + TPM buckets. Provider-aware defaults auto-configured from `core/rate_limits.py` (Anthropic/OpenAI, tiers 1-4). Supports tier overrides via config or CLI flags.

### Aggregation & Results (`simulation/aggregation.py`, `simulation/timeline.py`)

`compute_timestep_summary()` aggregates per-timestep metrics (exposure rate, position distributions, sentiment/conviction averages). `compute_final_aggregates()` produces simulation-wide summaries. `compute_outcome_distributions()` builds per-outcome histograms. `compute_timeline_aggregates()` creates time-series data for visualization. `TimelineManager` handles crash-safe JSONL event streaming.

### Cost Estimation (`simulation/estimator.py`)

`entropy estimate` runs a simplified SIR-like propagation model to predict LLM calls per timestep without making any API calls. Token counts estimated from persona size + scenario content. Pricing from `core/pricing.py` model database. Supports `--verbose` for per-timestep breakdown.

---

## LLM Integration (`entropy/core/llm.py`)

All LLM calls go through this file — never call providers directly elsewhere. Two-zone routing:

### Pipeline Zone (phases 1-2)

Configured via `entropy config set pipeline.*`:

| Function | Default Model | Tools | Use |
|----------|--------------|-------|-----|
| `simple_call()` | haiku / gpt-5-mini | none | Sufficiency checks, simple extractions |
| `reasoning_call()` | sonnet / gpt-5 | none | Attribute selection, hydration, scenario compilation |
| `agentic_research()` | sonnet / gpt-5 | web_search | Distribution hydration with real-world data |

### Simulation Zone (phase 3)

Configured via `entropy config set simulation.*`:

| Function | Default Model | Use |
|----------|--------------|-----|
| Pass 1 reasoning | pivotal model (gpt-5 / sonnet) | Agent role-play, freeform reaction |
| Pass 2 classification | routine model (gpt-5-mini / haiku) | Outcome extraction from narrative |

Two-pass model routing: Pass 1 uses `simulation.model` or `simulation.pivotal_model`. Pass 2 uses `simulation.routine_model`. CLI: `--model`, `--pivotal-model`, `--routine-model`.

### Provider Abstraction (`entropy/core/providers/`)

`LLMProvider` base class with `OpenAIProvider`, `ClaudeProvider`, and Azure OpenAI support (via `api_format` config). Factory functions `get_pipeline_provider()` and `get_simulation_provider()` read from `EntropyConfig`.

Base class provides `_retry_with_validation()` — shared validation-retry loop used by both providers' `reasoning_call()` and `agentic_research()`. Both providers implement `_with_retry()` / `_with_retry_async()` for transient API errors with exponential backoff (`2^attempt + random(0,1)` seconds, max 3 retries).

All calls use structured output (`response_format: json_schema`). Failed validations are fed back as "PREVIOUS ATTEMPT FAILED" prompts for self-correction.

---

## Data Models (`entropy/core/models/`)

All Pydantic v2. Key hierarchy:

- `population.py`: `PopulationSpec` -> `AttributeSpec` -> `SamplingConfig` -> `Distribution` (Normal/Lognormal/Uniform/Beta/Categorical/Boolean) / `Modifier` / `GroundingInfo`, `SpecMeta`, `GroundingSummary`
- `scenario.py`: `ScenarioSpec` -> `Event` (`EventType`), `SeedExposure` (channels + rules), `InteractionConfig` (`InteractionType`), `SpreadConfig` (`SpreadModifier`), `OutcomeConfig` (`OutcomeDefinition`, `OutcomeType`), `SimulationConfig` (`TimestepUnit`), `ScenarioMeta`
- `simulation.py`: `AgentState` (with public/private position/sentiment/conviction), `ConvictionLevel`, `CONVICTION_MAP`, `ExposureRecord`, `MemoryEntry`, `PeerOpinion`, `ReasoningContext` (memory_trace, peer opinions), `ReasoningResponse` (reasoning, public_statement, sentiment, conviction, will_share, outcomes), `SimulationEvent` (`SimulationEventType`), `SimulationRunConfig`, `TimestepSummary`
- `network.py`: `Edge`, `NodeMetrics`, `NetworkMetrics`, `NetworkConfig`
- `sampling.py`: `SamplingStats`, `SamplingResult`
- `results.py`: `SimulationSummary`, `AgentFinalState`, `SegmentAggregate`, `TimelinePoint`, `RunMeta`, `SimulationResults`
- `validation.py`: `Severity`, `ValidationIssue`, `ValidationResult`

YAML serialization via `to_yaml()`/`from_yaml()` on `PopulationSpec`, `ScenarioSpec`, and `NetworkConfig`.

---

## Validation (`entropy/population/validator/`)

Two layers for population specs:
- **Structural** (`structural.py`): ERROR-level — type/modifier compatibility, range violations, distribution params, dependency cycles, condition syntax, formula references, duplicates, strategy consistency
- **Semantic** (`semantic.py`): WARNING-level — no-op detection, modifier stacking, categorical option reference validity

Scenario validation (`entropy/scenario/validator.py`): attribute reference validity, edge type references, probability ranges.

---

## Config (`entropy/config.py`)

`EntropyConfig` with `PipelineConfig` and `SimZoneConfig` zones. Resolution order: programmatic > env vars > config file (`~/.config/entropy/config.json`) > defaults. CLI flags override at command level before reaching config.

**`PipelineConfig`** fields: `provider` (default: `"openai"`), `model_simple`, `model_reasoning`, `model_research` (all default: `""` = provider default).

**`SimZoneConfig`** fields: `provider` (default: `"openai"`), `model`, `pivotal_model`, `routine_model` (all default: `""` = provider default), `max_concurrent` (default: `50`), `rate_tier` (default: `None` = Tier 1), `rpm_override`, `tpm_override` (default: `None`), `api_format` (default: `""` = auto, supports `"responses"` for OpenAI or `"chat_completions"` for Azure).

**`EntropyConfig`** non-zone fields: `db_path` (default: `"./storage/entropy.db"`), `default_population_size` (default: `1000`).

API keys always from env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`. Azure also requires `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` (default: `"2025-03-01-preview"`), `AZURE_OPENAI_DEPLOYMENT`.

Three providers supported: `openai`, `claude`, `azure_openai`.

For package use: `from entropy.config import configure, EntropyConfig`.

---

## File Formats

| Format | Files | Notes |
|--------|-------|-------|
| YAML | Population specs, scenario specs, persona configs, network configs | Human-readable, version-controllable |
| JSON | Agents, networks, simulation results | Array of objects (`_id` field), streaming-friendly |
| SQLite | Simulation state | Tables: `agent_states`, `exposures`, `memory_traces`, `timeline`, `timestep_summaries`, `shared_to`, `simulation_metadata` |
| JSONL | Timeline | Streaming, crash-safe event log |

---

## Tests

pytest + pytest-asyncio. Fixtures in `tests/conftest.py` include seeded RNG (`Random(42)`), minimal/complex population specs, sample agents, network topologies (linear chain, star graph), and distribution fixtures. 580+ tests across 24 test files:

- `test_models.py`, `test_network.py`, `test_sampler.py`, `test_scenario.py`, `test_simulation.py`, `test_validator.py` — core logic
- `test_engine.py` — mock-based engine integration (seed exposure, flip resistance, conviction-gated sharing, chunked reasoning, checkpointing, resume logic, metadata lifecycle, progress state wiring)
- `test_conviction.py` — conviction bucketing, level mapping, map consistency
- `test_propagation.py` — seed exposure, network propagation, share probability, channel credibility
- `test_stopping.py` — stopping condition parsing, evaluation, convergence detection, compound conditions
- `test_memory_traces.py` — memory trace sliding window, multi-touch reasoning triggers, state aggregations
- `test_reasoning_prompts.py` — prompt construction, schema generation, sentiment tone mapping
- `test_integration_timestep.py` — full timestep loop with mocked LLM, multi-timestep dynamics
- `test_progress.py` — SimulationProgress thread-safe state (begin_timestep, record_agent_done, position counts, snapshot isolation)
- `test_compiler.py` — scenario compiler pipeline with mocked LLM calls, auto-configuration
- `test_providers.py` — provider response extraction, transient error retry, validation-retry exhaustion, source URL extraction (mocked HTTP)
- `test_rate_limiter.py` — token bucket, dual-bucket rate limiter, `for_provider` factory
- `test_estimator.py` — cost estimation, pricing lookup, token estimation
- `test_cli.py` — CLI smoke tests (`config show/set`, `validate`, `--version`)
- `test_scenario_validator.py` — scenario-specific validation rules
- `test_network_config_generator.py` — LLM-generated network config
- `test_paths.py` — path utilities

CI: `.github/workflows/test.yml` — lint (ruff check + format) and test (pytest, matrix: Python 3.11/3.12/3.13) via `astral-sh/setup-uv@v4`. Triggers on push/PR to `main`/`dev`.
