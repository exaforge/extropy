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

### 6. Network Generation (`network/generator.py`)

Hybrid algorithm: similarity-based edge probability with degree correction, calibrated via binary search to hit target avg_degree, then Watts-Strogatz rewiring (5%) for small-world properties.

Edge probability: `base_rate * sigmoid(similarity) * degree_factor_a * degree_factor_b`.

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

Per-timestep loop:

1. Apply seed exposures from scenario rules (`propagation.py`)
2. Propagate through network — agents with `will_share=True` spread to neighbors
3. Select agents to reason — first exposure OR multi-touch threshold exceeded (default: 3 new exposures since last reasoning)
4. **Two-pass async LLM reasoning** (`reasoning.py`):
   - **Pass 1** (pivotal model): Role-play prompt with persona + event + exposure history + peer statements. Freeform response — no enum labels, no anchoring. Produces natural language reaction, sentiment, conviction level, public statement, and sharing intent.
   - **Pass 2** (routine model): Classification prompt that takes the Pass 1 narrative and maps it to the scenario's outcome categories. Cheap and fast.
   - This two-pass approach eliminates the central tendency bias that plagues single-pass structured extraction.
5. Update state (`state.py`) — SQLite-backed with indexed tables for agent_states, exposures, timeline
6. Check stopping conditions (`stopping.py`) — Compound conditions like `"exposure_rate > 0.95 and no_state_changes_for > 10"`, convergence detection via position distribution variance

### Conviction System

Agents have categorical conviction levels rather than arbitrary floats:

| Level | Float | Meaning |
|-------|-------|---------|
| `very_uncertain` | 0.1 | Barely formed opinion |
| `leaning` | 0.3 | Tentative position |
| `moderate` | 0.5 | Reasonably confident |
| `firm` | 0.7 | Strong position |
| `absolute` | 0.9 | Unwavering |

The LLM picks from these labels. Agents never see numbers — only the categorical descriptions.

### Memory

Each agent maintains a 3-entry sliding window memory trace. Memory entries include the timestep, a summary of what they processed, and how it affected their thinking. This gives agents continuity across reasoning rounds without unbounded context growth.

### Persona System

`population/persona/` + `simulation/persona.py`: The `entropy persona` command generates a `PersonaConfig` via 5-step LLM pipeline (structure -> boolean -> categorical -> relative -> concrete phrasings). At simulation time, agents are rendered computationally using this config — no per-agent LLM calls.

Relative attributes (personality, attitudes) are positioned against population stats via z-scores ("I'm much more price-sensitive than most people"). Concrete attributes use format specs for proper number/time rendering.

**Trait salience**: When the scenario defines `decision_relevant_attributes`, those traits are grouped first in the persona under "Most Relevant to This Decision", ensuring the LLM focuses on what matters.

### Rate Limiting (`core/rate_limiter.py`)

Token bucket rate limiter with dual RPM + TPM buckets. Provider-aware defaults auto-configured from `core/rate_limits.py`. Supports tier overrides via config or CLI flags.

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

### Provider Abstraction (`entropy/core/providers/`)

`LLMProvider` base class with `OpenAIProvider` and `ClaudeProvider` implementations. Factory functions `get_pipeline_provider()` and `get_simulation_provider()` read from `EntropyConfig`.

All calls use structured output (`response_format: json_schema`). Failed validations are fed back as "PREVIOUS ATTEMPT FAILED" prompts for self-correction.

---

## Data Models (`entropy/core/models/`)

All Pydantic v2. Key hierarchy:

- `population.py`: `PopulationSpec` -> `AttributeSpec` -> `SamplingConfig` -> `Distribution` / `Modifier` / `Constraint`
- `scenario.py`: `ScenarioSpec` -> `Event`, `SeedExposure` (channels + rules), `InteractionConfig`, `SpreadConfig`, `OutcomeConfig`
- `simulation.py`: `AgentState`, `ConvictionLevel`, `MemoryEntry`, `ReasoningContext`, `ReasoningResponse`, `Pass1Response`, `Pass2Response`, `SimulationRunConfig`, `TimestepSummary`
- `network.py`: `Edge`, `NodeMetrics`, `NetworkMetrics`
- `validation.py`: `ValidationIssue`, `ValidationResult`

YAML serialization via `to_yaml()`/`from_yaml()` on `PopulationSpec` and `ScenarioSpec`.

---

## Validation (`entropy/population/validator/`)

Two layers for population specs:
- **Structural** (`structural.py`): ERROR-level — type/modifier compatibility, range violations, distribution params, dependency cycles, condition syntax, formula references, duplicates, strategy consistency
- **Semantic** (`semantic.py`): WARNING-level — no-op detection, modifier stacking, categorical option reference validity

Scenario validation (`entropy/scenario/validator.py`): attribute reference validity, edge type references, probability ranges.

---

## Config (`entropy/config.py`)

`EntropyConfig` with `PipelineConfig` and `SimZoneConfig` zones. Resolution order: CLI flags > env vars > config file (`~/.config/entropy/config.json`) > defaults.

API keys always from env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

`SimZoneConfig` fields: `provider`, `model`, `pivotal_model`, `routine_model`, `max_concurrent`, `rate_tier`, `rpm_override`, `tpm_override`.

---

## File Formats

| Format | Files | Notes |
|--------|-------|-------|
| YAML | Population specs, scenario specs, persona configs | Human-readable, version-controllable |
| JSON | Agents, networks, simulation results | Array of objects (`_id` field), streaming-friendly |
| SQLite | Simulation state | Indexed tables for agent_states, exposures, timeline |
| JSONL | Timeline | Streaming, crash-safe event log |

---

## Tests

pytest + pytest-asyncio. Fixtures in `tests/conftest.py` include seeded RNG (`Random(42)`), minimal/complex population specs, and sample agents.
