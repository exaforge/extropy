# Architecture

Extropy has three phases, each mapping to a package under `extropy/`.

## CLI Pipeline

```
extropy spec → extropy scenario → extropy persona → extropy sample → extropy network → extropy simulate → extropy results
```

All commands operate within a **study folder** — a directory containing `study.db` and scenario subdirectories. Data is keyed by `scenario_id` rather than `population_id`.

---

## Phase 1: Population Creation (`extropy/population/`)

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

Four sub-steps, each using different LLM tiers:

- **Independent** (`hydrators/independent.py`) — `agentic_research()` with web search finds real-world distributions with source URLs
- **Derived** (`hydrators/derived.py`) — `reasoning_call()` specifies deterministic formulas (e.g., `years_experience = age - 26`)
- **Conditional base** (`hydrators/conditional.py`) — `agentic_research()` finds base distributions for dependent attributes
- **Conditional modifiers** (`hydrators/conditional.py`) — `reasoning_call()` specifies how values shift based on other attributes

### 4. Constraint Binding (`spec_builder/binder.py`)

Topological sort (Kahn's algorithm) resolves attribute dependencies into a valid sampling order.

### 5. Sampling (`sampler/core.py`)

Iterates through `sampling_order`, routing each attribute by strategy. Supports 6 distribution types: normal, lognormal, uniform, beta, categorical, boolean.

### 6. Household Sampling (`sampler/households.py`)

When `household_mode: true`:
- Sample primary adults first
- Generate correlated partners (shared attributes: location, income correlation)
- Generate NPC dependents (children, elderly) based on household type
- Household types: singles, couples, single parents, couples with kids, multi-generational
- `agent_focus` controls who reasons: `primary`, `couples`, `families`, `all`

### 7. Network Generation (`network/`)

Hybrid algorithm: similarity-based edge probability with degree correction, calibrated via simulated annealing to hit target metrics, then Watts-Strogatz rewiring for small-world properties.

**Edge types** (structural, from attributes):
- `partner` — from `partner_id` field (weight: 1.0)
- `household` — same `household_id` (weight: 0.9)
- `coworker` — same `occupation_category` + region (weight: 0.6)
- `neighbor` — same region + similar age (weight: 0.4)
- `congregation` — same religious affiliation + high religiosity (weight: 0.4)
- `school_parent` — both have school-age kids + same region (weight: 0.35)

Remaining degree filled with `acquaintance` or `online_contact` based on attribute similarity.

---

## Phase 2: Scenario Compilation (`extropy/scenario/`)

**Compiler** (`compiler.py`) orchestrates 5 steps: parse event -> generate exposure rules -> determine interaction model -> define outcomes -> assemble spec.

### Event Types

| Type | Examples |
|------|----------|
| `announcement` | Company policy, product feature, organizational change |
| `news` | Breaking news, industry report, research findings |
| `policy_change` | Government regulation, tax change, zoning decision |
| `product_launch` | New product, service update, feature rollout |
| `rumor` | Unconfirmed reports, speculation, leaked information |
| `emergency` | Crisis, natural disaster, security incident |
| `observation` | Behavioral change noticed in environment |

### Exposure Channels

| Channel | Reach | Trust |
|---------|-------|-------|
| `broadcast` | High, many agents | Varies |
| `targeted` | Filtered by attributes | Higher |
| `organic` | Network-dependent | High |

### Outcome Types

| Type | Schema | Use |
|------|--------|-----|
| `categorical` | Enum options | Known decision space |
| `boolean` | Yes/No | Binary decisions |
| `float` | Range [-1,1] or [0,1] | Intensity measures |
| `open_ended` | Free text | Unknown space, discover post-hoc |

### Timeline Mode

- **Static** — Single event, opinions evolve over time
- **Evolving** — Multiple events injected at specified timesteps

---

## Phase 3: Simulation (`extropy/simulation/`)

### Engine (`engine.py`)

Per-timestep loop with 5 phases:

#### 1. Apply Exposures (`_apply_exposures`)

Apply seed exposures, then timeline exposures (if a timeline event fires), then network propagation via conviction-gated sharing.

Timeline exposures stamp provenance metadata on each exposure:
- `info_epoch` (the originating timeline timestep)
- `force_rereason` (whether committed agents should re-reason for that epoch)

Network exposures inherit epoch provenance from the source agent, so downstream re-reasoning can be driven by provenance rather than content keyword matching.

#### 2. Agent Reasoning (`_reason_agents`)

Select agents to reason, split into chunks, run two-pass async LLM reasoning.

Selection gates:
- First-time aware agents always reason.
- Non-committed agents reason on multi-touch threshold.
- Committed agents can re-reason when exposed to a newer forced `info_epoch`.
- A de-dup guard ensures at most one re-reason per agent per epoch.

**Pass 1** (strong model): Agent role-plays in first person with no categorical enums. Produces:
- `reasoning` — internal monologue
- `public_statement` — what they'd say publicly
- `sentiment` — emotional valence
- `conviction` — 0-100 confidence score
- `will_share` — whether they'll discuss with others

**Pass 2** (fast model): Classify free-text reasoning into scenario-defined outcomes.

#### 3. Process Results (`_process_reasoning_chunk`)

- Bounded confidence opinion update
- Conviction-based flip resistance
- Private opinion tracking (separate from public)
- State persistence to DB

#### 4. Conviction Decay

Non-reasoning agents experience gradual conviction decay, preventing stale states.

#### 5. Stopping Check

Compound conditions: explicit stop conditions, timeline-aware convergence/quiescence, max timesteps.

- `max_timesteps` and explicit `stop_conditions` are always evaluated.
- Auto `converged` and auto `simulation_quiescent` are suppressed when future timeline events exist (unless overridden by `simulation.allow_early_convergence` or CLI `--early-convergence`).

### Fidelity Tiers

| Tier | Conversations | Memory | Cognitive Features | Cost/Agent |
|------|---------------|--------|-------------------|------------|
| `low` | None | Last 5 traces | Basic | ~$0.03 |
| `medium` | Top 1 edge (partner/closest) | All traces | Standard | ~$0.04 |
| `high` | Top 2-3 edges | All + beliefs | THINK vs SAY, repetition detection | ~$0.05 |

### Conversations (medium/high fidelity)

- Agents request `talk_to` actions during reasoning
- Multi-turn exchanges (2 turns at medium, 3 at high)
- Both participants update state independently
- Supports agent-NPC conversations (kids, elderly parents)

### THINK vs SAY (high fidelity)

Explicit separation between internal monologue (raw, honest) and public statement (socially filtered).

### Repetition Detection (high fidelity)

If reasoning is >70% similar to previous timestep, agent gets nudged to consider what's actually changed.

### Two-Pass Reasoning

Single-pass reasoning caused 83% of agents to pick safe middle options (central tendency bias). Splitting role-play from classification fixes this.

### Conviction System

Agents output a 0-100 integer score. Bucketed immediately:

| Score | Float | Level | Meaning |
|-------|-------|-------|---------|
| 0-15 | 0.1 | `very_uncertain` | Barely formed |
| 16-35 | 0.3 | `leaning` | Tentative |
| 36-60 | 0.5 | `moderate` | Reasonably confident |
| 61-85 | 0.7 | `firm` | Strong position |
| 86-100 | 0.9 | `absolute` | Unwavering |

### Semantic Peer Influence

Agents see neighbors' `public_statement` + sentiment tone, NOT position labels. Influence is semantic — agents are swayed by arguments, not categorical stances.

### Memory

Each agent maintains a sliding window memory trace (configurable by fidelity). Entries include timestep, summary of what they processed, and how it affected their thinking.

### Persona System (`population/persona/` + `simulation/persona.py`)

The `extropy persona` command generates a `PersonaConfig` via 5-step LLM pipeline. At simulation time, agents are rendered computationally — no per-agent LLM calls.

- **Relative attributes** positioned via z-scores ("I'm much more price-sensitive than most")
- **Concrete attributes** use format specs for proper number/time rendering
- **Trait salience** groups decision-relevant attributes first

### Checkpointing & Resume

Each phase commits separately. On crash:
1. Detect crashed-mid-timestep or last-completed-timestep
2. Skip already-processed agents
3. Resume from checkpoint

---

## LLM Integration (`extropy/core/llm.py`)

All LLM calls go through this file. Two-zone routing:

### Pipeline Zone (phases 1-2)

| Function | Use |
|----------|-----|
| `simple_call()` | Sufficiency checks, simple extractions |
| `reasoning_call()` | Attribute selection, hydration, scenario compilation |
| `agentic_research()` | Distribution hydration with web search |

### Simulation Zone (phase 3)

| Pass | Model | Use |
|------|-------|-----|
| Pass 1 | strong model | Agent role-play, freeform reaction |
| Pass 2 | fast model | Outcome extraction from narrative |

### Provider Abstraction (`extropy/core/providers/`)

`LLMProvider` base class with `OpenAIProvider`, `ClaudeProvider`, and Azure OpenAI support. All calls use structured output (`response_format: json_schema`).

---

## Data Models (`extropy/core/models/`)

All Pydantic v2:

- `population.py`: `PopulationSpec`, `AttributeSpec`, `SamplingConfig`, distributions, modifiers
- `scenario.py`: `ScenarioSpec`, `Event`, `SeedExposure`, `OutcomeConfig`, `ScenarioSimConfig`
- `simulation.py`: `AgentState` (public/private position/sentiment/conviction + info epochs), `ExposureRecord` (channel/source + provenance), `ReasoningContext`, `ReasoningResponse`
- `network.py`: `Edge`, `NetworkConfig`, `NetworkMetrics`

---

## Storage (`extropy/storage/`)

Canonical store: `study.db` (SQLite) in the study folder root.

### Study Folder Structure

```
my-study/
├── study.db                    # Canonical data store
├── population.v1.yaml          # Base population spec
├── scenario/
│   └── my-scenario/
│       ├── scenario.v1.yaml    # Scenario spec (references base_population)
│       ├── persona.v1.yaml     # Persona rendering config
│       └── network-config.yaml # Optional custom network config
└── results/
    └── my-scenario/            # Simulation outputs
```

### Tables

| Table | Contents | Key |
|-------|----------|-----|
| `agents` | Sampled agent attributes (JSON) | `scenario_id` |
| `network_edges` | Social graph edges with weights and types | `scenario_id` |
| `agent_states` | Current simulation state per agent | `run_id` |
| `exposures` | Exposure records with source/channel plus epoch provenance (`info_epoch`, `force_rereason`) | `run_id` |
| `memory_traces` | Agent memory entries | `run_id` |
| `timeline` | Simulation events (JSONL-style) | `run_id` |
| `timestep_summaries` | Per-timestep aggregates | `run_id` |
| `simulation_runs` | Run metadata and status | `run_id` |
| `simulation_metadata` | Checkpoint state | `run_id` |
| `chat_sessions` | Post-sim agent chat sessions | `session_id` |

### Scenario-Centric Keying

Agents and network edges are keyed by `scenario_id`, not `population_id`. This allows:
- Multiple scenarios to share a base population spec
- Each scenario to have its own extended attributes merged at sample time
- Clear association between agents/network and their scenario context

---

## Configuration (`extropy/config.py`)

Resolution order: programmatic > env vars > config file > defaults

### CLI Zone

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `human` | `human` = interactive prompts, rich output. `agent` = JSON output, exit codes, no prompts |

### Models Zone

| Field | Default | Description |
|-------|---------|-------------|
| `fast` | provider default | Fast model for pipeline |
| `strong` | provider default | Strong model for pipeline |

### Simulation Zone

| Field | Default | Description |
|-------|---------|-------------|
| `fast` | `= models.fast` | Fast model for Pass 2 |
| `strong` | `= models.strong` | Strong model for Pass 1 |
| `max_concurrent` | 50 | Max concurrent LLM calls |
| `rate_tier` | 1 | Provider rate limit tier |

### Providers

Built-in: `openai`, `anthropic`, `azure`, `openrouter`, `deepseek`

Custom providers via `providers.<name>.base_url` and `providers.<name>.api_key_env`.

---

## File Formats

| Format | Files |
|--------|-------|
| YAML | Population specs, scenario specs, persona configs, network configs |
| SQLite | `study.db` — canonical simulation state |
| JSON | Result exports, legacy artifacts |
| JSONL | Timeline events, data exports |

---

## Tests

pytest + pytest-asyncio. Key coverage:

- `test_engine.py` — engine integration with mocked LLM
- `test_conviction.py` — conviction bucketing and level mapping
- `test_propagation.py` — exposure propagation and sharing
- `test_stopping.py` — stopping conditions and convergence
- `test_memory_traces.py` — memory window and multi-touch triggers
- `test_household_sampling.py` — household sampling and partner correlation
- `test_conversations.py` — multi-turn conversation dynamics

CI: `.github/workflows/test.yml` — lint (ruff) + test (Python 3.11/3.12/3.13)
