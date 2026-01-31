# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Entropy Is

Entropy is a predictive intelligence framework that simulates how real human populations respond to scenarios. It creates synthetic populations grounded in real-world statistical data, enriches them with LLM-extrapolated psychographic attributes, connects them via social networks, and runs agent-based simulations where each agent reasons individually via LLM calls. The output is not a poll — it's a simulation of emergent collective behavior.

Competitor reference: [Aaru](https://aaru.com) operates in the same space (multi-agent population simulation for predictive intelligence). Entropy differentiates through its grounding pipeline — every attribute distribution is researched from real-world sources with citations, not just LLM-generated.

## Commands

```bash
pip install -e ".[dev]"      # Install with dev deps
cp .env.example .env         # Set OPENAI_API_KEY

pytest                       # Run all tests
pytest tests/test_sampler.py # Single test file
pytest -k "test_name"        # Single test by name

ruff check .                 # Lint
ruff format .                # Format
```

CLI entry point: `entropy` (defined in `pyproject.toml` → `entropy.cli:app`). Python >=3.11.

## Pipeline (6 sequential commands)

```
entropy spec → entropy extend → entropy sample → entropy network → entropy scenario → entropy simulate
                                                                                            │
                                                                                     entropy results
```

Each command produces a file consumed by the next. `entropy validate` is a utility runnable at any point. `entropy results` is a viewer for simulation output.

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

6. **Network generation** (`network/generator.py`) — Hybrid algorithm: similarity-based edge probability with degree correction, calibrated via binary search to hit target avg_degree, then Watts-Strogatz rewiring (5%) for small-world properties. Edge probability: `base_rate * sigmoid(similarity) * degree_factor_a * degree_factor_b`.

### Phase 2: Scenario Compilation (`entropy/scenario/`)

**Compiler** (`compiler.py`) orchestrates 5 steps: parse event → generate exposure rules → determine interaction model → define outcomes → assemble spec.

- **Event types**: product_launch, policy_change, pricing_change, technology_release, organizational_change, market_event, crisis_event
- **Exposure channels**: broadcast, targeted, organic — with per-timestep rules containing conditions and probabilities
- **Outcomes**: categorical (enum options), boolean, float (with range), open_ended
- Auto-configures simulation parameters based on population size (<500: 50 timesteps, ≤5000: 100, >5000: 168)

### Phase 3: Simulation (`entropy/simulation/`)

**Engine** (`engine.py`) runs per-timestep loop:
1. Apply seed exposures from scenario rules (`propagation.py`)
2. Propagate through network — agents with `will_share=True` spread to neighbors
3. Select agents to reason — first exposure OR multi-touch threshold exceeded (default: 3 new exposures since last reasoning)
4. **Batch async LLM reasoning** (`reasoning.py`) — Semaphore-controlled (50 concurrent), each agent gets a first-person prompt with their persona + event + exposure history + peer opinions. Response schema is dynamically built from outcome definitions.
5. Update state (`state.py`) — SQLite-backed with indexed tables for agent_states, exposures, timeline
6. Check stopping conditions (`stopping.py`) — Compound conditions like `"exposure_rate > 0.95 and no_state_changes_for > 10"`, convergence detection via position distribution variance

**Persona system** (`persona.py` + `spec_builder/persona_template.py`): Hybrid narrative template intro (Jinja2-like, LLM-generated) + structured characteristics list grouped by category with personality/attitudes first.

## LLM Integration (`entropy/core/llm.py`)

All LLM calls go through this file — never call OpenAI directly elsewhere. Three tiers:

| Function | Model | Tools | Use |
|----------|-------|-------|-----|
| `simple_call()` | gpt-5-mini | none | Sufficiency checks, simple extractions |
| `reasoning_call()` | gpt-5 | none | Attribute selection, hydration, scenario compilation. Supports validator callback + retry |
| `agentic_research()` | gpt-5 | web_search | Distribution hydration with real-world data. Extracts source URLs |
| `simple_call_async()` | configurable | none | Batch simulation reasoning (async) |

All calls use structured output (`response_format: json_schema`). Failed validations are fed back as "PREVIOUS ATTEMPT FAILED" prompts for self-correction.

## Data Models (`entropy/core/models/`)

All Pydantic v2. Key hierarchy:

- `population.py`: `PopulationSpec` → `AttributeSpec` → `SamplingConfig` → `Distribution` / `Modifier` / `Constraint`
- `scenario.py`: `ScenarioSpec` → `Event`, `SeedExposure` (channels + rules), `InteractionConfig`, `SpreadConfig`, `OutcomeConfig`
- `simulation.py`: `AgentState`, `ReasoningContext`, `ReasoningResponse`, `SimulationRunConfig`, `TimestepSummary`
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
- "Position" = first required categorical outcome (used for aggregation and peer influence display)
- The `extend` command (not `overlay` — recently renamed) is where persona templates are generated, not `spec`
- Network config defaults in `network/config.py` are currently hardcoded for the German surgeons example and need generalization

## Tests

pytest + pytest-asyncio. Fixtures in `tests/conftest.py` include seeded RNG (`Random(42)`), minimal/complex population specs, and sample agents. Six test files covering models, network, sampler, scenario, simulation, validator.

## File Formats

- Population/scenario specs: YAML
- Agents: JSON (array of objects with `_id`)
- Network: JSON (`{meta, nodes, edges}`)
- Simulation state: SQLite
- Timeline: JSONL (streaming, crash-safe)
- Results: JSON files in output directory
