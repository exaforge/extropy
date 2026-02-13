# CLAUDE.md

This file gives practical guidance to coding agents working in this repository.

## What Extropy Is

Extropy is a Python framework + CLI for agent-based population simulation.

It:
- builds synthetic populations from statistically grounded attribute distributions,
- adds scenario-relevant psychographic/context attributes,
- connects sampled agents in a social network,
- simulates multi-timestep opinion/behavior updates with two-pass LLM reasoning,
- outputs segmented distributional results.

## Core CLI Pipeline

```bash
extropy spec -> extropy extend -> extropy sample -> extropy network -> extropy persona -> extropy scenario -> extropy simulate
                                                                                                          |              |
                                                                                                   extropy estimate    extropy results
```

Utility commands: `extropy validate`, `extropy config`.

CLI entry point: `extropy` (from `pyproject.toml`), Python `>=3.11`.

## Local Dev Commands

```bash
pip install -e ".[dev]"

# tests / lint
pytest
ruff check .
ruff format .

# config
extropy config show
extropy config set pipeline.provider claude
extropy config set simulation.provider openai
```

## High-Level Architecture

Three phases map to top-level packages:

1. Population creation (`extropy/population/`)
- Sufficiency check (`spec_builder/sufficiency.py`)
- Attribute selection (`spec_builder/selector.py`)
- Hydration (`spec_builder/hydrator.py`, `hydrators/`)
- Dependency binding (`spec_builder/binder.py`)
- Sampling (`sampler/core.py`)
- Network generation (`network/`)

2. Scenario compilation (`extropy/scenario/`)
- Compiler (`compiler.py`) orchestrates parse -> exposure -> interaction -> outcomes -> assembly
- Event types in code: `announcement`, `news`, `rumor`, `policy_change`, `product_launch`, `emergency`, `observation`
- Outcomes: `categorical`, `boolean`, `float`, `open_ended`

3. Simulation (`extropy/simulation/`)
- Engine (`engine.py`) loops over timestep phases
- Seed + network exposure propagation
- Two-pass reasoning (role-play then classification)
- Memory traces (sliding window)
- Conviction-aware sharing + flip resistance
- Checkpoint/resume via `simulation_metadata`

## LLM Integration

All LLM calls go through `extropy/core/llm.py`.

Two-zone routing:
- Pipeline zone (phases 1-2): `simple_call`, `reasoning_call`, `agentic_research`
- Simulation zone (phase 3): `simple_call_async`

Providers are created via `extropy/core/providers/`:
- `openai`
- `claude`
- `azure_openai`

Use provider factories (`get_pipeline_provider`, `get_simulation_provider`) rather than provider-specific calls in feature code.

## Configuration

Config lives in `extropy/config.py` and `~/.config/extropy/config.json`.

Resolution order (highest first):
1. Programmatic config (`configure(...)` / constructed `ExtropyConfig`)
2. Environment variables
3. Config file
4. Dataclass defaults

Defaults in code:
- `pipeline.provider = "openai"`
- `simulation.provider = "openai"`

API keys are env-only:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AZURE_OPENAI_API_KEY`

Azure extras:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION` (defaults to `2025-03-01-preview`)
- `AZURE_OPENAI_DEPLOYMENT`

## Data Models

Primary Pydantic models are under `extropy/core/models/`:
- `population.py`: `PopulationSpec`, attribute/distribution/sampling models
- `scenario.py`: `ScenarioSpec`, event/exposure/interaction/outcomes/simulation config
- `simulation.py`: `AgentState`, `ReasoningContext`, `ReasoningResponse`, conviction maps, timestep summaries
- `network.py`, `results.py`, `validation.py`, `sampling.py`

YAML I/O helpers exist on core spec/config models (`to_yaml` / `from_yaml`).

## Validation

Population validation (`extropy/population/validator/`):
- Structural: hard errors
- Semantic: warnings

Scenario validation: `extropy/scenario/validator.py`.

## Conventions

- Use `_id` from agent JSON as primary ID (fallback to index string).
- Network edges are traversed bidirectionally during simulation.
- Keep LLM reasoning pass semantic; classification pass handles structured outcome extraction.
- Prefer data-driven network behavior through `NetworkConfig` over hardcoded social rules.

## Testing & CI

Tests are in `tests/` and run with `pytest`.

CI (`.github/workflows/test.yml`) runs:
- lint: `ruff check`, `ruff format --check`
- tests on Python 3.11/3.12/3.13 (via `uv`)

## File Formats

- YAML: population/spec/network/persona/scenario configs
- JSON: sampled agents, network, result artifacts
- SQLite: simulation state/checkpoints
- JSONL: timeline events
