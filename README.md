# Extropy

[![CI](https://github.com/exaforge/extropy/actions/workflows/test.yml/badge.svg)](https://github.com/exaforge/extropy/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/extropy.svg)](https://pypi.org/project/extropy/)
[![Python](https://img.shields.io/pypi/pyversions/extropy.svg)](https://pypi.org/project/extropy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predictive intelligence through agent-based population simulation. Create synthetic populations grounded in real-world data, simulate how they respond to events, and watch opinions emerge through social networks.

Not a survey. Not a poll. A simulation of collective human behavior.

## What It Does

You describe a population and a scenario. Extropy builds statistically grounded synthetic agents, connects them in a social network, and has each one reason individually about the event using an LLM. Opinions form, spread through the network, and evolve — producing distributional predictions you can segment and analyze.

```
extropy spec → extropy extend → extropy sample → extropy network → extropy persona → extropy scenario → extropy simulate
                                                                                                  │              │
                                                                                           extropy estimate    extropy results
```

## Install

```bash
pip install extropy
```

Or from source:

```bash
git clone https://github.com/exaforge/extropy.git
cd extropy
pip install -e ".[dev]"
```

## Setup

```bash
# API keys (in .env or exported)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Or for Azure OpenAI:
# export AZURE_OPENAI_API_KEY=...
# export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
# export AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Configure providers (openai, claude, or azure_openai)
extropy config set pipeline.provider claude      # Claude for population/scenario building
extropy config set simulation.provider openai    # OpenAI for agent reasoning
extropy config show
```

## Quick Start

```bash
# Build a population
extropy spec "500 Austin TX commuters who drive into downtown for work" -o austin/base.yaml
extropy extend austin/base.yaml -s "Response to a $15/day downtown congestion tax" -o austin/population.yaml
extropy sample austin/population.yaml -o austin/agents.json --seed 42
extropy network austin/agents.json -o austin/network.json -p austin/population.yaml --seed 42
extropy persona austin/population.yaml --agents austin/agents.json

# Compile and run a scenario
extropy scenario -p austin/population.yaml -a austin/agents.json -n austin/network.json -o austin/scenario.yaml
extropy simulate austin/scenario.yaml -o austin/results/ --seed 42

# View results
extropy results austin/results/
extropy results austin/results/ --segment income
```

### What Comes Out

Outcomes are defined per-scenario — categorical, float, boolean, or open-ended. You choose what to measure.

```
═══════════════════════════════════════════════════════════
SIMULATION RESULTS: austin_congestion_tax
═══════════════════════════════════════════════════════════

Population: 500 agents | Duration: 47 timesteps | Model: gpt-5
Stopped: exposure_rate > 0.95 and no_state_changes_for > 5

EXPOSURE
────────────────────────────────────────
Final exposure rate: 96.8%
Reasoning calls: 1,847
Average conviction: 0.64 (moderate-to-firm)

OUTCOMES
────────────────────────────────────────
commute_response (categorical):
  drive_and_pay          38%  ███████████████░░░░░
  switch_to_transit      24%  █████████░░░░░░░░░░░
  shift_schedule         19%  ███████░░░░░░░░░░░░░
  telework_more          12%  ████░░░░░░░░░░░░░░░░
  undecided               7%  ██░░░░░░░░░░░░░░░░░░

sentiment (float, -1 to 1):
  mean: -0.18  std: 0.41  min: -0.9  max: 0.7

willingness_to_pay (boolean):
  yes: 42%  no: 58%

protest_likelihood (float, 0 to 1):
  mean: 0.31  std: 0.28

SEGMENT: income
────────────────────────────────────────
< $50k:   drive_and_pay 22% | switch_to_transit 14% | protest 41%
$50-100k: drive_and_pay 40% | switch_to_transit 28% | shift_schedule 21%
> $100k:  drive_and_pay 51% | switch_to_transit 31% | telework_more 14%
```

Each agent reasoned individually. A low-income commuter with no transit access reacts differently than a tech worker near a rail stop — not because we scripted it, but because their attributes, persona, and social context led them there.

The scenario YAML controls what gets tracked:

```yaml
outcomes:
  suggested_outcomes:
  - name: commute_response
    type: categorical
    options: [drive_and_pay, switch_to_transit, shift_schedule, telework_more, undecided]
  - name: sentiment
    type: float
    range: [-1.0, 1.0]
  - name: willingness_to_pay
    type: boolean
  - name: protest_likelihood
    type: float
    range: [0.0, 1.0]
```

## How It Works

**Population creation** — An LLM discovers relevant attributes (demographics, psychographics, scenario-specific), then researches real-world distributions with citations. Agents are sampled from these distributions respecting all dependencies. A social network connects them based on attribute similarity with small-world properties — the network config (which attributes matter, what edge types exist, who influences whom) is generated by the LLM from the population spec.

**Persona rendering** — Each agent gets a first-person narrative built from their attributes. Relative traits are positioned against population statistics ("I'm much more price-sensitive than most people"). Generated once per population, applied computationally per agent.

**Two-pass reasoning** — Pass 1: the agent role-plays their reaction in natural language (no enum labels, no anchoring). Pass 2: a cheap model classifies the freeform response into outcome categories. This eliminates the central tendency bias that plagues single-pass structured extraction.

**Network propagation** — Agents share information through social connections. Edge types, spread modifiers, and decay control how opinions travel. Multi-touch re-reasoning lets agents update their position after hearing from multiple peers.

## Documentation

- **[CLI Reference](docs/commands.md)** — Every command with arguments, options, and examples
- **[Architecture](docs/architecture.md)** — How the system works under the hood
- **[Use Cases](docs/use-cases.md)** — What Extropy can simulate, with examples

## Development

```bash
pip install -e ".[dev]"
pytest                    # Run tests
ruff check .              # Lint
ruff format .             # Format
```

## License

[MIT](LICENSE)
