<p align="center">
  <img src="assets/logo.png" alt="Extropy" width="600">
</p>

<p align="center">
  <a href="https://github.com/exaforge/extropy/actions/workflows/test.yml"><img src="https://github.com/exaforge/extropy/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/extropy-run/"><img src="https://img.shields.io/pypi/v/extropy-run.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/extropy-run/"><img src="https://img.shields.io/pypi/pyversions/extropy-run.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

<p align="center">
  <strong>Predictive intelligence through agent-based population simulation.</strong>
</p>

Extropy creates synthetic populations grounded in real-world distributions, connects them in social networks, and simulates how they respond to events — each agent reasoning individually via LLM. Opinions form, spread, and evolve. Get distributional predictions by segment with reasoning traces explaining why.

**Simulate anything:** Netflix price hikes. Policy changes. Product launches. Crisis response. ASI breakout. Alien contact. Any scenario where humans form opinions, make decisions, and influence each other.

## Install

```bash
pip install extropy-run
```

Requires Python 3.11+. Set your API key:

```bash
export OPENAI_API_KEY=sk-...      # or ANTHROPIC_API_KEY
```

## Quick Start

```bash
STUDY=runs/congestion-tax
DB=$STUDY/study.db
mkdir -p $STUDY

# Build population
extropy spec "500 Austin TX commuters" -o $STUDY/base.yaml -y
extropy extend $STUDY/base.yaml -s "Response to $15/day congestion tax" -o $STUDY/population.yaml -y
extropy sample $STUDY/population.yaml --study-db $DB --seed 42
extropy network --study-db $DB -p $STUDY/population.yaml --seed 42

# Compile and run
extropy scenario -p $STUDY/population.yaml --study-db $DB -o $STUDY/scenario.yaml -y
extropy simulate $STUDY/scenario.yaml --study-db $DB -o $STUDY/results --seed 42

# Results
extropy results --study-db $DB
extropy results --study-db $DB --segment income
```

## What You Get

```
Simulation Summary
Agents: 500
Aware: 484 (96.8%)
Positions:
  - drive_and_pay: 190 (38.0%)
  - switch_to_transit: 120 (24.0%)
  - shift_schedule: 95 (19.0%)
  - telework_more: 60 (12.0%)
  - undecided: 19 (3.8%)
```

Each agent reasoned individually. Low-income commuters with no transit access react differently than tech workers near rail — not scripted, but emergent from their attributes and social context.

Export full data for deeper analysis:
```bash
extropy export states --study-db $DB --to states.jsonl
extropy query sql --study-db $DB --sql "SELECT * FROM agent_states" --format json
```

## How It Works

1. **Population** — LLM discovers attributes, researches real-world distributions with citations, samples agents respecting dependencies
2. **Network** — Connects agents by attribute similarity with small-world properties; edge types (coworker, neighbor, partner) affect information flow
3. **Two-pass reasoning** — Pass 1: agent role-plays reaction in natural language. Pass 2: cheap model classifies into outcomes. Eliminates central tendency bias
4. **Propagation** — Opinions spread through network; multi-touch re-reasoning lets agents update after hearing from peers

## Features

- **Household mode** — Family units with correlated partners and NPC dependents
- **Fidelity tiers** — Low/medium/high controlling conversations, memory depth, cognitive features
- **Conversations** — Agents talk to each other, both update state independently
- **Timeline events** — Evolving scenarios with new information at specified timesteps
- **Open-ended outcomes** — Let agents tell you what they'd do; discover categories post-hoc

## Documentation

- **[CLI Reference](docs/commands.md)** — All commands and flags
- **[Architecture](docs/architecture.md)** — System internals

## Development

```bash
git clone https://github.com/exaforge/extropy.git
cd extropy
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
