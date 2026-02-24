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

<p align="center">
  <a href="https://extropy.run">Website</a> ·
  <a href="https://extropy.run/blog/extropy-01">Announcement</a> ·
  <a href="docs/commands.md">CLI Reference</a> ·
  <a href="docs/architecture.md">Architecture</a>
</p>

---

Extropy creates synthetic populations grounded in real-world distributions, connects them in social networks, and simulates how they respond to events — each agent reasoning individually via LLM.

**Simulate anything:** Policy changes. Pricing decisions. Product launches. Crisis response. Any scenario where humans form opinions, make decisions, and influence each other.

## Install

```bash
pip install extropy-run
export OPENAI_API_KEY=sk-...
# or: ANTHROPIC_API_KEY=... / AZURE_OPENAI_API_KEY=...
```

Requires Python 3.11+. [uv](https://github.com/astral-sh/uv) recommended.

## Quick Start

```bash
# Create study folder and build population spec
extropy spec "Austin TX commuters" -o congestion-tax -y
cd congestion-tax

# Create scenario with events and outcomes
extropy scenario "Response to $15/day congestion tax" -o congestion-tax -y
extropy persona -s congestion-tax -y

# Sample agents and generate network
extropy sample -s congestion-tax -n 500 --seed 42
extropy network -s congestion-tax --seed 42

# Run simulation
extropy simulate -s congestion-tax --seed 42

# View results
extropy results
extropy results segment income
```

## How It Works

1. **Population** — LLM discovers attributes, researches real-world distributions, samples agents
2. **Network** — Builds structural + similarity edges; edge types affect information flow
3. **Two-pass reasoning** — Agent role-plays reaction, then classifier extracts outcomes
4. **Propagation** — Opinions spread through network; agents update after hearing from peers

## Features

| Feature | Description |
|:--|:--|
| **Population** | |
| Any geography | US, Japan, India, Brazil — define attributes with your distributions |
| Real grounding | LLM researches actual demographics, cites sources |
| Household mode | Correlated partners, NPC dependents, assortative mating |
| Agent focus | Primary adult, couples, or full families as reasoning agents |
| **Network** | |
| Structural edges | Partner, household, coworker, neighbor, congregation, school parent |
| Similarity edges | Acquaintances and online contacts from attribute similarity |
| Small-world | Calibrated clustering coefficient and path lengths |
| **Simulation** | |
| Two-pass reasoning | Role-play first, classify second — reduces schema anchoring in outcome extraction |
| Conversations | Agents talk to each other; both update state independently |
| Memory | Persistent memory traces with fidelity-based prompt trimming |
| Conviction | Affects sharing probability and flip resistance |
| THINK vs SAY | Internal monologue separate from public statement |
| Timeline events | New information injected at specified timesteps |
| **Outcomes** | |
| Categorical | Known decision space (buy/wait/skip) |
| Boolean | Binary decisions (will share, will switch) |
| Float | Intensity measures (sentiment, likelihood) |
| Open-ended | Free text — discover categories post-hoc |

## Development

```bash
git clone https://github.com/exaforge/extropy.git && cd extropy
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
