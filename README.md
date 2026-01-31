# Entropy

Predictive intelligence through agent-based population simulation. Create synthetic populations grounded in real-world data, simulate how they respond to events, and watch opinions emerge through social networks.

Not a survey. Not a poll. A simulation of collective human behavior.

## What It Does

You describe a population and a scenario. Entropy builds statistically grounded synthetic agents, connects them in a social network, and has each one reason individually about the event using an LLM. Opinions form, spread through the network, and evolve — producing distributional predictions you can segment and analyze.

```
entropy spec → entropy extend → entropy sample → entropy network → entropy persona → entropy scenario → entropy simulate
                                                                                                               │
                                                                                                        entropy results
```

## Install

```bash
pip install entropy-predict
```

Or from source:

```bash
git clone https://github.com/exaforge/entropy.git
cd entropy
pip install -e ".[dev]"
```

## Setup

```bash
# API keys (in .env or exported)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Configure providers
entropy config set pipeline.provider claude      # Claude for population/scenario building
entropy config set simulation.provider openai    # OpenAI for agent reasoning
entropy config show
```

## Quick Start

```bash
# Build a population
entropy spec "500 Austin TX commuters who drive into downtown for work" -o austin/base.yaml
entropy extend austin/base.yaml -s "Response to a $15/day downtown congestion tax" -o austin/population.yaml
entropy sample austin/population.yaml -o austin/agents.json --seed 42
entropy network austin/agents.json -o austin/network.json --seed 42
entropy persona austin/population.yaml --agents austin/agents.json

# Compile and run a scenario
entropy scenario -p austin/population.yaml -a austin/agents.json -n austin/network.json -o austin/scenario.yaml
entropy simulate austin/scenario.yaml -o austin/results/ --seed 42

# View results
entropy results austin/results/
entropy results austin/results/ --segment income
```

### What Comes Out

```
Outcome Distributions:
  commute_response:
    drive_and_pay          44%  ████████████████░░░░
    switch_to_transit      21%  ████████░░░░░░░░░░░░
    shift_schedule         21%  ████████░░░░░░░░░░░░
    undecided               7%  ██░░░░░░░░░░░░░░░░░░
    telework_more           3%  █░░░░░░░░░░░░░░░░░░░

  sentiment: mean -0.12 (std 0.37)
  conviction: mean 0.61
```

Each agent reasoned individually. A low-income commuter with no transit access reacts differently than a tech worker near a rail stop — not because we scripted it, but because their attributes, persona, and social context led them there.

## How It Works

**Population creation** — An LLM discovers relevant attributes (demographics, psychographics, scenario-specific), then researches real-world distributions with citations. Agents are sampled from these distributions respecting all dependencies. A social network connects them based on attribute similarity with small-world properties.

**Persona rendering** — Each agent gets a first-person narrative built from their attributes. Relative traits are positioned against population statistics ("I'm much more price-sensitive than most people"). Generated once per population, applied computationally per agent.

**Two-pass reasoning** — Pass 1: the agent role-plays their reaction in natural language (no enum labels, no anchoring). Pass 2: a cheap model classifies the freeform response into outcome categories. This eliminates the central tendency bias that plagues single-pass structured extraction.

**Network propagation** — Agents share information through social connections. Edge types, spread modifiers, and decay control how opinions travel. Multi-touch re-reasoning lets agents update their position after hearing from multiple peers.

## Documentation

- **[CLI Reference](docs/commands.md)** — Every command with arguments, options, and examples
- **[Architecture](docs/architecture.md)** — How the system works under the hood

## Development

```bash
pip install -e ".[dev]"
pytest                    # Run tests
ruff check .              # Lint
ruff format .             # Format
```

## License

MIT
