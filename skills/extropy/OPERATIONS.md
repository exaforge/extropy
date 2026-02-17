# Operations

End-to-end execution guide for Extropy simulations.

## Pipeline Overview

```
extropy spec -> extend -> sample -> network -> persona -> scenario -> simulate -> results
```

All artifacts flow through a **study database** (`study.db`). This is the canonical store for agents, networks, simulation state, and results.

## Quick Start (Minimal Pipeline)

```bash
STUDY=runs/my-study
DB=$STUDY/study.db
mkdir -p $STUDY

# 1. Build population
extropy spec "500 Austin TX commuters" -o $STUDY/base.yaml
extropy extend $STUDY/base.yaml -s "Response to $15/day congestion tax" -o $STUDY/population.yaml
extropy sample $STUDY/population.yaml --study-db $DB --seed 42

# 2. Build network
extropy network --study-db $DB -p $STUDY/population.yaml --seed 42

# 3. Compile scenario
extropy scenario -p $STUDY/population.yaml --study-db $DB -o $STUDY/scenario.yaml

# 4. Estimate cost, then run
extropy estimate $STUDY/scenario.yaml --study-db $DB
extropy simulate $STUDY/scenario.yaml --study-db $DB -o $STUDY/results --seed 42

# 5. View results
extropy results --study-db $DB
extropy results --study-db $DB --segment income
```

## Command Reference

### `extropy spec`

Builds a base population specification from natural language.

```bash
extropy spec "<population description>" -o <output.yaml> [--yes]
```

- Discovers 25-40 relevant attributes (demographics, psychographics, behavioral)
- Researches real-world distributions with citations
- Pauses for confirmation unless `--yes`

### `extropy extend`

Adds scenario-specific attributes to a base spec.

```bash
extropy extend <base.yaml> -s "<scenario description>" -o <output.yaml> [--yes]
```

- Adds context-specific attitudes and behavioral attributes
- New attributes can depend on base attributes (creates correlations)

### `extropy sample`

Samples concrete agents from population distributions.

```bash
extropy sample <population.yaml> --study-db <db> [--count N] [--seed S] [--report]
```

Key flags:
- `--count` / `-n`: Number of agents to sample (required)
- `--seed`: Random seed for reproducibility (always set this)
- `--report` / `-r`: Show distribution summaries after sampling
- `--household-mode`: Enable household-based sampling (couples, dependents)

Household mode creates family units with:
- Correlated partner demographics (age, education, race via assortative mating)
- Shared economic attributes (household_income, savings)
- NPC dependents (kids with names, ages, school status)

### `extropy network`

Generates social network connecting agents.

```bash
extropy network --study-db <db> -p <population.yaml> [--quality-profile <profile>] [--seed S]
```

Key flags:
- `-p` / `--population`: Generate network config from population spec (recommended)
- `--quality-profile`: `fast` | `balanced` (default) | `strict`
- `--avg-degree`: Target connections per agent (default: 20)
- `--validate` / `-v`: Print network quality metrics

Network creates structural edges first (partner, household, coworker, neighbor, congregation, school_parent), then fills with similarity-based edges.

### `extropy persona`

Generates persona rendering config (first-person narrative templates).

```bash
extropy persona <population.yaml> --study-db <db> [-o <output.yaml>] [--preview]
```

- Generated once per population, applied computationally per agent
- Relative attributes positioned against population stats ("much more price-sensitive than most")
- Auto-detected by simulate if follows naming convention

### `extropy scenario`

Compiles scenario specification from population + network.

```bash
extropy scenario -p <population.yaml> --study-db <db> -o <output.yaml> [--yes]
```

Generates:
- Event definition (type, source, credibility)
- Exposure channels and rules
- Spread configuration
- Outcome definitions
- Simulation parameters (timesteps, fidelity, stopping conditions)

For evolving scenarios, add timeline events in the YAML after generation.

### `extropy estimate`

Predicts simulation cost without API calls.

```bash
extropy estimate <scenario.yaml> --study-db <db> [--verbose]
```

Shows:
- Predicted LLM calls (reasoning + classification)
- Token estimates
- USD cost by model
- `--verbose`: Per-timestep breakdown

### `extropy simulate`

Runs the simulation.

```bash
extropy simulate <scenario.yaml> --study-db <db> -o <results_dir> [--seed S] [--fidelity <level>]
```

Key flags:
- `--seed`: Random seed (always set for reproducibility)
- `--fidelity`: `low` | `medium` (default) | `high`
- `--merged-pass`: Single-pass reasoning (cheaper, test for quality)
- `--chunk-size`: Agents per checkpoint (default: 50)
- `--threshold`: Multi-touch re-reasoning threshold (default: 3)

Fidelity controls:
| Feature | low | medium | high |
|---------|-----|--------|------|
| Conversations | None | Top 1 edge | Top 2-3 edges |
| Memory traces | Last 5 | All | All + beliefs |
| Peer opinions | Top 5 | Top 5 | Top 10 + demographics |
| THINK vs SAY | No | No | Yes |
| Repetition detection | No | No | Yes |

Resume: If interrupted, rerun the exact same command. Checkpoints auto-resume.

### `extropy results`

Displays simulation outcomes.

```bash
extropy results --study-db <db> [--segment <attr>] [--timeline] [--agent <id>]
```

- Default: Aggregate outcome distributions
- `--segment`: Break down by attribute (income, age_bracket, etc.)
- `--timeline`: Show dynamics over time
- `--agent`: Deep dive on single agent's reasoning chain

Export artifacts:
```bash
extropy export states --study-db $DB --to states.jsonl
extropy export agents --study-db $DB --to agents.jsonl
extropy export elaborations --study-db $DB --to elaborations.csv
extropy export posts --study-db $DB --to posts.json
```

## Reproducibility Rules

1. **Always set `--seed`** on `sample`, `network`, and `simulate`
2. **Log all overrides**: models, thresholds, rate settings, fidelity
3. **Use `--yes`** for unattended/scripted runs
4. **Pin scenario/config revisions** before comparing variants

## Study Database Schema

The `study.db` SQLite database contains:

| Table | Contents |
|-------|----------|
| `agents` | Sampled agent attributes |
| `households` | Household groupings and shared attributes |
| `network_edges` | Social graph edges with types and weights |
| `agent_states` | Per-timestep agent state (position, sentiment, conviction) |
| `exposures` | Exposure records by channel and timestep |
| `conversations` | Agent-agent dialogue transcripts |
| `social_posts` | Public statements posted by agents |
| `memory_traces` | Reasoning history per agent |
| `timestep_summaries` | Aggregate metrics per timestep |
| `simulation_metadata` | Checkpoint and run state |

## Batch Operations

### Directory Structure

```
runs/
  <study>/
    study.db           # Canonical database
    base.yaml          # Base population spec
    population.yaml    # Extended population spec
    scenario.yaml      # Scenario spec
    results/           # Simulation outputs
    variants/          # Parameter sweep variants
      thresh-2/
      thresh-4/
      seed-43/
```

### Parameter Sweeps

Vary one axis at a time unless explicitly requested otherwise.

Common axes:
- `--seed` (confidence sweep)
- `--threshold` (re-reasoning sensitivity)
- `--fidelity` (cost vs quality)
- `--model` / `--routine-model` (model comparison)

Pattern:
```bash
# Baseline
extropy simulate scenario.yaml --study-db study.db -o results/baseline --seed 42

# Threshold variants
extropy simulate scenario.yaml --study-db study.db -o results/thresh-2 --seed 42 --threshold 2
extropy simulate scenario.yaml --study-db study.db -o results/thresh-4 --seed 42 --threshold 4

# Seed sweep for confidence
for seed in 42 43 44 45 46; do
  extropy simulate scenario.yaml --study-db study.db -o results/seed-$seed --seed $seed
done
```

### Variant Comparison

Only compare variants when:
- Same population (same sample seed)
- Same scenario spec
- Same config (except the axis being tested)

Report:
1. What changed (single axis)
2. Delta in key outcomes
3. Delta in cost/runtime
4. Confidence assessment

## Configuration

```bash
# View current config
extropy config show

# Set providers
extropy config set pipeline.provider claude      # For spec/extend/scenario
extropy config set simulation.provider openai    # For agent reasoning

# Set models
extropy config set simulation.model gpt-5-mini
extropy config set simulation.routine_model gpt-5-mini  # For Pass 2 classification

# Rate limits
extropy config set simulation.rate_tier 2       # 1-4, higher = more generous
```

API keys are environment variables only:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AZURE_OPENAI_API_KEY` (+ `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`)

## Validation

```bash
extropy validate <population.yaml>   # Population spec
extropy validate <scenario.yaml>     # Scenario spec (auto-detected)
```

Run after any manual edits to specs.

## Inspection & Debugging

### `extropy inspect`

Inspect study database entities.

```bash
extropy inspect summary --study-db $DB                    # Overview of agents, edges, runs
extropy inspect agent --study-db $DB --agent-id agent_042 # Single agent details
extropy inspect network --study-db $DB                    # Network topology stats
extropy inspect network-status --study-db $DB --network-run-id <id>  # Calibration progress
```

### `extropy query`

Run read-only SQL against study database.

```bash
extropy query sql --study-db $DB --sql "SELECT * FROM agent_states WHERE position = 'protest'" --format json
extropy query sql --study-db $DB --sql "SELECT position, COUNT(*) FROM agent_states GROUP BY position"
```

Only `SELECT`, `WITH`, `EXPLAIN` allowed. Use for ad-hoc analysis not covered by `results`.

### `extropy report`

Generate JSON reports for downstream processing.

```bash
extropy report run --study-db $DB -o run-report.json        # Run summary (counts, positions)
extropy report network --study-db $DB -o network-report.json # Network stats (edges, types)
```

## Post-Simulation Chat

### `extropy chat`

Interactive conversation with simulated agents using their DB-backed state.

```bash
# Interactive REPL
extropy chat --study-db $DB --run-id <id> --agent-id agent_042

# Non-interactive (for automation)
extropy chat ask --study-db $DB --run-id <id> --agent-id agent_042 \
  --prompt "Why did you change your mind?" --json
```

REPL commands: `/context`, `/timeline <n>`, `/history`, `/exit`

Useful for:
- Understanding individual agent reasoning
- Testing counterfactuals ("what if X happened?")
- Generating quotes for reports
