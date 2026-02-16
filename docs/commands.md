# CLI Reference

Complete reference for all Extropy CLI commands, flags, and options.

---

## Pipeline Overview

```
extropy spec ──> extropy extend ──> extropy sample ──> extropy network ──> extropy persona ──> extropy scenario ──> extropy simulate
                                                                                                                     │              │
                                                                                                              extropy estimate    extropy results
```

Canonical data store: `study.db` (SQLite). All commands read from and write to this database.

---

## extropy spec

Build a population specification from a natural language description.

```bash
extropy spec "500 Austin TX commuters" -o austin/base.yaml -y
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `description` | string | yes | Natural language population description |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--output` | `-o` | path | required | Output YAML file path |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |

---

## extropy extend

Extend a base population spec with scenario-relevant attributes.

```bash
extropy extend austin/base.yaml -s "Response to $15/day congestion tax" -o austin/population.yaml -y
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `base_spec` | path | yes | Path to base population spec YAML |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | required | Scenario description |
| `--output` | `-o` | path | required | Output merged spec YAML |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |

---

## extropy sample

Sample concrete agents from a population specification.

```bash
extropy sample austin/population.yaml --study-db austin/study.db --seed 42
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `spec_file` | path | yes | Population spec YAML to sample from |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--study-db` | | path | required | Canonical study database path |
| `--population-id` | | string | `default` | Population ID in study DB |
| `--count` | `-n` | int | spec size | Number of agents to sample |
| `--seed` | | int | random | Random seed for reproducibility |
| `--report` | `-r` | flag | false | Show distribution summaries after sampling |
| `--skip-validation` | | flag | false | Skip spec validation before sampling |

---

## extropy network

Generate a social network connecting sampled agents.

```bash
extropy network --study-db austin/study.db -p austin/population.yaml --seed 42
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--study-db` | | path | required | Canonical study database path |
| `--population-id` | | string | `default` | Population ID to load agents from |
| `--network-id` | | string | `default` | Network ID to write |
| `--population` | `-p` | path | | Population spec YAML (generates network config via LLM) |
| `--network-config` | `-c` | path | | Custom network config YAML file |
| `--quality-profile` | | string | `balanced` | Quality profile: `fast`, `balanced`, `strict` |
| `--avg-degree` | | int | 20 | Target average connections per agent |
| `--rewire-prob` | | float | 0.05 | Watts-Strogatz rewiring probability |
| `--seed` | | int | random | Random seed for reproducibility |
| `--validate` | `-v` | flag | false | Print network validation metrics |
| `--resume` | | flag | false | Resume from checkpoint in study DB |

#### Resource Tuning Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-restarts` | int | profile | Number of calibration restarts |
| `--max-iters` | int | profile | Max iterations per restart |
| `--calibration-metric` | string | profile | Metric to optimize |
| `--min-degree` | int | profile | Minimum node degree |
| `--clustering-target` | float | profile | Target clustering coefficient |
| `--clustering-tolerance` | float | profile | Clustering tolerance |

---

## extropy persona

Generate persona rendering configuration for a population.

```bash
extropy persona austin/population.yaml --study-db austin/study.db -y
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `spec_file` | path | yes | Population spec YAML |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--study-db` | | path | | Study DB to load agents from (preferred) |
| `--population-id` | | string | `default` | Population ID when using --study-db |
| `--agents` | `-a` | path | | Legacy: sampled agents JSON file |
| `--output` | `-o` | path | `{spec}.persona.yaml` | Output persona config path |
| `--preview/--no-preview` | | flag | true | Show sample persona before saving |
| `--agent` | | int | 0 | Agent index for preview |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |
| `--show` | `-s` | flag | false | Preview existing config without regenerating |

---

## extropy scenario

Compile a scenario specification from population and network.

```bash
extropy scenario -p austin/population.yaml --study-db austin/study.db -o austin/scenario.yaml -y
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--population` | `-p` | path | required | Population spec YAML |
| `--study-db` | | path | required | Canonical study database path |
| `--population-id` | | string | `default` | Population ID in study DB |
| `--network-id` | | string | `default` | Network ID in study DB |
| `--description` | `-d` | string | from spec | Scenario description |
| `--output` | `-o` | path | `{pop}.scenario.yaml` | Output scenario spec path |
| `--timeline` | | path | | Timeline events YAML for evolving scenarios |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |

---

## extropy simulate

Run the simulation engine.

```bash
extropy simulate austin/scenario.yaml --study-db austin/study.db -o austin/results --seed 42
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `scenario_file` | path | yes | Scenario spec YAML |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--output` | `-o` | path | required | Output results directory |
| `--study-db` | | path | required | Canonical study database path |
| `--seed` | | int | random | Random seed for reproducibility |
| `--fidelity` | `-f` | string | `medium` | Fidelity tier: `low`, `medium`, `high` |
| `--merged-pass` | | flag | false | Single-pass reasoning (cheaper, less accurate) |
| `--threshold` | `-t` | int | 3 | Multi-touch threshold for re-reasoning |
| `--chunk-size` | | int | 50 | Agents per reasoning chunk (checkpoint granularity) |
| `--resume` | | flag | false | Resume from checkpoint |
| `--run-id` | | string | auto | Simulation run ID |

#### Model Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--strong` | `-m` | string | config | Strong model for Pass 1 (`provider/model`) |
| `--fast` | | string | config | Fast model for Pass 2 (`provider/model`) |

#### Rate Limiting Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rate-tier` | int | config | Provider rate limit tier (1-4) |
| `--rpm-override` | int | | Override requests per minute |
| `--tpm-override` | int | | Override tokens per minute |
| `--max-concurrent` | int | config | Max concurrent LLM calls |

#### Output Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--quiet` | `-q` | flag | false | Suppress progress output |
| `--verbose` | `-v` | flag | false | Show detailed logs |
| `--debug` | | flag | false | Show debug-level logs |

---

## extropy estimate

Predict simulation cost without making API calls.

```bash
extropy estimate austin/scenario.yaml --study-db austin/study.db
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `scenario_file` | path | yes | Scenario spec YAML |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--study-db` | | path | required | Canonical study database path |
| `--strong` | `-m` | string | config | Strong model for Pass 1 (`provider/model`) |
| `--fast` | | string | config | Fast model for Pass 2 (`provider/model`) |
| `--threshold` | `-t` | int | 3 | Multi-touch threshold |
| `--verbose` | `-v` | flag | false | Show per-timestep breakdown |

---

## extropy results

Display simulation results from the study database.

```bash
extropy results --study-db austin/study.db
extropy results --study-db austin/study.db --segment income
extropy results --study-db austin/study.db --timeline
extropy results --study-db austin/study.db --agent agent_042
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--study-db` | | path | required | Canonical study database path |
| `--run-id` | | string | latest | Simulation run ID |
| `--segment` | `-s` | string | | Attribute to segment results by |
| `--timeline` | `-t` | flag | false | Show timeline view |
| `--agent` | `-a` | string | | Show single agent details |

---

## extropy validate

Validate a population or scenario spec.

```bash
extropy validate austin/population.yaml
extropy validate austin/scenario.yaml
extropy validate austin/population.yaml --strict
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `spec_file` | path | yes | Spec file to validate |

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--strict` | flag | false | Treat warnings as errors (population specs only) |

Auto-detects file type: `*.scenario.yaml` runs scenario validation, other `*.yaml` runs population validation.

---

## extropy config

View and modify configuration.

```bash
extropy config show
extropy config set models.fast openai/gpt-5-mini
extropy config set simulation.strong anthropic/claude-sonnet-4.5
extropy config reset
```

### Arguments

| Name | Type | Description |
|------|------|-------------|
| `action` | string | Action: `show`, `set`, `reset` |
| `key` | string | Config key (for `set`) |
| `value` | string | Value to set (for `set`) |

### Available Keys

| Key | Description |
|-----|-------------|
| `models.fast` | Fast model for pipeline (`provider/model`) |
| `models.strong` | Strong model for pipeline (`provider/model`) |
| `simulation.fast` | Fast model for simulation Pass 2 |
| `simulation.strong` | Strong model for simulation Pass 1 |
| `simulation.max_concurrent` | Max concurrent LLM calls |
| `simulation.rate_tier` | Rate limit tier (1-4) |
| `simulation.rpm_override` | RPM override |
| `simulation.tpm_override` | TPM override |
| `show_cost` | Show cost tracking |
| `providers.<name>.base_url` | Custom provider base URL |
| `providers.<name>.api_key_env` | Custom provider API key env var |

---

## extropy export

Export data from the study database.

### extropy export agents

```bash
extropy export agents --study-db austin/study.db --to agents.jsonl
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--population-id` | string | `default` | Population ID |
| `--to` | path | required | Output file path |

### extropy export edges

```bash
extropy export edges --study-db austin/study.db --to edges.jsonl
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--network-id` | string | `default` | Network ID |
| `--to` | path | required | Output file path |

### extropy export states

```bash
extropy export states --study-db austin/study.db --to states.jsonl
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--run-id` | string | latest | Simulation run ID |
| `--to` | path | required | Output file path |

---

## extropy inspect

Inspect study database entities.

### extropy inspect summary

```bash
extropy inspect summary --study-db austin/study.db
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--population-id` | string | `default` | Population ID |
| `--network-id` | string | `default` | Network ID |
| `--run-id` | string | latest | Simulation run ID |

### extropy inspect agent

```bash
extropy inspect agent --study-db austin/study.db --agent-id agent_042
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--agent-id` | string | required | Agent ID |
| `--run-id` | string | latest | Simulation run ID |

### extropy inspect network

```bash
extropy inspect network --study-db austin/study.db
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--network-id` | string | `default` | Network ID |
| `--top` | int | 10 | Number of top-degree nodes to show |

### extropy inspect network-status

```bash
extropy inspect network-status --study-db austin/study.db --network-run-id <id>
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--network-run-id` | string | required | Network generation run ID |

---

## extropy query

Run read-only SQL queries against the study database.

```bash
extropy query sql --study-db austin/study.db --sql "SELECT * FROM agents LIMIT 10"
```

### extropy query sql

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--sql` | string | required | Read-only SQL statement |
| `--limit` | int | 1000 | Max rows to return |
| `--format` | string | `table` | Output format: `table`, `json`, `jsonl` |

Only `SELECT`, `WITH`, and `EXPLAIN` queries are allowed.

---

## extropy report

Generate JSON reports from simulation data.

### extropy report run

```bash
extropy report run --study-db austin/study.db -o report.json
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--study-db` | | path | required | Study database path |
| `--run-id` | | string | latest | Simulation run ID |
| `--output` | `-o` | path | required | Output JSON file |

### extropy report network

```bash
extropy report network --study-db austin/study.db -o network-report.json
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--study-db` | | path | required | Study database path |
| `--network-id` | | string | `default` | Network ID |
| `--output` | `-o` | path | required | Output JSON file |

---

## extropy chat

Interactive chat with simulated agents.

### Interactive REPL

```bash
extropy chat --study-db austin/study.db --run-id <id> --agent-id agent_042
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--run-id` | string | required | Simulation run ID |
| `--agent-id` | string | required | Agent ID |
| `--session-id` | string | auto | Chat session ID |

REPL commands: `/context`, `/timeline <n>`, `/history`, `/exit`

### extropy chat ask

Non-interactive API for automation.

```bash
extropy chat ask --study-db austin/study.db --run-id <id> --agent-id agent_042 \
  --prompt "What changed your mind?" --json
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Study database path |
| `--run-id` | string | required | Simulation run ID |
| `--agent-id` | string | required | Agent ID |
| `--prompt` | string | required | Question to ask |
| `--session-id` | string | auto | Chat session ID |
| `--json` | flag | false | Output JSON response |

---

## extropy migrate

Migrate legacy artifacts to the study database.

### extropy migrate legacy

```bash
extropy migrate legacy --study-db austin/study.db \
  --agents-file agents.json --network-file network.json
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study-db` | path | required | Target study database |
| `--agents-file` | path | | Legacy agents JSON |
| `--network-file` | path | | Legacy network JSON |
| `--population-spec` | path | | Population spec for provenance |
| `--population-id` | string | `default` | Population ID |
| `--network-id` | string | `default` | Network ID |

### extropy migrate scenario

```bash
extropy migrate scenario --input old-scenario.yaml --study-db austin/study.db -o new-scenario.yaml
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | | path | required | Legacy scenario YAML |
| `--study-db` | | path | required | Study database path |
| `--population-id` | | string | `default` | Population ID |
| `--network-id` | | string | `default` | Network ID |
| `--output` | `-o` | path | `{input}.db-first.yaml` | Output scenario path |

---

## Environment Variables

### API Keys (required)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |

### Azure Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | `2025-03-01-preview` | Azure API version |
| `AZURE_OPENAI_DEPLOYMENT` | | Azure deployment name |

---

## Quick Reference

```bash
# Full pipeline
STUDY=runs/my-study
DB=$STUDY/study.db
mkdir -p $STUDY

extropy spec "500 Austin TX commuters" -o $STUDY/base.yaml -y
extropy extend $STUDY/base.yaml -s "Response to $15/day congestion tax" -o $STUDY/population.yaml -y
extropy sample $STUDY/population.yaml --study-db $DB --seed 42
extropy network --study-db $DB -p $STUDY/population.yaml --seed 42
extropy persona $STUDY/population.yaml --study-db $DB -y
extropy scenario -p $STUDY/population.yaml --study-db $DB -o $STUDY/scenario.yaml -y

# Estimate cost
extropy estimate $STUDY/scenario.yaml --study-db $DB

# Run simulation
extropy simulate $STUDY/scenario.yaml --study-db $DB -o $STUDY/results --seed 42

# View results
extropy results --study-db $DB
extropy results --study-db $DB --segment income
extropy results --study-db $DB --timeline
extropy results --study-db $DB --agent agent_042

# Validate
extropy validate $STUDY/population.yaml
extropy validate $STUDY/scenario.yaml

# Config
extropy config show
extropy config set simulation.strong anthropic/claude-sonnet-4.5
```
