# CLI Reference

Complete reference for all Extropy CLI commands, flags, and options.

---

## Pipeline Overview

```
extropy spec ──> extropy scenario ──> extropy persona ──> extropy sample ──> extropy network ──> extropy simulate ──> extropy results
                                                                                                      │
                                                                                               extropy estimate
```

All commands operate within a **study folder** — a directory containing `study.db` and scenario subdirectories. Commands auto-detect the study folder from the current working directory.

**Study folder structure:**
```
my-study/
├── study.db                    # Canonical data store (SQLite)
├── population.v1.yaml          # Base population spec
├── scenario/
│   └── my-scenario/
│       ├── scenario.v1.yaml    # Scenario spec
│       └── persona.v1.yaml     # Persona config
└── results/
    └── my-scenario/            # Simulation outputs
```

---

## extropy spec

Build a population specification from a natural language description.

```bash
extropy spec "Austin TX commuters" -o population.v1.yaml -y
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

## extropy scenario

Create a scenario spec with extended attributes and event configuration.

```bash
extropy scenario -s "Response to $15/day congestion tax" -o scenario/congestion-tax -y
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | required | Scenario description |
| `--output` | `-o` | path | required | Output directory (creates `scenario.v1.yaml`) |
| `--base-population` | | string | `population.v1` | Base population reference (`name` or `name.vN`) |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |

The scenario spec includes:
- Extended attributes specific to the scenario
- Event definition (type, content, source)
- Seed exposure rules
- Interaction model
- Outcome configuration
- Simulation settings

---

## extropy persona

Generate persona rendering configuration for a scenario.

```bash
extropy persona -s congestion-tax -y
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--preview/--no-preview` | | flag | true | Show sample persona before saving |
| `--agent` | | int | 0 | Agent index for preview |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |
| `--show` | | flag | false | Preview existing config without regenerating |

---

## extropy sample

Sample agents from a scenario's merged population spec.

```bash
extropy sample -s congestion-tax -n 500 --seed 42
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--count` | `-n` | int | required | Number of agents to sample |
| `--seed` | | int | random | Random seed for reproducibility |
| `--report` | `-r` | flag | false | Show distribution summaries after sampling |
| `--skip-validation` | | flag | false | Skip spec validation before sampling |

Sampling process:
1. Loads scenario's `base_population` spec
2. Merges with scenario's `extended_attributes`
3. Validates the merged spec
4. Samples agents
5. Saves to `study.db` keyed by `scenario_id`

---

## extropy network

Generate a social network connecting sampled agents.

```bash
extropy network -s congestion-tax --seed 42
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | | Output network JSON (defaults to scenario folder) |
| `--network-config` | `-c` | path | | Custom network config YAML |
| `--generate-config` | | flag | false | Generate network config via LLM |
| `--seed` | | int | random | Random seed for reproducibility |
| `--no-metrics` | | flag | false | Skip network metrics calculation |
| `--resume` | | flag | false | Resume from checkpoint |

#### Performance Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--candidate-mode` | string | `all` | Candidate selection: `all`, `blocked`, `sampled` |
| `--candidate-pool-multiplier` | float | 4.0 | Multiplier for candidate pool size |
| `--block-attr` | string | | Attribute for block-based candidate selection |
| `--similarity-workers` | int | auto | Parallel workers for similarity computation |
| `--similarity-chunk-size` | int | 100 | Chunk size for similarity batches |

#### Checkpointing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | path | | Path to study.db for checkpointing |
| `--checkpoint-every` | int | 5 | Checkpoint frequency (iterations) |

---

## extropy simulate

Run the simulation engine.

```bash
extropy simulate -s congestion-tax --seed 42
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | `results/{scenario}/` | Output results directory |
| `--seed` | | int | random | Random seed for reproducibility |
| `--fidelity` | `-f` | string | `medium` | Fidelity tier: `low`, `medium`, `high` |
| `--merged-pass` | | flag | false | Single-pass reasoning (cheaper, less accurate) |
| `--threshold` | `-t` | int | 3 | Multi-touch threshold for re-reasoning |
| `--chunk-size` | | int | 50 | Agents per reasoning chunk |
| `--resume` | | flag | false | Resume from checkpoint |
| `--run-id` | | string | auto | Simulation run ID |

#### Model Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--strong` | `-m` | string | config | Strong model for Pass 1 (`provider/model`) |
| `--fast` | | string | config | Fast model for Pass 2 (`provider/model`) |

#### Rate Limiting

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

## extropy results

Display simulation results. Supports subcommands for different views.

```bash
extropy results                              # summary (default)
extropy results timeline                     # timestep progression
extropy results segment income               # segment by attribute
extropy results agent agent_042              # single agent details
```

### Shared Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | | Filter by scenario |
| `--run-id` | | string | latest | Simulation run ID |

### extropy results summary

Default view when no subcommand is given. Shows agent count, awareness rate, and position distribution.

### extropy results timeline

Shows timestep-by-timestep progression including new exposures, agents reasoned, shares, and exposure rate.

### extropy results segment \<attribute\>

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `attribute` | string | yes | Agent attribute to segment by |

### extropy results agent \<agent_id\>

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `agent_id` | string | yes | Agent ID to inspect |

**Output modes:**
- Human mode (`cli.mode: human`): Rich terminal formatting
- Agent mode (`cli.mode: agent`): Structured JSON output

---

## extropy query

Query and export raw data from the study database.

### extropy query agents

Dump agent attributes.

```bash
extropy query agents                         # print to stdout
extropy query agents --to agents.jsonl       # write JSONL file
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--to` | path | | Write JSONL to file |
| `--population-id` | string | `default` | Population ID |

### extropy query edges

Dump network edges.

```bash
extropy query edges --to edges.jsonl
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--to` | path | | Write JSONL to file |
| `--network-id` | string | `default` | Network ID |

### extropy query states

Dump agent states for a simulation run.

```bash
extropy query states --to states.jsonl
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | latest | Simulation run ID |
| `--to` | path | | Write JSONL to file |

### extropy query summary

Show study entity counts (agents, edges, simulation states, timesteps, events).

```bash
extropy query summary
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | latest | Simulation run ID |
| `--population-id` | string | `default` | Population ID |
| `--network-id` | string | `default` | Network ID |

### extropy query network

Show network statistics (edge count, average weight, top-degree nodes).

```bash
extropy query network
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--network-id` | string | `default` | Network ID |
| `--top` | int | 10 | Number of top-degree nodes to show |

### extropy query network-status \<network-run-id\>

Show network calibration progress.

```bash
extropy query network-status <run-id>
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `network_run_id` | string | yes | Network generation run ID |

### extropy query sql \<sql\>

Run a read-only SQL query against the study database.

```bash
extropy query sql "SELECT count(*) FROM agents"
extropy query sql "SELECT * FROM agent_states LIMIT 10" --format json
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `sql` | string | yes | Read-only SQL statement |

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--limit` | int | 1000 | Max rows to return |
| `--format` | string | `table` | Output format: `table`, `json`, `jsonl` |

Only `SELECT`, `WITH`, and `EXPLAIN` queries are allowed.

---

## extropy estimate

Predict simulation cost without making API calls.

```bash
extropy estimate -s congestion-tax
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name |
| `--strong` | `-m` | string | config | Strong model for Pass 1 |
| `--fast` | | string | config | Fast model for Pass 2 |
| `--threshold` | `-t` | int | 3 | Multi-touch threshold |
| `--verbose` | `-v` | flag | false | Show per-timestep breakdown |

---

## extropy validate

Validate a population or scenario spec.

```bash
extropy validate population.v1.yaml
extropy validate scenario/congestion-tax/scenario.v1.yaml
extropy validate population.v1.yaml --strict
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
| `cli.mode` | CLI mode: `human` (interactive) or `agent` (JSON output) |
| `show_cost` | Show cost tracking |
| `providers.<name>.base_url` | Custom provider base URL |
| `providers.<name>.api_key_env` | Custom provider API key env var |

---

## extropy chat

Interactive chat with simulated agents. Uses the same study folder auto-detection as other commands.

### Interactive REPL

```bash
extropy chat                                  # auto-detect study, use latest run/first agent
extropy chat --run-id run_123 --agent-id a_42
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | latest | Simulation run ID |
| `--agent-id` | string | first agent in run | Agent ID |
| `--session-id` | string | auto | Chat session ID |

REPL commands: `/context`, `/timeline <n>`, `/history`, `/exit`

### extropy chat list

Show recent runs and sample agents so users can pick chat targets quickly.

```bash
extropy chat list
extropy chat list --limit-runs 5
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--limit-runs` | int | `10` | Number of recent runs to list |
| `--agents-per-run` | int | `5` | Number of sample agent IDs per run |

### extropy chat ask

Non-interactive API for automation.

```bash
extropy chat ask --prompt "What changed your mind?"
extropy chat ask --run-id r1 --agent-id a1 --prompt "What changed?"
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | latest | Simulation run ID |
| `--agent-id` | string | first agent in run | Agent ID |
| `--prompt` | string | required | Question to ask |
| `--session-id` | string | auto | Chat session ID |

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
cd my-study  # study folder

extropy spec "Austin TX commuters" -o population.v1.yaml -y
extropy scenario -s "Response to $15/day congestion tax" -o scenario/congestion-tax -y
extropy persona -s congestion-tax -y
extropy sample -s congestion-tax -n 500 --seed 42
extropy network -s congestion-tax --seed 42

# Estimate cost
extropy estimate -s congestion-tax

# Run simulation
extropy simulate -s congestion-tax --seed 42

# View results
extropy results
extropy results timeline
extropy results segment income
extropy results agent agent_042

# Query data
extropy query agents --to agents.jsonl
extropy query states --to states.jsonl
extropy query summary
extropy query network
extropy query sql "SELECT count(*) FROM agents"

# Validate
extropy validate population.v1.yaml
extropy validate scenario/congestion-tax/scenario.v1.yaml

# Config
extropy config show
extropy config set simulation.strong anthropic/claude-sonnet-4.5
extropy config set cli.mode agent  # for AI harnesses
extropy config set cli.mode human  # for terminal users (default)
```
