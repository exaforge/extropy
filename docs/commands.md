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

## Global Options

All commands support these global options:

| Flag | Description |
|------|-------------|
| `--json` | Output machine-readable JSON instead of human-friendly text |
| `--version` | Show version and exit |
| `--cost` | Show cost summary after command completes |
| `--study PATH` | Study folder path (auto-detected from cwd if not specified) |

---

## extropy spec

Generate a population spec from a natural language description.

```bash
# Create new study folder with population.v1.yaml
extropy spec "German surgeons" -o surgeons

# Create with custom name (surgeons/hospital-staff.v1.yaml)
extropy spec "German surgeons" -o surgeons/hospital-staff

# Iterate on existing (from within study folder)
cd surgeons && extropy spec "add income distribution"
# Creates population.v2.yaml

# Explicit file path
extropy spec "farmers" -o my-spec.yaml
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `description` | string | yes | Natural language population description |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--output` | `-o` | path | | Output path: study folder, folder/name, or explicit .yaml file |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |
| `--answers` | | string | | JSON with pre-supplied clarification answers (for agent mode) |
| `--use-defaults` | | flag | false | Use defaults for ambiguous values instead of prompting |

---

## extropy scenario

Create a scenario with extended attributes and event configuration.

```bash
# Create new scenario
extropy scenario "AI diagnostic tool adoption" -o ai-adoption

# Pin population version
extropy scenario "vaccine mandate" -o vaccine @pop:v1

# Rebase existing scenario to new population
extropy scenario ai-adoption --rebase @pop:v2
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `description` | string | yes | Scenario description (what event/situation to simulate) |
| `population_ref` | string | no | Population version reference: `@pop:v1`, `@pop:latest`, or path to YAML |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--output` | `-o` | string | required | Scenario name (creates `scenario/{name}/scenario.v1.yaml`) |
| `--rebase` | | string | | Rebase existing scenario to new population version (e.g. `@pop:v2`) |
| `--timeline` | | string | `auto` | Timeline mode: `auto` (LLM decides), `static` (single event), `evolving` (multi-event) |
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
# Generate for a scenario (auto-versions)
extropy persona -s ai-adoption

# Pin scenario version
extropy persona -s ai-adoption@v1

# Preview existing config
extropy persona -s ai-adoption --show
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | | Output file (default: `scenario/{name}/persona.vN.yaml`) |
| `--preview/--no-preview` | | flag | true | Show a sample persona before saving |
| `--agent` | | int | 0 | Which agent to use for preview |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |
| `--show` | | flag | false | Preview existing persona config without regenerating |

---

## extropy sample

Sample agents from a scenario's merged population spec.

```bash
extropy sample -s ai-adoption -n 500
extropy sample -s ai-adoption -n 1000 --seed 42 --report
extropy sample -n 500  # auto-selects scenario if only one exists
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--count` | `-n` | int | required | Number of agents to sample |
| `--seed` | | int | random | Random seed for reproducibility |
| `--report` | `-r` | flag | false | Show distribution summaries and stats |
| `--skip-validation` | | flag | false | Skip validator errors |

**Exit codes:** 0 = Success, 1 = Validation error, 2 = File not found, 3 = Sampling error

Sampling process:
1. Loads scenario's `base_population` spec
2. Merges with scenario's `extended_attributes`
3. Validates the merged spec
4. Samples agents
5. Saves to `study.db` keyed by `scenario_id`

---

## extropy network

Generate a social network from sampled agents.

```bash
extropy network -s ai-adoption                             # Uses LLM-generated config (default)
extropy network -s ai-adoption --avg-degree 15 --seed 42   # Custom degree and seed
extropy network -s ai-adoption --no-generate-config        # Flat network, no similarity structure
extropy network -s ai-adoption -c custom-network.yaml      # Load custom config
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | | Optional JSON export path (non-canonical) |
| `--network-config` | `-c` | path | | Custom network config YAML file |
| `--save-config` | | path | | Save the (generated or loaded) network config to YAML |
| `--generate-config` | | flag | true | Generate network config via LLM from population spec (default: enabled) |
| `--avg-degree` | | float | 20.0 | Target average degree (connections per agent) |
| `--rewire-prob` | | float | 0.05 | Watts-Strogatz rewiring probability |
| `--seed` | | int | random | Random seed for reproducibility |
| `--validate` | `-v` | flag | false | Print validation metrics |
| `--no-metrics` | | flag | false | Skip computing node metrics (faster) |

#### Quality & Candidate Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quality-profile` | string | `balanced` | Quality profile: `fast`, `balanced`, `strict` |
| `--candidate-mode` | string | `blocked` | Similarity candidate mode: `exact`, `blocked` |
| `--candidate-pool-multiplier` | float | 12.0 | Blocked mode candidate pool size as multiple of avg_degree |
| `--block-attr` | string (repeatable) | auto | Blocking attribute(s). If omitted, auto-selects top attributes |
| `--similarity-workers` | int | 1 | Worker processes for similarity computation |
| `--similarity-chunk-size` | int | 64 | Row chunk size for similarity worker tasks |

#### Checkpointing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | path | | DB path for similarity checkpointing (must be study.db) |
| `--resume` | flag | false | Resume similarity and calibration checkpoints from study.db |
| `--checkpoint-every` | int | 250 | Write checkpoint every N processed similarity rows |

#### Resource Tuning

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resource-mode` | string | `auto` | Resource tuning mode: `auto`, `manual` |
| `--safe-auto-workers/--unsafe-auto-workers` | flag | true | Conservative auto tuning for laptops/VMs |
| `--max-memory-gb` | float | | Optional memory budget cap for auto resource tuning |

---

## extropy simulate

Run a simulation from a scenario spec.

```bash
extropy simulate -s ai-adoption
extropy simulate -s ai-adoption --seed 42 --strong gpt-4o
extropy simulate -s ai-adoption --fidelity high
```

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | `results/{scenario}/` | Output results directory |
| `--seed` | | int | random | Random seed for reproducibility |
| `--fidelity` | `-f` | string | `medium` | Fidelity level: `low`, `medium`, `high` |
| `--merged-pass` | | flag | false | Use single merged reasoning pass instead of two-pass (experimental) |
| `--threshold` | `-t` | int | 3 | Multi-touch threshold for re-reasoning |
| `--chunk-size` | | int | 50 | Agents per reasoning chunk for checkpointing |

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

#### Checkpointing & Resume

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | auto | Explicit run id (required with --resume) |
| `--resume` | flag | false | Resume an existing run from study DB checkpoints |
| `--checkpoint-every-chunks` | int | 1 | Persist simulation chunk checkpoints every N chunks |

#### Database Tuning

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--writer-queue-size` | int | 256 | Max reasoning chunks buffered before DB writer backpressure |
| `--db-write-batch-size` | int | 100 | Number of chunks applied per DB writer transaction |
| `--retention-lite` | flag | false | Reduce retained payload volume (drops full raw reasoning text) |

#### Resource Tuning

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resource-mode` | string | `auto` | Resource tuning mode: `auto`, `manual` |
| `--safe-auto-workers/--unsafe-auto-workers` | flag | true | Conservative auto tuning for laptop/VM environments |
| `--max-memory-gb` | float | | Optional memory budget cap for auto resource tuning |

#### Output Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--quiet` | `-q` | flag | false | Suppress progress output |
| `--verbose` | `-v` | flag | false | Show detailed logs |
| `--debug` | | flag | false | Show debug-level logs (very verbose) |

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
extropy estimate -s ai-adoption
extropy estimate -s ai-adoption --strong openai/gpt-5
extropy estimate -s ai-adoption --strong openai/gpt-5 --fast openai/gpt-5-mini -v
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
extropy validate population.v1.yaml                       # Population spec
extropy validate scenario/congestion-tax/scenario.v1.yaml # Versioned scenario spec
extropy validate my-scenario.scenario.yaml                # Legacy scenario spec
extropy validate population.v1.yaml --strict              # Treat warnings as errors
```

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `spec_file` | path | yes | Spec file to validate |

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--strict` | flag | false | Treat warnings as errors (population specs only) |

Auto-detects file type based on naming:
- `*.scenario.yaml` or `*.scenario.yml` → scenario spec validation
- `scenario.yaml` or `scenario.yml` → scenario spec validation
- `scenario.v{N}.yaml` or `scenario.v{N}.yml` → scenario spec validation (versioned)
- Other `*.yaml` files → population spec validation

Supports both flows for scenario validation:
- **New flow**: `meta.base_population` references versioned population (e.g., `population.v2`)
- **Legacy flow**: `meta.population_spec` + `meta.study_db` file paths

**Exit codes:** 0 = Success (valid spec), 1 = Validation error (invalid spec), 2 = File not found

---

## extropy config

View and modify configuration.

```bash
extropy config show
extropy config set models.fast openai/gpt-5-mini
extropy config set simulation.strong anthropic/claude-sonnet-4.5
extropy config set simulation.strong openrouter/anthropic/claude-sonnet-4.5
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

Interactive chat with simulated agents. Auto-detects study folder from current working directory.

### Interactive REPL

```bash
cd austin  # study folder
extropy chat
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | latest | Simulation run ID |
| `--agent-id` | string | auto | Agent ID (auto-selects first agent if not specified) |
| `--session-id` | string | auto | Chat session ID |

REPL commands: `/context`, `/timeline <n>`, `/history`, `/exit`

### extropy chat list

Show recent runs and sample agents so users can pick chat targets quickly.

```bash
cd austin && extropy chat list
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--limit-runs` | int | 10 | Number of recent runs to list |
| `--agents-per-run` | int | 5 | Number of sample agent IDs per run |

### extropy chat ask

Non-interactive API for automation.

```bash
cd austin && extropy chat ask --prompt "What changed your mind?"
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | latest | Simulation run ID |
| `--agent-id` | string | auto | Agent ID (auto-selects first agent if not specified) |
| `--prompt` | string | required | Question to ask |
| `--session-id` | string | auto | Chat session ID |

**Output modes:**
- Human mode (`cli.mode: human`): Rich terminal formatting
- Agent mode (`cli.mode: agent`): Structured JSON output

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
# Create study folder with population spec
extropy spec "Austin TX commuters" -o my-study
cd my-study

# Create scenario and persona config
extropy scenario "Response to $15/day congestion tax" -o congestion-tax
extropy persona -s congestion-tax -y

# Sample agents and generate network (LLM config by default)
extropy sample -s congestion-tax -n 500 --seed 42
extropy network -s congestion-tax --seed 42

# Estimate cost before running
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

# Validate specs
extropy validate population.v1.yaml                         # Population spec
extropy validate scenario/congestion-tax/scenario.v1.yaml   # Versioned scenario

# Config
extropy config show
extropy config set simulation.strong anthropic/claude-sonnet-4.5
extropy config set cli.mode agent  # for AI harnesses
extropy config set cli.mode human  # for terminal users (default)
```
