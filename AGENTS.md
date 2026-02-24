# AGENTS.md

Operational guide for AI agents and automation tools using the Extropy CLI.

## Automation Mode

Use agent mode for machine-readable output and non-interactive behavior:

```bash
extropy config set cli.mode agent
```

Agent mode behavior:
- JSON output shape from command handlers
- deterministic exit codes
- no interactive prompt loops (clarification requests return exit code `2`)

## CLI Pipeline

```bash
extropy spec -> extropy scenario -> extropy persona -> extropy sample -> extropy network -> extropy simulate -> extropy results
```

| Command | Purpose |
|---------|---------|
| `spec` | Create/iterate `population.vN.yaml` |
| `scenario` | Create/iterate `scenario/{name}/scenario.vN.yaml` |
| `persona` | Generate `persona.vN.yaml` for a scenario |
| `sample` | Sample agents from merged base + extended attributes |
| `network` | Build social graph for sampled agents |
| `simulate` | Run timestep simulation |
| `results` | Read run outcomes (`summary`, `timeline`, `segment`, `agent`) |
| `query` | Export/query raw DB-backed entities |
| `chat` | Chat with simulated agents (`list`, `ask`) |
| `validate` | Validate population/scenario/persona YAML |
| `config` | View/set configuration |

## Non-Interactive Usage

### Pre-supply clarifications

```bash
extropy spec "German surgeons" -o surgeons --answers '{"location":"Bavaria"}'
extropy spec "German surgeons" -o surgeons --use-defaults
```

### Skip confirmations

```bash
extropy scenario "AI adoption in radiology" -o ai-radiology -y
extropy persona -s ai-radiology -y
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Validation/general error |
| `2` | Clarification required (agent mode) |
| `3` | File/study/scenario not found |
| `4` | Sampling error |
| `5` | Network error |
| `6` | Simulation error |
| `7` | Scenario error |
| `10` | User cancelled |

## Querying Data

```bash
# study/entity summary
extropy query summary

# agents / edges / states export (JSONL)
extropy query agents --to agents.jsonl
extropy query edges --to edges.jsonl
extropy query states --to states.jsonl

# read-only SQL
extropy query sql "SELECT run_id, status FROM simulation_runs ORDER BY started_at DESC LIMIT 5" --format json
```

## Chat API

```bash
# list recent runs and sample agents
extropy chat list

# non-interactive chat turn
extropy chat ask --run-id <run_id> --agent-id <agent_id> --prompt "What changed your mind?"
```

Note: REPL chat mode is not for automation. Use `chat ask`.

## Configuration Keys

```bash
# output behavior
extropy config set cli.mode agent

# pipeline models (spec/scenario/persona)
extropy config set models.fast anthropic/claude-sonnet-4-6
extropy config set models.strong anthropic/claude-sonnet-4-6

# simulation models (reasoning/classification)
extropy config set simulation.strong azure/gpt-5-mini
extropy config set simulation.fast azure/gpt-5-mini

# runtime controls
extropy config set simulation.rate_tier 2
extropy config set show_cost true
```

## Environment Variables

Set at least one provider key used by your configured models:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
export DEEPSEEK_API_KEY=sk-...

# Azure (supported names)
export AZURE_API_KEY=...
export AZURE_ENDPOINT=https://<resource>.services.ai.azure.com/
# Legacy aliases still supported:
# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
```

## Global Flags

| Flag | Purpose |
|------|---------|
| `--version` | Print version and exit |
| `--cost` | Show command cost footer |
| `--study PATH` | Explicit study folder path |

## Study Folder Structure

```
my-study/
├── study.db
├── population.v1.yaml
├── scenario/
│   └── my-scenario/
│       ├── scenario.v1.yaml
│       └── persona.v1.yaml
└── results/
    └── my-scenario/
```

## Typical Automation Flow

```bash
# 1) Create study + base population
extropy spec "Austin TX commuters" -o my-study --use-defaults
cd my-study
extropy config set cli.mode agent

# 2) Build scenario + persona
extropy scenario "Congestion pricing response" -o congestion-tax -y
extropy persona -s congestion-tax -y

# 3) Sample and network
extropy sample -s congestion-tax -n 500 --seed 42 --strict-gates
extropy network -s congestion-tax --seed 42 --quality-profile balanced --validate

# 4) Simulate
extropy simulate -s congestion-tax --seed 42 --fidelity medium

# 5) Read outputs
extropy results -s congestion-tax summary
extropy results -s congestion-tax timeline
extropy query states --to states.jsonl
```
