# AGENTS.md

Instructions for AI agents and automation tools using the Extropy CLI.

## Agent Mode

Set agent mode for structured JSON output and non-interactive operation:

```bash
extropy config set cli.mode agent
```

Or use the `--json` flag per-command:

```bash
extropy results --json
extropy query summary --json
```

**Agent mode behavior:**
- All output is JSON (parseable, no ANSI formatting)
- Exit codes indicate success/failure (0 = success, non-zero = error)
- No interactive prompts — commands fail fast if missing required input
- Clarification requests return exit code 2 with structured error

## CLI Pipeline

```bash
extropy spec → extropy scenario → extropy persona → extropy sample → extropy network → extropy simulate → extropy results
```

| Command | Purpose |
|---------|---------|
| `spec` | Create population spec from description |
| `scenario` | Create scenario with events and outcomes |
| `persona` | Generate persona rendering config |
| `sample` | Sample agents from merged spec |
| `network` | Generate social network |
| `simulate` | Run simulation |
| `results` | View results (summary, timeline, segment, agent) |
| `query` | Export raw data (agents, edges, states, SQL) |
| `chat` | Chat with simulated agents |
| `estimate` | Predict simulation cost |
| `validate` | Validate spec files |
| `config` | View/set configuration |

## Non-Interactive Usage

### Pre-supplying answers

For `spec` command, use `--answers` to skip clarification prompts:

```bash
extropy spec "German surgeons" -o surgeons --answers '{"location": "Bavaria"}'
```

Or use `--use-defaults` to accept default values:

```bash
extropy spec "German surgeons" -o surgeons --use-defaults
```

### Skip confirmations

Use `-y` / `--yes` to skip confirmation prompts:

```bash
extropy scenario "AI adoption" -o ai-adoption -y
extropy persona -s ai-adoption -y
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error / validation failure |
| 2 | Clarification needed (agent mode) / file not found |
| 3 | Sampling error |

## Querying Data

Export raw data for downstream processing:

```bash
# Agents as JSONL
extropy query agents --to agents.jsonl

# Network edges
extropy query edges --to edges.jsonl

# Simulation states
extropy query states --to states.jsonl

# Arbitrary SQL (read-only)
extropy query sql "SELECT * FROM agent_states WHERE aware = 1" --format json
```

## Chat API

Non-interactive chat for automation:

```bash
extropy chat ask \
  --study-db study.db \
  --agent-id agent_042 \
  --prompt "What changed your mind?" \
  --json
```

List available agents:

```bash
extropy chat list --study-db study.db --json
```

**Note:** The interactive REPL (`extropy chat`) requires a TTY and is not suitable for automation. Use `extropy chat ask` instead.

## Configuration Keys

```bash
extropy config set cli.mode agent           # Enable agent mode globally
extropy config set models.strong openai/gpt-4o
extropy config set models.fast openai/gpt-4o-mini
extropy config set simulation.strong anthropic/claude-sonnet-4.5
extropy config set simulation.fast anthropic/claude-haiku-3.5
extropy config set show_cost true           # Show cost after commands
```

## Environment Variables

Required API keys (set at least one):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
export DEEPSEEK_API_KEY=sk-...
export AZURE_API_KEY=...
export AZURE_ENDPOINT=https://<resource>.services.ai.azure.com/
```

## Global Flags

All commands accept:

| Flag | Purpose |
|------|---------|
| `--json` | JSON output (overrides cli.mode) |
| `--cost` | Show cost summary after command |
| `--study PATH` | Explicit study folder path |

## Study Folder Structure

```
my-study/
├── study.db                    # Canonical SQLite store
├── population.v1.yaml          # Base population spec
├── scenario/
│   └── my-scenario/
│       ├── scenario.v1.yaml    # Scenario spec
│       └── persona.v1.yaml     # Persona config
└── results/
    └── my-scenario/            # Simulation outputs
```

## Typical Automation Flow

```bash
# Setup
cd my-study
extropy config set cli.mode agent

# Build pipeline
extropy spec "Austin TX commuters" -o . --use-defaults
extropy scenario "Congestion tax response" -o congestion-tax -y
extropy persona -s congestion-tax -y
extropy sample -s congestion-tax -n 500 --seed 42
extropy network -s congestion-tax --seed 42

# Estimate before running
extropy estimate -s congestion-tax --json

# Run simulation
extropy simulate -s congestion-tax --seed 42

# Extract results
extropy results --json
extropy query agents --to agents.jsonl
extropy query states --to states.jsonl
```
