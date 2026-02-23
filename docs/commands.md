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

**Note:** These are root-level Typer options. Place them before the subcommand, e.g. `extropy --json spec "Austin commuters" -o study`.

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

### What `spec` does in current flow

1. Runs sufficiency check (may ask clarifications; in agent mode returns structured questions).
2. Selects attributes (strategy, scope, dependencies, semantic metadata).
3. Runs split hydration for distributions/formulas/modifiers.
4. Binds constraints + computes dependency-safe sampling order.
5. Builds and validates `PopulationSpec`.
6. Saves versioned output YAML.

**Stage ownership notes:**
- Spec stage does **not** persist household config (household modeling is scenario-owned).
- Name generation is **not** part of spec generation; names are generated at sampling/runtime.

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

### Validation Failure Behavior

- If spec validation fails, CLI writes a versioned invalid artifact next to target output (`population.v1.yaml` -> `population.v1.invalid.v1.yaml`, then `.v2`, `.v3`, ...).
- Command exits non-zero after writing the invalid artifact.

---

## extropy scenario

Create a scenario with scenario-specific attributes and simulation configuration.

The scenario command is essentially a **mini spec builder** — it discovers and researches attributes that are specific to this scenario but not in the base population spec. For example, a "vaccine adoption" scenario might add `vaccine_hesitancy` and `prior_flu_shot` attributes that wouldn't exist in a general population spec.

```bash
# Create new scenario
extropy scenario "AI diagnostic tool adoption" -o ai-adoption

# Pin population version
extropy scenario "vaccine mandate" -o vaccine @pop:v1

# Rebase existing scenario to new population
extropy scenario "rebase marker" -o ai-adoption --rebase @pop:v2
```

### What the Scenario Command Does

1. **Runs sufficiency check** — infers duration/type/unit/focus hints and asks clarifications if needed
2. **Discovers scenario-specific attributes** — identifies extension attributes not already in base population
3. **Hydrates extension + household config** — researches distributions and scenario household semantics
4. **Binds constraints** — validates dependencies and sampling order for extension attrs
5. **Compiles scenario dynamics** — builds event, exposure, interaction/spread, timeline, and outcomes
6. **Validates scenario contract** — deterministic checks before save (base+extended refs, literals, channels, timeline, outcomes)
7. **Saves versioned artifact** — `scenario/{name}/scenario.vN.yaml` (or versioned `.invalid` on fail-hard)

### Sufficiency Behavior

- Sufficiency is intentionally lenient, but deterministic post-processing adds guardrails:
  - explicit timeline markers (for example `week 1`, `month 0`) force evolving mode
  - static scenarios must have an explicit timestep unit (or trigger a clarification question)
- In agent mode, insufficiency returns structured questions with exit code `2`.
- `--use-defaults` retries sufficiency automatically using defaults from those clarification questions.

### Arguments

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `description` | string | yes | Scenario description (what event/situation to simulate) |
| `population_ref` | string | no | Population version reference: `@pop:v1` or `@pop:latest` |

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--output` | `-o` | string | required | Scenario name (creates `scenario/{name}/scenario.v1.yaml`) |
| `--rebase` | | string | | Rebase existing scenario to new population version (e.g. `@pop:v2`) |
| `--timeline` | | string | `auto` | Timeline mode: `auto` (LLM decides), `static` (single event), `evolving` (multi-event) |
| `--timestep-unit` | | string | inferred | Override timestep unit: `hour`, `day`, `week`, `month`, `year` |
| `--max-timesteps` | | int | inferred | Override simulation horizon |
| `--use-defaults` | | flag | false | Auto-answer sufficiency clarifications with defaults |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |

### Scenario Spec Contents

The generated `scenario.v1.yaml` includes:

- **`extended_attributes`** — Scenario-specific attributes with full distribution specs (same format as population attributes)
- **`event`** — Event definition (type, content, source, credibility, ambiguity, emotional valence)
- **`timeline`** — For evolving scenarios: subsequent events at different timesteps
- **`seed_exposure`** — Channels and rules for initial exposure
- **`interaction`** — How agents interact about the event
- **`spread`** — How information propagates through the network
- **`outcomes`** — What to measure from each agent
- **`simulation`** — Timestep config, stopping conditions, convergence settings
- **`household_config` + `agent_focus_mode`** — Scenario-owned household semantics for sample stage
- **`sampling_semantic_roles`** — Scenario-level semantic role mappings used by sampling/runtime checks
- **`identity_dimensions`** (optional) — Identity activation hints consumed by simulation prompts

### Validation Failure Behavior

- If no scenario extension attributes are discovered, scenario creation hard-fails and writes a versioned JSON invalid artifact.
- If compile fails mid-pipeline, scenario creation hard-fails and writes a versioned JSON invalid artifact.
- If final scenario validation fails, CLI writes versioned YAML invalid artifact (`scenario.vN.invalid.vK.yaml`) and exits non-zero.

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

### What `persona` does in current flow

1. Resolves scenario and loads `scenario.vN.yaml`.
2. Loads referenced base population and merges `extended_attributes`.
3. Runs persona generation pipeline (structure, boolean/categorical/relative/concrete phrasings).
4. Validates generated config against merged attributes (`validate_persona_config`).
5. Saves versioned output YAML (`persona.vN.yaml`).

Notes:
- If sampled agents already exist, persona generation computes `population_stats` at generation time.
- If not, stats can be backfilled later at simulation runtime.

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | | Output file (default: `scenario/{name}/persona.vN.yaml`) |
| `--preview/--no-preview` | | flag | true | Reserved flag (currently not used as a separate generation gate) |
| `--agent` | | int | 0 | Which agent to use for preview |
| `--yes` | `-y` | flag | false | Skip confirmation prompts |
| `--show` | | flag | false | Preview existing persona config without regenerating |

### Validation Failure Behavior

- If generation fails, CLI writes a versioned JSON invalid artifact and exits non-zero.
- If persona validation fails, CLI writes versioned YAML invalid artifact (`persona.vN.invalid.vK.yaml`) and exits non-zero.
- `extropy validate persona.vN.yaml` (or `.invalid`) runs persona-specific validation against merged base+extended attributes.

---

## extropy sample

Sample agents from a scenario's merged population spec.

```bash
extropy sample -s ai-adoption -n 500
extropy sample -s ai-adoption -n 1000 --seed 42 --report
extropy sample -n 500  # auto-selects scenario if only one exists
```

### What `sample` does in current flow

1. Resolves scenario and requires persona config pre-flight.
2. Loads base population + scenario extension and builds merged spec.
3. Recomputes merged sampling order via topological sort.
4. Validates merged spec.
5. Samples agents using scenario household config/focus/semantic roles.
6. Runs deterministic post-sample rule-pack gate.
7. Saves agents and run metadata to `study.db`.

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--count` | `-n` | int | required | Number of agents to sample |
| `--seed` | | int | random | Random seed for reproducibility |
| `--report` | `-r` | flag | false | Show distribution summaries and stats |
| `--skip-validation` | | flag | false | Skip validator errors |
| `--strict-gates` | | flag | false | Promote high-risk warnings and post-sample condition warnings to fail-hard |

**Exit codes:** 0 = Success, 1 = Validation error, 3 = File not found, 4 = Sampling error

Sampling process:
1. Loads scenario's `base_population` spec
2. Merges with scenario's `extended_attributes`
3. Recomputes merged dependency order (topological sort)
4. Validates the merged spec
5. Samples agents
6. Applies rule-pack gate (`impossible`/`implausible`)
7. Saves to `study.db` keyed by `scenario_id`

### Validation/Gate Failure Behavior

- Missing persona config blocks sampling pre-flight.
- Merged-order cycles or merged-spec validation failures write versioned JSON invalid artifacts and exit non-zero.
- Post-sampling gate failure writes versioned JSON invalid artifact (`sample.invalid.vN.json`) and exits non-zero.

---

## extropy network

Generate a social network from sampled agents.

```bash
extropy network -s ai-adoption                             # Uses LLM-generated config (default)
extropy network -s ai-adoption --avg-degree 15 --seed 42   # Custom degree and seed
extropy network -s ai-adoption --no-generate-config        # Flat network, no similarity structure
extropy network -s ai-adoption -c custom-network.yaml      # Load custom config
```

### What `network` does in current flow

1. Resolves study + scenario and verifies sampled agents exist.
2. Loads scenario + base population and builds merged attribute context (base + extension) for config generation.
3. Resolves config in this order:
   - explicit `--network-config`
   - latest auto-detected `scenario/<name>/*.network-config.yaml`
   - LLM-generated config (`--generate-config`, default)
   - empty config fallback (`--no-generate-config`)
4. Applies CLI overrides, quality profile defaults, and resource auto-tuning.
5. Generates network (with metrics unless `--no-metrics`).
6. Evaluates topology gate and persists result to `study.db`.
7. Optionally exports a non-canonical JSON copy with `--output`.

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | | Optional JSON export path (non-canonical) |
| `--network-config` | `-c` | path | | Custom network config YAML file |
| `--save-config` | | path | | Save the (generated or loaded) network config to YAML |
| `--generate-config` | | flag | true | Generate network config via LLM from population spec (default: enabled) |
| `--avg-degree` | | float | unset | Override target average degree (otherwise keep config value) |
| `--rewire-prob` | | float | unset | Override rewiring probability (otherwise keep config value) |
| `--seed` | | int | unset | Override config seed (if unset, generator picks seed) |
| `--validate` | | flag | false | Print validation metrics |
| `--no-metrics` | | flag | false | Skip computing node metrics (faster) |

#### Quality & Candidate Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--quality-profile` | string | `balanced` | Quality profile: `fast`, `balanced`, `strict` |
| `--candidate-mode` | string | `blocked` | Similarity candidate mode: `exact`, `blocked` |
| `--candidate-pool-multiplier` | float | 12.0 | Blocked mode candidate pool size as multiple of avg_degree |
| `--block-attr` | string (repeatable) | auto | Blocking attribute(s). If omitted, auto-selects top attributes |
| `--similarity-workers` | int | 0 | Worker processes for similarity computation (`0` = auto) |
| `--similarity-chunk-size` | int | 64 | Row chunk size for similarity worker tasks |

#### Checkpointing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | path | | DB path for checkpointing (must resolve to the same file as `study.db`) |
| `--resume` | flag | false | Resume similarity and calibration checkpoints from study.db |
| `--checkpoint-every` | int | 250 | Write checkpoint every N processed similarity rows |

#### Resource Tuning

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resource-mode` | string | `auto` | Resource tuning mode: `auto`, `manual` |
| `--safe-auto-workers/--unsafe-auto-workers` | flag | true | Conservative auto tuning for laptops/VMs |
| `--max-memory-gb` | float | | Optional memory budget cap for auto resource tuning |

### Validation/Gate Failure Behavior

- Missing sampled agents blocks network generation pre-flight.
- Invalid option values (`quality_profile`, `candidate_mode`, `topology_gate`, checkpoint mismatch) exit non-zero.
- Strict topology-gate failures (`quality.accepted=false` with strict gate and `N>=50`) exit non-zero:
  - by default, command saves a quarantined network artifact and does not report canonical success,
  - if quarantine is disabled via advanced flag, command still exits non-zero.

### Notes

- Generated configs can be auto-saved into `scenario/<name>/network-config.seed*.yaml`.
- Use `extropy query network-status <network_run_id>` to inspect calibration/progress records.

---

## extropy simulate

Run a simulation from a scenario spec.

```bash
extropy simulate -s ai-adoption
extropy simulate -s ai-adoption --seed 42 --strong anthropic/claude-sonnet-4-6
extropy simulate -s ai-adoption --fidelity high
extropy simulate -s asi-announcement --early-convergence off
```

### What `simulate` does in current flow

1. Resolves study folder and scenario.
2. Pre-flight checks required upstream artifacts:
   - sampled agents exist for scenario,
   - network edges exist for scenario,
   - persona config exists for scenario.
3. Validates runtime flags (`--resume`/`--run-id`, `--resource-mode`, `--early-convergence`).
4. Resolves effective models/rate limits from CLI overrides then config defaults.
5. Runs simulation loop:
   - seed + timeline + network exposures,
   - chunked reasoning (two-pass by default, merged with `--merged-pass`),
   - medium/high conversation interleaving,
   - timestep summary + stopping checks.
6. Persists run state to canonical `study.db` and writes results artifacts to `results/{scenario}/` (or `--output`).

### Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (auto-selects if only one exists) |
| `--output` | `-o` | path | `results/{scenario}/` | Output results directory |
| `--seed` | | int | random | Random seed for reproducibility |
| `--fidelity` | `-f` | string | `medium` | Fidelity level: `low`, `medium`, `high` |
| `--merged-pass` | | flag | false | Use single merged reasoning pass instead of two-pass (experimental) |
| `--threshold` | `-t` | int | 3 | Multi-touch threshold for re-reasoning |
| `--early-convergence` | | string | `auto` | Override convergence auto-stop policy: `auto`, `on`, `off` |
| `--chunk-size` | | int | 50 | Agents per reasoning chunk for checkpointing |

#### Early Convergence Override

`--early-convergence` controls whether convergence/quiescence auto-stops can end a run early.

- `auto` (default): use scenario YAML value (`simulation.allow_early_convergence`), else engine auto-rule.
- `on`: force-enable early convergence auto-stops for this run.
- `off`: force-disable early convergence auto-stops for this run.

Precedence:
1. CLI flag (`on`/`off`) wins.
2. Scenario YAML (`simulation.allow_early_convergence`) is used when CLI is `auto`.
3. If both are unset (`auto` + YAML `null`), engine auto-rule applies:
   `convergence/quiescence auto-stop only when no future timeline events remain`.

#### Model Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--strong` | | string | config | Strong model for Pass 1 (`provider/model`) |
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

### Runtime Notes

- `--resume` requires an explicit `--run-id`.
- Scenario lookup is scenario-name first, with legacy id fallback for older studies.
- `--early-convergence auto` uses scenario YAML value when set; otherwise runtime auto-rule applies (do not early-stop while future timeline events remain).
- `low` fidelity skips conversations; `medium` and `high` enable conversations with stricter per-agent caps at lower fidelity.
- `--retention-lite` drops full raw reasoning payload retention to reduce DB/storage volume.

### Failure Behavior

- Missing study folder/scenario/persona/agents/network fails pre-flight and exits non-zero.
- Invalid flag values (for example bad `--resource-mode` or `--early-convergence`) fail fast and exit non-zero.
- Runtime exceptions mark the simulation run as `failed` in `simulation_runs` and return non-zero.
- Successful completion updates run status to `completed` or `stopped` (when a stop condition ends early).

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
extropy query agents                              # print to stdout (uses latest run's scenario)
extropy query agents --to agents.jsonl            # write JSONL file
extropy query agents -s congestion-tax            # explicit scenario
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--to` | | path | | Write JSONL to file |
| `--scenario` | `-s` | string | auto | Scenario name (resolved from latest run if not specified) |
| `--run-id` | | string | | Simulation run ID (used to resolve scenario if not specified) |

### extropy query edges

Dump network edges.

```bash
extropy query edges --to edges.jsonl
extropy query edges -s congestion-tax --to edges.jsonl
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--to` | | path | | Write JSONL to file |
| `--scenario` | `-s` | string | auto | Scenario name (resolved from latest run if not specified) |
| `--run-id` | | string | | Simulation run ID (used to resolve scenario if not specified) |

### extropy query states

Dump agent states for a simulation run.

```bash
extropy query states --to states.jsonl
extropy query states --run-id abc123 --to states.jsonl
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | string | latest | Simulation run ID |
| `--to` | path | | Write JSONL to file |

### extropy query summary

Show study entity counts (agents, edges, simulation states, timesteps, events).

```bash
extropy query summary
extropy query summary -s congestion-tax
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--run-id` | | string | latest | Simulation run ID |
| `--scenario` | `-s` | string | auto | Scenario name (resolved from latest run if not specified) |

### extropy query network

Show network statistics (edge count, average weight, top-degree nodes).

```bash
extropy query network
extropy query network -s congestion-tax
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--scenario` | `-s` | string | auto | Scenario name (resolved from latest run if not specified) |
| `--run-id` | | string | | Simulation run ID (used to resolve scenario if not specified) |
| `--top` | | int | 10 | Number of top-degree nodes to show |

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
| `--strong` | | string | config | Strong model for Pass 1 |
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

**Exit codes:** 0 = Success (valid spec), 1 = Validation error (invalid spec), 3 = File not found

---

## extropy config

View and modify configuration.

```bash
extropy config show
extropy config set models.fast openai/gpt-5-mini
extropy config set simulation.strong anthropic/claude-sonnet-4-6
extropy config set simulation.strong openrouter/anthropic/claude-sonnet-4-6
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
extropy config set simulation.strong anthropic/claude-sonnet-4-6
extropy config set cli.mode agent  # for AI harnesses
extropy config set cli.mode human  # for terminal users (default)
```
