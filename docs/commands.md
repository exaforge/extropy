# CLI Reference

A hands-on walkthrough of every command in order, using a single example from start to finish.

---

## The Example: Austin Congestion Tax

A city council in Austin, TX announces a **$15/day congestion tax** for driving into the downtown core during peak hours (7-10am, 4-7pm). The goal is to reduce traffic and fund public transit expansion. We want to simulate how 500 Austin commuters respond — who complies, who switches to transit, who protests, and who just pays and moves on.

This is the kind of question Entropy was built for: a heterogeneous population with different incomes, commute methods, values, and trust levels — all reacting to the same policy change.

---

## Pipeline Overview

```
entropy spec ──> entropy extend ──> entropy sample ──> entropy network ──> entropy persona ──> entropy scenario ──> entropy simulate
                                                                                                                     │              │
                                                                                                              entropy estimate    entropy results
```

Each step produces a file that feeds into the next:

| Step | Command | Input | Output |
|------|---------|-------|--------|
| 1 | `entropy spec` | Natural language description | `base.yaml` (population spec) |
| 2 | `entropy extend` | `base.yaml` + scenario description | `population.yaml` (merged spec) |
| 3 | `entropy sample` | `population.yaml` | `agents.json` (concrete agents) |
| 4 | `entropy network` | `agents.json` | `network.json` (social graph) |
| 5 | `entropy persona` | `population.yaml` + `agents.json` | `population.persona.yaml` (persona config) |
| 6 | `entropy scenario` | `population.yaml` + `agents.json` + `network.json` | `scenario.yaml` (executable spec) |
| 7 | `entropy simulate` | `scenario.yaml` | `results/` (simulation output) |

You can also run `entropy validate` at any point to check a spec file, `entropy estimate` to preview simulation cost, and `entropy results` to inspect simulation output.

---

## Configuration

Before starting, configure your providers and models. You can mix and match providers for different phases of the pipeline.

```bash
# Set pipeline (steps 1-6) to use Claude
entropy config set pipeline.provider claude

# Set simulation (step 7) to use OpenAI
entropy config set simulation.provider openai

# Optionally override the simulation model
entropy config set simulation.model gpt-5-mini

# View current config
entropy config show
```

### API Keys

Set your API keys as environment variables (or in a `.env` file):

```bash
export ANTHROPIC_API_KEY=sk-ant-...       # For Claude
export OPENAI_API_KEY=sk-...              # For OpenAI
```

### Programmatic Configuration (Package Use)

When using entropy as a library, configure programmatically — no files needed:

```python
from entropy.config import configure, EntropyConfig, PipelineConfig, SimZoneConfig

configure(EntropyConfig(
    pipeline=PipelineConfig(provider="claude"),
    simulation=SimZoneConfig(provider="openai", model="gpt-5-mini"),
))
```

---

## Step 1: Define the Base Population

```bash
entropy spec "500 Austin TX commuters who drive into downtown for work" \
  -o austin/base.yaml
```

This takes a natural language description and builds a **population specification** — a YAML file describing who these people are, statistically.

### What happens under the hood

1. **Sufficiency check** — Extracts population size (500), geography (Austin, TX), and checks if the description is specific enough.
2. **Attribute discovery** — An LLM identifies 25-40 relevant attributes: `age`, `income`, `zip_code`, `commute_distance_miles`, `vehicle_type`, `has_transit_access`, etc.
3. **Distribution research** — For each attribute, researches real-world distributions using web search and LLM reasoning. e.g., "Austin median household income is $85,000" becomes `distribution: lognormal(mean=85000, std=35000)`.
4. **Constraint binding** — Resolves dependencies between attributes (e.g., `commute_time` depends on `commute_distance_miles` and `commute_method`) and determines a valid sampling order.
5. **Validation** — Checks for circular dependencies, invalid distributions, and structural issues.

### Human checkpoints

The command pauses twice for confirmation:
1. After attribute discovery — review the list of attributes before expensive distribution research.
2. After spec assembly — review the final spec before saving.

Use `-y` to skip both prompts (useful for scripting).

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `description` | Natural language population description |
| **Opt** | `--output` / `-o` | Output YAML file path **(required)** |
| **Opt** | `--yes` / `-y` | Skip confirmation prompts |

### Output

A YAML file (`base.yaml`) with:
- **Meta** — size, geography, description
- **Attributes** — each with type, distribution, dependencies, constraints, and grounding sources
- **Sampling order** — topologically sorted for correct dependency resolution

---

## Step 2: Extend with Scenario Attributes

```bash
entropy extend austin/base.yaml \
  -s "Response to a new $15/day downtown congestion tax during peak hours" \
  -o austin/population.yaml
```

The base spec knows *who* these commuters are (demographics, commute patterns). This step adds *how they'll think about the scenario* — behavioral and psychological attributes that don't exist in any census.

### What it adds

Given the base attributes and the scenario description, the LLM discovers new attributes like:
- `price_sensitivity` — derived from `income` and `monthly_transportation_cost`
- `environmental_values` — how much they care about emissions reduction
- `trust_in_local_government` — affects whether they see the tax as legitimate
- `transit_switching_feasibility` — derived from `has_transit_access` and `commute_distance_miles`
- `protest_propensity` — derived from `political_engagement` and `price_sensitivity`

These new attributes can **depend on base attributes**, creating realistic correlations. A low-income commuter with no transit access will have different `protest_propensity` than a high-income commuter who could easily switch to rail.

### Persona template

This step also generates a **persona template** — a natural language template that will be filled in per-agent during simulation:

> *"You are a {age}-year-old {occupation} living in {neighborhood}. You commute {commute_distance_miles} miles to downtown Austin by {commute_method}. Your household income is ${income}. You {transit_access_description}..."*

This is what the LLM reads when reasoning as each agent.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `base_spec` | Path to base population spec YAML |
| **Opt** | `--scenario` / `-s` | Scenario description **(required)** |
| **Opt** | `--output` / `-o` | Output merged spec YAML **(required)** |
| **Opt** | `--yes` / `-y` | Skip confirmation prompts |

### Output

A merged YAML file (`population.yaml`) containing all base attributes + new scenario attributes + persona template. This is the complete population definition.

---

## Step 3: Sample Concrete Agents

```bash
entropy sample austin/population.yaml \
  -o austin/agents.json \
  -n 500 \
  --seed 42
```

The population spec defines *distributions* (e.g., "age: Normal(38, 12)"). This step **samples** 500 concrete agents from those distributions, respecting all dependencies and constraints.

### What you get

Each agent is a dictionary of concrete values:

```json
{
  "agent_id": "agent_001",
  "age": 34,
  "income": 62000,
  "zip_code": "78745",
  "commute_method": "personal_vehicle",
  "commute_distance_miles": 14.2,
  "has_transit_access": false,
  "price_sensitivity": 0.78,
  "trust_in_local_government": 0.35,
  "environmental_values": 0.42
}
```

Agent 001 is a 34-year-old making $62k, driving 14 miles from south Austin with no transit access. High price sensitivity, low government trust. This person will react very differently to the congestion tax than a $150k tech worker who lives near a rail stop.

### The `--report` flag

Add `--report` / `-r` to see distribution summaries after sampling — means, standard deviations, and top categorical values. Useful for sanity-checking that sampled agents match the spec.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `spec_file` | Population spec YAML to sample from |
| **Opt** | `--output` / `-o` | Output file path (.json or .db) **(required)** |
| **Opt** | `--count` / `-n` | Number of agents (defaults to `spec.meta.size`) |
| **Opt** | `--seed` | Random seed for reproducibility |
| **Opt** | `--format` / `-f` | Output format: `json` or `sqlite` (default: `json`) |
| **Opt** | `--report` / `-r` | Show distribution summaries after sampling |
| **Opt** | `--skip-validation` | Skip spec validation before sampling |

### Output

A JSON file (`agents.json`) or SQLite database (`agents.db`) containing all sampled agents with their attributes.

---

## Step 4: Build the Social Network

```bash
entropy network austin/agents.json \
  -o austin/network.json \
  -p austin/population.yaml \
  --seed 42
```

Agents don't exist in isolation. This step creates a **social network graph** connecting agents based on attribute similarity — people who live in the same neighborhood, work in similar industries, or share commute patterns are more likely to be connected.

### Network configuration

The network's social structure is driven by a `NetworkConfig` that defines which attributes create connections, what types of relationships exist, and who influences whom. Three ways to provide it:

1. **LLM-generated** (`-p population.yaml`) — The LLM reads the population spec and generates attribute weights, edge type rules, influence factors, and degree multipliers tailored to that population.
2. **Manual YAML** (`-c config.yaml`) — Load a hand-crafted or previously saved config file.
3. **No config** — Produces a flat network with no similarity structure (all edges are "peer" type).

Auto-detection: if `{population_stem}.network-config.yaml` exists alongside the population file, it's loaded automatically.

Save a generated config for inspection/editing with `--save-config`:

```bash
entropy network austin/agents.json -o austin/network.json \
  -p austin/population.yaml --save-config austin/network-config.yaml
```

### How connections form

The network uses a **Watts-Strogatz small-world model** with attribute-based similarity weighting:

- **Homophily** — Agents with matching attributes (defined by `attribute_weights` in the config) get higher connection probability. The LLM picks which attributes matter for each population.
- **Weak ties** — Random rewiring (controlled by `--rewire-prob`) creates cross-cluster bridges, modeling how information spreads beyond tight-knit groups.
- **Edge types** — Connections are typed via priority-ordered rules (e.g., same workplace → `colleague`, same neighborhood → `neighbor`). These matter during simulation because information spreads differently through close colleagues vs. acquaintances.
- **Influence asymmetry** — Influence factors define who sways whom (e.g., senior employees influence junior ones more). Supports ordinal, boolean, and numeric factor types.

### The `--validate` flag

Add `-v` to print network quality metrics:

```bash
entropy network austin/agents.json -o austin/network.json -p austin/population.yaml --validate
```

This shows clustering coefficient, average path length, modularity, and flags anything outside expected ranges for a realistic social network.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `agents_file` | Agents JSON file |
| **Opt** | `--output` / `-o` | Output network JSON file **(required)** |
| **Opt** | `--population` / `-p` | Population spec YAML — generates network config via LLM |
| **Opt** | `--network-config` / `-c` | Custom network config YAML file |
| **Opt** | `--save-config` | Save the generated/loaded config to YAML |
| **Opt** | `--avg-degree` | Target average connections per agent (default: `20`) |
| **Opt** | `--rewire-prob` | Watts-Strogatz rewiring probability (default: `0.05`) |
| **Opt** | `--seed` | Random seed for reproducibility |
| **Opt** | `--validate` / `-v` | Print network validation metrics |
| **Opt** | `--no-metrics` | Skip computing node metrics (faster for large populations) |

### Output

A JSON file (`network.json`) containing nodes (agent IDs) and weighted, typed edges.

---

## Step 5: Generate Persona Configuration

```bash
entropy persona austin/population.yaml \
  --agents austin/agents.json \
  -o austin/population.persona.yaml
```

This step generates a **persona configuration** — instructions for how to render each agent's attributes into a first-person narrative that the LLM reads during simulation.

### Why this matters

The persona is what the LLM sees when reasoning as an agent. A flat list of attributes like `age: 34, income: 62000, price_sensitivity: 0.78` doesn't help the LLM *embody* the agent. The persona system converts these into first-person statements that create genuine perspective-taking:

> *"I'm 34 years old... I'm much more price-sensitive than most people..."*

The difference between "puppetry" (the LLM referencing external data) and "embodiment" (the LLM internalizing a worldview) is critical for simulation accuracy.

### The 5-step generation pipeline

1. **Structure** — Classify each attribute as `concrete` (keep exact values) or `relative` (position against population), and group them thematically.

2. **Boolean phrasings** — Generate true/false phrases: *"I own my home"* vs *"I rent my home"*.

3. **Categorical phrasings** — Generate per-option phrases: *"I drive a pickup truck"*, *"I take the bus"*.

4. **Relative phrasings** — Generate 5-tier positioning labels based on z-scores: *"I'm far more price-sensitive than most people"* (z > 1) vs *"I'm about average"* (|z| < 0.3).

5. **Concrete phrasings** — Generate templates with format specs: *"I drive {value} miles to downtown"* with `.1f` formatting, or *"I start work around {value}"* with `time12` formatting (8.5 -> "8:30 AM").

### Scalability

The persona config is generated **once per population** via LLM, then applied to all agents computationally. There are no per-agent LLM calls for persona rendering — just template substitution and z-score lookups.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `spec_file` | Population spec YAML |
| **Opt** | `--agents` / `-a` | Sampled agents JSON (for population statistics) |
| **Opt** | `--output` / `-o` | Output path (default: `{spec_stem}.persona.yaml`) |
| **Opt** | `--preview` / `--no-preview` | Show sample persona before saving (default: on) |
| **Opt** | `--agent` | Which agent to preview (default: `0`) |
| **Opt** | `--yes` / `-y` | Skip confirmation prompts |
| **Opt** | `--show` / `-s` | Preview existing persona config without regenerating |

### Output

A YAML file (`population.persona.yaml`) containing:
- **Intro template** — Narrative opening paragraph template
- **Treatments** — Per-attribute classification (concrete vs relative)
- **Groups** — Thematic groupings with labels ("About Me", "My Commute", etc.)
- **Phrasings** — Templates for boolean, categorical, relative, and concrete attributes
- **Population stats** — Mean/std/min/max for relative positioning

The simulation engine auto-detects this file when running — no need to pass it explicitly if it follows the naming convention `{population_stem}.persona.yaml`.

---

## Step 6: Compile the Scenario

```bash
entropy scenario \
  -p austin/population.yaml \
  -a austin/agents.json \
  -n austin/network.json \
  -o austin/scenario.yaml
```

This compiles everything into an **executable scenario specification** — the complete instruction set for the simulation engine.

### What gets generated

1. **Event definition** — The congestion tax announcement: event type (`policy_change`), source (`Austin City Council`), credibility score, ambiguity level.

2. **Seed exposure rules** — How agents first learn about the event:
   - *Official city notice* — broadcast to all agents at timestep 0
   - *Local news coverage* — broadcast with high reach at timestep 0
   - *Social media discussion* — organic spread starting timestep 1
   - *Employer HR notification* — targeted to downtown office workers at timestep 1
   - *Neighborhood group chats* — targeted by `zip_code` clusters at timestep 2

3. **Interaction model** — How agents discuss and process the event (broadcast response, direct conversation, etc.)

4. **Spread configuration** — How information and opinions propagate through the network. Includes edge-type modifiers (e.g., neighbor connections amplify sharing about local policy).

5. **Outcome definitions** — What to measure: `compliance_intent` (categorical: comply/switch_transit/protest/avoid), `sentiment` (float: -1 to 1), `willingness_to_pay` (boolean), etc.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Opt** | `--population` / `-p` | Population spec YAML **(required)** |
| **Opt** | `--agents` / `-a` | Sampled agents JSON **(required)** |
| **Opt** | `--network` / `-n` | Network JSON **(required)** |
| **Opt** | `--description` / `-d` | Scenario description (defaults to spec metadata) |
| **Opt** | `--output` / `-o` | Output path (defaults to `{population_stem}.scenario.yaml`) |
| **Opt** | `--yes` / `-y` | Skip confirmation prompts |

### Output

A YAML file (`scenario.yaml`) containing the complete scenario specification: event, exposure rules, interaction model, spread config, outcomes, and simulation parameters.

---

## Step 7: Run the Simulation

```bash
entropy simulate austin/scenario.yaml \
  -o austin/results/ \
  --seed 42
```

This is where it all comes together. The simulation engine loads the scenario spec, hydrates agents with their personas, and runs timestep-by-timestep simulation with LLM-powered reasoning.

### What happens each timestep

1. **Seed exposure** — Agents learn about the congestion tax through channels defined in the scenario (city notices, news, social media).
2. **Network propagation** — Exposed agents share information through their social connections. Edge types and spread modifiers control how fast and far information travels.
3. **Two-pass agent reasoning** — Each newly-exposed agent receives their persona + the event description + what they've heard from connections. **Pass 1**: the agent role-plays their reaction in natural language (no enums, no anchoring). **Pass 2**: a cheap model classifies the freeform response into outcome categories. This eliminates central tendency bias. Agents are processed in chunks (default 50, configurable via `--chunk-size`) with per-chunk SQLite commits for crash safety.
4. **State update** — Agent states are updated with position, sentiment, conviction, and a public statement summarizing their stance. Agents who receive information from multiple sources may re-evaluate (controlled by `--threshold`).
5. **Stopping check** — The engine checks if exposure has saturated, opinions have converged, or max timesteps are reached.

### Live progress display

In normal mode, the CLI shows a live-updating display with per-agent progress and decision distribution bars:

```
Timestep 1/100 | 267/513 agents (52%) | Exposure: 65.3% | 5m 23s

  pay_for_extra_members   52% ████████████████░░░░
  remove_shared_access    31% ██████████░░░░░░░░░░
  cancel_or_downgrade     12% ████░░░░░░░░░░░░░░░░
  switch_to_competitor     5% ██░░░░░░░░░░░░░░░░░░
```

Position counts are cumulative across all timesteps. In verbose mode (`-v`), per-agent log lines include the agent's position and the engine logs periodic summary blocks every 50 agents with distribution and averages.

### Checkpointing and resume

The simulation checkpoints after each chunk of agents. If the process crashes or is interrupted (`Ctrl-C`), rerunning the same command with the same `--output` directory automatically resumes from where it left off — completed timesteps are skipped, and already-processed agents within a partial timestep are not re-reasoned.

### What emerges

Agent 001 (low-income, no transit, south Austin) reasons: *"I can't afford $15/day, that's $300/month. There's no bus from my neighborhood. This feels targeted at people like me."* Outcome: **protest**.

Agent 247 (tech worker, lives near rail, $150k) reasons: *"$15 is annoying but I could take the train. Less traffic sounds nice actually."* Outcome: **switch_transit**.

Agent 389 (small business owner, downtown) reasons: *"My employees can't get to work if they can't afford to drive in. This will kill my business."* Outcome: **protest** — but for completely different reasons than Agent 001.

These aren't scripted responses. They emerge from each agent's unique combination of attributes, persona, and reasoning.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `scenario_file` | Scenario spec YAML |
| **Opt** | `--output` / `-o` | Output results directory **(required)** |
| **Opt** | `--model` / `-m` | LLM model for both passes (default: from `entropy config`) |
| **Opt** | `--pivotal-model` | Model override for Pass 1 reasoning |
| **Opt** | `--routine-model` | Model override for Pass 2 classification |
| **Opt** | `--threshold` / `-t` | Multi-touch threshold for re-reasoning (default: `3`) |
| **Opt** | `--rate-tier` | Provider rate limit tier 1-4 (default: from config) |
| **Opt** | `--rpm-override` | Override requests per minute limit |
| **Opt** | `--tpm-override` | Override tokens per minute limit |
| **Opt** | `--chunk-size` | Agents per reasoning chunk for checkpointing (default: `50`) |
| **Opt** | `--seed` | Random seed for reproducibility |
| **Opt** | `--persona` / `-p` | Persona config YAML (auto-detected if not specified) |
| **Opt** | `--quiet` / `-q` | Suppress progress output |
| **Opt** | `--verbose` / `-v` | Show detailed logs |
| **Opt** | `--debug` | Show debug-level logs |

### Output

A results directory containing:
- `simulation.db` — SQLite database with full simulation state
- `timeline.jsonl` — Event-by-event timeline
- `agent_states.json` — Final state of every agent
- `by_timestep.json` — Per-timestep metrics (exposure, sentiment, conviction, position distributions)
- `outcome_distributions.json` — Aggregate outcome distributions
- `meta.json` — Run configuration and metadata

---

## Viewing Results

```bash
entropy results austin/results/
```

Display a summary of simulation outcomes — exposure rates, outcome distributions, and convergence information.

### Segment by attribute

```bash
entropy results austin/results/ --segment income
```

Break down outcomes by an attribute. This is where the insights live: *"Low-income commuters (<$50k) are 4x more likely to protest than high-income commuters (>$100k), who mostly comply or switch to transit."*

### Timeline view

```bash
entropy results austin/results/ --timeline
```

See how opinions evolved over time — when protest sentiment peaked, when transit-switching accelerated, etc.

### Single agent deep-dive

```bash
entropy results austin/results/ --agent agent_001
```

Inspect one agent's full reasoning chain: what they heard, from whom, when, and how their position evolved.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `results_dir` | Results directory from simulation |
| **Opt** | `--segment` / `-s` | Attribute to segment results by |
| **Opt** | `--timeline` / `-t` | Show timeline view |
| **Opt** | `--agent` / `-a` | Show single agent details |

---

## Utility: Validate Specs

```bash
entropy validate austin/population.yaml        # population spec
entropy validate austin/scenario.yaml           # scenario spec (auto-detected)
entropy validate austin/population.yaml --strict  # treat warnings as errors
```

Validate a spec file at any point in the pipeline. Auto-detects file type based on naming: `*.scenario.yaml` runs scenario spec validation, `*.yaml` runs population spec validation. Checks for structural issues, distribution validity, formula syntax, dependency cycles, and scenario-specific rules.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `spec_file` | Spec file to validate (`.yaml` or `.scenario.yaml`) |
| **Opt** | `--strict` | Treat warnings as errors (population specs only) |

---

## Estimate Simulation Cost

```bash
entropy estimate austin/scenario.yaml
entropy estimate austin/scenario.yaml --verbose
entropy estimate austin/scenario.yaml --pivotal-model gpt-5 --routine-model gpt-5-mini
```

Predict the cost of a simulation run without making any API calls. Uses a simplified SIR-like propagation model to estimate how many agents reason per timestep, then calculates token counts and USD costs from the model pricing database.

### What you get

```
COST ESTIMATE: austin_congestion_tax
──────────────────────────────────────
Population: 500 agents | Avg degree: 20
Max timesteps: 50 | Effective: 38 (early stopping predicted)

Models: gpt-5 (Pass 1) / gpt-5-mini (Pass 2)

LLM CALLS
  Pass 1 (reasoning):     1,847 calls
  Pass 2 (classification): 1,847 calls

TOKENS
  Pass 1 input:    2.4M tokens
  Pass 1 output:   184K tokens
  Pass 2 input:    923K tokens
  Pass 2 output:   92K tokens

COST
  Pass 1:  $4.12
  Pass 2:  $0.38
  Total:   $4.50
```

### The `--verbose` flag

Shows a per-timestep breakdown: how many agents are newly exposed, how many reason, and cumulative cost at each timestep. Useful for understanding the cost curve and where early stopping kicks in.

### Arguments & options

| | Name | Description |
|---|---|---|
| **Arg** | `scenario_file` | Scenario spec YAML |
| **Opt** | `--model` / `-m` | Model for both passes |
| **Opt** | `--pivotal-model` | Model for Pass 1 reasoning |
| **Opt** | `--routine-model` | Model for Pass 2 classification |
| **Opt** | `--threshold` / `-t` | Multi-touch threshold (default: `3`) |
| **Opt** | `--verbose` / `-v` | Show per-timestep breakdown |

---

## Managing Configuration

```bash
entropy config show
entropy config set pipeline.provider claude
entropy config set simulation.model gpt-5-mini
entropy config reset
```

Entropy uses a **two-zone configuration** system. The **pipeline** zone controls which provider and models are used for population and scenario building (steps 1-6). The **simulation** zone controls agent reasoning (step 7). This lets you use a powerful model for building (e.g., Claude) and a fast/cheap model for simulation (e.g., GPT-5-mini).

Three providers are supported: `openai`, `claude`, and `azure_openai`.

Config is stored at `~/.config/entropy/config.json` and managed exclusively through this command.

### Available Keys

| Key | Description | Default |
|-----|-------------|---------|
| `pipeline.provider` | LLM provider for steps 1-6 (`openai`, `claude`, or `azure_openai`) | `openai` |
| `pipeline.model_simple` | Model for simple calls (sufficiency checks) | provider default |
| `pipeline.model_reasoning` | Model for reasoning calls (attribute selection, hydration) | provider default |
| `pipeline.model_research` | Model for research calls (web search + reasoning) | provider default |
| `simulation.provider` | LLM provider for step 7 (`openai`, `claude`, or `azure_openai`) | `openai` |
| `simulation.model` | Model for agent reasoning | provider default |
| `simulation.pivotal_model` | Model for Pass 1 (role-play reasoning) | provider default |
| `simulation.routine_model` | Model for Pass 2 (classification) | provider default |
| `simulation.max_concurrent` | Max concurrent LLM calls during simulation | `50` |
| `simulation.rate_tier` | Rate limit tier (1-4, higher = more generous limits) | `None` (Tier 1) |
| `simulation.rpm_override` | Override requests per minute limit | `None` |
| `simulation.tpm_override` | Override tokens per minute limit | `None` |
| `simulation.api_format` | API format override (`responses` for OpenAI, `chat_completions` for Azure) | `""` (auto) |

### Resolution Order

Config values are resolved in this order (first wins):

1. Programmatic (`EntropyConfig` constructed in code / CLI flag overrides)
2. Environment variable (e.g., `SIMULATION_MODEL`, `PIPELINE_PROVIDER`)
3. Config file (`~/.config/entropy/config.json`)
4. Hardcoded defaults

### Environment Variables

API keys are always read from environment variables (never stored in config):

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Azure API version (default: `2025-03-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT` | Azure OpenAI deployment name |

Other environment variable overrides:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | Legacy: override both pipeline and simulation provider |
| `PIPELINE_PROVIDER` | Override pipeline provider |
| `SIMULATION_PROVIDER` | Override simulation provider |
| `MODEL_SIMPLE` | Override pipeline simple model |
| `MODEL_REASONING` | Override pipeline reasoning model |
| `MODEL_RESEARCH` | Override pipeline research model |
| `SIMULATION_MODEL` | Override simulation model |
| `SIMULATION_PIVOTAL_MODEL` | Override Pass 1 model |
| `SIMULATION_ROUTINE_MODEL` | Override Pass 2 model |
| `SIMULATION_RATE_TIER` | Override rate limit tier |
| `SIMULATION_RPM_OVERRIDE` | Override RPM limit |
| `SIMULATION_TPM_OVERRIDE` | Override TPM limit |
| `SIMULATION_API_FORMAT` | Override API format |
| `DB_PATH` | Override database path |
| `DEFAULT_POPULATION_SIZE` | Override default population size |

---

## Quick Reference

```bash
# Full pipeline
entropy spec "500 Austin TX commuters who drive into downtown for work" -o austin/base.yaml
entropy extend austin/base.yaml -s "Response to a $15/day downtown congestion tax" -o austin/population.yaml
entropy sample austin/population.yaml -o austin/agents.json --seed 42
entropy network austin/agents.json -o austin/network.json -p austin/population.yaml --seed 42
entropy persona austin/population.yaml --agents austin/agents.json
entropy scenario -p austin/population.yaml -a austin/agents.json -n austin/network.json -o austin/scenario.yaml

# Estimate cost before running
entropy estimate austin/scenario.yaml

# Run simulation
entropy simulate austin/scenario.yaml -o austin/results/ --seed 42

# View results
entropy results austin/results/
entropy results austin/results/ --segment income
entropy results austin/results/ --timeline

# Validate at any point
entropy validate austin/population.yaml
entropy validate austin/scenario.yaml
```
