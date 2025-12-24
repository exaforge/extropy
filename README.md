# Entropy

Simulate how populations respond to scenarios. Create synthetic populations grounded in real-world data, simulate how they react to events, and watch opinions evolve through social networks over time.

## Installation

```bash
# Clone and install
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Quick Start

```bash
# Phase 1: Generate a population and sample agents
entropy spec "500 German surgeons" -o surgeons_base.yaml
entropy overlay surgeons_base.yaml -s "AI diagnostic tool adoption" -o surgeons.yaml
entropy sample surgeons.yaml -o agents.json
entropy network agents.json -o network.json

# Phase 2: Create a scenario
entropy scenario "AI diagnostic tool announcement" \
  -p surgeons.yaml -a agents.json -n network.json -o scenario.yaml

# Phase 3: Run simulation
entropy simulate scenario.yaml -o results/
entropy results results/
```

## Three Phases

| Phase                            | What It Does                                                | LLM                | Status         |
| -------------------------------- | ----------------------------------------------------------- | ------------------ | -------------- |
| **Phase 1: Population Creation** | Generate population specs, sample agents, create networks   | OpenAI API (GPT-5) | ✅ Implemented |
| **Phase 2: Scenario Injection**  | Define events, exposure rules, interaction models, outcomes | OpenAI API (GPT-5) | ✅ Implemented |
| **Phase 3: Simulation**          | Agents respond; opinions evolve with social influence       | OpenAI API (GPT-5) | ✅ Implemented |

---

## End-to-End Sample Run

This walkthrough demonstrates the complete Entropy pipeline from creating a population to running a simulation.

### Step 1: Create Base Population Spec

Generate a base population specification from a natural language description. The architect layer discovers relevant demographic and population-specific attributes.

```bash
entropy spec "500 German surgeons" -o surgeons_base.yaml
```

**What happens:**

- Validates description has enough context
- Discovers 25-40 base attributes (age, specialty, income, etc.)
- Researches distributions from authoritative sources
- Creates a YAML spec with sampling instructions

**Output:** `surgeons_base.yaml` - Base population blueprint

### Step 2: Add Scenario-Specific Attributes (Overlay)

Layer scenario-specific attributes on the base population. This adds attributes relevant to how agents will respond to the scenario.

```bash
entropy overlay surgeons_base.yaml \
  -s "AI diagnostic tool adoption for surgery planning" \
  -o surgeons.yaml
```

**What happens:**

- Loads base population spec
- Discovers NEW scenario-specific attributes (e.g., `tech_adoption_tendency`, `ai_trust_level`, `workflow_flexibility`)
- New attributes can depend on base attributes
- Researches distributions for new attributes
- Merges into final spec with recomputed sampling order

**Output:** `surgeons.yaml` - Complete population spec with scenario attributes

### Step 3: Validate Population Spec (Optional)

Check the spec for structural correctness before sampling.

```bash
entropy validate surgeons.yaml
```

**What happens:**

- Checks distribution parameters are valid
- Verifies dependencies are resolvable
- Confirms no circular references
- Validates modifier conditions

### Step 4: Sample Agents

Generate synthetic agents from the population spec.

```bash
entropy sample surgeons.yaml -o agents.json --seed 42
```

**What happens:**

- Samples each attribute in topological order
- Applies conditional modifiers based on other attributes
- Computes derived attributes from formulas
- Generates unique agent IDs

**Output:** `agents.json` - List of 500 agent dictionaries

### Step 5: Generate Social Network

Create a realistic social network connecting the agents.

```bash
entropy network agents.json -o network.json --avg-degree 20
```

**What happens:**

- Groups agents by profession/location for clustering
- Uses Watts-Strogatz model for small-world properties
- Weights edges by relationship type
- Computes network metrics (centrality, clustering)

**Output:** `network.json` - Nodes and weighted edges

### Step 6: Create Scenario Spec

Transform a natural language scenario into a machine-readable spec.

```bash
entropy scenario "Hospital announces new AI diagnostic tool for surgery planning" \
  -p surgeons.yaml -a agents.json -n network.json -o ai_tool.yaml
```

**What happens:**

- Parses event (type, content, credibility, emotional valence)
- Generates exposure rules (who learns when, via what channel)
- Defines interaction model (how agents discuss)
- Specifies outcomes to measure (adoption_intent, sentiment, etc.)

**Output:** `ai_tool.yaml` - Complete scenario specification

### Step 7: Validate Scenario (Optional)

Verify the scenario spec is consistent with population and network.

```bash
entropy validate-scenario ai_tool.yaml
```

**What happens:**

- Checks attribute references in `when` clauses exist
- Verifies edge types match network
- Validates probabilities are in [0, 1]
- Confirms referenced files exist

### Step 8: Run Simulation

Execute the scenario against the population.

```bash
entropy simulate ai_tool.yaml -o results/ --seed 42
```

**What happens:**

- Initializes agent states in SQLite database
- For each timestep:
  - Applies seed exposures from Phase 2 rules
  - Propagates through network from sharing agents
  - Agents reason via LLM about the event
  - Updates positions, sentiments, sharing intent
- Checks stopping conditions (exposure rate, convergence)
- Exports all results

**Output:** `results/` directory with:

- `simulation.db` - SQLite database
- `timeline.jsonl` - Event stream
- `agent_states.json` - Final states
- `meta.json` - Run configuration

### Step 9: View Results

Analyze the simulation outcomes.

```bash
# Summary view
entropy results results/

# Segment by specialty
entropy results results/ --segment specialty

# View timeline
entropy results results/ --timeline

# Single agent details
entropy results results/ --agent agent_042
```

---

### Complete Copy-Paste Example

```bash
# ═══════════════════════════════════════════════════════════════
# PHASE 1: POPULATION CREATION
# ═══════════════════════════════════════════════════════════════

# Create base population spec (interactive, ~2-3 minutes)
entropy spec "500 German surgeons" -o surgeons_base.yaml

# Add scenario-specific attributes via overlay (~1-2 minutes)
entropy overlay surgeons_base.yaml \
  -s "AI diagnostic tool adoption for surgery planning" \
  -o surgeons.yaml

# Validate the merged spec
entropy validate surgeons.yaml

# Sample agents
entropy sample surgeons.yaml -o agents.json --seed 42 --report

# Generate network
entropy network agents.json -o network.json --avg-degree 20 --validate

# ═══════════════════════════════════════════════════════════════
# PHASE 2: SCENARIO COMPILATION
# ═══════════════════════════════════════════════════════════════

# Create scenario spec
entropy scenario "Hospital announces mandatory AI diagnostic tool for all surgeries" \
  -p surgeons.yaml -a agents.json -n network.json -o ai_mandate.yaml

# Validate scenario
entropy validate-scenario ai_mandate.yaml

# ═══════════════════════════════════════════════════════════════
# PHASE 3: SIMULATION
# ═══════════════════════════════════════════════════════════════

# Run simulation
entropy simulate ai_mandate.yaml -o results/ --seed 42

# View results
entropy results results/
entropy results results/ --segment specialty
entropy results results/ --timeline
```

---

### Non-Interactive Mode (Automation)

For scripts and CI/CD, use `--yes` to skip confirmation prompts:

```bash
entropy spec "500 German surgeons" -o surgeons_base.yaml --yes
entropy overlay surgeons_base.yaml -s "AI tool adoption" -o surgeons.yaml --yes
entropy sample surgeons.yaml -o agents.json --seed 42
entropy network agents.json -o network.json --seed 42
entropy scenario "AI tool announcement" \
  -p surgeons.yaml -a agents.json -n network.json -o scenario.yaml --yes
entropy simulate scenario.yaml -o results/ --seed 42 --quiet
```

---

## Phase 1: Population Creation

Phase 1 creates synthetic populations: defining attributes, sampling agents, and generating social networks.

### `entropy spec`

Generate a population spec from a natural language description.

```bash
entropy spec "<description>" -o <output.yaml> [--yes]
```

**Arguments:**

- `description`: Natural language population description (e.g., "2000 Netflix subscribers in the US")
- `-o, --output`: Output YAML file path (required)
- `-y, --yes`: Skip confirmation prompts (for automation)

**Pipeline:**

1. **Step 0**: Validates description has enough context
2. **Step 1**: Discovers 25-40 relevant attributes
3. **Human checkpoint**: Review/edit attributes
4. **Step 2**: Researches distributions (with sub-step progress):
   - 2a: Independent attribute distributions
   - 2b: Derived attribute formulas
   - 2c: Conditional base distributions
   - 2d: Conditional modifiers
5. **Step 3**: Binds constraints and determines sampling order
6. **Human checkpoint**: Review and save spec

**Example:**

```bash
entropy spec "500 German surgeons" -o surgeons.yaml
entropy spec "1000 Indian farmers in Maharashtra" -o farmers.yaml
```

### `entropy overlay`

Layer scenario-specific attributes on an existing population spec.

```bash
entropy overlay <base.yaml> -s "<scenario>" -o <output.yaml> [--yes]
```

**Arguments:**

- `base`: Path to existing population spec YAML
- `-s, --scenario`: Scenario description
- `-o, --output`: Output YAML file path (required)
- `-y, --yes`: Skip confirmation prompts

**How it works:**

- Loads base population (e.g., 35 attributes)
- Discovers NEW scenario-specific attributes (e.g., 8 attributes)
- New attributes can depend on base attributes
- Merges into final spec with recomputed sampling order

**Example:**

```bash
entropy overlay farmers.yaml \
  -s "Drought-resistant seed adoption decision" \
  -o farmers_seeds.yaml
```

### `entropy validate`

Validate a population spec for structural correctness.

```bash
entropy validate <spec.yaml> [--strict]
```

**Arguments:**

- `spec`: Population spec YAML file to validate
- `--strict`: Treat warnings as errors

**Checks:**

- Type/modifier compatibility
- Range violations
- Weight validity
- Distribution parameters
- Dependencies and formulas
- Strategy consistency

**Example:**

```bash
entropy validate surgeons.yaml
entropy validate surgeons.yaml --strict
```

### `entropy fix`

Auto-fix modifier condition option references.

```bash
entropy fix <spec.yaml> [-o <output.yaml>] [--dry-run] [-c <confidence>]
```

**Arguments:**

- `spec`: Population spec YAML file to fix
- `-o, --output`: Output file (defaults to overwriting input)
- `-n, --dry-run`: Preview fixes without applying them
- `-c, --confidence`: Minimum fuzzy match confidence 0-1 (default: 0.6)

**What it fixes:**
The LLM sometimes generates modifier `when` conditions that reference categorical options with inconsistent naming (e.g., `'University hospital'` instead of `'University_hospital'`). This command uses fuzzy matching to automatically correct these mismatches.

**Example:**

```bash
# Preview fixes without applying
entropy fix surgeons.yaml --dry-run

# Fix in place
entropy fix surgeons.yaml

# Save to new file
entropy fix surgeons.yaml -o fixed.yaml

# Stricter matching (80% confidence required)
entropy fix surgeons.yaml -c 0.8
```

**Output:**

```
✓ Loaded: 500 German surgeons (35 attributes)

Found 4 fix(es):

  ai_tool_awareness[0]:
    - 'University hospital'
    + 'University_hospital' (confidence: 95%)

  income[2]:
    - 'Senior/Oberarzt'
    + 'Senior_Oberarzt' (confidence: 95%)

═══════════════════════════════════════════════════════════════
✓ Fixed spec saved to surgeons.yaml
═══════════════════════════════════════════════════════════════
```

### `entropy sample`

Generate agents from a population spec.

```bash
entropy sample <spec.yaml> -o <output.json> [options]
```

**Arguments:**

- `spec`: Population spec YAML file
- `-o, --output`: Output file path (.json or .db)
- `-n, --count`: Number of agents (default: spec.meta.size)
- `--seed`: Random seed for reproducibility
- `-f, --format`: Output format: json or sqlite
- `-r, --report`: Show distribution summaries and stats
- `--skip-validation`: Skip validator errors

**Example:**

```bash
entropy sample surgeons.yaml -o agents.json
entropy sample surgeons.yaml -n 500 -o agents.json --seed 42
entropy sample surgeons.yaml -o agents.json --report
```

### `entropy network`

Generate a social network from sampled agents.

```bash
entropy network <agents.json> -o <network.json> [options]
```

**Arguments:**

- `agents`: Agents JSON file
- `-o, --output`: Output network JSON file
- `--avg-degree`: Target average degree (default: 20.0)
- `--rewire-prob`: Watts-Strogatz rewiring probability (default: 0.05)
- `--seed`: Random seed for reproducibility
- `-v, --validate`: Print validation metrics
- `--no-metrics`: Skip computing node metrics (faster)

**Example:**

```bash
entropy network agents.json -o network.json
entropy network agents.json -o network.json --avg-degree 25 --validate
entropy network agents.json -o network.json --seed 42
```

---

## Phase 2: Scenario Compiler

Phase 2 transforms natural language scenario descriptions into machine-readable scenario specs that Phase 3 executes.

### `entropy scenario`

Create a scenario spec from a natural language description.

```bash
entropy scenario "<description>" \
  -p <population.yaml> \
  -a <agents.json> \
  -n <network.json> \
  -o <scenario.yaml> \
  [--yes]
```

**Arguments:**

- `description`: Natural language scenario (e.g., "Netflix announces $3 price increase")
- `-p, --population`: Population spec YAML file (required)
- `-a, --agents`: Sampled agents JSON file (required)
- `-n, --network`: Network JSON file (required)
- `-o, --output`: Output scenario YAML file (required)
- `-y, --yes`: Skip confirmation prompts

**Pipeline:**

1. **Step 1**: Parse scenario → Event definition (type, content, source, credibility)
2. **Step 2**: Generate seed exposure → How agents learn about the event
3. **Step 3**: Determine interaction model → How agents discuss and respond
4. **Step 4**: Define outcomes → What to measure (cancel_intent, sentiment, etc.)
5. **Step 5**: Assemble and validate → Complete scenario spec

**Example:**

```bash
entropy scenario "Netflix announces $3 price increase" \
  -p netflix_users.yaml -a agents.json -n network.json -o scenario.yaml
```

**Output preview:**

```
Creating scenario for: "Netflix announces $3 price increase"

Step 1/5: Parsing event definition... ✓
Step 2/5: Generating seed exposure rules... ✓
Step 3/5: Determining interaction model... ✓
Step 4/5: Defining outcomes... ✓
Step 5/5: Assembling scenario spec... ✓

┌──────────────────────────────────────────────────────────┐
│                  SCENARIO SPEC READY                      │
└──────────────────────────────────────────────────────────┘

Event: announcement — "Netflix announces $3/month price increase..."
Source: Netflix (credibility: 0.95)

Exposure Channels:
  • email_notification (broadcast)
  • social_media (organic)
  • word_of_mouth (organic)

Seed Exposure Rules: 3
  • email_notification: all subscribers at t=0
  • social_media: 30% exposure at t=1

Interaction Model: passive_observation + direct_conversation
Share Probability: 0.35

Outcomes:
  • cancel_intent (categorical): will_cancel, considering, staying, undecided
  • sentiment (float): -1 to 1
  • share_behavior (boolean)
  • downgrade_intent (boolean)

Simulation: 168 timesteps (hours)

[Y] Save  [n] Cancel
```

### `entropy validate-scenario`

Validate a scenario spec against its referenced files.

```bash
entropy validate-scenario <scenario.yaml>
```

**Arguments:**

- `scenario`: Scenario spec YAML file to validate

**Checks:**

- All attribute references in `when` clauses exist in population spec
- All edge type references exist in network
- All probabilities are in valid range [0, 1]
- All timesteps are valid
- Outcome definitions are complete
- Channel references are valid
- Referenced files exist and are consistent

**Example:**

```bash
entropy validate-scenario scenario.yaml
```

**Output:**

```
Validating scenario: price_increase.yaml

File References:
  ✓ Population spec: netflix_users.yaml
  ✓ Agents file: agents.json (5000 agents)
  ✓ Network file: network.json (47,832 edges)

✓ Scenario spec is valid
```

---

## Scenario Spec Structure

A generated scenario YAML spec contains:

```yaml
meta:
  name: "netflix_price_increase"
  description: "Netflix announces $3/month price increase"
  population_spec: "netflix_users.yaml"
  agents_file: "agents.json"
  network_file: "network.json"
  created_at: "2024-01-15T10:30:00Z"

event:
  type: "announcement"
  content: "Netflix announces that subscription prices will increase..."
  source: "Netflix"
  credibility: 0.95
  ambiguity: 0.2
  emotional_valence: -0.3

seed_exposure:
  channels:
    - name: "email_notification"
      description: "Direct email from Netflix to subscribers"
      reach: "broadcast"
      credibility_modifier: 1.0
    - name: "social_media"
      reach: "organic"
      credibility_modifier: 0.7

  rules:
    - channel: "email_notification"
      when: "true"
      probability: 0.98
      timestep: 0
    - channel: "social_media"
      when: "age < 45"
      probability: 0.4
      timestep: 1

interaction:
  primary_model: "passive_observation"
  secondary_model: "direct_conversation"
  description: "Users see reactions on social media and discuss with household"

spread:
  share_probability: 0.35
  share_modifiers:
    - when: "sentiment < -0.5"
      multiply: 1.8
    - when: "age < 30"
      multiply: 1.3
  decay_per_hop: 0.1

outcomes:
  suggested_outcomes:
    - name: "cancel_intent"
      type: "categorical"
      options: ["will_cancel", "seriously_considering", "might_consider", "staying"]
      required: true
    - name: "sentiment"
      type: "float"
      range: [-1.0, 1.0]
      required: true
  capture_full_reasoning: true

simulation:
  max_timesteps: 168
  timestep_unit: "hour"
  stop_conditions:
    - "exposure_rate > 0.95 and no_state_changes_for > 10"
```

---

## Population Spec Structure

A generated population YAML spec contains:

```yaml
meta:
  description: "500 German surgeons"
  size: 500
  geography: Germany

grounding:
  overall: medium
  sources_count: 12
  strong_count: 15
  medium_count: 12
  low_count: 5

attributes:
  - name: age
    type: int
    category: universal
    sampling:
      strategy: independent
      distribution:
        type: normal
        mean: 47
        std: 10
        min: 28
        max: 70
    grounding:
      level: strong
      method: researched
      source: "German Medical Association statistics"

  - name: years_experience
    type: int
    category: population_specific
    sampling:
      strategy: conditional
      distribution:
        type: normal
        mean_formula: "age - 28"
        std: 3
      modifiers:
        - when: "specialty == 'neurosurgery'"
          add: 2

sampling_order:
  - age
  - gender
  - specialty
  - years_experience
  # ... (respects dependencies)
```

---

## Phase 3: Simulation Engine

Phase 3 executes scenarios against populations, simulating opinion dynamics with agent reasoning and network propagation.

### `entropy simulate`

Run a simulation from a scenario spec.

```bash
entropy simulate <scenario.yaml> -o <results_dir> [options]
```

**Arguments:**

- `scenario`: Scenario spec YAML file (required)
- `-o, --output`: Output results directory (required)
- `-m, --model`: LLM model for agent reasoning (default: gpt-5-mini)
- `-t, --threshold`: Multi-touch threshold for re-reasoning (default: 3)
- `--seed`: Random seed for reproducibility
- `-q, --quiet`: Suppress progress output

**Pipeline:**

1. Load scenario, population spec, agents, and network
2. Initialize agent states in SQLite database
3. For each timestep:
   - Apply seed exposures based on Phase 2 rules
   - Propagate through network from sharing agents
   - Agents who are newly aware (or have N+ new exposures) reason via LLM
   - Update states with positions, sentiments, and sharing intent
4. Check stopping conditions (max timesteps, exposure rate, convergence)
5. Export results to output directory

**Example:**

```bash
entropy simulate scenario.yaml -o results/
entropy simulate scenario.yaml -o results/ --model gpt-5-nano --seed 42
```

**Output:**

```
Simulating: scenario.yaml
Output: results/
Model: gpt-5-mini | Threshold: 3

⠋ Timestep 45/168 (27%) | Exposure: 67.2% | 1m 23s

════════════════════════════════════════════════════════════
✓ Simulation complete
════════════════════════════════════════════════════════════

Duration: 32m 14s (87 timesteps)
Stopped: exposure_rate > 0.95
Reasoning calls: 18,432
Final exposure rate: 96.1%

Outcome Distributions:
  cancel_intent: will_cancel:18.4%, seriously_considering:27.1%, staying:23.3%
  sentiment: mean=-0.34

Results saved to: results/
```

### `entropy results`

Display simulation results.

```bash
entropy results <results_dir> [options]
```

**Arguments:**

- `results_dir`: Results directory from simulation (required)
- `-s, --segment`: Attribute to segment by
- `-t, --timeline`: Show timeline view
- `-a, --agent`: Show single agent details

**Views:**

1. **Summary (default):**

```bash
entropy results results/
```

Shows overall statistics, exposure rates, and outcome distributions.

2. **Segment breakdown:**

```bash
entropy results results/ --segment plan_tier
```

Breaks down results by any agent attribute.

3. **Timeline:**

```bash
entropy results results/ --timeline
```

Shows metrics over time: exposure rate, sentiment, positions.

4. **Agent details:**

```bash
entropy results results/ --agent agent_001
```

Shows single agent's attributes, state, and reasoning.

**Example output (summary):**

```
═══════════════════════════════════════════════════════════════
SIMULATION RESULTS: netflix_price_increase
═══════════════════════════════════════════════════════════════

Population: 5,000 agents
Duration: 87 timesteps
Model: gpt-5-mini

EXPOSURE
────────────────────────────────────────
Final exposure rate: 96.1%
Total exposures: 24,891
Reasoning calls: 18,432

OUTCOMES
────────────────────────────────────────
cancel_intent:
  will_cancel          18.4%  ████████░░░░░░░░░░░░
  seriously_considering 27.1%  ████████████░░░░░░░░
  might_consider       31.2%  ██████████████░░░░░░
  staying              23.3%  ██████████░░░░░░░░░░

sentiment: mean -0.34 (std 0.42)
```

### Results Directory Structure

```
results/
├── simulation.db              # SQLite database with all state
├── timeline.jsonl             # Streaming event log
├── agent_states.json          # Final state per agent
├── by_timestep.json           # Metrics over time
├── outcome_distributions.json # Final outcome distributions
└── meta.json                  # Run configuration
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (future phases)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=llama-3.2-3b
```
