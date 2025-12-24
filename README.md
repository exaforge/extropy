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
entropy spec "500 German surgeons" -o surgeons.yaml
entropy sample surgeons.yaml -o agents.json
entropy network agents.json -o network.json

# Phase 2: Create a scenario
entropy scenario "AI diagnostic tool announcement" \
  -p surgeons.yaml -a agents.json -n network.json -o scenario.yaml
```

## Three Phases

| Phase | What It Does | LLM | Status |
|-------|--------------|-----|--------|
| **Phase 1: Population Creation** | Generate population specs, sample agents, create networks | OpenAI API (GPT-5) | ✅ Implemented |
| **Phase 2: Scenario Injection** | Define events, exposure rules, interaction models, outcomes | OpenAI API (GPT-5) | ✅ Implemented |
| **Phase 3: Simulation** | Agents respond; opinions evolve with social influence | LM Studio (local) | 📋 Planned |

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

## Complete Workflow Example

```bash
# Phase 1: Create population
entropy spec "5000 Netflix subscribers in the US" -o netflix_users.yaml
entropy sample netflix_users.yaml -o agents.json --seed 42
entropy network agents.json -o network.json --avg-degree 20

# Phase 2: Create scenario
entropy scenario "Netflix announces $3/month price increase for all tiers" \
  -p netflix_users.yaml -a agents.json -n network.json -o price_increase.yaml

# Validate everything
entropy validate netflix_users.yaml
entropy validate-scenario price_increase.yaml

# Phase 3: Run simulation (coming soon)
# entropy simulate price_increase.yaml -o results/
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
