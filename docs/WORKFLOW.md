# Entropy Workflow Documentation

A comprehensive guide to how Entropy works: from population spec creation through scenario overlay to agent-based simulation.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Population Creation](#phase-1-population-creation)
   - [Step 0: Context Sufficiency Check](#step-0-context-sufficiency-check)
   - [Step 1: Attribute Selection](#step-1-attribute-selection)
   - [Step 2: Distribution Hydration](#step-2-distribution-hydration-4-sub-steps)
   - [Step 3: Constraint Binding](#step-3-constraint-binding)
   - [Final Validation System](#final-validation-system)
   - [Sampling](#sampling)
   - [Network Generation](#network-generation)
3. [Phase 1.5: Scenario Overlay](#phase-15-scenario-overlay)
4. [Phase 2: Scenario Compilation](#phase-2-scenario-compilation)
5. [Phase 3: Simulation Engine](#phase-3-simulation-engine)
6. [File Structure Reference](#file-structure-reference)
7. [Data Flow Diagram](#data-flow-diagram)

---

## Overview

Entropy is an agent-based simulation framework that:

1. **Creates synthetic populations** - Generates realistic agent populations with statistically grounded attributes
2. **Defines scenarios** - Compiles natural language events into machine-readable specifications
3. **Simulates responses** - Runs LLM-powered agent reasoning with social network propagation

```
                    ┌─────────────────┐
                    │ "500 German     │
                    │  surgeons"      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  PHASE 1:       │
                    │  Population     │
                    │  Creation       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  surgeons_      │
                    │  base.yaml      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐     │              │
     │  PHASE 1.5:     │     │              │
     │  Overlay        │     │              │
     │  (Scenario-     │     │              │
     │   specific)     │     │              │
     └────────┬────────┘     │              │
              │              │              │
     ┌────────▼────────┐     │              │
     │  surgeons.yaml  │     │              │
     │  (merged)       │     │              │
     └────────┬────────┘     │              │
              │              │              │
     ┌────────▼────────┐     │              │
     │  entropy sample │─────┤              │
     │  agents.json    │     │              │
     └────────┬────────┘     │              │
              │              │              │
     ┌────────▼────────┐     │              │
     │  entropy network│─────┤              │
     │  network.json   │     │              │
     └────────┬────────┘     │              │
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  PHASE 2:       │
                    │  Scenario       │
                    │  Compiler       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  scenario.yaml  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  PHASE 3:       │
                    │  Simulation     │
                    │  Engine         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  results/       │
                    │  - simulation.db│
                    │  - timeline.jsonl│
                    │  - agent_states.json│
                    └─────────────────┘
```

---

## Phase 1: Population Creation

**CLI Command:** `entropy spec "500 German surgeons" -o surgeons_base.yaml`

**Purpose:** Transform a natural language population description into a complete, validated PopulationSpec YAML file with statistically grounded distributions.

### Step 0: Context Sufficiency Check

**File:** `entropy/population/architect/sufficiency.py`

**Purpose:** Validate that the input description has enough context to proceed.

**What it checks:**

- **WHO** - Clear identity (profession, demographic group)
- **SIZE** - Number of agents (defaults to 1000 if not specified)
- **GEOGRAPHY** - Geographic scope (optional but helpful)

**LLM Call:** `simple_call()` with `gpt-5-mini`

**Input:** Natural language description

```
"500 German surgeons"
```

**Output:** `SufficiencyResult`

```python
SufficiencyResult(
    sufficient=True,
    size=500,
    geography="Germany",
    clarifications_needed=[]
)
```

**What can go wrong:**

- Description too vague (e.g., "people", "users")
- No identifiable population group
- Returns clarification questions if insufficient

---

### Step 1: Attribute Selection

**File:** `entropy/population/architect/selector.py`

**Purpose:** Discover 25-40 relevant attributes for the population.

**LLM Call:** `reasoning_call()` with `gpt-5` (reasoning enabled)

**Attribute Categories:**

| Category            | Count | Examples                                   |
| ------------------- | ----- | ------------------------------------------ |
| Universal           | 8-12  | age, gender, income, education, location   |
| Population-specific | 10-18 | specialty, years_experience, employer_type |
| Context-specific    | 0-5   | (only if product/service mentioned)        |
| Personality         | 5-8   | Big Five traits, risk_tolerance            |

**Sampling Strategies:**

| Strategy      | Description                               | Dependencies         |
| ------------- | ----------------------------------------- | -------------------- |
| `independent` | Sampled directly from distribution        | None                 |
| `derived`     | Computed from formula (zero variance)     | Must have depends_on |
| `conditional` | Probabilistic relationship with modifiers | Must have depends_on |

**Output:** List of `DiscoveredAttribute`

```python
DiscoveredAttribute(
    name="years_experience",
    type="int",
    category="population_specific",
    description="Years of surgical experience",
    strategy="conditional",
    depends_on=["age", "specialty"]
)
```

**Automatic Fixes Applied:**

- If `strategy=independent` but has `depends_on`, switches to `conditional`
- If `strategy=derived/conditional` but no `depends_on`, switches to `independent`
- Ensures Big Five traits use exact canonical names

---

### Step 2: Distribution Hydration (4 Sub-steps)

**File:** `entropy/population/architect/hydrator.py`

**Purpose:** Research and specify distributions for all attributes.

**Key Feature: LLM Retry with Validation**

Each hydration sub-step uses the LLM retry-with-validation loop:

```
┌────────────────────────────────────────────────────────────┐
│                   LLM CALL WITH VALIDATION                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Call LLM (agentic_research or reasoning_call)          │
│                                                            │
│  2. Parse response into HydratedAttribute objects          │
│                                                            │
│  3. Run sub-step validation function:                      │
│     - validate_independent_hydration() for Step 2a         │
│     - validate_derived_hydration() for Step 2b             │
│     - validate_conditional_base() for Step 2c              │
│     - validate_modifiers() for Step 2d                     │
│                                                            │
│  4. If validation errors:                                  │
│     ├── Format errors as feedback prompt                   │
│     ├── Prepend to original prompt                         │
│     ├── Call on_retry callback (for CLI progress)          │
│     └── Re-call LLM with error context (up to max_retries) │
│                                                            │
│  5. Return validated result                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

The LLM functions support these validation parameters:

- `validator`: Callback `(dict) -> (bool, str)` to validate response
- `max_retries`: Maximum retry attempts (default: 2)
- `on_retry`: Callback `(attempt, max, error_summary)` for progress updates
- `previous_errors`: Error string prepended to prompt for retry

#### Step 2a: Independent Attributes

**Function:** `hydrate_independent()`

**LLM Call:** `agentic_research()` with `gpt-5` + web search

**What it does:**

- Searches real-world data sources
- Finds statistical distributions from government data, professional associations, academic studies
- Assigns grounding levels (strong/medium/low)

**Intermediate Validation:** `validate_independent_hydration()`

| Check                     | Error Condition                  |
| ------------------------- | -------------------------------- |
| Distribution exists       | `distribution is None`           |
| Categorical has options   | `options` list is empty          |
| Options/weights match     | `len(options) != len(weights)`   |
| Weights sum to 1.0        | `abs(sum(weights) - 1.0) > 0.02` |
| Boolean probability valid | `probability_true < 0 or > 1`    |
| Std is positive           | `std < 0`                        |
| Std is non-zero           | `std == 0` (should be derived)   |
| Min < Max                 | `min >= max`                     |
| Beta alpha/beta positive  | `alpha <= 0 or beta <= 0`        |

**Output per attribute:**

```yaml
name: age
type: int
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
```

#### Step 2b: Derived Attributes

**Function:** `hydrate_derived()`

**LLM Call:** `reasoning_call()` with `gpt-5` (NO web search)

**What it does:**

- Specifies deterministic formulas
- No variance - same inputs always produce same output

**Intermediate Validation:** `validate_derived_hydration()`

| Check                    | Error Condition                                  |
| ------------------------ | ------------------------------------------------ |
| Formula exists           | `formula is None or empty`                       |
| Formula syntax valid     | `ast.parse()` raises `SyntaxError`               |
| References in depends_on | Formula references attribute not in `depends_on` |
| References exist         | Formula references unknown attribute name        |

**Example formulas:**

```python
# Categorical binning
age_bracket = "'18-24' if age < 25 else '25-34' if age < 35 else '35-44' if age < 45 else '45+'"

# Boolean flag
is_senior = "years_experience >= 15"

# Mathematical
bmi = "weight / (height ** 2)"
```

#### Step 2c: Conditional Base Distributions

**Function:** `hydrate_conditional_base()`

**LLM Call:** `agentic_research()` with `gpt-5` + web search

**What it does:**

- Researches BASE distributions for conditional attributes
- Can use `mean_formula` for continuous dependencies
- Does NOT include modifiers yet

**Intermediate Validation:** `validate_conditional_base()`

| Check                         | Error Condition                                     |
| ----------------------------- | --------------------------------------------------- |
| Distribution exists           | `distribution is None`                              |
| mean_formula references valid | References attribute not in `depends_on`            |
| Conditional has variance      | Has `mean_formula` but no `std` (should be derived) |

**Output:**

```yaml
name: years_experience
sampling:
  strategy: conditional
  distribution:
    type: normal
    mean_formula: "age - 28" # Dynamic mean based on age
    std: 3
  depends_on: [age, specialty]
```

#### Step 2d: Conditional Modifiers

**Function:** `hydrate_conditional_modifiers()`

**LLM Call:** `agentic_research()` with `gpt-5` + web search

**What it does:**

- Specifies how base distributions shift based on dependencies
- Different modifier fields for different distribution types

**Intermediate Validation:** `validate_modifiers()`

| Check                      | Severity | Error Condition                                                    |
| -------------------------- | -------- | ------------------------------------------------------------------ |
| Condition references valid | ERROR    | `when` references attribute not in `depends_on`                    |
| Numeric uses multiply/add  | ERROR    | Numeric dist uses `weight_overrides` or `probability_override`     |
| Categorical uses weights   | ERROR    | Categorical dist uses `multiply`, `add`, or `probability_override` |
| Boolean uses probability   | ERROR    | Boolean dist uses `multiply`, `add`, or `weight_overrides`         |
| Weight keys exist          | ERROR    | `weight_overrides` key not in distribution options                 |
| Not a no-op                | WARNING  | `multiply=1.0, add=0, no overrides`                                |

**Modifier Type Rules:**

| Distribution Type                | Allowed Fields         | Forbidden Fields                       |
| -------------------------------- | ---------------------- | -------------------------------------- |
| normal, lognormal, uniform, beta | `multiply`, `add`      | weight_overrides, probability_override |
| categorical                      | `weight_overrides`     | multiply, add, probability_override    |
| boolean                          | `probability_override` | multiply, add, weight_overrides        |

**Example modifier:**

```yaml
modifiers:
  - when: "specialty == 'Neurosurgery'"
    multiply: 1.0
    add: 2 # Neurosurgeons train longer
  - when: "employer_type == 'University_hospital'"
    multiply: 1.1
    add: 1
```

#### Final Step: Strategy Consistency Check

**Function:** `validate_strategy_consistency()`

After all sub-steps complete, this validates that each attribute has the correct fields for its declared strategy:

| Strategy      | Required Fields          | Forbidden Fields               |
| ------------- | ------------------------ | ------------------------------ |
| `independent` | distribution             | formula, modifiers, depends_on |
| `derived`     | formula, depends_on      | distribution, modifiers        |
| `conditional` | distribution, depends_on | formula                        |

**Sources Collected:**

- All web search URLs are collected across retries
- Used for grounding summary

---

### Constraint Types

Constraints define rules for attribute values. There are two categories:

#### Spec-Level Constraints (`spec_expression`)

These validate the **YAML specification itself**, not individual agents. They are checked during spec validation, NOT during sampling.

**Use for:**
- `sum(weights)==1` - validates categorical weights sum to 1
- `weights[0]+weights[1]==1` - validates weight arrays
- `len(options) > 0` - validates options exist

**Example:**
```yaml
constraints:
  - type: spec_expression
    expression: sum(weights)==1
    reason: Category weights must sum to 1.
```

#### Agent-Level Constraints (`expression`)

These are validated against **each sampled agent**. Violations are reported but don't block sampling.

**Use for:**
- `children_count <= household_size - 1` - validates logical relationships between agent attributes
- `years_experience <= age - 23` - validates derived relationships

**Example:**
```yaml
constraints:
  - type: expression
    expression: children_count <= max(0, household_size - 1)
    reason: Children cannot exceed household size minus one adult.
```

#### Hard Constraints (`hard_min`, `hard_max`)

These are enforced during sampling via clamping. Values outside bounds are automatically clipped.

```yaml
constraints:
  - type: hard_min
    value: 0
    reason: Cannot be negative.
  - type: hard_max
    value: 100
    reason: Maximum allowed value.
```

---

### Dynamic Bounds with Formula

For attributes where valid bounds depend on other attributes, use `min_formula` and `max_formula`:

```yaml
children_count:
  sampling:
    strategy: conditional
    distribution:
      type: normal
      mean_formula: "max(0, household_size - 2)"
      std: 0.9
      min: 0
      max_formula: "max(0, household_size - 1)"  # Dynamic upper bound
    depends_on: [household_size]
```

**Supported distributions:** `normal`, `lognormal`, `beta`

**Formula precedence:** When both static and formula bounds are provided, **formula takes precedence**.

**Available in formulas:**
- Any attribute in `depends_on`
- Built-in functions: `max()`, `min()`, `abs()`, `round()`, `int()`, `float()`

**Benefits:**
- Guarantees zero constraint violations for the bounded attribute
- More efficient than sampling + validation + rejection
- Self-documenting specification

---

### Step 3: Constraint Binding

**File:** `entropy/population/architect/binder.py`

**Purpose:**

1. Build dependency graph
2. Detect circular dependencies
3. Compute topological sort for sampling order
4. Convert to final `AttributeSpec` objects

**Algorithm:** Kahn's algorithm for topological sort

**Output:**

- `List[AttributeSpec]` - Final attribute definitions
- `sampling_order` - Order to sample attributes
- `warnings` - Any issues found (unknown dependencies removed)

**Errors Detected:**

- `CircularDependencyError` - A depends on B, B depends on A
- Unknown dependency references (removed with warning)
- Strategy/dependency inconsistencies

---

### Final Validation System

**File:** `entropy/population/validator/`

**Structure:**

```
validator/
├── __init__.py      # Main entry point
├── syntactic.py     # ERROR checks (Categories 1-9)
├── semantic.py      # WARNING checks (Categories 10-12)
└── fixer.py         # Auto-fix for common LLM errors
```

**Note:** This is the FINAL validation gate before sampling. The intermediate validations in Step 2 catch errors early during hydration, while this catches any remaining issues in the assembled spec.

#### Syntactic Checks (ERROR - Blocks Sampling)

| Category                       | What It Checks                               |
| ------------------------------ | -------------------------------------------- |
| 1. Type/Modifier Compatibility | multiply/add on categorical → ERROR          |
| 2. Range Violations            | beta distribution add > 0.5 → ERROR          |
| 3. Weight Validity             | categorical weights don't sum to 1.0 → ERROR |
| 4. Distribution Parameters     | negative std, min >= max → ERROR             |
| 5. Dependency Validation       | references non-existent attribute → ERROR    |
| 6. Condition Syntax            | invalid Python in `when` clause → ERROR      |
| 7. Formula Validation          | invalid formula syntax → ERROR               |
| 8. Duplicate Detection         | same attribute name twice → ERROR            |
| 9. Strategy Consistency        | derived without formula → ERROR              |

#### Semantic Checks (WARNING - Sampling Proceeds)

| Category                     | What It Checks                     |
| ---------------------------- | ---------------------------------- |
| 10. No-Op Detection          | modifier with multiply=1.0, add=0  |
| 11. Modifier Stacking        | extreme combined modifier effects  |
| 12. Condition Value Validity | option name mismatch in conditions |

#### Auto-Fixer

**File:** `entropy/population/validator/fixer.py`

**CLI:** `entropy fix surgeons.yaml`

**What it fixes:**

- Modifier conditions with wrong option names
- Uses fuzzy matching (SequenceMatcher)
- E.g., `'University hospital'` → `'University_hospital'`

**Example:**

```
Found 4 fix(es):

  ai_tool_awareness[0]:
    - 'University hospital'
    + 'University_hospital' (confidence: 95%)
```

---

### Sampling

**File:** `entropy/population/sampler/core.py`

**CLI:** `entropy sample surgeons.yaml -o agents.json --seed 42`

**Process:**

1. **Load spec and validate**
2. **Initialize RNG** with seed for reproducibility
3. **For each agent:**

   - Iterate through `sampling_order`
   - For each attribute:
     - **independent:** Sample from distribution
     - **derived:** Evaluate formula
     - **conditional:** Sample with modifiers applied
   - Apply hard constraints (clamping)
   - Assign agent ID

4. **Collect statistics:**
   - Means/stds for numeric attributes
   - Counts for categorical/boolean
   - Modifier trigger counts
   - Constraint violations

**Output:** `agents.json`

```json
{
  "meta": {
    "spec": "500 German surgeons",
    "count": 500,
    "seed": 42,
    "generated_at": "2024-01-15T10:30:00Z"
  },
  "agents": [
    {
      "_id": "agent_000",
      "age": 42,
      "gender": "male",
      "specialty": "Cardiac_surgery",
      "years_experience": 14,
      ...
    },
    ...
  ]
}
```

---

### Network Generation

**File:** `entropy/network/generator.py`

**CLI:** `entropy network agents.json -o network.json --avg-degree 20`

**Algorithm:**

1. **Attribute Similarity** - Calculate similarity between agents
2. **Watts-Strogatz Model** - Small-world network properties
3. **Edge Types** - colleague, professional_contact, etc.
4. **Weight Assignment** - Based on relationship strength

**Network Metrics Computed:**

- Node count, edge count
- Average degree
- Clustering coefficient
- Average path length
- Modularity
- Degree assortativity

**Output:** `network.json`

```json
{
  "meta": {
    "node_count": 500,
    "edge_count": 5000,
    "avg_degree": 20.0
  },
  "nodes": [...],
  "edges": [
    {
      "source": "agent_000",
      "target": "agent_042",
      "edge_type": "colleague",
      "weight": 0.85
    },
    ...
  ]
}
```

---

## Phase 1.5: Scenario Overlay

**CLI:** `entropy overlay surgeons_base.yaml -s "AI diagnostic tool adoption" -o surgeons.yaml`

**Purpose:** Layer scenario-specific behavioral attributes on an existing population.

**Key Difference from Base Spec:**

- Selector receives `context=base_spec.attributes`
- Only discovers NEW attributes (5-15 typical)
- New attributes CAN depend on base attributes
- Hydrator can reference base attributes in formulas/modifiers

**Process:**

1. **Load base spec** (e.g., 35 attributes)
2. **Attribute Selection** - Pass base as context, discover NEW attributes
3. **Hydration** - Research distributions, can reference base attrs (with intermediate validation per sub-step)
4. **Binding** - Topological sort with cross-layer dependencies
5. **Merge** - `base_spec.merge(overlay_spec)`

**New Attributes Example:**

```
• tech_adoption_tendency (float) ← depends on: age, education
• ai_trust_level (float) ← depends on: years_experience
• workflow_flexibility (float)
• change_resistance (float) ← depends on: age, employer_type
```

**Output:** Merged spec with all 43 attributes

---

## Phase 2: Scenario Compilation

**CLI:** `entropy scenario "AI announcement" -p surgeons.yaml -a agents.json -n network.json -o scenario.yaml`

**File:** `entropy/scenario/compiler.py`

**Purpose:** Transform natural language scenario into machine-readable spec.

### Input Files

| File            | How It's Used                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `surgeons.yaml` | **Loaded and parsed** - Used for attribute references in exposure rules                                                 |
| `agents.json`   | **Path stored only** - NOT loaded during compilation (saves memory/time for large files). Phase 3 loads it when needed. |
| `network.json`  | **Loaded for validation** - Edge types extracted for spread modifier validation                                         |

**Why agents.json is not loaded:**

- Large agent files (10k+ agents) would consume significant memory
- Agent content is not needed for scenario compilation
- Only the file path is stored in `meta.agents_file`
- Phase 3 (simulation) loads agents when it actually needs them

### Step 1: Parse Scenario

**File:** `entropy/scenario/parser.py`

**LLM Call:** `reasoning_call()` with `gpt-5`

**Extracts:**

```yaml
event:
  type: announcement
  content: "Hospital announces mandatory AI diagnostic tool for all surgeries..."
  source: "Hospital Administration"
  credibility: 0.95
  ambiguity: 0.2
  emotional_valence: -0.1
```

### Step 2: Generate Seed Exposure

**File:** `entropy/scenario/exposure.py`

**LLM Call:** `reasoning_call()` with `gpt-5`

**Generates:**

```yaml
seed_exposure:
  channels:
    - name: email_notification
      description: "Official hospital email"
      reach: broadcast
      credibility_modifier: 1.0
    - name: staff_meeting
      reach: targeted
      credibility_modifier: 0.95
    - name: word_of_mouth
      reach: organic
      credibility_modifier: 0.7

  rules:
    - channel: email_notification
      when: "true"
      probability: 0.98
      timestep: 0
    - channel: staff_meeting
      when: "employer_type == 'University_hospital'"
      probability: 0.80
      timestep: 1
```

### Step 3: Determine Interaction Model

**File:** `entropy/scenario/interaction.py`

**LLM Call:** `reasoning_call()` with `gpt-5`

**Receives:** Network summary (edge types, node count) for context (injected into LLM prompt)

**Generates:**

```yaml
interaction:
  primary_model: passive_observation
  secondary_model: direct_conversation
  description: "Surgeons observe peer reactions and discuss informally"

spread:
  share_probability: 0.35
  share_modifiers:
    - when: "sentiment < -0.5"
      multiply: 1.8
      add: 0.0
    - when: "age < 40"
      multiply: 1.3
      add: 0.0
  decay_per_hop: 0.1
```

### Step 4: Define Outcomes

**File:** `entropy/scenario/outcomes.py`

**LLM Call:** `reasoning_call()` with `gpt-5`

**Generates:**

```yaml
outcomes:
  suggested_outcomes:
    - name: adoption_intent
      type: categorical
      options: [will_adopt, considering, resistant, strongly_opposed]
      required: true
    - name: sentiment
      type: float
      range: [-1.0, 1.0]
      required: true
    - name: will_advocate
      type: boolean
      description: "Will actively promote or oppose"
  capture_full_reasoning: true
```

### Step 5: Assemble and Validate

**File:** `entropy/scenario/validator.py`

**Validation Checks:**

- All attribute references in `when` clauses exist in population spec
- All channel references are defined
- Probabilities are in [0, 1]
- Timesteps are non-negative
- Outcome definitions are complete
- Referenced files exist (population_spec, network_file)
- Edge type references in spread modifiers exist in network

**Note on agents.json validation:**

- The file path is validated to exist
- Agent count consistency check is SKIPPED (agents not loaded)
- This is handled in Phase 3 when agents are actually loaded

**Final Output:** `scenario.yaml`

```yaml
meta:
  name: ai_tool_announcement
  description: "AI tool announcement"
  population_spec: surgeons.yaml # Path to population spec
  agents_file: agents.json # Path only - NOT loaded during compilation
  network_file: network.json # Loaded for edge type validation
  created_at: "2024-01-15T10:30:00Z"

event: ...
seed_exposure: ...
interaction: ...
spread: ...
outcomes: ...
simulation: ...
```

---

## Phase 3: Simulation Engine

**CLI:** `entropy simulate scenario.yaml -o results/ --model gpt-5-mini --seed 42`

**File:** `entropy/simulation/engine.py`

### Initialization

1. **Load all inputs:**

   - ScenarioSpec (from scenario.yaml)
   - PopulationSpec (from meta.population_spec)
   - **Agents JSON** (from meta.agents_file - **loaded here, not in Phase 2**)
   - Network JSON (from meta.network_file)

2. **Initialize State Manager** (`entropy/simulation/state.py`)

   - SQLite database for agent states
   - Tracks exposure history
   - Tracks reasoning history

3. **Generate Personas** (`entropy/simulation/persona.py`)
   - Create natural language persona for each agent
   - Used in reasoning prompts

### Simulation Loop

**For each timestep:**

```
┌────────────────────────────────────────────────────────────┐
│ TIMESTEP t                                                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. SEED EXPOSURE                                          │
│     - Apply Phase 2 exposure rules for timestep t          │
│     - Record ExposureRecord for each exposed agent         │
│                                                            │
│  2. NETWORK PROPAGATION                                    │
│     - Get all agents with will_share=True                  │
│     - For each sharer's neighbors:                         │
│       - Calculate share probability (with modifiers)       │
│       - Record exposure if successful                      │
│                                                            │
│  3. IDENTIFY AGENTS TO REASON                              │
│     - Newly aware agents (first exposure)                  │
│     - Multi-touch agents (N+ new exposures since last)     │
│                                                            │
│  4. AGENT REASONING (LLM)                                  │
│     - Build ReasoningContext (persona, exposures, peers)   │
│     - Call LLM with structured output                      │
│     - Extract: position, sentiment, will_share, outcomes   │
│     - Update AgentState                                    │
│                                                            │
│  5. CHECK STOPPING CONDITIONS                              │
│     - exposure_rate > 0.95                                 │
│     - no_state_changes_for > 10 timesteps                  │
│     - max_timesteps reached                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Agent Reasoning

**File:** `entropy/simulation/reasoning.py`

**Prompt Structure:**

```
## Who You Are
[Generated persona from agent attributes]

## What Happened
[Event content from scenario]

Source: [Event source]

## How You Learned About This
- You learned via email_notification (timestep 0)
- A contact in your network told you about this (timestep 2)

## What People Around You Think
- A colleague of yours is resistant
- A professional_contact of yours is considering

## Your Response
Based on who you are and what you've learned, respond naturally...
```

**Response Schema (Generated from Outcomes):**

```json
{
  "type": "object",
  "properties": {
    "reasoning": { "type": "string" },
    "will_share": { "type": "boolean" },
    "adoption_intent": { "enum": ["will_adopt", "considering", "resistant", "strongly_opposed"] },
    "sentiment": { "type": "number", "minimum": -1, "maximum": 1 }
  },
  "required": ["reasoning", "will_share", "adoption_intent", "sentiment"]
}
```

### Propagation Logic

**File:** `entropy/simulation/propagation.py`

**Seed Exposure:**

```python
for rule in scenario.seed_exposure.rules:
    if rule.timestep == current_timestep:
        for agent in agents:
            if eval_condition(rule.when, agent):
                if random() < rule.probability:
                    state_manager.record_exposure(agent_id, exposure)
```

**Network Propagation:**

```python
for sharer_id in state_manager.get_sharers():
    for neighbor_id, edge_data in get_neighbors(network, sharer_id):
        prob = calculate_share_probability(agent, edge_data, spread_config)
        if random() < prob:
            state_manager.record_exposure(neighbor_id, exposure)
```

### Stopping Conditions

**File:** `entropy/simulation/stopping.py`

**Default Conditions:**

- `exposure_rate > 0.95` - 95% of agents aware
- `no_state_changes_for > 10` - Convergence detected
- `max_timesteps` reached

### Output Files

**Directory:** `results/`

| File                         | Contents                       |
| ---------------------------- | ------------------------------ |
| `simulation.db`              | SQLite database with all state |
| `timeline.jsonl`             | Streaming event log            |
| `agent_states.json`          | Final state per agent          |
| `by_timestep.json`           | Metrics over time              |
| `outcome_distributions.json` | Final outcome distributions    |
| `meta.json`                  | Run configuration              |

---

## File Structure Reference

```
entropy/
├── cli.py                          # All CLI commands
├── config.py                       # Environment settings
│
├── core/
│   ├── llm.py                      # LLM client (simple_call, reasoning_call, agentic_research)
│   │                               # Supports: validator callback, max_retries, on_retry callback
│   └── models/
│       ├── population.py           # Population specs, distributions, sampling
│       ├── scenario.py             # Scenario specs, events, exposure, outcomes
│       ├── simulation.py           # Agent state, reasoning context
│       └── results.py              # Aggregation models
│
├── population/
│   ├── architect/
│   │   ├── sufficiency.py          # Step 0: Context check
│   │   ├── selector.py             # Step 1: Attribute discovery
│   │   ├── hydrator.py             # Step 2: Distribution research (orchestrator)
│   │   ├── hydrator_utils.py       # Step 2: Validation functions per sub-step
│   │   │                           #   - validate_independent_hydration()
│   │   │                           #   - validate_derived_hydration()
│   │   │                           #   - validate_conditional_base()
│   │   │                           #   - validate_modifiers()
│   │   │                           #   - validate_strategy_consistency()
│   │   └── binder.py               # Step 3: Dependency graph
│   │
│   ├── sampler/
│   │   ├── core.py                 # Main sampling loop
│   │   ├── distributions.py        # Distribution sampling
│   │   ├── modifiers.py            # Modifier application
│   │   └── eval_safe.py            # Safe formula evaluation
│   │
│   └── validator/
│       ├── __init__.py             # validate_spec() - FINAL validation gate
│       ├── syntactic.py            # ERROR checks (Categories 1-9)
│       ├── semantic.py             # WARNING checks (Categories 10-12)
│       └── fixer.py                # Auto-fix
│
├── network/
│   ├── generator.py                # Network creation
│   ├── similarity.py               # Agent similarity
│   ├── metrics.py                  # Network metrics
│   └── config.py                   # Network config
│
├── scenario/
│   ├── compiler.py                 # Main orchestrator
│   │                               # NOTE: agents.json path stored, NOT loaded
│   ├── parser.py                   # Event parsing
│   ├── exposure.py                 # Seed exposure rules
│   ├── interaction.py              # Interaction model (receives network summary)
│   ├── outcomes.py                 # Outcome definitions
│   └── validator.py                # Scenario validation
│
├── simulation/
│   ├── engine.py                   # Main simulation loop
│   │                               # NOTE: Loads agents.json here
│   ├── state.py                    # SQLite state manager
│   ├── persona.py                  # Persona generation
│   ├── reasoning.py                # LLM reasoning calls
│   ├── propagation.py              # Exposure propagation
│   ├── stopping.py                 # Stopping conditions
│   ├── timeline.py                 # Event logging
│   └── aggregation.py              # Result aggregation
│
└── results/
    ├── reader.py                   # Results loading
    └── formatter.py                # Display formatting
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: POPULATION                            │
└─────────────────────────────────────────────────────────────────────────┘

  "500 German surgeons"
         │
         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ check_sufficiency│──────│ SufficiencyResult│
  │ (gpt-5-mini)     │      │ size=500         │
  └──────────────────┘      │ geography=Germany│
                            └────────┬─────────┘
                                     │
         ┌───────────────────────────┘
         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ select_attributes│──────│ List[Discovered  │
  │ (gpt-5 reasoning)│      │   Attribute]     │
  └──────────────────┘      │ 35 attributes    │
                            └────────┬─────────┘
                                     │
         ┌───────────────────────────┘
         ▼
  ┌──────────────────┐
  │ hydrate_attributes│
  │ (gpt-5 + search) │
  └────────┬─────────┘
           │
           │  ┌──────────────────────────────────────────────────────────┐
           │  │ FOR EACH SUB-STEP:                                       │
           │  │   1. Call LLM (agentic_research or reasoning_call)       │
           │  │   2. Parse response                                      │
           │  │   3. Run validation function                             │
           │  │   4. If errors: retry with error feedback (up to 2x)     │
           │  └──────────────────────────────────────────────────────────┘
           │
           ├──▶ hydrate_independent() ──▶ validate_independent_hydration()
           ├──▶ hydrate_derived() ──▶ validate_derived_hydration()
           ├──▶ hydrate_conditional_base() ──▶ validate_conditional_base()
           └──▶ hydrate_conditional_modifiers() ──▶ validate_modifiers()
                            │
                            ▼
                   validate_strategy_consistency()
                            │
                            ▼
                   ┌──────────────────┐
                   │ List[Hydrated    │
                   │   Attribute]     │
                   └────────┬─────────┘
                            │
         ┌──────────────────┘
         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ bind_constraints │──────│ List[AttributeSpec]│
  │ (topological sort│      │ sampling_order   │
  └──────────────────┘      └────────┬─────────┘
                                     │
         ┌───────────────────────────┘
         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ validate_spec    │──────│ ValidationResult │  ◀── FINAL validation gate
  │ (syntactic +     │      │ errors, warnings │
  │  semantic)       │      └────────┬─────────┘
  └──────────────────┘               │
                                     │ (errors → FIX or FAIL)
         ┌───────────────────────────┘
         ▼
  ┌──────────────────┐
  │ build_spec       │──────▶ surgeons_base.yaml
  └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1.5: OVERLAY                               │
└─────────────────────────────────────────────────────────────────────────┘

  surgeons_base.yaml + "AI tool adoption"
         │
         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ select_attributes│──────│ 8 NEW attributes │
  │ (context=base)   │      │ can depend on    │
  └──────────────────┘      │ base attributes  │
                            └────────┬─────────┘
                                     │
         (hydrate with sub-step validation, bind, final validate)
                                     │
                                     ▼
                   ┌──────────────────┐
                   │ base.merge(overlay)│──────▶ surgeons.yaml (43 attrs)
                   └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                      SAMPLING & NETWORK                                  │
└─────────────────────────────────────────────────────────────────────────┘

  surgeons.yaml
         │
         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ sample_population│──────│ agents.json      │
  │ (500 agents)     │      │ 500 agent dicts  │
  └──────────────────┘      └────────┬─────────┘
                                     │
         ┌───────────────────────────┘
         ▼
  ┌──────────────────┐      ┌──────────────────┐
  │ generate_network │──────│ network.json     │
  │ (Watts-Strogatz) │      │ nodes + edges    │
  └──────────────────┘      └─────────────────-┘


┌─────────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: SCENARIO                                  │
└─────────────────────────────────────────────────────────────────────────┘

  "AI tool announcement" + surgeons.yaml + agents.json + network.json
         │
         ├──▶ Load surgeons.yaml (parsed for attribute validation)
         ├──▶ Store agents.json PATH ONLY (not loaded - saves memory)
         └──▶ Load network.json (for edge type validation)
         │
         ├──▶ parse_scenario() ──▶ Event
         ├──▶ generate_seed_exposure() ──▶ SeedExposure
         ├──▶ determine_interaction_model(network_summary) ──▶ InteractionConfig, SpreadConfig
         └──▶ define_outcomes() ──▶ OutcomeConfig
                            │
                            ▼
                   ┌──────────────────┐
                   │ validate_scenario│
                   │ (network loaded, │
                   │  agents NOT loaded)│
                   └────────┬─────────┘
                            │
                            ▼
                   scenario.yaml
                   (stores paths to agents.json, not content)


┌─────────────────────────────────────────────────────────────────────────┐
│                       PHASE 3: SIMULATION                                │
└─────────────────────────────────────────────────────────────────────────┘

  scenario.yaml ─────────────────────────────────────────────┐
  surgeons.yaml (from meta.population_spec) ─────────────────┤
  agents.json (from meta.agents_file) ◀── LOADED HERE ───────┤
  network.json (from meta.network_file) ─────────────────────┘
         │
         ▼
  ┌──────────────────┐
  │ SimulationEngine │
  │   .run()         │
  └────────┬─────────┘
           │
           │  ┌─────────────────────────────────────────────────────┐
           │  │ FOR EACH TIMESTEP t:                                │
           │  │                                                     │
           │  │  1. apply_seed_exposures(t)                        │
           │  │     └──▶ new agents become aware                   │
           │  │                                                     │
           │  │  2. propagate_through_network(t)                   │
           │  │     └──▶ sharers spread to neighbors               │
           │  │                                                     │
           │  │  3. get_agents_to_reason()                         │
           │  │     └──▶ newly aware + multi-touch                 │
           │  │                                                     │
           │  │  4. FOR EACH agent to reason:                      │
           │  │     │                                               │
           │  │     ├──▶ build_reasoning_context()                 │
           │  │     │    └──▶ persona + exposures + peer opinions  │
           │  │     │                                               │
           │  │     ├──▶ reason_agent() [LLM CALL]                 │
           │  │     │    └──▶ gpt-5-mini structured output         │
           │  │     │                                               │
           │  │     └──▶ update_agent_state()                      │
           │  │          └──▶ position, sentiment, will_share      │
           │  │                                                     │
           │  │  5. evaluate_stopping_conditions()                 │
           │  │     └──▶ exposure_rate, convergence, max_timesteps │
           │  │                                                     │
           │  └─────────────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────┐
  │ _export_results()│
  └────────┬─────────┘
           │
           ▼
  results/
  ├── simulation.db
  ├── timeline.jsonl
  ├── agent_states.json
  ├── by_timestep.json
  ├── outcome_distributions.json
  └── meta.json
```

---

## LLM Usage Summary

| Function             | Model              | Features                               | Use Case                              |
| -------------------- | ------------------ | -------------------------------------- | ------------------------------------- |
| `simple_call()`      | gpt-5-mini         | Fast, cheap, no reasoning              | Sufficiency check, agent reasoning    |
| `reasoning_call()`   | gpt-5              | Reasoning + validator + retry          | Attribute selection, scenario parsing |
| `agentic_research()` | gpt-5 + web_search | Reasoning + search + validator + retry | Distribution research                 |

**Retry-with-Validation Parameters:**

| Parameter         | Type                      | Description                                         |
| ----------------- | ------------------------- | --------------------------------------------------- |
| `validator`       | `(dict) -> (bool, str)`   | Validates response, returns (is_valid, error_msg)   |
| `max_retries`     | `int`                     | Max retry attempts if validation fails (default: 2) |
| `on_retry`        | `(attempt, max, summary)` | Callback when retry begins                          |
| `previous_errors` | `str`                     | Error feedback prepended to prompt                  |

---

## Validation Summary

### Phase 1: Population Creation

| When                    | Function                           | File                  | Blocking     |
| ----------------------- | ---------------------------------- | --------------------- | ------------ |
| Step 2a (each LLM call) | `validate_independent_hydration()` | hydrator_utils.py     | Retry        |
| Step 2b (each LLM call) | `validate_derived_hydration()`     | hydrator_utils.py     | Retry        |
| Step 2c (each LLM call) | `validate_conditional_base()`      | hydrator_utils.py     | Retry        |
| Step 2d (each LLM call) | `validate_modifiers()`             | hydrator_utils.py     | Retry        |
| After all hydration     | `validate_strategy_consistency()`  | hydrator_utils.py     | Warning      |
| Before save             | `validate_spec()`                  | validator/**init**.py | ERROR blocks |

### Phase 2: Scenario Compilation

| When           | Function              | File                  | Blocking     |
| -------------- | --------------------- | --------------------- | ------------ |
| After assembly | `validate_scenario()` | scenario/validator.py | ERROR blocks |

---

## Error Handling Summary

| Phase            | Error Type           | File              | Resolution                        |
| ---------------- | -------------------- | ----------------- | --------------------------------- |
| Sufficiency      | Insufficient context | sufficiency.py    | Returns clarification questions   |
| Selection        | Strategy mismatch    | selector.py       | Auto-corrects strategy/depends_on |
| Hydration 2a     | Invalid distribution | hydrator_utils.py | Retry with error feedback         |
| Hydration 2b     | Invalid formula      | hydrator_utils.py | Retry with error feedback         |
| Hydration 2c     | Invalid conditional  | hydrator_utils.py | Retry with error feedback         |
| Hydration 2d     | Invalid modifier     | hydrator_utils.py | Retry with error feedback         |
| Binding          | Circular dependency  | binder.py         | Raises CircularDependencyError    |
| Final Validation | Type mismatch        | syntactic.py      | ERROR - blocks sampling           |
| Final Validation | Option mismatch      | semantic.py       | WARNING - auto-fixable            |
| Sampling         | Formula error        | eval_safe.py      | Raises SamplingError              |
| Scenario         | Missing reference    | validator.py      | ERROR - blocks simulation         |
| Simulation       | LLM failure          | reasoning.py      | Retry with backoff                |

---

## Quick Reference: CLI Commands

```bash
# Phase 1: Population Creation
entropy spec "500 German surgeons" -o surgeons_base.yaml
entropy overlay surgeons_base.yaml -s "AI tool adoption" -o surgeons.yaml
entropy validate surgeons.yaml [--strict]
entropy fix surgeons.yaml [-o fixed.yaml] [--dry-run]
entropy sample surgeons.yaml -o agents.json [--seed 42] [--report]
entropy network agents.json -o network.json [--avg-degree 20] [--validate]

# Phase 2: Scenario Compilation
# Note: agents.json path is stored in scenario, not loaded during compilation
entropy scenario "AI announcement" -p surgeons.yaml -a agents.json -n network.json -o scenario.yaml
entropy validate-scenario scenario.yaml

# Phase 3: Simulation
# Note: agents.json is loaded here when simulation starts
entropy simulate scenario.yaml -o results/ [--model gpt-5-mini] [--seed 42]
entropy results results/ [--segment specialty] [--timeline] [--agent agent_042]
```
