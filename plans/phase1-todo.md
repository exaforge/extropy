## Complete Phase 1 Pipeline

---

## ARCHITECT LAYER

### Step 0: Context Sufficiency Check

**Model:** GPT-5-mini (no web search)
**Cost:** ~$0.001
**Time:** ~2s

```
Input:  "500 German surgeons"
Output: {sufficient: true} or {sufficient: false, missing: ["specialty focus?", "region?"]}
```

**Prompt:**

```
Is this population description sufficient to generate a synthetic population?
"{context}"

Check for:
- Clear population identity (who are they?)
- Approximate size (or use default 1000)
- Geographic scope (or assume global)

If insufficient, list what's missing.
```

---

### Step 1: Attribute Selection

**Model:** GPT-5-mini (no web search)
**Cost:** ~$0.01
**Time:** ~5s

```
Input:  "500 German surgeons"
Output: List of attributes with types and categories
```

**Prompt:**

```
For population: "German surgeons"

Discover ALL relevant attributes across these categories:

1. UNIVERSAL (every human has)
   - age, gender, income, location, education, marital_status, etc.
   - Use geography-appropriate options (German states, not US)

2. POPULATION-SPECIFIC (what defines THIS population)
   - Profession-specific: specialty, years_practice, hospital_type
   - Cultural/economic factors unique to them

3. CONTEXT-SPECIFIC (if a product/service mentioned)
   - Relationship attributes: tenure, usage, satisfaction
   - Skip if no context entity

4. PERSONALITY (if variance matters)
   - Big Five traits
   - Domain-specific: risk_tolerance, work_life_balance

For each attribute return:
- name (snake_case)
- type (int, float, categorical)
- category (universal/population_specific/context_specific/personality)
- description (one line)
- suggested_dependencies (list of other attribute names this depends on)
```

**Example Output:**

```yaml
attributes:
  - { name: age, type: int, category: universal, depends_on: [] }
  - { name: gender, type: categorical, category: universal, depends_on: [] }
  - { name: location, type: categorical, category: universal, depends_on: [] }
  - { name: income_eur, type: int, category: universal, depends_on: [specialty, hospital_type, years_practice] }
  - { name: specialty, type: categorical, category: population_specific, depends_on: [] }
  - { name: years_practice, type: int, category: population_specific, depends_on: [age] }
  - { name: hospital_type, type: categorical, category: population_specific, depends_on: [] }
  - { name: surgeries_per_month, type: int, category: population_specific, depends_on: [specialty] }
  - { name: risk_tolerance, type: float, category: personality, depends_on: [] }
```

---

### ✋ Human Check #1: Attribute Confirmation

**CLI Interface:**

```
╭────────────────────────────────────────────────────────────╮
│  DISCOVERED ATTRIBUTES (German surgeons)                   │
╰────────────────────────────────────────────────────────────╯

Universal:
  • age (int)
  • gender (categorical)
  • location (categorical) - German states
  • income_eur (int) ← depends on: specialty, hospital_type, years_practice

Population-specific:
  • specialty (categorical)
  • years_practice (int) ← depends on: age
  • hospital_type (categorical)
  • surgeries_per_month (int) ← depends on: specialty

Personality:
  • risk_tolerance (float)

[Y] Proceed  [e] Edit  [a] Add  [r] Remove  [n] Cancel
>
```

**Edit mode allows:**

- Remove attribute: `r years_practice`
- Add attribute: `a on_call_frequency categorical`
- Change dependency: `d income_eur add location`

---

### Step 2: Attribute Hydration (Distribution Research)

**Model:** GPT-5 + agentic web search
**Cost:** ~$0.10-0.50
**Time:** ~60-180s

```
Input:  Confirmed attribute list
Output: Distribution parameters + correlations for each
```

**Prompt:**

```
For population "German surgeons", research distributions for these attributes:

[list of confirmed attributes]

For each attribute, find:
1. Distribution type and parameters:
   - int/float: {type: normal|uniform|exponential, min, max, mean, std}
   - categorical: {options: [...], weights: [...]}

2. Constraints:
   - Hard bounds (age >= 28 for surgeons)
   - Derived formulas (years_practice = f(age))

3. Correlations:
   - Which attributes influence this one?
   - Multiplicative modifiers (specialty affects income)

Use real data from medical associations, salary surveys, demographic studies.
```

**Example Output:**

```yaml
attributes:
  - name: age
    distribution: { type: normal, mean: 47, std: 10, min: 28, max: 72 }
    constraints:
      - { type: hard_min, value: 28, reason: "post-residency" }
    sources: ["German Medical Association 2024"]

  - name: years_practice
    sampling_strategy: derived
    formula: "age - 28 - uniform(0, 5)"
    constraints:
      - { type: hard_min, value: 0 }
      - { type: hard_max, formula: "age - 28" }

  - name: specialty
    distribution:
      type: categorical
      options: [general, orthopedics, cardiology, neurology, ...]
      weights: [0.25, 0.15, 0.12, 0.08, ...]
    sources: ["Bundesärztekammer statistics"]

  - name: income_eur
    distribution: { type: normal, mean: 180000, std: 60000, min: 70000, max: 500000 }
    modifiers:
      - { when: "specialty == cardiology", multiply: 1.25 }
      - { when: "specialty == general", multiply: 0.85 }
      - { when: "hospital_type == private_clinic", multiply: 1.40 }
      - { when: "years_practice > 20", multiply: 1.20 }
    sources: ["Stepstone Salary Report 2024", "Marburger Bund survey"]

grounding_level: medium
sources_count: 23
```

---

### Step 3: Constraint Binding

**Model:** None (deterministic processing)
**Cost:** $0
**Time:** ~100ms

```
Input:  Hydrated attributes with constraints
Output: Dependency graph + sampling order
```

**Process:**

1. Build dependency graph from `depends_on` and `modifiers`
2. Topological sort → sampling order
3. Classify each attribute:
   - `independent` → sample from distribution directly
   - `derived` → compute from formula
   - `conditional` → sample with modifiers applied

**Output:**

```yaml
sampling_order:
  1. age          # independent
  2. gender       # independent
  3. location     # independent
  4. specialty    # independent
  5. hospital_type # independent
  6. years_practice # derived from age
  7. surgeries_per_month # conditional on specialty
  8. income_eur   # conditional on specialty, hospital_type, years_practice
  9. risk_tolerance # independent

constraint_checks:
  - {attr: age, check: ">= 28"}
  - {attr: years_practice, check: "<= age - 28"}
  - {attr: income_eur, check: ">= 70000"}
```

---

### Step 4: Population Spec Generated

```yaml
# german_surgeons_spec.yaml
population: "German surgeons"
size: 500
geography: "Germany"
created: "2024-11-27T10:30:00Z"

grounding:
  level: medium
  sources: 23

attributes:
  - name: age
    type: int
    category: universal
    sampling: independent
    distribution: { type: normal, mean: 47, std: 10, min: 28, max: 72 }

  - name: years_practice
    type: int
    category: population_specific
    sampling: derived
    formula: "max(0, age - 28 - uniform(0, 5))"

  - name: income_eur
    type: int
    category: universal
    sampling: conditional
    distribution: { type: normal, mean: 180000, std: 60000 }
    modifiers:
      - { when: "specialty == cardiology", multiply: 1.25 }
      - { when: "hospital_type == private_clinic", multiply: 1.40 }
    constraints: [{ type: hard_min, value: 70000 }]

  # ... all other attributes ...

sampling_order: [age, gender, location, specialty, hospital_type, years_practice, ...]
```

---

### ✋ Human Check #2: Spec Confirmation

```
╭────────────────────────────────────────────────────────────╮
│  POPULATION SPEC READY                                     │
╰────────────────────────────────────────────────────────────╯

German surgeons (500 agents)
Grounding: 🟡 Medium (23 sources)

Attributes:
  age           int    28-72, mean 47
  gender        cat    male 62%, female 38%
  specialty     cat    general 25%, orthopedics 15%, cardiology 12%...
  years_practice int   derived from age
  income_eur    int    70k-500k, mean 180k (modified by specialty, hospital)
  ...

Sampling order: age → gender → location → specialty → hospital_type → years_practice → ...

[Y] Generate 500 agents  [s] Save spec only  [e] Edit  [n] Cancel
>
```

---

## SAMPLER LAYER

### Step 5: Agent Sampling

**Model:** None (pure local computation)
**Cost:** $0
**Time:** ~1-5s for 500 agents

```python
for i in range(500):
    agent = {}

    for attr in sampling_order:
        if attr.sampling == "independent":
            value = sample_distribution(attr.distribution)

        elif attr.sampling == "derived":
            value = evaluate_formula(attr.formula, agent)

        elif attr.sampling == "conditional":
            base = sample_distribution(attr.distribution)
            for mod in attr.modifiers:
                if evaluate_condition(mod.when, agent):
                    base *= mod.multiply
            value = base

        # Apply hard constraints
        value = apply_constraints(value, attr.constraints, agent)

        agent[attr.name] = value

    agents.append(agent)
```

---

### Step 6: Network Generation

**Model:** None (graph algorithm)
**Time:** ~1s

Same as before: Watts-Strogatz + similarity weighting

---

### Step 7: Persona Synthesis

**Model:** Template-based (no LLM)
**Time:** ~1s

Generate natural language persona from attributes dict.

---

## Summary: Constraint Approach

We use a **hybrid approach**:

| Constraint Type                   | Handling Method                         |
| --------------------------------- | --------------------------------------- |
| Hard bounds (age ≥ 28)            | In distribution params (min/max)        |
| Derived values (years_practice)   | Formula evaluation, not sampling        |
| Correlations (income ~ specialty) | Multiplicative modifiers                |
| Post-validation                   | Final constraint check, clamp if needed |

**No rejection sampling** — everything is computed/conditioned correctly the first time.

---

Great questions. Let me address both:

---

## 1. Is This Generalizable + Hallucination Safe?

### Generalizability: ✅ YES

| Population         | Works? | Why                                                |
| ------------------ | ------ | -------------------------------------------------- |
| German surgeons    | ✅     | Discovers specialty, years_practice, hospital_type |
| Indian farmers     | ✅     | Discovers caste, land_acres, irrigation_access     |
| Crypto traders     | ✅     | Discovers portfolio_size, preferred_chain          |
| US Netflix users   | ✅     | Discovers tenure, plan_tier, satisfaction          |
| Evangelical voters | ✅     | Discovers church_attendance, biblical_literalism   |

The 4-category taxonomy + dynamic dict = works for anything.

---

### Hallucination Safety: ⚠️ MEDIUM RISK

| Step                          | Hallucination Risk | Mitigation                                     |
| ----------------------------- | ------------------ | ---------------------------------------------- |
| Step 0: Sufficiency           | 🟢 Low             | Simple yes/no                                  |
| Step 1: Attribute Selection   | 🟡 Medium          | **Human check catches bad attributes**         |
| Step 2: Distribution Research | 🔴 High            | Web search grounds it, but can still fabricate |
| Step 3: Constraint Binding    | 🟢 None            | Deterministic                                  |
| Sampling                      | 🟢 None            | Deterministic                                  |

**What can still go wrong:**

- LLM cites non-existent sources
- Plausible-but-wrong statistics (says 30% when it's 20%)
- Misses recent changes (outdated data)

**Possible mitigations:**

```yaml
# Add to spec
grounding:
  level: medium
  sources:
    - url: "https://bundesaerztekammer.de/..."
      claim: "62% of surgeons are male"
      verified: false # Human can spot-check
```

**For critical applications:** Add source verification step (expensive but safer).

---

## 2. CLI vs Chat-Based CLI

### Current: Command-Based

```bash
entropy create "500 German surgeons" --name surgeons
# [Y/n/e] prompts
```

### Alternative: Chat-Based

```bash
$ entropy

entropy> create 500 German surgeons

I found these attributes for German surgeons:
  • age (28-72)
  • specialty (cardiology, orthopedics...)
  • years_practice (derived from age)

Would you like to modify?

entropy> remove surgeries_per_month

Removed. Anything else?

entropy> add on_call_frequency categorical

Added. Ready to research distributions?

entropy> yes

Researching... 23 sources found.

entropy> looks good, generate

✓ 500 agents created, saved as "german_surgeons"
```

---

### Comparison

| Aspect          | Command CLI                      | Chat CLI                      |
| --------------- | -------------------------------- | ----------------------------- |
| **Editing**     | Clunky (`e` mode, type commands) | Natural ("remove X", "add Y") |
| **Iteration**   | Restart required for big changes | Fluid back-and-forth          |
| **Automation**  | ✅ Scriptable                    | ❌ Hard to script             |
| **Power users** | ✅ Faster                        | Slower (more typing)          |
| **New users**   | Confusing                        | ✅ More intuitive             |
| **Testing**     | ✅ Easy                          | Harder                        |

---

### Recommendation: Both

```bash
# Chat mode for interactive use
entropy
# or
entropy chat

# Command mode for automation / scripts
entropy create --from-spec surgeons.yaml --name batch_1
entropy create --from-spec surgeons.yaml --name batch_2 --size 1000
```

**Same backend, two interfaces.**

For the MVP: **Start with command CLI** (easier to build, test, debug), add chat mode later.

---

## Summary

| Question            | Answer                                                       |
| ------------------- | ------------------------------------------------------------ |
| Generalizable?      | ✅ Yes — dynamic dict + 4 categories                         |
| Hallucination safe? | ⚠️ Medium — human checks + sources help, but not bulletproof |
| CLI approach?       | Command CLI for MVP, chat CLI as enhancement                 |
