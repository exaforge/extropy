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
# Generate a population spec
entropy spec "500 German surgeons" -o surgeons.yaml

# Layer a scenario on an existing population
entropy overlay surgeons.yaml -s "AI diagnostic tool adoption" -o surgeons_ai.yaml
```

## CLI Commands

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

**Example output:**
```
✓ Context sufficient (size: 500, geography: Germany)
✓ Discovered 32 attributes

Step 2a: Researching independent distributions... 0:45
Step 2b: Specifying derived formulas... 0:52
Step 2c: Researching conditional distributions... 1:15
Step 2d: Specifying conditional modifiers... 1:28
✓ Researched distributions (1:28, 12 sources)

✓ Spec saved to surgeons.yaml
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
# Base population
entropy spec "1000 Indian farmers in Maharashtra" -o farmers.yaml

# Layer adoption scenario
entropy overlay farmers.yaml \
  -s "Drought-resistant seed adoption decision" \
  -o farmers_seeds.yaml
```

## Three Phases

| Phase | What It Does | LLM | Status |
|-------|--------------|-----|--------|
| **Phase 1: Population Creation** | Generate population specs from natural language | OpenAI API (GPT-5) | ✅ Implemented |
| **Phase 2: Scenario Injection** | Define events and decisions for populations | OpenAI API | 📋 Planned |
| **Phase 3: Simulation** | Agents respond; opinions evolve with social influence | LM Studio (local) | 📋 Planned |

## Population Spec Structure

A generated YAML spec contains:

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
