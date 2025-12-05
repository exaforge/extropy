# CLAUDE.md - AI Assistant Guide for Entropy

## Project Overview

**Entropy** is a population simulation system that creates synthetic populations grounded in real-world data and simulates how they respond to scenarios over time.

**Purpose**: Generate realistic agent-based populations from natural language descriptions, layer scenario-specific attributes on them, and simulate opinion evolution through social networks.

**Current Status**: Phase 1 (Population Creation) - Architect Layer is implemented and working.

## Three-Phase Architecture

| Phase | What It Does | LLM | Status |
|-------|--------------|-----|--------|
| **Phase 1: Population Creation** | Generate population specs from natural language | OpenAI API (GPT-5) | ✅ In progress |
| **Phase 2: Scenario Injection** | Layer scenario-specific attributes on populations | OpenAI API | 📋 Planned |
| **Phase 3: Simulation** | Agents respond; opinions evolve via social influence | LM Studio (local) | 📋 Planned |

## Repository Structure

```
entropy/
├── pyproject.toml              # Python package config
├── README.md                   # User-facing documentation
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore patterns
│
├── entropy/                    # Main package
│   ├── __init__.py
│   ├── cli.py                  # Typer-based CLI (main entry point)
│   ├── config.py               # Pydantic settings from .env
│   ├── llm.py                  # OpenAI API wrappers (3 functions)
│   ├── spec.py                 # Population spec data models
│   │
│   └── architect/              # Phase 1: Spec generation pipeline
│       ├── __init__.py         # Exports main functions
│       ├── sufficiency.py      # Step 0: Context validation
│       ├── selector.py         # Step 1: Attribute discovery
│       ├── hydrator.py         # Step 2: Distribution research
│       └── binder.py           # Step 3: Dependency graph + spec assembly
│
├── plans/                      # Design documents
│   ├── phase1.md               # Phase 1 technical spec
│   ├── phase1-todo.md          # Phase 1 task breakdown
│   └── phase1-process.md       # Phase 1 process documentation
│
├── tests/                      # Test suite
│   └── __init__.py
│
├── test_search.py              # Ad-hoc testing script
│
├── surgeons_base.yaml          # Example: base population spec
├── surgeons_ai.yaml            # Example: base + overlay merged spec
│
├── logs/                       # LLM request/response logs (gitignored)
├── storage/                    # SQLite database (gitignored, future)
└── data/cache/                 # Research cache (gitignored, future)
```

**Total code files**: ~12 Python files in `entropy/`

## Key Concepts

### 1. Population Spec

A `PopulationSpec` is a complete blueprint for generating a population. It contains:
- **Metadata**: description, size, geography, timestamps
- **Attributes**: 25-40 attribute definitions with distributions and dependencies
- **Grounding**: quality metrics and source citations
- **Sampling order**: topologically sorted order respecting dependencies

Specs are serialized to YAML files and can be:
- Created from scratch (`entropy spec`)
- Merged with overlays (`entropy overlay`)
- Loaded and inspected

### 2. Attribute Categories

Every attribute belongs to one of four categories:

1. **Universal**: Demographic attributes everyone has (age, gender, income, location)
2. **Population-specific**: What makes this population unique (specialty for surgeons, land_acres for farmers)
3. **Context-specific**: Relationship to a product/service (subscription_tenure, satisfaction)
4. **Personality**: Behavioral/psychological traits (Big Five, risk_tolerance)

### 3. Sampling Strategies

Attributes use one of three sampling strategies:

- **Independent**: Sample directly from distribution (e.g., age ~ Normal(47, 10))
- **Derived**: Compute from formula (e.g., `years_practice = age - 28 - uniform(0, 5)`)
- **Conditional**: Sample then apply modifiers (e.g., income modified by specialty, hospital_type)

### 4. Overlay Mode

The architect pipeline supports two modes:

- **Base mode**: Create a complete population from scratch
- **Overlay mode**: Add scenario-specific attributes to an existing population
  - New attributes can depend on base attributes
  - Base attributes are preserved
  - Sampling order recomputed to handle cross-layer dependencies

## Architect Layer Pipeline

The Phase 1 pipeline has **4 steps + 2 human checkpoints**:

### Step 0: Sufficiency Check
- **File**: `entropy/architect/sufficiency.py`
- **Function**: `check_sufficiency(description: str) -> SufficiencyResult`
- **LLM**: GPT-5-mini (simple_call, no web search)
- **Purpose**: Validate description has enough context; extract size and geography
- **Time**: ~2s

### Step 1: Attribute Selection
- **File**: `entropy/architect/selector.py`
- **Function**: `select_attributes(description, size, geography, context=None) -> list[DiscoveredAttribute]`
- **LLM**: GPT-5 (reasoning_call, no web search)
- **Purpose**: Discover 25-40 relevant attributes across 4 categories
- **Time**: ~5-10s
- **Overlay**: When `context` provided, only discovers NEW attributes

👤 **Human Checkpoint #1**: User reviews and confirms attributes

### Step 2: Distribution Research (Hydration)
- **File**: `entropy/architect/hydrator.py`
- **Function**: `hydrate_attributes(attributes, description, geography, context=None) -> tuple[list[HydratedAttribute], list[str]]`
- **LLM**: GPT-5 (agentic_research with web search)
- **Purpose**: Research real-world distributions for each attribute
- **Returns**: Hydrated attributes + source URLs
- **Time**: ~60-180s

### Step 3: Constraint Binding
- **File**: `entropy/architect/binder.py`
- **Function**: `bind_constraints(attributes, context=None) -> tuple[list[AttributeSpec], list[str]]`
- **LLM**: None (deterministic)
- **Purpose**: Build dependency graph, topological sort, validate constraints
- **Raises**: `CircularDependencyError` if cycles detected
- **Time**: ~100ms

### Step 4: Spec Assembly
- **File**: `entropy/architect/binder.py`
- **Function**: `build_spec(...) -> PopulationSpec`
- **Purpose**: Assemble final PopulationSpec with metadata and grounding summary
- **Time**: ~10ms

👤 **Human Checkpoint #2**: User reviews spec summary and chooses to save

## LLM Usage Patterns

The project uses OpenAI API with three distinct calling patterns defined in `entropy/llm.py`:

### 1. `simple_call(prompt, response_schema, model="gpt-5-mini")`
- **Use for**: Fast, cheap tasks with no reasoning needed
- **Features**: Structured output (JSON schema), no reasoning, no web search
- **Examples**: Sufficiency checks, simple classification
- **Model**: gpt-5-mini

### 2. `reasoning_call(prompt, response_schema, model="gpt-5", reasoning_effort="low")`
- **Use for**: Tasks requiring reasoning but not external data
- **Features**: Structured output + reasoning, no web search
- **Examples**: Attribute selection, logical analysis
- **Model**: gpt-5

### 3. `agentic_research(prompt, response_schema, model="gpt-5", reasoning_effort="low")`
- **Use for**: Research tasks requiring real-world data
- **Features**: Structured output + reasoning + web search
- **Returns**: `(structured_data, source_urls)`
- **Examples**: Distribution research
- **Model**: gpt-5 with web search tool

**All three functions**:
- Return parsed JSON matching the schema
- Log requests/responses to `logs/` directory
- Use strict JSON schemas for reliability

## Data Models (entropy/spec.py)

### Core Types

```python
# Distributions
NormalDistribution(mean, std, min?, max?)
UniformDistribution(min, max)
CategoricalDistribution(options: list[str], weights: list[float])
BooleanDistribution(probability_true: float)

# Sampling configuration
SamplingConfig(
    strategy: "independent" | "derived" | "conditional",
    distribution: Distribution | None,
    formula: str | None,
    depends_on: list[str],
    modifiers: list[Modifier]
)

# Grounding information
GroundingInfo(
    level: "strong" | "medium" | "low",
    method: "researched" | "extrapolated" | "estimated" | "computed",
    source: str | None,
    note: str | None
)
```

### Spec Models

```python
# Full attribute specification
AttributeSpec(
    name: str,
    type: "int" | "float" | "categorical" | "boolean",
    category: "universal" | "population_specific" | "context_specific" | "personality",
    description: str,
    sampling: SamplingConfig,
    grounding: GroundingInfo,
    constraints: list[Constraint]
)

# Complete population spec
PopulationSpec(
    meta: SpecMeta,
    grounding: GroundingSummary,
    attributes: list[AttributeSpec],
    sampling_order: list[str]
)
```

### Pipeline-Specific Types

```python
# Step 1 output
DiscoveredAttribute(name, type, category, description, depends_on)

# Step 2 output
HydratedAttribute(DiscoveredAttribute + sampling + grounding + constraints)

# Step 3 output
AttributeSpec (final form)
```

## Development Conventions

### Code Style

1. **Type hints**: Use everywhere (function signatures, class attributes)
2. **Docstrings**: Every public function has docstring with Args/Returns/Raises
3. **Imports**: Absolute imports from package root (`from entropy.llm import ...`)
4. **Error handling**: Raise specific exceptions (`CircularDependencyError`, `ValueError`)

### File Organization

- **One responsibility per file**: `selector.py` only does attribute selection
- **Public API in `__init__.py`**: Architect layer exports 5 functions
- **Models separate**: All Pydantic models in `spec.py`
- **No circular imports**: Strict dependency tree

### LLM Interaction

- **Always use structured output**: Define JSON schemas, use Pydantic models
- **Log everything**: All LLM calls logged to `logs/` with timestamps
- **Fail gracefully**: Handle LLM errors, validate responses
- **Prompt engineering**: Detailed prompts with examples and constraints

### Geographic/Cultural Awareness

- **Don't assume Western defaults**: Prompts explicitly request local context
- **Use local units**: EUR for Germany, INR for India (not always USD)
- **Use local systems**: Bundesland for Germany, states for India
- **Consider local structures**: Caste in India, credit scores may not exist everywhere

## Common Workflows

### Creating a New Population Spec

```bash
# Interactive mode (default)
entropy spec "500 German surgeons" -o surgeons.yaml

# Non-interactive mode (for testing/automation)
entropy spec "500 German surgeons" -o surgeons.yaml --yes
```

**What happens**:
1. Sufficiency check validates description
2. Attribute selection discovers ~30 attributes
3. **Human checkpoint**: Review/edit attributes
4. Distribution research finds real data (~60-180s)
5. Constraint binding builds dependency graph
6. **Human checkpoint**: Review/save spec
7. Spec saved to YAML

### Creating an Overlay (Scenario)

```bash
entropy overlay surgeons_base.yaml \
  -s "AI diagnostic tool adoption" \
  -o surgeons_ai.yaml
```

**What happens**:
1. Load base spec (e.g., 35 attributes)
2. Discover NEW scenario attributes (e.g., 8 attributes)
3. **Human checkpoint**: Review new attributes
4. Research distributions (NEW attributes can reference base)
5. Bind constraints (handles cross-layer dependencies)
6. **Human checkpoint**: Review merged spec
7. Save merged spec (43 total attributes, recomputed sampling order)

### Adding a New Architect Step

If adding a new step to the pipeline:

1. Create file: `entropy/architect/mystep.py`
2. Define main function with clear signature
3. Add to exports in `entropy/architect/__init__.py`
4. Update CLI in `entropy/cli.py` to call it
5. Add logging if LLM involved
6. Document in this file

## Testing Strategy

**Current state**: Minimal automated tests, mostly manual validation

**Testing approach**:
- **Ad-hoc scripts**: `test_search.py` for quick experiments
- **Real examples**: `surgeons_base.yaml`, `surgeons_ai.yaml` as regression tests
- **LLM logs**: `logs/` directory for debugging LLM behavior
- **Human validation**: Two checkpoints in pipeline

**Future**: Add pytest tests when patterns stabilize

## Configuration

### Environment Variables (.env)

```bash
# Required for Phase 1
OPENAI_API_KEY=sk-...

# Future phases
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=llama-3.2-3b

# Paths
DB_PATH=./storage/entropy.db

# Defaults
DEFAULT_POPULATION_SIZE=1000
```

Loaded via `entropy/config.py` using Pydantic Settings.

## Important Constraints

### Attribute Limits

- **Base mode**: 25-40 attributes total
  - Universal: 8-12
  - Population-specific: 10-18
  - Context-specific: 0-5
  - Personality: 5-8
- **Overlay mode**: 5-15 new attributes

These limits prevent:
- Overwhelming users with too many attributes
- LLM confusion from huge contexts
- Sampling complexity explosion

### Dependency Constraints

- **Max 3 dependencies per attribute**: Prevents deep dependency chains
- **No circular dependencies**: Enforced by topological sort
- **Only known attributes**: Dependencies must reference existing/context attributes

### Grounding Levels

Attributes are grounded at three levels:

- **Strong**: Direct data from authoritative sources (census, medical associations)
- **Medium**: Extrapolated from related data or regional approximations
- **Low**: Estimated/assumed based on domain knowledge

Overall spec grounding computed as:
- **Strong**: ≥60% of attributes strongly grounded
- **Medium**: ≥50% of attributes medium/strongly grounded
- **Low**: Otherwise

## CLI Commands (Current)

```bash
# Generate population spec
entropy spec "<description>" -o <file.yaml> [--yes]

# Layer scenario on base population
entropy overlay <base.yaml> -s "<scenario>" -o <output.yaml> [--yes]
```

**Flags**:
- `--yes, -y`: Skip human confirmation prompts (for automation)
- `--output, -o`: Output file path (required)
- `--scenario, -s`: Scenario description (for overlay)

## Future Phases (Not Yet Implemented)

### Phase 2: Scenario Injection
- Define events/information/decisions
- Inject into populations
- Store scenarios in DB

### Phase 3: Simulation
- Agent response generation
- Opinion propagation via social network
- Timeline tracking
- Results analysis

## Debugging Tips

### LLM Issues

1. **Check logs**: `logs/` directory has full request/response JSONs
2. **Inspect schemas**: Validate JSON schemas match Pydantic models
3. **Test prompts**: Use `test_search.py` pattern for quick iterations
4. **Try different models**: `gpt-5-mini` vs `gpt-5` vs reasoning effort levels

### Constraint Issues

1. **CircularDependencyError**: Check `depends_on` in attribute definitions
2. **Invalid dependencies**: Ensure referenced attributes exist
3. **Sampling order**: Inspect `sampling_order` in generated YAML

### Performance Issues

1. **Slow research**: Hydration takes 60-180s (expected with web search)
2. **Check reasoning_effort**: "low" vs "medium" vs "high"
3. **Monitor API usage**: Check OpenAI dashboard for costs

## Key Design Decisions

### Why YAML for specs?

- Human-readable for review/editing
- Version control friendly
- Easy to share/reproduce
- Pydantic handles serialization

### Why two-mode pipeline (base + overlay)?

- **Reusability**: Create base population once, test multiple scenarios
- **Efficiency**: Don't re-research demographics for each scenario
- **Separation of concerns**: Demographics vs. behavior

### Why human checkpoints?

- **Quality control**: Catch LLM hallucinations early
- **Customization**: Users know their domain better than LLM
- **Trust**: Users see what's happening, can validate

### Why topological sort for sampling?

- **Correctness**: Ensures dependencies sampled before dependents
- **Determinism**: Same seed → same population
- **Clarity**: Explicit sampling order is debuggable

## Common Issues & Solutions

### Issue: LLM discovers wrong attributes
**Solution**: Use Human Checkpoint #1 to remove/modify before expensive research step

### Issue: Missing sources or low grounding
**Solution**: LLM did its best; consider adding manual sources or accepting "low" grounding

### Issue: Circular dependency detected
**Solution**: Review attribute dependencies, break cycle by removing unnecessary dependency

### Issue: Overlay rediscovers base attributes
**Solution**: Check selector.py context handling; should filter out existing attributes

### Issue: Geographic context ignored
**Solution**: Ensure geography parameter passed through pipeline; check prompts use it

## Code Navigation Quick Reference

| Task | File | Function |
|------|------|----------|
| CLI entry points | `cli.py` | `spec_command()`, `overlay_command()` |
| Attribute discovery | `architect/selector.py` | `select_attributes()` |
| Distribution research | `architect/hydrator.py` | `hydrate_attributes()` |
| Dependency graph | `architect/binder.py` | `bind_constraints()`, `_topological_sort()` |
| Spec I/O | `spec.py` | `PopulationSpec.to_yaml()`, `.from_yaml()` |
| LLM calls | `llm.py` | `simple_call()`, `reasoning_call()`, `agentic_research()` |
| Config | `config.py` | `get_settings()` |

## Working with This Codebase

### When modifying the pipeline:

1. **Understand the flow**: Each step depends on previous step's output
2. **Maintain contracts**: Don't break function signatures used by CLI
3. **Add logging**: LLM interactions should be logged
4. **Update schemas**: Keep JSON schemas in sync with Pydantic models
5. **Test with real examples**: Use `entropy spec` end-to-end

### When adding new features:

1. **Check phase plan**: Consult `plans/phase1.md` for architecture
2. **Follow patterns**: Match existing code style and conventions
3. **Consider overlay mode**: Will this work with base + overlay?
4. **Document**: Update this file and function docstrings

### When fixing bugs:

1. **Check logs first**: `logs/` directory often reveals root cause
2. **Reproduce manually**: Use CLI to recreate issue
3. **Inspect YAML**: Generated specs show what actually happened
4. **Test both modes**: Base and overlay may behave differently

## Glossary

- **Architect Layer**: Phase 1 pipeline that generates PopulationSpecs
- **Attribute**: A single variable that describes agents (age, income, specialty)
- **Base Population**: A population spec without scenario-specific attributes
- **Binding**: Process of resolving dependencies and determining sampling order
- **Context**: Existing attributes provided to overlay mode
- **Grounding**: How well-supported an attribute is by real data
- **Hydration**: Process of researching distributions for attributes
- **Overlay**: Scenario-specific attributes layered on a base population
- **Sampling Strategy**: How an attribute's value is determined (independent/derived/conditional)
- **Spec**: Short for PopulationSpec - a complete population blueprint

## Related Documentation

- `README.md`: User-facing usage documentation
- `plans/phase1.md`: Detailed technical specification for Phase 1
- `plans/phase1-todo.md`: Task breakdown and implementation checklist
- `.env.example`: Environment variable template

## Version

This documentation is for **Entropy v0.1.0** (Phase 1 in development).

Last updated: 2025-12-05
