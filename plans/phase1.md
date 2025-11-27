# Entropy: Technical Specification v2

## Project Overview

Entropy simulates how populations respond to scenarios. Create synthetic populations grounded in real-world data, simulate how they react to events, and watch opinions evolve through social networks over time.

---

## Three Phases

| Phase                            | What It Does                                          | LLM                            |
| -------------------------------- | ----------------------------------------------------- | ------------------------------ |
| **Phase 1: Population Creation** | Create N agents from natural language context         | OpenAI API                     |
| **Phase 2: Scenario Injection**  | Define what happens to this population                | OpenAI API                     |
| **Phase 3: Simulation**          | Agents respond; opinions evolve with social influence | LM Studio (local SLM, batched) |

---

## Phase 1: Population Creation

### Input

```
"2000 Netflix subscribers in the US"
```

### Output

- 2000 unique agents with complete profiles
- Social network graph
- Natural language personas
- Metadata (grounding level, sources)

### Pipeline

```
Context Parsing (OpenAI)
    ↓
Research (Web Search + OpenAI for extraction/gaps)
    ↓
Distribution Building (Code)
    ↓
Agent Sampling (Statistics)
    ↓
Network Generation (Code)
    ↓
Persona Synthesis (OpenAI, batched)
    ↓
Storage (SQLite)
```

### Step Details

**1. Context Parsing**

- LLM parses natural language → structured context
- 1 OpenAI call

**2. Research**

- Web search for real demographics, behavioral data
- OpenAI extracts structured stats from search results
- OpenAI fills gaps if research insufficient
- 3-5 search queries, 2-4 OpenAI calls

**3. Distribution Building**

- Code transforms research → probability distributions
- No LLM

**4. Agent Sampling**

- Statistical sampling from distributions
- Correlations enforced mathematically
- No LLM

**5. Network Generation**

- Graph algorithm connects agents by similarity
- No LLM

**6. Persona Synthesis**

- OpenAI generates natural language persona from structured attributes
- Batched: ~50 agents per call
- 2000 agents = ~40 OpenAI calls

**7. Storage**

- SQLite persistence

---

## Tech Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| Language         | Python 3.11+                   |
| CLI              | Typer                          |
| API              | FastAPI                        |
| Database         | SQLite                         |
| LLM (Phase 1, 2) | OpenAI API                     |
| LLM (Phase 3)    | LM Studio (local)              |
| Web Search       | OpenAI web search or Brave API |
| Statistics       | NumPy, SciPy                   |
| Graph            | NetworkX                       |
| Validation       | Pydantic                       |

---

## File Structure

```
entropy/
├── pyproject.toml
├── README.md
├── .env.example
│
├── entropy/
│   ├── __init__.py
│   ├── cli.py                      # Typer CLI
│   ├── api.py                      # FastAPI app
│   ├── config.py                   # Config management
│   ├── db.py                       # SQLite operations
│   │
│   ├── llm.py                      # LLM clients (OpenAI + LM Studio)
│   ├── search.py                   # Web search
│   │
│   ├── population.py               # Phase 1: all population creation logic
│   ├── scenario.py                 # Phase 2: scenario injection
│   ├── simulation.py               # Phase 3: simulation engine
│   │
│   └── models.py                   # All Pydantic models
│
├── tests/
│   ├── test_population.py
│   ├── test_scenario.py
│   └── test_simulation.py
│
├── data/
│   ├── base_distributions.json     # Census, psychographic baselines
│   └── cache/                      # Research cache
│
└── storage/                        # SQLite + populations (gitignored)
```

**12 files in entropy/. That's it.**

---

## CLI Commands

```bash
# Phase 1
entropy create "<context>" --name <name>
entropy list
entropy inspect <name>
entropy delete <name>

# Phase 2
entropy scenario "<description>" --population <name> --name <scenario_name>

# Phase 3
entropy simulate <population> --scenario <scenario_name>
entropy simulate <population> --scenario <scenario_name> --mode continuous --duration 7d

# Results
entropy results <population>
entropy results <population> --by <attribute>
entropy results <population> --timeline
```

---

## Configuration

```bash
# .env

OPENAI_API_KEY=sk-...
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=llama-3.2-3b

DB_PATH=./storage/entropy.db
DEFAULT_POPULATION_SIZE=1000
```

---

## Data Models

```python
# entropy/models.py

class ParsedContext(BaseModel):
    size: int
    base_population: str
    context_type: str
    context_entity: str | None
    geography: str | None
    filters: list[str]

class Agent(BaseModel):
    id: str
    demographics: dict
    psychographics: dict
    cognitive: dict
    information_env: dict
    situation: dict
    connections: list[dict]
    persona: str
    state: dict = {}

class Population(BaseModel):
    name: str
    size: int
    context_raw: str
    context_parsed: ParsedContext
    grounding_level: str
    sources: list[str]
    agents: list[Agent]
    created_at: datetime

class Scenario(BaseModel):
    name: str
    population_name: str
    description: str
    event_type: str
    channels: list[str]
    created_at: datetime

class SimulationResult(BaseModel):
    population_name: str
    scenario_name: str
    mode: str
    results: dict
    timeline: list[dict] | None
    created_at: datetime
```

---

## LLM Usage Summary

| Step                | LLM       | Calls (2000 agents)     |
| ------------------- | --------- | ----------------------- |
| Context parsing     | OpenAI    | 1                       |
| Research extraction | OpenAI    | 2-4                     |
| Gap filling         | OpenAI    | 0-2                     |
| Persona synthesis   | OpenAI    | ~40 (batched)           |
| **Phase 1 Total**   | OpenAI    | ~45-50                  |
| Simulation          | LM Studio | 2000+ (batched locally) |

---

## API Endpoints (FastAPI)

```python
# For future web UI

POST   /populations              # Create population
GET    /populations              # List populations
GET    /populations/{name}       # Get population details
DELETE /populations/{name}       # Delete population

POST   /scenarios                # Create scenario
GET    /scenarios                # List scenarios

POST   /simulations              # Run simulation
GET    /simulations/{id}         # Get results
GET    /simulations/{id}/timeline # Get timeline (continuous mode)
```

---

## Phase 1 Deliverables

1. `entropy create` command working end-to-end
2. Web search → research extraction
3. Statistical agent generation with correlations
4. Network generation
5. Persona synthesis (OpenAI batched)
6. SQLite storage
7. `entropy list`, `entropy inspect`, `entropy delete`

---

## Success Criteria

| Metric                      | Target                  |
| --------------------------- | ----------------------- |
| Creation time (2000 agents) | < 3 minutes             |
| OpenAI calls (Phase 1)      | < 60                    |
| Agent uniqueness            | 100%                    |
| Reproducibility             | Same seed → same agents |

---

## Answers

1. **Persona synthesis now uses OpenAI** — batched, ~50 agents per call

2. **Yes** — file structure supports Phase 2, 3, and FastAPI for web UI

3. **Updated** — OpenAI for Phase 1/2, LM Studio for Phase 3 simulation

**It's better quality, but not necessary. Here's the tradeoff:**

---

## Option A: LLM Persona Synthesis (Current Plan)

**How it works:**

```
Send 50 agents' structured attributes to OpenAI
→ Get back 50 natural language personas
```

**Pros:**

- More natural, varied language
- Can capture nuance ("She's the type who...")
- Personas feel less formulaic

**Cons:**

- ~40 API calls for 2000 agents
- Cost: ~$0.50-1.00 per population (GPT-4o-mini)
- Adds ~60-90 seconds to creation time

---

## Option B: Template Persona Synthesis (No LLM)

**How it works:**

```python
persona = f"""
You are a {age}-year-old {gender} in {location}.
You work as a {occupation}, earning {income_range}.
You are {openness_desc}. You tend to be {conscientiousness_desc}.
As a Netflix subscriber for {tenure}, you {satisfaction_desc}.
"""
```

**Pros:**

- Instant (milliseconds)
- Free
- Fully deterministic

**Cons:**

- Formulaic, repetitive feel
- Less natural variation
- "Mad libs" quality

---

## My Recommendation

**Start with templates. Switch to LLM if quality suffers.**

Why:

1. The persona is just context for the simulation LLM—it doesn't need to be literary
2. What matters is the structured attributes being accurate, not the prose
3. You can always upgrade later
4. Saves ~$1 per population and ~90 seconds

**The simulation LLM (Phase 3) is what makes agents feel real, not the persona text.**

---

## Cost Comparison (2000 agents)

| Approach                   | Phase 1 Cost | Time  |
| -------------------------- | ------------ | ----- |
| Template personas          | ~$0.05       | ~30s  |
| LLM personas (GPT-4o-mini) | ~$0.80       | ~2min |
| LLM personas (GPT-4o)      | ~$8.00       | ~2min |

At scale (100 populations): $5 vs $80 vs $800.

---

**Decision: Use templates for Phase 1 MVP. Revisit if simulation quality needs better persona priming.**

Want me to update the spec?
