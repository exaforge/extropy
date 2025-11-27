<!-- 20100bd4-6d7f-479e-b396-f92ce82b6481 2e27fd88-5256-46d9-941d-a79975cf34f7 -->
# Entropy Phase 1 Implementation

## Agent Schema Design

Based on social science models, each agent will have:

```python
demographics = {
    "age": int,
    "gender": str,
    "income": int,
    "education": str,  # high_school, bachelors, masters, doctorate
    "occupation": str,
    "location": {"state": str, "urban_rural": str},
    "ethnicity": str,
    "marital_status": str,
    "household_size": int
}

psychographics = {
    "openness": float,        # Big Five (0-1 scale)
    "conscientiousness": float,
    "extraversion": float,
    "agreeableness": float,
    "neuroticism": float,
    "values": list[str],      # e.g., ["family", "achievement", "security"]
    "interests": list[str]
}

cognitive = {
    "information_processing": str,  # analytical, intuitive, balanced
    "openness_to_change": float,    # 0-1
    "trust_in_institutions": float, # 0-1
    "confirmation_bias": float,     # 0-1, affects processing of contradicting info
    "persuadability": float         # 0-1, affects social influence susceptibility
}

information_env = {
    "news_sources": list[str],
    "social_media": list[str],
    "media_hours_daily": float,
    "trust_in_media": float,  # 0-1
    "exposure_rate": float    # 0-1, how likely to see new info
}

# DYNAMIC - populated by research phase based on context
situation = {}  # Empty base, filled per context (Netflix: tenure, plan, satisfaction, etc.)

# Network position
network = {
    "connections": list[dict],  # [{target_id, strength, type}]
    "influence_score": float    # Derived from network position
}

# Mutable state for simulation (Phase 3)
state = {
    "beliefs": dict,        # Updated during simulation
    "exposures": list,      # Events they've seen
    "emotional_state": str  # Current state
}
```

**Key: `situation` is fully dynamic.** Research phase determines what attributes matter for the context, LLM generates schema + distributions, sampling fills values. No templates.

## Implementation Order

### 1. Project Scaffolding

- Create `pyproject.toml` with dependencies (typer, fastapi, openai, numpy, scipy, networkx, pydantic)
- Set up folder structure per spec
- Create `.env.example`

### 2. Core Infrastructure

- `entropy/config.py` - Pydantic settings from .env
- `entropy/models.py` - All data models (ParsedContext, Agent, Population, etc.)
- `entropy/db.py` - SQLite CRUD for populations

### 3. LLM and Search

- `entropy/llm.py` - OpenAI client wrapper with web search capability
- `entropy/search.py` - High-level research functions using OpenAI's web search tool

### 4. Population Pipeline (`entropy/population.py`)

The core logic, implementing each pipeline step:

1. **parse_context()** - Single OpenAI call to extract structured context
2. **research_demographics()** - Web search + extraction for real data
3. **build_distributions()** - Convert research to scipy distributions with correlations
4. **sample_agents()** - Statistical sampling using copulas for correlated attributes
5. **generate_network()** - Hybrid Watts-Strogatz + similarity-weighted edges
6. **synthesize_personas()** - Template-based persona generation
7. **create_population()** - Orchestrates the full pipeline

### 5. CLI (`entropy/cli.py`)

- `entropy create "<context>" --name <name>` - Full pipeline
- `entropy list` - Show all populations
- `entropy inspect <name>` - Show population details
- `entropy delete <name>` - Remove population

### 6. Stub Files for Future Phases

- `entropy/scenario.py` - Placeholder for Phase 2
- `entropy/simulation.py` - Placeholder for Phase 3
- `entropy/api.py` - FastAPI stub for future web UI

## Key Implementation Details

**Correlations**: Use Gaussian copulas to sample correlated attributes (age-income, education-income, etc.)

**Network**: Watts-Strogatz with k=6, p=0.1, then weight edges by agent similarity (shared demographics, psychographics)

**Template Persona**: Jinja2-style templating that reads naturally:

```
"A {age}-year-old {gender} from {location}, working as a {occupation}. 
{pronoun} tends to be {openness_desc} and {extraversion_desc}..."
```

**Caching**: Cache web search results in `data/cache/` to avoid redundant API calls for similar queries

## Files to Create

| File | Purpose |

|------|---------|

| `pyproject.toml` | Dependencies, project metadata |

| `entropy/__init__.py` | Package init |

| `entropy/config.py` | Environment config |

| `entropy/models.py` | Pydantic models |

| `entropy/db.py` | SQLite operations |

| `entropy/llm.py` | OpenAI client |

| `entropy/search.py` | Web search wrapper |

| `entropy/population.py` | Phase 1 pipeline |

| `entropy/cli.py` | Typer CLI |

| `entropy/scenario.py` | Phase 2 stub |

| `entropy/simulation.py` | Phase 3 stub |

| `entropy/api.py` | FastAPI stub |

### To-dos

- [ ] Create project structure, pyproject.toml, .env.example
- [ ] Implement Pydantic models in models.py
- [ ] Implement config.py with Pydantic settings
- [ ] Implement SQLite CRUD operations in db.py
- [ ] Implement OpenAI client with web search in llm.py
- [ ] Implement research functions in search.py
- [ ] Implement full population pipeline in population.py
- [ ] Implement Typer CLI commands in cli.py
- [ ] Create placeholder files for Phase 2/3 (scenario.py, simulation.py, api.py)