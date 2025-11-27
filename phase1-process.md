Here's the complete call graph for Phase 1:

## File Dependency Diagram

```
User runs: python -m entropy.cli create "2000 Netflix subscribers in the US" --name netflix_us
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  cli.py                                                                      │
│  ├── create() command                                                        │
│  │   ├── db.population_exists(name)          → Check if name taken          │
│  │   ├── population.create_population(...)   → THE MAIN PIPELINE            │
│  │   ├── db.save_population(population)      → Save to SQLite               │
│  │   └── Display rich output                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ calls
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  population.py                                                               │
│  └── create_population(context, name, seed)                                  │
│      │                                                                       │
│      ├── Step 1: parse_context(context)                                      │
│      │   └── llm.chat_completion() ─────────────────────────────┐           │
│      │       Returns: ParsedContext                              │           │
│      │                                                           │           │
│      ├── Step 2: search.conduct_research(parsed) ───────────────│───┐       │
│      │   Returns: ResearchData                                   │   │       │
│      │                                                           │   │       │
│      ├── Step 3: Distributions(research, parsed)    ← No LLM     │   │       │
│      │   (builds scipy distributions from research)              │   │       │
│      │                                                           │   │       │
│      ├── Step 4: sample_agents(distributions, n)    ← No LLM     │   │       │
│      │   └── sample_agent() × n times                            │   │       │
│      │       ├── distributions.sample_age()                      │   │       │
│      │       ├── distributions.sample_income(age, edu)           │   │       │
│      │       ├── sample_psychographics(age, edu)                 │   │       │
│      │       ├── sample_cognitive(psych, edu)                    │   │       │
│      │       ├── sample_information_env(age, psych)              │   │       │
│      │       └── distributions.sample_situation()                │   │       │
│      │   Returns: list[Agent]                                    │   │       │
│      │                                                           │   │       │
│      ├── Step 5: generate_network(agents)           ← No LLM     │   │       │
│      │   └── NetworkX Watts-Strogatz + similarity weighting      │   │       │
│      │   Returns: agents with connections                        │   │       │
│      │                                                           │   │       │
│      ├── Step 6: synthesize_personas(agents, parsed) ← No LLM    │   │       │
│      │   └── synthesize_persona() × n (template-based)           │   │       │
│      │   Returns: agents with persona strings                    │   │       │
│      │                                                           │   │       │
│      └── Returns: Population(name, agents, research, ...)        │   │       │
└──────────────────────────────────────────────────────────────────│───│───────┘
                                                                   │   │
                          ┌────────────────────────────────────────┘   │
                          │                                            │
                          ▼                                            │
┌─────────────────────────────────────────────────────────────────────────────┐
│  llm.py                                                              │       │
│  ├── get_openai_client()        → Returns OpenAI client              │       │
│  ├── chat_completion()          → Structured output from LLM ◄───────┘       │
│  └── web_search()               → OpenAI web search tool ◄───────────────┐   │
└──────────────────────────────────────────────────────────────────────────│───┘
                                                                           │
                          ┌────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  search.py                                                                   │
│  └── conduct_research(parsed_context)                                        │
│      │                                                                       │
│      ├── research_demographics(context)                                      │
│      │   ├── Build queries: "Netflix demographics 2024", etc.               │
│      │   ├── For each query:                                                │
│      │   │   ├── _load_from_cache(query) OR                                 │
│      │   │   └── llm.web_search(query) → _save_to_cache()                   │
│      │   └── llm.chat_completion() → Extract DemographicStats               │
│      │                                                                       │
│      ├── research_situation(context)                                         │
│      │   ├── Build queries: "Netflix customer behavior", "churn rate", etc. │
│      │   ├── For each query:                                                │
│      │   │   ├── _load_from_cache(query) OR                                 │
│      │   │   └── llm.web_search(query) → _save_to_cache()                   │
│      │   └── llm.chat_completion() → Extract SituationResearch (DYNAMIC!)   │
│      │                                                                       │
│      ├── assess_research_quality() → "strong" / "medium" / "low"            │
│      │                                                                       │
│      ├── fill_research_gaps() (if low)                                       │
│      │   └── llm.chat_completion() → Fill missing distributions             │
│      │                                                                       │
│      └── Returns: ResearchData(demographics, situation_schema, sources)      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  models.py                                                                   │
│  ├── ParsedContext     ← Output of context parsing                          │
│  ├── ResearchData      ← Output of research                                 │
│  ├── SituationSchema   ← Dynamic attribute definitions                      │
│  ├── Demographics      ← Agent demographics                                 │
│  ├── Psychographics    ← Big Five + values + interests                      │
│  ├── Cognitive         ← Processing style, biases                           │
│  ├── InformationEnvironment                                                  │
│  ├── Network           ← Connections + influence score                      │
│  ├── Agent             ← Complete agent with all above                      │
│  └── Population        ← Collection of agents + metadata                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  config.py                                                                   │
│  └── get_settings()    → Loads from .env (OPENAI_API_KEY, DB_PATH, etc.)    │
│      Used by: llm.py, db.py, search.py                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  db.py                                                                       │
│  ├── save_population()      → SQLite insert                                 │
│  ├── load_population()      → SQLite query                                  │
│  ├── list_populations()     → For `entropy list`                            │
│  ├── delete_population()    → For `entropy delete`                          │
│  └── population_exists()    → Check before create                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Summary: Who Calls Who

| File            | Calls                      | Called By           |
| --------------- | -------------------------- | ------------------- |
| `cli.py`        | population, db             | User (command line) |
| `population.py` | llm, search, models        | cli                 |
| `search.py`     | llm, config, models        | population          |
| `llm.py`        | config (for API key)       | population, search  |
| `models.py`     | (none - just data classes) | everyone            |
| `config.py`     | (none - just settings)     | llm, search, db     |
| `db.py`         | config, models             | cli                 |

## LLM Calls Breakdown

```
cli.create()
 └── population.create_population()
      ├── parse_context()
      │   └── llm.chat_completion()              ← 1 call
      │
      └── search.conduct_research()
           ├── research_demographics()
           │   ├── llm.web_search() × 2-3        ← 2-3 calls
           │   └── llm.chat_completion()         ← 1 call
           │
           ├── research_situation()
           │   ├── llm.web_search() × 3-4        ← 3-4 calls
           │   └── llm.chat_completion()         ← 1 call
           │
           └── fill_research_gaps() (maybe)
               └── llm.chat_completion()         ← 0-1 call
                                                 ─────────
                                          Total: ~8-12 calls
```
