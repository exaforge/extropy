# CLI Refactor Plan

## Overview

Simplify the CLI pipeline from the current fragmented flow to a clean, linear pipeline with automatic file naming, study folder management, and scenario-centric design.

**New Pipeline:**
```
spec → scenario → persona → sample → network → simulate → results
```

**Key Changes:**
1. `scenario` absorbs `extend` - one command does attribute extension + scenario config
2. Auto-versioned file naming - no manual `-o` for every file
3. Study folder structure - everything organized under study directory
4. `study.db` inferred from cwd - no `--study-db` everywhere
5. Scenario-centric keying - downstream commands key off scenario, not population

---

## Study Folder Structure

```
{study-name}/
  study.db                          # canonical SQLite store
  population.v1.yaml                # base population spec
  population.v2.yaml                # iterations
  scenario/
    {scenario-name}/
      scenario.v1.yaml              # scenario spec (includes extended attrs)
      scenario.v2.yaml              # iterations
      persona.v1.yaml               # persona config for this scenario
    {another-scenario}/
      scenario.v1.yaml
      persona.v1.yaml
```

**Default study folder:** `study-{timestamp}` if not specified

**study.db keying:** Changes from `population_id` to `scenario_id` as primary key for:
- `agents` table
- `network_edges` table
- `simulation_runs` table

---

## Command Specifications

### 1. `extropy spec`

Creates base population spec with attribute definitions and distributions.

```bash
# Create new study
extropy spec "500 German surgeons who work in hospitals" -o surgeons
# → creates surgeons/
# → creates surgeons/study.db
# → creates surgeons/population.v1.yaml

# Custom population name
extropy spec "German surgeons" -o surgeons/hospital-surgeons
# → surgeons/hospital-surgeons.v1.yaml

# Iterate on existing (from within study folder)
cd surgeons
extropy spec "German surgeons, add income distribution"
# → population.v2.yaml (auto-increment)

# From anywhere, specify study
extropy --study surgeons spec "updated description"
# → surgeons/population.v2.yaml
```

**Flags:**
| Flag | Required | Description |
|------|----------|-------------|
| `-o, --output` | Yes (if not in study) | Study folder or `folder/name` |
| `--study` | No | Global flag to specify study folder |

**Changes from current:**
- Remove `size` from spec - moved to `sample -n`
- Auto-create study folder + study.db
- Auto-version files

**Output:** `population.vN.yaml` with:
- `meta.description`
- `meta.geography`
- `meta.agent_focus`
- `attributes[]` with full distribution specs
- `sampling_order[]`
- `grounding` summary

---

### 2. `extropy scenario`

Creates scenario spec by extending population with scenario-specific attributes AND defining event/exposure/outcome config. **Absorbs current `extend` command.**

```bash
cd surgeons

# Create new scenario
extropy scenario "AI diagnostic tool adoption, announced via hospital email,
                  measure adoption intent and trust over 2 weeks" -o ai-adoption
# → scenario/ai-adoption/scenario.v1.yaml
# Uses latest population.vN.yaml by default

# Pin population version
extropy scenario "vaccine mandate" -o vaccine @pop:v1
# → scenario/vaccine/scenario.v1.yaml with base_population: population.v1

# Use explicit latest
extropy scenario "..." -o test @pop:latest
# → resolves to current latest, stores concrete version

# Iterate on existing scenario
extropy scenario "AI adoption, add social media exposure channel" -o ai-adoption
# → scenario/ai-adoption/scenario.v2.yaml

# Rebase to new population (cheap, no LLM re-run)
extropy scenario ai-adoption --rebase @pop:v2
# → validates compatibility
# → creates scenario.v2.yaml with base_population: population.v2
# → copies attrs/config from scenario.v1.yaml
```

**Flags:**
| Flag | Required | Description |
|------|----------|-------------|
| `-o, --output` | Yes (for new) | Scenario name |
| `@pop:vN` | No | Pin population version |
| `@pop:latest` | No | Explicit latest (resolves at creation) |
| `--rebase` | No | Cheap update to new population |

**`--rebase` safety detection:**
1. Load scenario's attribute dependencies (what base attrs does it reference?)
2. Check new population has all required attributes
3. If missing → error: "Unsafe rebase: scenario depends on `income` which doesn't exist in population.v2"
4. If safe → create new scenario version with updated reference

**Output:** `scenario.vN.yaml` with:
```yaml
meta:
  name: ai-adoption
  description: "AI diagnostic tool adoption..."
  base_population: population.v2  # concrete reference, never "latest"
  created_at: ...

# Extended attributes (what extend currently does)
attributes:
  - name: ai_familiarity
    type: categorical
    ...
  - name: tech_anxiety
    type: float
    ...

# Scenario config (current scenario command)
event:
  type: announcement
  content: "Hospital introducing AI diagnostic tool..."
  source: hospital_admin
  credibility: 0.85

seed_exposure:
  channels: [...]
  rules: [...]

interaction:
  primary_model: deliberation
  ...

spread:
  share_probability: 0.3
  ...

outcomes:
  suggested_outcomes:
    - name: adoption_intent
      type: categorical
      options: [will_adopt, considering, resistant, opposed]
    ...

simulation:
  max_timesteps: 14
  timestep_unit: day
  ...
```

**Validation:**
1. Load base population
2. Merge attributes (base + scenario)
3. Recompute sampling_order via topological sort
4. Run structural validation on merged spec
5. If invalid → save as `scenario.vN.invalid.yaml`
6. If valid → save as `scenario.vN.yaml`

---

### 3. `extropy persona`

Generates persona rendering config from scenario's full attribute set.

```bash
cd surgeons

extropy persona -s ai-adoption
# → scenario/ai-adoption/persona.v1.yaml

# Iterate
extropy persona -s ai-adoption
# → persona.v2.yaml if v1 exists

# Pin scenario version
extropy persona -s ai-adoption@v1
```

**Flags:**
| Flag | Required | Description |
|------|----------|-------------|
| `-s, --scenario` | Yes (if multiple) | Scenario name, auto-select if only one |
| `@vN` | No | Pin scenario version |

**What it does:**
1. Load scenario spec (for full attribute list including base + extended)
2. Run LLM pipeline to generate phrasings (5 steps)
3. Save persona config

**Note:** Population stats (mean/std for relative positioning) are computed at render time from sampled agents, NOT stored in persona.yaml. This keeps persona independent of sample size.

**Output:** `persona.vN.yaml` with:
- `intro_template`
- `treatments[]`
- `groups[]`
- `phrasings.boolean[]`
- `phrasings.categorical[]`
- `phrasings.relative[]`
- `phrasings.concrete[]`
- NO `population_stats` (computed at render time)

---

### 4. `extropy sample`

Samples agents from merged population + scenario attributes.

```bash
cd surgeons

extropy sample -s ai-adoption -n 500
# → loads scenario.vN.yaml + referenced population
# → merges attributes
# → validates merged spec
# → samples 500 agents
# → saves to study.db (scenario_id=ai-adoption)

# With seed
extropy sample -s ai-adoption -n 500 --seed 42

# Re-sample (overwrites existing agents for this scenario)
extropy sample -s ai-adoption -n 1000 --seed 123

# Pin versions
extropy sample -s ai-adoption@v1 -n 500
```

**Flags:**
| Flag | Required | Description |
|------|----------|-------------|
| `-s, --scenario` | Yes (if multiple) | Scenario name |
| `-n, --count` | Yes | Number of agents to sample |
| `--seed` | No | Random seed for reproducibility |
| `@vN` | No | Pin scenario version |

**Pre-flight validation:**
1. Check persona.vN.yaml exists for this scenario → error if missing
2. Load scenario + population
3. Merge attributes
4. Validate merged spec (structural checks)
5. If valid → proceed to sample
6. If invalid → error with details

**Output:** Agents saved to `study.db`:
- `agents` table keyed by `scenario_id`
- Each agent has all attributes (base + scenario)

---

### 5. `extropy network`

Generates social network from sampled agents.

```bash
cd surgeons

extropy network -s ai-adoption
# → loads agents from study.db (scenario_id=ai-adoption)
# → generates network
# → saves to study.db

# With config
extropy network -s ai-adoption --avg-degree 15 --seed 42

# Generate network config from population spec
extropy network -s ai-adoption --generate-config
```

**Flags:**
| Flag | Required | Description |
|------|----------|-------------|
| `-s, --scenario` | Yes (if multiple) | Scenario name |
| `--avg-degree` | No | Target average degree |
| `--seed` | No | Random seed |
| `--generate-config` | No | Generate config via LLM |
| Other existing flags | No | Quality profile, workers, etc. |

**Pre-flight:**
- Check agents exist for this scenario → error if missing

**Output:** Network saved to `study.db`:
- `network_edges` table keyed by `scenario_id`

---

### 6. `extropy simulate`

Runs simulation using scenario spec + agents + network.

```bash
cd surgeons

extropy simulate -s ai-adoption
# → loads scenario spec
# → loads agents + network from study.db
# → loads persona config
# → runs simulation
# → saves results to study.db

# With options
extropy simulate -s ai-adoption --seed 42 --strong openai/gpt-5 --fast openai/gpt-5-mini
```

**Flags:**
| Flag | Required | Description |
|------|----------|-------------|
| `-s, --scenario` | Yes (if multiple) | Scenario name |
| `--seed` | No | Random seed |
| `--strong` | No | Strong model for reasoning |
| `--fast` | No | Fast model for classification |
| Other existing flags | No | Rate limits, checkpointing, etc. |

**Pre-flight:**
- Check agents exist → error if missing
- Check network exists → error if missing
- Check persona config exists → error if missing

**Persona rendering:**
- Load `persona.vN.yaml` for phrasings
- Compute population stats from agents in study.db (fresh, not stale)
- Render personas at simulation time

**Output:** Results saved to `study.db`:
- `simulation_runs` table
- `agent_states` table
- `timestep_summaries` table
- etc.

---

### 7. `extropy results`

View simulation results.

```bash
cd surgeons

extropy results -s ai-adoption
# → shows latest run for this scenario

extropy results -s ai-adoption --timeline
extropy results -s ai-adoption --segment age_group
extropy results -s ai-adoption --agent agent_123
```

**Flags:**
| Flag | Required | Description |
|------|----------|-------------|
| `-s, --scenario` | No | Scenario name (latest if omitted) |
| `--run-id` | No | Specific run ID |
| `--timeline` | No | Show timeline view |
| `--segment` | No | Segment by attribute |
| `--agent` | No | Show single agent |

---

### 8. `extropy validate`

Validates spec files. Updated to handle new structure.

```bash
# Validate population spec
extropy validate population.v1.yaml

# Validate scenario spec (auto-detects, loads + validates merged)
extropy validate scenario/ai-adoption/scenario.v1.yaml
# → loads referenced population
# → merges attributes
# → validates merged result

# Validate persona config
extropy validate scenario/ai-adoption/persona.v1.yaml
```

**Auto-detection:**
- `population.*.yaml` → population validation
- `scenario.*.yaml` → scenario validation (includes merge validation)
- `persona.*.yaml` → persona config validation

---

## Global `--study` Flag

All commands support `--study` to specify study folder from anywhere:

```bash
extropy --study ~/projects/surgeons sample -s ai-adoption -n 500
extropy --study surgeons simulate -s ai-adoption
```

If not specified, inferred from cwd:
- If cwd contains `study.db` → use cwd
- If cwd is inside a study folder → find parent with `study.db`
- Otherwise → error "Not in a study folder. Use --study or -o to specify."

---

## Deprecations & Removals

| Command | Action |
|---------|--------|
| `extend` | **Remove** - absorbed into `scenario` |
| `--study-db` flag | **Remove** - use `--study` or infer from cwd |
| `spec --size` / `meta.size` | **Move** to `sample -n` |

---

## Database Schema Changes

### New: `scenario_id` as primary key

```sql
-- agents table
ALTER TABLE agents ADD COLUMN scenario_id TEXT;
-- Primary key becomes (scenario_id, agent_id)

-- network_edges table
ALTER TABLE network_edges ADD COLUMN scenario_id TEXT;
-- Primary key becomes (scenario_id, source_id, target_id)

-- simulation_runs table already has scenario linkage via population_id
-- Update to use scenario_id instead
```

### Migration

No migration for existing study.db files. Old format won't work with new CLI. Users re-run pipeline.

---

## Implementation Phases

### Phase 1: Foundation
1. Add `--study` global flag to CLI app
2. Add study folder detection logic
3. Add auto-versioning logic for YAML files
4. Update `study.db` schema with `scenario_id`

### Phase 2: Spec Command
1. Remove `size` from spec (move to sample)
2. Add auto study folder creation
3. Add auto-versioning
4. Update sufficiency check (no size required)

### Phase 3: Scenario Command (Big One)
1. Absorb `extend` logic into scenario
2. New scenario YAML schema with `attributes` section
3. Add `@pop:vN` syntax parsing
4. Add `--rebase` flag with safety detection
5. Merge + validate flow
6. Auto-versioning

### Phase 4: Persona Command
1. Update to use `-s scenario` instead of spec file
2. Remove population stats from output (compute at render time)
3. Auto-versioning

### Phase 5: Sample Command
1. Change from spec file input to `-s scenario`
2. Add `-n count` as required flag
3. Merge population + scenario at sample time
4. Pre-flight persona check
5. Save with `scenario_id` key

### Phase 6: Network Command
1. Change to `-s scenario` input
2. Save with `scenario_id` key

### Phase 7: Simulate Command
1. Change to `-s scenario` input
2. Update persona rendering to compute stats at runtime
3. Load from `scenario_id` keyed tables

### Phase 8: Results & Validate
1. Update results to use `-s scenario`
2. Update validate to handle merged scenario validation

### Phase 9: Cleanup
1. Remove `extend` command
2. Remove `--study-db` flag
3. Update docs
4. Update tests

---

## File Changes Summary

| File | Change |
|------|--------|
| `extropy/cli/app.py` | Add `--study` global flag, study detection |
| `extropy/cli/commands/spec.py` | Remove size, add auto-versioning |
| `extropy/cli/commands/scenario.py` | Absorb extend, new schema, `@pop:` syntax |
| `extropy/cli/commands/extend.py` | **Delete** |
| `extropy/cli/commands/persona.py` | `-s scenario` input, remove stats storage |
| `extropy/cli/commands/sample.py` | `-s scenario` + `-n count`, merge logic |
| `extropy/cli/commands/network.py` | `-s scenario` input |
| `extropy/cli/commands/simulate.py` | `-s scenario`, runtime stats |
| `extropy/cli/commands/results.py` | `-s scenario` input |
| `extropy/cli/commands/validate.py` | Merged scenario validation |
| `extropy/cli/commands/__init__.py` | Remove extend import |
| `extropy/storage/study_db.py` | Add `scenario_id` to schema |
| `extropy/core/models/scenario.py` | Add `attributes` section to schema |
| `extropy/population/persona/renderer.py` | Compute stats at render time |
| `extropy/cli/utils.py` | Add versioning helpers |
| New: `extropy/cli/study.py` | Study folder detection, versioning logic |

---

## Open Questions

1. **Config command** - should `extropy config` also be study-aware? Currently global.

2. **Other commands** - `inspect`, `query`, `report`, `export`, `chat`, `migrate` - leave as-is for now? They may need `--study` support later.

3. **Backwards compat** - should we support `--study-db` with deprecation warning for a version? Or hard break?
