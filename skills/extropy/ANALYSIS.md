# Analysis

Post-run analytics, interpretation, and data science workflows.

## Primary Artifacts

After simulation, these artifacts contain your data:

| Artifact | Contents | Location |
|----------|----------|----------|
| `study.db` | Canonical database with all state | Specified via `--study-db` |
| `meta.json` | Run metadata, cost, stop reason | `results/meta.json` |
| `by_timestep.json` | Time-series aggregates | `results/by_timestep.json` |

Export additional data:
```bash
extropy export states --study-db study.db --to states.jsonl
extropy export agents --study-db study.db --to agents.jsonl
extropy export elaborations --study-db study.db --to elaborations.csv
extropy export conversations --study-db study.db --to conversations.json
extropy export posts --study-db study.db --to posts.json
```

## Analysis Sequence

### 1. Run Summary

Start with high-level metrics:
```bash
extropy results --study-db study.db
```

Key metrics:
- **Population size**: How many agents
- **Timesteps completed**: How long the simulation ran
- **Stop reason**: Why it stopped (exposure saturation, convergence, max timesteps)
- **Exposure rate**: What fraction heard about the event
- **Average conviction**: How certain agents are

### 2. Outcome Distributions

Headline results - what did people decide?

```
commute_response (categorical):
  drive_and_pay          38%
  switch_to_transit      24%
  shift_schedule         19%
  telework_more          12%
  undecided               7%
```

For float outcomes, check mean, std, min, max.
For boolean outcomes, check yes/no split.

### 3. Segment Analysis

Break down outcomes by attributes:
```bash
extropy results --study-db study.db --segment income
extropy results --study-db study.db --segment age_bracket
extropy results --study-db study.db --segment political_orientation
```

Look for:
- **Heterogeneous effects**: Do segments differ meaningfully?
- **Vulnerable groups**: Which segments have extreme outcomes?
- **Surprising patterns**: Segments behaving unexpectedly?

Example insight:
> "Low-income commuters (<$50k) are 4x more likely to protest than high-income (>$100k), who mostly comply or switch to transit."

### 4. Timeline Dynamics

How did opinions evolve?
```bash
extropy results --study-db study.db --timeline
```

Look for:
- **Exposure curve**: S-curve (normal) vs flat (spread problem) vs instant (seed too high)
- **Opinion shifts**: When did positions change?
- **Convergence point**: When did dynamics stop?
- **Inflection points**: Sudden changes worth investigating

### 5. Agent Deep Dives

Inspect individual reasoning chains:
```bash
extropy results --study-db study.db --agent agent_0042
```

Shows:
- Agent attributes
- Exposure history (when, through what channel, from whom)
- Reasoning traces per timestep
- Position/sentiment/conviction trajectory
- Conversations participated in

Use for:
- **Representative agents**: Typical member of a segment
- **Outliers**: Unusual trajectories worth explaining
- **Mechanism investigation**: Why did this agent change?

### 6. Conversation Analysis

If fidelity was medium or high:
```bash
extropy export conversations --study-db study.db --to convos.json
```

Look for:
- **State-changing conversations**: Which dialogues shifted positions?
- **Influence patterns**: Who influences whom?
- **Relationship effects**: Do partner conversations differ from coworker?

The most impactful conversations (by sentiment + conviction delta) often reveal the mechanisms of opinion change.

### 7. Social Post Analysis

If agents posted:
```bash
extropy export posts --study-db study.db --to posts.json
```

Analyze:
- **Post volume over time**: When was discourse most active?
- **Sentiment trajectory**: How did public mood evolve?
- **Position representation**: What positions got amplified?

## Confidence and Stability

### Single Run Caveat

One run with one seed is a single sample. Treat with caution.

Always state:
> "Based on single seed (42). Run confidence sweep for stability assessment."

### Confidence Sweep

Run same scenario with multiple seeds:
```bash
for seed in 42 43 44 45 46; do
  extropy simulate scenario.yaml --study-db study-$seed.db -o results-$seed --seed $seed
done
```

Then aggregate:
- Mean outcome distribution across seeds
- Standard deviation of key metrics
- Min/max range

### Stability Assessment

| Variance | Interpretation |
|----------|----------------|
| Low (std < 5% of mean) | Stable finding, high confidence |
| Medium (std 5-15%) | Moderate stability, report range |
| High (std > 15%) | Unstable, investigate causes |

Unstable outcomes may indicate:
- Bifurcation points (small changes cause large differences)
- Sensitive to initial conditions
- Underspecified population or scenario

## Comparative Analysis

When comparing variants (threshold sweep, message test, etc.):

### Comparison Template

1. **What changed**: Single axis (threshold 2 vs 3 vs 4)
2. **Delta in key outcomes**:
   - Baseline: 38% drive_and_pay
   - Thresh-2: 42% (+4pp)
   - Thresh-4: 35% (-3pp)
3. **Delta in dynamics**:
   - Thresh-2: Faster convergence (12 vs 18 timesteps)
   - Thresh-4: More opinion volatility
4. **Delta in cost**:
   - Thresh-2: $4.20 (+5%)
   - Thresh-4: $3.80 (-5%)
5. **Confidence assessment**: Stable across seeds? Sample size sufficient?

### Valid Comparison Criteria

Only compare variants directly when:
- Same population (same sample seed)
- Same scenario spec (or controlled difference)
- Same config (except the axis being tested)
- Same seed policy

## Mechanism Investigation

Beyond "what happened" - explain "why."

### Trace Analysis

For unexpected outcomes, trace back:
1. What was the agent's initial state?
2. What exposures did they receive?
3. From whom? (peer influence)
4. What conversations did they have?
5. How did their reasoning evolve?

### Pattern Recognition

Look for:
- **Attribute clusters**: Do certain attribute combinations predict outcomes?
- **Network effects**: Did outcomes cluster in network neighborhoods?
- **Temporal patterns**: Did early movers influence later decisions?
- **Conversation effects**: Which conversations were pivotal?

### Mechanism Hypotheses

Frame findings as testable mechanisms:
> "Low-income agents with no transit access defaulted to protest because the policy offers no viable alternative."

> "High-trust agents accepted the policy faster, but their conviction remained lower."

> "Coworker influence dominated for workplace-relevant scenarios; partner influence dominated for household-budget scenarios."

## Elaboration Analysis (Open-Ended Outcomes)

For `type: open_ended` outcomes:

```bash
extropy export elaborations --study-db study.db --to elaborations.csv
```

This exports:
- `agent_id`
- All demographics
- All outcome values
- Raw elaboration text

### Manual Analysis

Read a sample (50-100) to identify themes.

### Automated Clustering

Use external DS tools:
```python
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

df = pd.read_csv("elaborations.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["elaboration"].tolist())
clusters = KMeans(n_clusters=5).fit_predict(embeddings)
df["cluster"] = clusters
```

Report discovered categories, not pre-defined ones.

## Reporting

### Decision Brief Structure

1. **Decision context**: What question are we answering?
2. **Key finding**: One-sentence headline
3. **Outcome distribution**: Primary metric
4. **Segment impacts**: Who is affected differently?
5. **Confidence**: Stability assessment
6. **Mechanism**: Why this happened
7. **Recommendation**: What to do
8. **Caveats**: What could change this

### Evidence Standards

Always cite:
- Exact file paths (`results/meta.json`)
- Specific metrics with context
- Sample sizes and seed counts
- Confidence intervals or ranges where applicable

### Uncertainty Communication

| Confidence | Language |
|------------|----------|
| High | "The simulation shows...", "Consistent across seeds..." |
| Medium | "The simulation suggests...", "With moderate confidence..." |
| Low | "Preliminary indication...", "Unstable across seeds, but..." |

## Quick Analysis Commands

```bash
# Full summary
extropy results --study-db study.db

# Segment breakdown
extropy results --study-db study.db --segment income

# Timeline view
extropy results --study-db study.db --timeline

# Single agent
extropy results --study-db study.db --agent agent_0042

# Export for external analysis
extropy export states --study-db study.db --to states.jsonl
extropy export elaborations --study-db study.db --to elaborations.csv
extropy export conversations --study-db study.db --to conversations.json
```

## Red Flags

Watch for these in analysis:

| Red Flag | Indicates |
|----------|-----------|
| 100% one outcome | Event unambiguous or schema problem |
| 0% exposure spread | Network or spread config issue |
| Timestep 1 convergence | Seed exposure too high |
| Identical reasoning traces | Persona or memory issue |
| Cost 10x estimate | Runaway dynamics or config error |
| High between-seed variance | Unstable bifurcation point |

If red flags appear, triage before reporting (see TROUBLESHOOTING.md).
