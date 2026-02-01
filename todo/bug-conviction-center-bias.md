# BUG: Conviction Level Center Bias

## Problem

~98% of agents cluster at `moderate` (~40%) and `firm` (~58%) conviction levels. The extremes (`very_uncertain`, `leaning`, `absolute`) are almost never chosen. This collapses opinion diversity and breaks conviction-dependent mechanics (flip resistance, sharing behavior, convergence detection).

## Root Cause

The `ConvictionLevel` enum is presented to the LLM as an ordered 5-option list with `moderate` at dead center (position 3/5):

```json
"conviction": {
  "type": "string",
  "enum": ["very_uncertain", "leaning", "moderate", "firm", "absolute"],
  "description": "How firmly do you hold this view?"
}
```

LLMs have well-documented center-bias when selecting from ordered enums. The prompt provides no scaffolding to guide the model away from the middle.

**Location:** `entropy/simulation/reasoning.py:179-183` (schema), `entropy/core/models/simulation.py:30-51` (enum definition)

## Evidence

From a 449-agent simulation (Austin homebuyers):
- `very_uncertain` (0.1): **0%**
- `leaning` (0.3): **~1%**
- `moderate` (0.5): **~41%**
- `firm` (0.7): **~58%**
- `absolute` (0.9): **0%**

Both extremes at exactly 0% across hundreds of agents is statistically impossible without positional bias.

## Why Two-Pass Didn't Fix This

The two-pass split fixed central tendency for **categorical outcomes** (the 83% middle-option problem from Pass 2 classification). But conviction is still selected from the same ordered enum in **Pass 1** with no intermediate reasoning step. The persona is rich but gets overridden by positional anchoring.

## Impact

- Flip resistance is uniform (nearly all agents have the same conviction threshold)
- Sharing behavior is uniform (conviction gates sharing, but all agents pass the gate)
- Convergence detection loses signal (sentiment variance is artificially low)
- The simulation produces less emergent behavior than it should

## Proposed Fixes (pick one)

### Option 1: Randomize enum order per agent (simplest)

Shuffle the conviction options before sending to the LLM. Breaks positional anchoring.

```python
# reasoning.py:179-183
import random
levels = [level.value for level in ConvictionLevel]
random.shuffle(levels)
"conviction": {
    "type": "string",
    "enum": levels,
    "description": "How firmly do you hold this view?",
}
```

Pros: One-line change. Cons: May not fully eliminate bias, just distributes it.

### Option 2: Two-step conviction

First ask a binary "more certain or less certain?", then narrow within that half.

```python
# Step 1: "Are you more on the certain side or uncertain side?"
# Step 2 (if certain): choose from ["moderate", "firm", "absolute"]
# Step 2 (if uncertain): choose from ["very_uncertain", "leaning", "moderate"]
```

Pros: Breaks 5-way center pull into smaller decisions. Cons: Extra LLM round-trip per agent (cost/latency).

### Option 3: Scalar conviction (recommended)

Ask for a 0-100 confidence score, then bucket into levels post-hoc.

```python
"conviction_score": {
    "type": "integer",
    "minimum": 0,
    "maximum": 100,
    "description": "How confident are you in your position? 0 = completely unsure, 100 = absolutely certain.",
}
# Post-hoc bucketing:
# 0-15 → very_uncertain, 16-35 → leaning, 36-60 → moderate, 61-85 → firm, 86-100 → absolute
```

Pros: Continuous scales distribute more naturally for LLMs. Cons: Need to tune bucket boundaries empirically.

### Option 4: Descriptive labels

Replace abstract names with semantic descriptions that the LLM engages with meaningfully.

```python
"conviction": {
    "type": "string",
    "enum": [
        "I really have no idea what to think",
        "I'm starting to lean one way but could easily change",
        "I have a view but could definitely be persuaded",
        "I'm fairly sure about this",
        "I am completely certain and nothing will change my mind",
    ],
}
# Map back to ConvictionLevel after extraction
```

Pros: Rich semantics reduce positional defaulting. Cons: Parsing/mapping overhead.

## Files to Change

- `entropy/simulation/reasoning.py:153-198` — `build_pass1_schema()`
- `entropy/simulation/reasoning.py:419-555` — conviction extraction in two-pass flow
- `entropy/core/models/simulation.py:30-51` — `ConvictionLevel` enum (if changing labels)
