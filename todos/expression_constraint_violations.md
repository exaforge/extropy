# Expression Constraint Violations in Sampler

**Priority:** High  
**Component:** `entropy/population/sampler/core.py`  
**Related:** `entropy/core/models/population.py`

---

## Summary

The sampling report shows "Expression Constraint Violations" that include both **false positives** (validator bugs) and **real violations** (sampling logic gaps). Both need to be fixed.

---

## Issue 1: FALSE POSITIVES (Validator Bug)

### Observed

```
Expression Constraint Violations:
  ⚠ gender: sum(weights)==1: 1000 agents
  ⚠ federal_state: sum(weights)==1: 1000 agents
  ⚠ urbanity: sum(weights)==1: 1000 agents
  ⚠ marital_status: sum(weights)==1: 1000 agents
  ⚠ citizenship_status: sum(weights)==1: 1000 agents
  ⚠ highest_education_level: weights[0]+weights[1]==1: 1000 agents
  ⚠ surgical_specialty: sum(weights)==1: 1000 agents
  ⚠ employer_type: sum(weights)==1: 1000 agents
```

### Root Cause

`_check_expression_constraints()` in `core.py` evaluates **every** expression constraint against **every** agent. But constraints like `sum(weights)==1` are **spec-level constraints** — they validate that categorical distribution weights sum to 1 in the YAML definition.

When evaluated against an agent context like `{"gender": "male", "age": 45}`, there's no `weights` variable, so the expression fails for all 1000 agents.

### Affected Constraints

| Constraint Type | Example | Should Check Against |
|-----------------|---------|---------------------|
| Spec-level | `sum(weights)==1` | Spec definition (before sampling) |
| Spec-level | `weights[0]+weights[1]==1` | Spec definition (before sampling) |
| Agent-level | `children_count <= household_size - 1` | Individual agents (during/after sampling) |

### Fix

In `_check_expression_constraints()`, skip constraints that reference spec-level variables:

```python
SPEC_LEVEL_KEYWORDS = {'weights', 'options', 'sum(weights)', 'len(options)'}

for constraint in attr.constraints:
    if constraint.type == "expression" and constraint.expression:
        # Skip spec-level constraints - they don't apply to agents
        if any(kw in constraint.expression for kw in SPEC_LEVEL_KEYWORDS):
            continue
        # ... rest of validation
```

**Alternative:** Add a new constraint type `spec_validation` vs `expression` to distinguish them at the schema level.

---

## Issue 2: REAL VIOLATIONS (Sampling Logic Gap)

### Observed

```
Expression Constraint Violations:
  ⚠ children_count: children_count <= max(0, household_size - 1): 105 agents
  ⚠ years_experience: years_experience <= max(0, age - 23): 152 agents
```

### Root Cause

The sampler doesn't **enforce** expression constraints during sampling — it only **reports** violations after.

| Attribute | Distribution | Problem |
|-----------|--------------|---------|
| `children_count` | Normal(μ=household_size-2, σ=0.9) | σ=0.9 can push value 2+ std above mean, exceeding `household_size-1` |
| `years_experience` | Normal(μ=age-27, σ=3.0) | σ=3 can push value above `age-23` constraint |

Example: household_size=3 → mean=1, but Normal(1, 0.9) can sample 3 → 3 children in 3-person household = impossible (0 adults).

### Fix Options

#### Option A: Add `min_formula` / `max_formula` to Distribution Schema

Extend `NormalDistribution` model to support dynamic bounds:

```python
class NormalDistribution(BaseModel):
    # ... existing fields ...
    min_formula: str | None = None  # e.g., "0"
    max_formula: str | None = None  # e.g., "household_size - 1"
```

Update `_sample_normal()` in `distributions.py` to evaluate these formulas and clamp.

**Pros:** Clean spec-level solution, LLM can generate these  
**Cons:** Schema change, need to update hydrator prompts

#### Option B: Enforce Expression Constraints During Sampling

After sampling each attribute, check expression constraints and clamp/resample if violated:

```python
def _sample_attribute(...):
    value = ... # sample as normal
    
    # Enforce expression constraints
    for constraint in attr.constraints:
        if constraint.type == "expression" and constraint.expression:
            context = {**agent, attr.name: value}
            if not eval_condition(constraint.expression, context):
                value = _clamp_to_expression_bound(value, constraint, agent)
    
    return value
```

**Pros:** No schema change, works with existing specs  
**Cons:** Parsing constraint expressions to extract bounds is complex

#### Option C: Reject and Resample

If sampled value violates constraint, resample (with max attempts):

```python
for attempt in range(MAX_RESAMPLE_ATTEMPTS):
    value = sample_distribution(...)
    if check_constraints(value, agent, attr):
        break
else:
    value = clamp_to_hard_bounds(value, attr)
```

**Pros:** Preserves distribution shape better  
**Cons:** Could be slow, might not converge

### Recommended Approach

**Option A (min_formula/max_formula)** is the cleanest long-term fix. The spec can express:

```yaml
children_count:
  distribution:
    type: normal
    mean_formula: "max(0, household_size - 2)"
    std: 0.9
    min: 0
    max_formula: "max(0, household_size - 1)"  # Dynamic cap
```

This lets the LLM generate correct specs and ensures zero violations.

---

## Files to Modify

| File | Change |
|------|--------|
| `entropy/population/sampler/core.py` | Skip spec-level constraints in `_check_expression_constraints()` |
| `entropy/core/models/population.py` | Add `min_formula`, `max_formula` to `NormalDistribution` (and other dist types) |
| `entropy/population/sampler/distributions.py` | Evaluate min/max formulas in `_sample_normal()`, etc. |
| `entropy/population/architect/hydrator_utils.py` | Update prompts to use min/max formulas when appropriate |
| `docs/WORKFLOW.md` | Document new formula fields |

---

## Acceptance Criteria

- [ ] `sum(weights)==1` constraints do NOT appear in agent violation reports
- [ ] `children_count` violations = 0 agents
- [ ] `years_experience` violations = 0 agents
- [ ] All numeric attributes with expression constraints have 0 violations
- [ ] Existing specs continue to work (backward compatible)

