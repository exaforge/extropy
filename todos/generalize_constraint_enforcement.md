# Implementation Plan: Generalizable Expression Constraint Enforcement

**Status:** In Progress  
**Priority:** High  
**Created:** 2024-12-25

---

## Executive Summary

This document describes the complete implementation to make expression constraint enforcement **generalizable** across ANY population spec, not just manually-edited examples.

**Problem:** The LLM sometimes creates expression constraints like `children_count <= household_size - 1` but forgets to add the corresponding `max_formula` that enforces the bound during sampling. This leads to constraint violations.

**Solution:** Add validation rules that detect this pattern and trigger LLM retry with prescriptive error messages telling it exactly what to fix.

---

## Part 1: What Was Already Implemented

### Phase 1: Core Infrastructure (DONE)

Added support for dynamic bounds in the sampling engine.

#### Files Modified

| File | Changes |
|------|---------|
| `entropy/core/models/population.py` | Added `min_formula`, `max_formula` fields to `NormalDistribution`, `LognormalDistribution`, `BetaDistribution`. Added `spec_expression` to `Constraint.type` Literal. |
| `entropy/population/sampler/distributions.py` | Added `_resolve_optional_param()` helper. Updated `_sample_normal()`, `_sample_lognormal()`, `_sample_beta()` to evaluate and apply formula bounds. |
| `entropy/population/sampler/modifiers.py` | Updated `_apply_numeric_modifiers()` to re-apply formula bounds after modifier multiplication/addition. |
| `entropy/population/sampler/core.py` | Updated `_check_expression_constraints()` to skip `spec_expression` type constraints (only validates `expression` type against agents). |
| `entropy/population/architect/hydrator.py` | Updated LLM prompts in `hydrate_independent()` and `hydrate_conditional_base()` to explain `spec_expression` vs `expression` and `min_formula`/`max_formula` usage. |
| `entropy/population/architect/hydrator_utils.py` | Updated JSON schemas to include `spec_expression` in constraint type enum. Added `min_formula`/`max_formula` to distribution schema. Updated `parse_distribution()` to handle formula fields. |
| `docs/WORKFLOW.md` | Added documentation for "Constraint Types" and "Dynamic Bounds with Formula" sections. |
| `tests/test_sampler.py` | Added `TestSpecExpressionConstraints` (2 tests) and `TestDynamicBounds` (8 tests including `test_modifier_respects_formula_bounds`). |

#### How Dynamic Bounds Work

**Old flow (without `max_formula`):**
```
1. Sample value from distribution
2. Apply static min/max bounds (if any)
3. Apply modifiers (multiply, add)
4. Re-apply static min/max bounds
5. Result: Value may exceed logical constraints
```

**New flow (with `max_formula`):**
```
1. Sample value from distribution
2. Evaluate max_formula using agent context -> dynamic bound
3. Apply resolved min/max bounds
4. Apply modifiers (multiply, add)
5. Re-evaluate max_formula -> re-apply dynamic bound
6. Result: Value always respects agent-specific constraints
```

**Example:**
```yaml
children_count:
  distribution:
    type: normal
    mean_formula: "max(0, household_size - 2)"
    std: 0.9
    min: 0
    max_formula: "max(0, household_size - 1)"  # Dynamic upper bound
  depends_on: [household_size]
```

For an agent with `household_size=4`:
- `max_formula` evaluates to `4 - 1 = 3`
- Sampled value is clamped to max 3
- After modifiers, value is re-clamped to max 3
- Result: `children_count <= 3` guaranteed

---

## Part 2: What's Missing (To Be Implemented)

### The Gap

The **sampler** is fully generalizable - it works with any attribute/formula combination.

The **problem** is the LLM doesn't always output `max_formula`/`min_formula` when it should. It creates constraints but forgets the enforcement mechanism.

### The Fix: Validation + Retry

Following the existing pattern in `WORKFLOW.md`, we add **validation rules** that:
1. Detect when the LLM made a mistake
2. Generate prescriptive error messages
3. Trigger retry with error feedback
4. LLM fixes its own mistakes

---

## Part 3: Validation Rules to Implement

### Rule 1: Spec-Level Constraint Type Detection

**What it detects:** When the LLM uses `type: expression` for constraints that reference spec-level variables (`weights`, `options`).

**Pattern (BAD):**
```yaml
constraints:
  - type: expression  # WRONG TYPE
    expression: "sum(weights)==1"
```

**Error message:**
```
ERROR in gender.constraints:
  Value: 'sum(weights)==1'
  Problem: constraint references spec-level variables (weights) but uses type='expression'
  Fix: Change to type='spec_expression' - this validates the YAML spec itself, not individual agents
```

### Rule 2: Missing Formula Bound Detection

**What it detects:** When the LLM creates an expression constraint like `attr <= expr` but doesn't add the corresponding `max_formula` to the distribution.

**Patterns to detect:**

| Expression Pattern | Should Add |
|-------------------|------------|
| `attr <= expr` | `max_formula: "expr"` |
| `attr < expr` | `max_formula: "expr - 1"` (for int) or `max_formula: "expr"` (for float) |
| `attr >= expr` | `min_formula: "expr"` |
| `attr > expr` | `min_formula: "expr + 1"` (for int) or `min_formula: "expr"` (for float) |

**Error message:**
```
ERROR in children_count:
  Value: constraint 'children_count <= max(0, household_size - 1)'
  Problem: distribution has no max_formula to enforce this bound during sampling
  Fix: Add to distribution: max_formula: 'max(0, household_size - 1)'
```

---

## Part 4: Files to Modify

| File | Changes |
|------|---------|
| `entropy/population/architect/hydrator_utils.py` | Add `_extract_bound_from_constraint()` helper. Add spec-level constraint check to `validate_independent_hydration()`. Add missing formula bound check to `validate_conditional_base()`. |
| `entropy/population/architect/quick_validate.py` | Add same validation to `validate_independent_response()` and `validate_conditional_base_response()` for fail-fast detection. |

---

## Part 5: Expected LLM Behavior After Fix

**Step 1:** LLM outputs (first attempt):
```yaml
children_count:
  distribution:
    type: normal
    mean_formula: "max(0, household_size - 2)"
    std: 0.9
    min: 0
    # NO max_formula
  constraints:
    - type: expression
      expression: "children_count <= household_size - 1"
```

**Step 2:** Validation catches error, triggers retry with:
```
ERROR: children_count has expression constraint 'children_count <= household_size - 1' 
but distribution has no max_formula. Add: max_formula: 'household_size - 1'
```

**Step 3:** LLM retries and outputs (corrected):
```yaml
children_count:
  distribution:
    type: normal
    mean_formula: "max(0, household_size - 2)"
    std: 0.9
    min: 0
    max_formula: "household_size - 1"  # ADDED
  constraints:
    - type: expression
      expression: "children_count <= household_size - 1"
```

**Step 4:** Validation passes. Sampling produces zero violations.

---

## Part 6: Acceptance Criteria

- [ ] `validate_independent_hydration()` catches `expression` type used for spec-level constraints
- [ ] `validate_conditional_base()` catches missing `max_formula` when constraint has `attr <= expr`
- [ ] `validate_conditional_base()` catches missing `min_formula` when constraint has `attr >= expr`
- [ ] Error messages are prescriptive (tell LLM exactly what to add)
- [ ] Errors trigger retry (not just warnings)
- [ ] Same validation in `quick_validate.py` for fail-fast detection
- [ ] All existing tests pass
- [ ] New tests for validation logic pass

---

## Part 7: Workflow Comparison

### Old Workflow (Before All Changes)

```
LLM outputs constraints
    |
No validation of constraint/formula alignment
    |
Spec saved with constraints but no formula bounds
    |
Sampling happens
    |
Values exceed logical bounds
    |
Constraint violations reported (105 agents, 152 agents, etc.)
    |
User manually edits YAML to add max_formula
```

### New Workflow (After All Changes)

```
LLM outputs constraints
    |
Validation checks:
  - Is expression type correct? (spec_expression vs expression)
  - Does constraint have matching formula bound?
    |
If errors: Retry with prescriptive feedback
  "Add to distribution: max_formula: 'household_size - 1'"
    |
LLM fixes its own mistakes
    |
Spec saved with BOTH constraints AND formula bounds
    |
Sampling happens
    |
Formula bounds enforce constraints during sampling
    |
Zero constraint violations
    |
Spec is clean, self-documenting, and correct
```
