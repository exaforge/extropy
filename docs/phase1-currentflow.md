# Phase 1: `entropy spec` Command — Full Architecture

This document describes the complete flow when running `entropy spec "500 German surgeons"`.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         entropy spec "500 German surgeons"               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 0: check_sufficiency()                                              │
│ ─────────────────────────────────────────────────────────────────────── │
│ Module: population/architect/sufficiency.py                              │
│ Model: gpt-5-mini (fast)                                                 │
│ Purpose: Check if description has enough info                            │
│ Output: SufficiencyResult(sufficient, size, geography, clarifications)   │
│ Validation: None (simple JSON schema enforcement)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: select_attributes()                                              │
│ ─────────────────────────────────────────────────────────────────────── │
│ Module: population/architect/selector.py                                 │
│ Model: gpt-5 with reasoning                                              │
│ Purpose: Discover relevant attributes + their strategies                 │
│ Output: list[DiscoveredAttribute] with:                                  │
│   - name, type, category, description                                    │
│   - strategy: independent | derived | conditional                        │
│   - depends_on: list of dependency names                                 │
│ Validation: JSON schema enforcement only                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│ INDEPENDENT attrs │  │   DERIVED attrs   │  │ CONDITIONAL attrs │
└───────────────────┘  └───────────────────┘  └───────────────────┘
            │                       │                       │
            ▼                       │                       │
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2a: hydrate_independent()                                           │
│ ─────────────────────────────────────────────────────────────────────── │
│ Module: population/architect/hydrator.py                                 │
│ Model: gpt-5 with WEB SEARCH                                             │
│ Purpose: Research real-world distributions                               │
│ Output: HydratedAttribute with distribution, grounding, constraints      │
│                                                                          │
│ VALIDATION (LLM Response - Fail-Fast):                                   │
│   └─ validate_independent_response() from validator/llm_response.py     │
│      ├─ Check JSON structure matches schema                              │
│      ├─ validate_distribution_data() - min<max, std>0, weights sum to 1 │
│      ├─ validate_formula_syntax() - invalid formula syntax               │
│      └─ If invalid → RETRY with error feedback (up to 3 times)          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2b: hydrate_derived()                                               │
│ ─────────────────────────────────────────────────────────────────────── │
│ Model: gpt-5 NO web search (formulas are deterministic)                  │
│ Purpose: Specify derivation formulas (e.g., years_experience = age - 27) │
│ Output: HydratedAttribute with formula, grounding                        │
│                                                                          │
│ VALIDATION (LLM Response - Fail-Fast):                                   │
│   └─ validate_derived_response()                                         │
│      ├─ Check formula is valid Python syntax (AST parse)                 │
│      ├─ Check formula references only declared depends_on                │
│      └─ If invalid → RETRY with error feedback                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2c: hydrate_conditional_base()                                      │
│ ─────────────────────────────────────────────────────────────────────── │
│ Model: gpt-5 with WEB SEARCH                                             │
│ Purpose: Research BASE distributions (before modifiers)                  │
│ Output: HydratedAttribute with base distribution, grounding              │
│                                                                          │
│ VALIDATION (LLM Response - Fail-Fast):                                   │
│   └─ validate_conditional_base_response()                                │
│      ├─ validate_distribution_data()                                     │
│      └─ If invalid → RETRY with error feedback                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2d: hydrate_conditional_modifiers()                                 │
│ ─────────────────────────────────────────────────────────────────────── │
│ Model: gpt-5 with WEB SEARCH                                             │
│ Purpose: Specify distribution modifiers (when: condition → adjustments)  │
│ Output: Updates HydratedAttribute.modifiers                              │
│                                                                          │
│ VALIDATION (LLM Response - Fail-Fast):                                   │
│   └─ validate_modifiers_response()                                       │
│      ├─ validate_condition_syntax() - each modifier.when                 │
│      ├─ validate_modifier_data() - type compatibility                    │
│      │   ├─ Numeric types can't have weight_overrides                    │
│      │   ├─ Categorical can't have multiply/add                          │
│      │   └─ multiply ∈ [0.01, 100], probability ∈ [0, 1]                 │
│      └─ If invalid → RETRY with error feedback                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: bind_constraints()                                               │
│ ─────────────────────────────────────────────────────────────────────── │
│ Module: population/architect/binder.py                                   │
│ Purpose: Build dependency graph + determine sampling order               │
│ Algorithm: Kahn's topological sort                                       │
│ Output: (list[AttributeSpec], sampling_order, warnings)                  │
│                                                                          │
│ VALIDATION (Structural):                                                 │
│   ├─ Remove unknown dependencies (logged as warnings)                    │
│   └─ Detect circular dependencies → CircularDependencyError             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: build_spec()                                                     │
│ ─────────────────────────────────────────────────────────────────────── │
│ Module: population/architect/binder.py                                   │
│ Purpose: Assemble final PopulationSpec                                   │
│ Output: PopulationSpec(meta, grounding, attributes, sampling_order)      │
│                                                                          │
│ NO VALIDATION HERE - just assembly                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ FINAL: validate_spec()                                                   │
│ ─────────────────────────────────────────────────────────────────────── │
│ Module: population/validator/                                            │
│ Purpose: Full spec validation before saving                              │
│                                                                          │
│ STRUCTURAL CHECKS (ERROR) - validator/structural.py:                     │
│   ├─ No duplicate attribute names                                        │
│   ├─ Sampling order includes all attrs                                   │
│   ├─ Sampling order respects dependencies                                │
│   ├─ Strategy matches fields (derived needs formula, etc.)               │
│   ├─ Type/modifier compatibility                                         │
│   ├─ Formula/condition syntax valid                                      │
│   ├─ Formula/condition references in depends_on                          │
│   └─ Distribution params valid (min<max, std>0, weights sum)            │
│                                                                          │
│ SEMANTIC CHECKS (WARNING) - validator/semantic.py:                       │
│   ├─ No-op modifier detection (multiply=1, add=0)                        │
│   ├─ Modifier stacking analysis (values pushed out of bounds)            │
│   └─ Condition value validity (refs valid categorical options)          │
│                                                                          │
│ Output: ValidationResult(issues=[...], valid=True/False)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           Save to YAML file
```

---

## Step 0: Sufficiency Check

| Property   | Value                                                            |
| ---------- | ---------------------------------------------------------------- |
| Module     | `population/architect/sufficiency.py`                            |
| Function   | `check_sufficiency()`                                            |
| Model      | `gpt-5-mini` (fast)                                              |
| Purpose    | Verify description has enough info                               |
| Output     | `SufficiencyResult(sufficient, size, geography, clarifications)` |
| Validation | JSON schema enforcement only                                     |

---

## Step 1: Attribute Selection

| Property | Value                                   |
| -------- | --------------------------------------- |
| Module   | `population/architect/selector.py`      |
| Function | `select_attributes()`                   |
| Model    | `gpt-5` with reasoning                  |
| Purpose  | Discover attributes + assign strategies |
| Output   | `list[DiscoveredAttribute]`             |

Each `DiscoveredAttribute` includes:

- `name`, `type`, `category`, `description`
- `strategy`: `independent` | `derived` | `conditional`
- `depends_on`: list of dependency names

---

## Step 2: Hydration (Split into 4 Sub-Steps)

Attributes are grouped by strategy and processed separately:

### Step 2a: `hydrate_independent()`

| Property | Value                                            |
| -------- | ------------------------------------------------ |
| Model    | `gpt-5` **with web search**                      |
| Purpose  | Research real-world distributions                |
| Output   | `HydratedAttribute` with distribution, grounding |

**LLM Response Validation** (`validate_independent_response()`):

- Check JSON structure matches schema
- `validate_distribution_data()` — min<max, std>0, weights sum to 1
- `validate_formula_syntax()` — formula syntax valid
- **On failure → RETRY with error feedback** (up to 3 times)

---

### Step 2b: `hydrate_derived()`

| Property | Value                         |
| -------- | ----------------------------- |
| Model    | `gpt-5` **no web search**     |
| Purpose  | Specify derivation formulas   |
| Example  | `years_experience = age - 27` |

**LLM Response Validation** (`validate_derived_response()`):

- Formula is valid Python syntax (AST parse)
- Formula references only declared `depends_on`
- **On failure → RETRY**

---

### Step 2c: `hydrate_conditional_base()`

| Property | Value                                          |
| -------- | ---------------------------------------------- |
| Model    | `gpt-5` **with web search**                    |
| Purpose  | Research BASE distributions (before modifiers) |

**LLM Response Validation** (`validate_conditional_base_response()`):

- `validate_distribution_data()`
- **On failure → RETRY**

---

### Step 2d: `hydrate_conditional_modifiers()`

| Property | Value                                             |
| -------- | ------------------------------------------------- |
| Model    | `gpt-5` **with web search**                       |
| Purpose  | Specify modifiers (when: condition → adjustments) |

**LLM Response Validation** (`validate_modifiers_response()`):

- `validate_condition_syntax()` — each `modifier.when`
- `validate_modifier_data()` — type compatibility:
  - Numeric types can't have `weight_overrides`
  - Categorical can't have `multiply`/`add`
  - `multiply ∈ [0.01, 100]`, `probability ∈ [0, 1]`
- **On failure → RETRY**

---

## Step 3: Constraint Binding

| Property  | Value                                             |
| --------- | ------------------------------------------------- |
| Module    | `population/architect/binder.py`                  |
| Function  | `bind_constraints()`                              |
| Algorithm | Kahn's topological sort                           |
| Output    | `(list[AttributeSpec], sampling_order, warnings)` |

**Validation**:

- Remove unknown dependencies (logged as warnings)
- Detect circular dependencies → `CircularDependencyError`

---

## Step 4: Build Spec

| Property | Value                                                         |
| -------- | ------------------------------------------------------------- |
| Module   | `population/architect/binder.py`                              |
| Function | `build_spec()`                                                |
| Output   | `PopulationSpec(meta, grounding, attributes, sampling_order)` |

No validation — just assembly.

---

## Final: Full Spec Validation

| Property | Value                   |
| -------- | ----------------------- |
| Module   | `population/validator/` |
| Function | `validate_spec()`       |

### Structural Checks (ERROR) — `structural.py`

| Check                       | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| Duplicate names             | No duplicate attribute names                          |
| Sampling order complete     | Includes all attrs                                    |
| Sampling order valid        | Respects dependencies                                 |
| Strategy consistency        | Derived needs formula, conditional needs distribution |
| Type/modifier compatibility | Numeric can't have weight_overrides, etc.             |
| Formula/condition syntax    | Valid Python (AST parse)                              |
| Formula/condition refs      | All refs in `depends_on`                              |
| Distribution params         | min<max, std>0, weights sum to 1                      |

### Semantic Checks (WARNING) — `semantic.py`

| Check             | Description                                |
| ----------------- | ------------------------------------------ |
| No-op modifiers   | multiply=1, add=0 has no effect            |
| Modifier stacking | Values pushed far out of bounds            |
| Condition values  | Refs valid categorical options (AST-based) |

---

## Validation Summary by Layer

| Layer            | When               | What                        | Severity      | On Failure          |
| ---------------- | ------------------ | --------------------------- | ------------- | ------------------- |
| **LLM Response** | During 2a/2b/2c/2d | JSON + syntax               | ERROR         | Retry with feedback |
| **Binding**      | Step 3             | Circular deps, unknown refs | ERROR/WARNING | Exception or logged |
| **Structural**   | `validate_spec()`  | Spec consistency            | ERROR         | Block save          |
| **Semantic**     | `validate_spec()`  | Potential issues            | WARNING       | Logged, continues   |
