# Validation Consolidation: Complete Refactoring Plan

## Executive Summary

Complete the work started in `issue1_validation_solution.md` by:
1. Creating shared validation primitives (`entropy/validation/`)
2. Removing remaining backwards compatibility aliases
3. Consolidating duplicate code across `population/validator/` and `scenario/validator.py`
4. Establishing consistent return types and naming conventions

---

## Part 1: Current State (The Mess)

### File Inventory

| File | Lines | Purpose | Returns | Problems |
|------|-------|---------|---------|----------|
| `core/models/validation.py` | 224 | Pydantic types | - | ✅ Good |
| `population/validator/common.py` | 86 | Expression extraction | `set[str]` | Has backwards compat aliases |
| `population/validator/syntactic.py` | 759 | ERROR checks | `list[ValidationIssue]` | Imports from `common.py` |
| `population/validator/semantic.py` | 266 | WARNING checks | `list[ValidationIssue]` | ✅ Clean |
| `population/validator/hydration.py` | 395 | HydratedAttribute validation | `list[str]` ❌ | Wrong return type |
| `population/validator/llm_response.py` | 715 | LLM output validation | `ValidationResult` | Duplicates constants |
| `population/validator/fixer.py` | 335 | Auto-fix conditions | `FixResult` | ✅ Clean |
| `population/validator/__init__.py` | 137 | Exports 30+ functions | - | Too bloated |
| `scenario/validator.py` | 572 | Scenario validation | `ValidationResult` | Duplicates expression parsing |

### Duplication Map

```
DUPLICATED CONSTANTS:
├── common.py          → BUILTIN_NAMES
├── llm_response.py    → ALLOWED_BUILTINS (same thing)
└── scenario/validator → inline in function

DUPLICATED FUNCTIONS:
├── common.py              → extract_names_from_expression()
├── scenario/validator.py  → _extract_attribute_references() (same logic)
└── llm_response.py        → inline AST parsing

BACKWARDS COMPAT ALIASES:
├── common.py:83-85        → extract_names_from_formula, extract_names_from_condition
├── llm_response.py:25-27  → ValidationError, QuickValidationResult
└── __init__.py            → exports of above
```

---

## Part 2: The A-F Grouping

### Group A: Expression/Formula Validation
**"Is this Python expression valid and does it reference known things?"**
- Extract variable names from expression
- Validate expression syntax (AST parse)
- Check references exist in `depends_on`

### Group B: Distribution Parameter Validation
**"Are the distribution parameters mathematically valid?"**
- `min < max`
- `std > 0`
- `weights.sum() ≈ 1.0`
- `probability ∈ [0, 1]`

### Group C: Structural Validation (ERROR)
**"Is the spec internally consistent?"**
- No duplicate attribute names
- Sampling order includes all attrs
- Sampling order respects deps
- Strategy matches fields
- Type/modifier compatibility

### Group D: Semantic Validation (WARNING)
**"This is valid but probably wrong"**
- No-op modifier (multiply=1, add=0)
- Modifier stacking exceeds bounds
- Condition refs non-existent option

### Group E: External Reference Validation
**"Do external references exist?"**
- File paths exist
- Attribute names exist in population
- Edge types exist in network

### Group F: LLM Response Validation
**"Did the LLM give us garbage?"**
- Validates raw `dict` from LLM for fail-fast retry

---

## Part 3: New Architecture

### File Tree

```
entropy/
├── core/models/
│   └── validation.py         # ✅ KEEP: Severity, ValidationIssue, ValidationResult
│
├── validation/                # 🆕 NEW: Shared primitives (Groups A & B)
│   ├── __init__.py
│   ├── expressions.py        # Group A
│   └── distributions.py      # Group B
│
├── population/validator/
│   ├── __init__.py           # Clean: 8 exports only
│   ├── structural.py         # 📝 RENAMED from syntactic.py (Group C)
│   ├── semantic.py           # ✅ KEEP (Group D)
│   ├── llm_response.py       # ✅ KEEP (Group F)
│   └── fixer.py              # ✅ KEEP
│   # ❌ DELETE: common.py, hydration.py
│
└── scenario/
    └── validator.py          # ✅ KEEP (Groups D + E)
```

### Group → File Mapping

| Group | Responsibility | File Location |
|-------|---------------|---------------|
| **A** | Expression/Formula | `validation/expressions.py` |
| **B** | Distribution Params | `validation/distributions.py` |
| **C** | Structural (ERROR) | `population/validator/structural.py` |
| **D** | Semantic (WARNING) | `semantic.py` + `scenario/validator.py` |
| **E** | External References | `scenario/validator.py` |
| **F** | LLM Response | `population/validator/llm_response.py` |

---

## Part 4: Style Consistency

### Return Type Rules

| Context | Returns |
|---------|---------|
| Shared primitives (A/B) | `str \| None` (error message or nothing) |
| Spec checks (C/D) | `list[ValidationIssue]` |
| LLM validation (F) | `ValidationResult` |
| Public orchestrator | `ValidationResult` |

### Naming Conventions

```python
# Shared primitives - verb_noun, private constants
_BUILTIN_NAMES = {...}
validate_weight_sum(weights) -> str | None
extract_names_from_expression(expr) -> set[str]

# Internal checks - _check_* prefix
_check_duplicates(attrs) -> list[ValidationIssue]

# Entry points - run_* internal, validate_* public
run_structural_checks(spec) -> list[ValidationIssue]
validate_spec(spec) -> ValidationResult
```

---

## Part 5: Migration Checklist

### Phase 1: Create Shared Primitives
- [ ] Create `entropy/validation/__init__.py`
- [ ] Create `entropy/validation/expressions.py`
- [ ] Create `entropy/validation/distributions.py`
- [ ] Update imports in `syntactic.py`, `llm_response.py`, `scenario/validator.py`
- [ ] Delete `common.py`

### Phase 2: Remove Backwards Compat
- [ ] Delete aliases from `llm_response.py`
- [ ] Clean `__init__.py` exports

### Phase 3: Clean Up Structure
- [ ] Rename `syntactic.py` → `structural.py`
- [ ] Handle `hydration.py` (merge or delete)
- [ ] Update `hydrator.py` imports

### Phase 4: Verification
- [ ] `uv run pytest tests/test_validator.py -v`
- [ ] `uv run pytest tests/test_sampler.py -v`
- [ ] `uv run pytest tests/ -v`

---

## Part 6: Files Changed Summary

| Action | File |
|--------|------|
| CREATE | `entropy/validation/__init__.py` |
| CREATE | `entropy/validation/expressions.py` |
| CREATE | `entropy/validation/distributions.py` |
| RENAME | `syntactic.py` → `structural.py` |
| MODIFY | `population/validator/__init__.py` |
| MODIFY | `population/validator/llm_response.py` |
| MODIFY | `scenario/validator.py` |
| MODIFY | `population/architect/hydrator.py` |
| DELETE | `population/validator/common.py` |
| DELETE | `population/validator/hydration.py` |

---

## Final Public API

```python
# Population spec validation
from entropy.population.validator import validate_spec
result = validate_spec(spec)  # ValidationResult

# Scenario spec validation
from entropy.scenario.validator import validate_scenario
result = validate_scenario(spec, population_spec, network)

# LLM response validation (internal)
from entropy.population.validator import validate_independent_response

# Auto-fix
from entropy.population.validator import fix_modifier_conditions
```
