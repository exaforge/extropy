# Issue 1: Validation Consolidation - COMPLETED

## Problem

Multiple validation pathways in `population/` each had their own validation patterns:

1. **`validator/__init__.py`**: Used `@dataclass` for `Severity`, `ValidationIssue`, `ValidationResult`
2. **`architect/quick_validate.py`**: Plain Python classes `ValidationError`, `QuickValidationResult`
3. **`core/models/scenario.py`**: Pydantic models `ValidationError`, `ValidationWarning`, `ValidationResult`
4. **`architect/hydrator_utils.py`**: Duplicated functions `_is_spec_level_constraint`, `_extract_bound_from_constraint`

This inconsistency led to:
- Code duplication
- Type confusion when passing validation results between modules
- Difficult maintenance

## Solution Implemented

### 1. Created Unified Validation Models: `core/models/validation.py`

```python
class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ValidationIssue(BaseModel):
    severity: Severity = Severity.ERROR
    category: str
    location: str
    message: str
    suggestion: str | None = None
    modifier_index: int | None = None
    value: str | None = None

class ValidationResult(BaseModel):
    issues: list[ValidationIssue] = []

    @property
    def valid(self) -> bool: ...
    @property
    def errors(self) -> list[ValidationIssue]: ...
    @property
    def warnings(self) -> list[ValidationIssue]: ...

    def format_for_retry(self) -> str: ...

# Backwards compatibility aliases
ValidationError = ValidationIssue
ValidationWarning = ValidationIssue
```

### 2. Deleted `quick_validate.py`, Moved Logic to `validator/llm_response.py`

All LLM response validation functions now live in one place:

```
entropy/population/validator/
├── __init__.py          # Exports all validation functions
├── llm_response.py      # LLM response validation (was quick_validate.py)
├── syntactic.py         # Spec validation: ERROR level checks
├── semantic.py          # Spec validation: WARNING level checks
└── fixer.py             # Auto-fix utilities
```

### 3. Removed Duplicated Functions from `hydrator_utils.py`

- `_is_spec_level_constraint` → now imported from `validator.llm_response`
- `_extract_bound_from_constraint` → now imported from `validator.llm_response`

## Files Changed

| File | Action |
|------|--------|
| `core/models/validation.py` | NEW - Unified Pydantic models |
| `core/models/__init__.py` | MODIFIED - Export validation types |
| `core/models/scenario.py` | MODIFIED - Import from validation.py |
| `population/validator/__init__.py` | MODIFIED - Remove @dataclass, export LLM validation |
| `population/validator/llm_response.py` | NEW - Moved from quick_validate.py |
| `population/validator/syntactic.py` | MODIFIED - Use core models |
| `population/validator/semantic.py` | MODIFIED - Use core models |
| `population/architect/quick_validate.py` | DELETED |
| `population/architect/__init__.py` | MODIFIED - Update imports |
| `population/architect/hydrator.py` | MODIFIED - Update imports |
| `population/architect/hydrator_utils.py` | MODIFIED - Remove duplicates, import from validator |
| `tests/test_sampler.py` | MODIFIED - Update imports |

## Key Principles Followed

1. **Models centralized** - Single source of truth in `core/models/validation.py`
2. **Logic in domain modules** - Validation logic stays in `population/validator/`
3. **No duplication** - Shared functions imported, not copied
4. **Backwards compatible** - Aliases maintained for existing code
5. **Field rename** - `attribute=` → `location=` (more generic)

## Commits

1. `7dd52ae` - refactor: consolidate validation types into core/models/validation.py
2. `b0a25c6` - refactor: delete quick_validate.py and merge into validator/
