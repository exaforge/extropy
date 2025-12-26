# Issue 1: Validation Consolidation

## Problem

Multiple validation pathways in `population/` each had their own validation patterns:

1. **`validator/__init__.py`**: Used `@dataclass` for `Severity`, `ValidationIssue`, `ValidationResult`
2. **`architect/quick_validate.py`**: Plain Python classes `ValidationError`, `QuickValidationResult`
3. **`core/models/scenario.py`**: Pydantic models `ValidationError`, `ValidationWarning`, `ValidationResult`

This inconsistency led to:
- Code duplication
- Type confusion when passing validation results between modules
- Difficult maintenance

## Solution

Consolidated all validation types into a single unified module: `entropy/core/models/validation.py`

### New Central Module: `core/models/validation.py`

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

    def add_error(...): ...
    def add_warning(...): ...
    def format_for_retry(self) -> str: ...
```

### Files Modified

1. **`core/models/validation.py`** - NEW: Unified Pydantic models
2. **`core/models/__init__.py`** - Added validation imports
3. **`core/models/scenario.py`** - Imports from validation.py (backwards compat)
4. **`population/validator/__init__.py`** - Removed @dataclass, imports from core
5. **`population/validator/syntactic.py`** - Uses core models, `location=` instead of `attribute=`
6. **`population/validator/semantic.py`** - Uses core models, `location=` instead of `attribute=`
7. **`population/architect/quick_validate.py`** - Uses core models with `_make_error()` helper

### Backwards Compatibility

Aliases maintained for existing code:
- `ValidationError = ValidationIssue`
- `ValidationWarning = ValidationIssue`
- `QuickValidationResult = ValidationResult`

### Key Changes

1. **Field rename**: `attribute=` → `location=` (more generic, works for scenario validation too)
2. **Single issues list**: `ValidationResult.issues` replaces separate `errors`, `warnings`, `info` lists
3. **Computed properties**: `valid`, `errors`, `warnings` are now computed from `issues`
4. **LLM retry support**: `format_for_retry()` and `for_llm_retry()` for fail-fast validation
