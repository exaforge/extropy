# Standardize CLI Validation & Command Structure

**Priority:** Medium
**Component:** `entropy/cli.py`
**Goal:** Eliminate redundant `validate-*` commands by implementing a type-aware validation system.

---

## The Problem

Currently, the CLI has fragmented validation commands:

- `entropy validate <file>` (for population specs)
- `entropy validate-scenario <file>` (for scenario specs)

This is inelegant. A user should be able to run `entropy validate <file>` and have the system **automatically detect** the file type and apply the correct validator.

## Proposed Solution: Type-Aware "Smart Validate"

### 1. Unified `validate` Command

Deprecate `validate-scenario`. The single `entropy validate` command will:

1.  Load the YAML file.
2.  Inspect the top-level keys or a specific `kind/type` field (if we add one) to determine the spec type.
3.  Dispatch to the appropriate validation logic.

### 2. Schema Identification Strategy

We can identify spec types by their unique top-level keys:

- **Population Spec:** Contains `attributes` and `meta`.
- **Scenario Spec:** Contains `event`, `seed_exposure`, and `meta`.
- **Network:** Contains `nodes` and `edges` (JSON).
- **Agents:** Contains `agents` list (JSON).

### 3. Implementation Plan

#### A. Refactor `cli.py`

```python
@app.command("validate")
def validate_command(file: Path):
    """Smart validation for any Entropy file type."""
    file_type = detect_file_type(file)

    if file_type == "population":
        validate_population_spec(file)
    elif file_type == "scenario":
        validate_scenario_spec(file)
    elif file_type == "network":
        validate_network_file(file)
    else:
        print("Unknown file type")
```

#### B. Future-Proofing: Explicit `kind` Field

Consider adding a `kind` field to all YAML/JSON artifacts for explicit typing, similar to Kubernetes resources:

```yaml
# surgeons.yaml
kind: PopulationSpec
meta: ...
```

```yaml
# scenario.yaml
kind: ScenarioSpec
meta: ...
```

## Tasks

1.  [ ] **Create `detect_file_type(path)` helper:** Logic to sniff YAML/JSON structure.
2.  [ ] **Refactor `entropy validate`:** Switch to dispatch logic.
3.  [ ] **Deprecate `entropy validate-scenario`:** Remove or alias it to the new command.
4.  [ ] **Add Network Validation:** Integrate the existing network validation logic into this unified command.

---

## Benefits

- **User Experience:** One command to remember (`entropy validate`).
- **Consistency:** Standardized output format for all file types.
- **Extensibility:** Easy to add new spec types (e.g., `ExperimentSpec`) later.
