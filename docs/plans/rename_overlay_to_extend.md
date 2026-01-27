# Rename Overlay → Extend + Scenario Auto-Naming

## Goal

1. Rename `overlay` → `extend` (clearer purpose)
2. Scenario auto-creates `.scenario.yaml` from spec
3. Keep current pipeline order

## New Flow
```
entropy spec "German surgeons" -o surgeons.yaml
entropy extend surgeons.yaml "AI adoption" -o surgeons_ai.yaml
entropy sample surgeons_ai.yaml -o surgeons_ai.agents.json
entropy network surgeons_ai.agents.json -o surgeons_ai.network.json
entropy scenario surgeons_ai.yaml --agents surgeons_ai.agents.json --network surgeons_ai.network.json
    → surgeons_ai.scenario.yaml (auto-named)
entropy simulate surgeons_ai.scenario.yaml
```

---

## Changes

### [MODIFY] [overlay.py → extend.py](file:///Users/devparagiri/Projects/entropy/entropy/cli/commands/overlay.py)

1. Rename file to `extend.py`
2. Rename command from `overlay` to `extend`
3. Store scenario description in spec metadata

### [MODIFY] [scenario.py](file:///Users/devparagiri/Projects/entropy/entropy/cli/commands/scenario.py)

1. Read description from spec metadata (no positional arg)
2. Auto-name output as `{input_stem}.scenario.yaml`
3. Update CLI signature

### [MODIFY] [commands/__init__.py](file:///Users/devparagiri/Projects/entropy/entropy/cli/commands/__init__.py)

Update import from `overlay` to `extend`.

### [MODIFY] Tests

Update test references from `overlay` to `extend`.

---

## Verification

```bash
uv run pytest tests/ -x -q
entropy extend --help
entropy scenario --help
```
