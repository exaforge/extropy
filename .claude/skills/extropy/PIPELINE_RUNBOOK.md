# Pipeline Runbook

Use this when autonomously building a new study from scratch.

## 0. Preconditions

1. Ensure output root exists (`runs/<study>`).
2. Ensure provider config is set (`extropy config show`).
3. Ensure required API key env vars exist for selected provider(s).

## 1. Build Artifacts in Order

1. `spec` -> `base.yaml`
2. `extend` -> `population.yaml`
3. `sample` -> `agents.json`
4. `network` -> `network.json`
5. `persona` -> `population.persona.yaml`
6. `scenario` -> `scenario.yaml`
7. `estimate` -> cost/volume preview
8. `simulate` -> `results/`
9. `results` -> summary and segments

## 2. Standard Paths

- `runs/<study>/base.yaml`
- `runs/<study>/population.yaml`
- `runs/<study>/agents.json`
- `runs/<study>/network.json`
- `runs/<study>/population.persona.yaml`
- `runs/<study>/scenario.yaml`
- `runs/<study>/results/`

## 3. Reproducibility

- Always set `--seed` on `sample`, `network`, and `simulate`.
- Pin key overrides in command logs: models, thresholds, rate settings, chunk size.
- For unattended execution, use non-interactive flags where available (`--yes`).

## 4. Fast Sanity Checks

After each stage, verify expected file exists and is parseable:

```bash
test -f <file>
```

For simulation output, verify:
- `meta.json`
- `by_timestep.json`
- `outcome_distributions.json`
- `agent_states.json`

## 5. If `scenario` has no description

Use `extropy extend` first with `-s`, or provide `extropy scenario --description`.
