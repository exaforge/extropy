# Operations

Use this file for end-to-end execution, run management, and repeatable experiment operations.

## 1) Preconditions

1. Ensure output root exists (`runs/<study>`).
2. Verify provider/runtime config (`extropy config show`).
3. Verify required API keys for active provider(s).
4. Confirm objective, success metric, and scope; if missing, run clarification from `QUALITY_TRIAGE_ESCALATION.md`.
5. Select a realism gate profile and schema map before strict realism checks.

## 2) End-to-End Build Order

1. `spec` -> `base.yaml`
2. `extend` -> `population.yaml`
3. `sample` -> `agents.json`
4. `network` -> `network.json`
5. `persona` -> `population.persona.yaml`
6. `scenario` -> `scenario.yaml`
7. `estimate` -> cost/volume preview
8. `simulate` -> `results/`
9. `results` -> aggregate + segment views

Minimal skeleton:

```bash
# population build
extropy spec "<population>" -o runs/<study>/base.yaml
extropy extend runs/<study>/base.yaml -s "<scenario>" -o runs/<study>/population.yaml
extropy sample runs/<study>/population.yaml -o runs/<study>/agents.json --seed 42
extropy network runs/<study>/agents.json -p runs/<study>/population.yaml -o runs/<study>/network.json --seed 42
extropy persona runs/<study>/population.yaml --agents runs/<study>/agents.json -o runs/<study>/population.persona.yaml

# scenario + sim
extropy scenario -p runs/<study>/population.yaml -a runs/<study>/agents.json -n runs/<study>/network.json -o runs/<study>/scenario.yaml
extropy estimate runs/<study>/scenario.yaml
extropy simulate runs/<study>/scenario.yaml -o runs/<study>/results --seed 42
```

## 3) Standard Output Paths

- `runs/<study>/base.yaml`
- `runs/<study>/population.yaml`
- `runs/<study>/agents.json`
- `runs/<study>/network.json`
- `runs/<study>/population.persona.yaml`
- `runs/<study>/scenario.yaml`
- `runs/<study>/results/`

## 4) Reproducibility Rules

- Always set `--seed` on `sample`, `network`, `simulate`.
- Log provider/model/threshold/chunk/rate overrides for every run.
- Prefer non-interactive flags when available (`--yes`).
- Never compare variants directly unless scenario/config/seed policy are comparable.

## 5) Autopilot Loop (After Every Stage)

1. Run command.
2. Wait for full exit.
3. Run stage quality checks from `QUALITY_TRIAGE_ESCALATION.md`.
4. If FAIL: apply smallest upstream fix; rerun only dependent downstream stages.
5. If same gate fails twice: escalate per policy.

## 6) Long-Run Conduct

For long-running `spec`, `extend`, `simulate`:
- Do not infer failure before process exit.
- Monitor health non-destructively.
- Judge outputs only after exit unless canceled/timeboxed by policy.

## 7) Batch + Variant Management

Use this structure:

```text
runs/
  <study_slug>/
    registry/
      runs.csv
      latest.txt
    specs/
      pop/
      persona/
      network-config/
      scenario/
    batches/
      <batch_id>/
        manifest.yaml
        variants/
          <variant_id>/
            inputs/
            results/
```

Canonical IDs:
- `study_slug`: lowercase kebab-case
- `batch_id`: `bYYYYMMDD-HHMM-<intent>-vNN`
- `variant_id`: `vr-<axis>-<value>-s<seed>`
- `scenario_rev`: `scn-vNN`
- `config_rev`: `cfg-vNN`

Revision rules:
- bump `scn-vNN` when event/exposure/outcome logic changes
- bump `cfg-vNN` when provider/model/rate/logic defaults change

## 8) Manifest Contract

Each batch should declare:
- study + batch_id
- scenario/config revisions
- objective
- base spec paths
- variant IDs, seeds, and explicit overrides

## 9) Registry Contract

Append one row per variant run in `registry/runs.csv` with:
- timestamp, study, batch_id, variant_id
- scenario_rev, config_rev, seed
- status, results_dir, notes

Update `registry/latest.txt` to the latest successful promoted baseline batch.

## 10) Sweep Patterns

Use one-axis sweeps unless user explicitly requests full factorial.

Common axes:
- seed
- threshold
- chunk size
- model/routine model
- provider and rate settings

Recommended sets:
1. Baseline + 2 to 3 sensitivity variants.
2. Confidence sweep with 5 to 10 seeds.

## 11) Resume + Recovery

If `simulation.db` exists and run interrupted:
- rerun same `extropy simulate ... -o <same results dir>` command
- do not change seed/config mid-resume

## 12) Minimum Batch Deliverables

1. Updated run registry rows
2. One experiment report (`EXPERIMENT_REPORT_TEMPLATE.md`)
3. At least two segment analyses
4. Stability/confidence section (or explicit single-seed caveat)
5. Gate status summary (PASS/WARN/FAIL) for each variant
