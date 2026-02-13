# Run Management Playbook

Use this for day-to-day operations around runs/configs.

## 1. Create New Batch

1. Pick `batch_id` using `RUN_VERSIONING.md`.
2. Write `manifest.yaml` with scenario/config revisions and variants.
3. Create directories for each variant.

## 2. Execute Variant

For each variant:
1. Resolve overrides from manifest.
2. Run `extropy estimate` first.
3. Run `extropy simulate ... -o <variant results dir>`.
4. Mark status in `runs.csv`.

## 3. Resume Failed Variant

If `simulation.db` exists, rerun the exact same simulate command with same output dir.

## 4. Promote Canonical Baseline

A baseline can be promoted when:
- run status is success
- no critical triage issues
- report generated

Record promoted baseline in `registry/latest.txt`.

## 5. Config Drift Control

Before launching a new batch:
1. Verify `extropy config show`.
2. Capture provider/model/rate settings into manifest.
3. If changed, bump `cfg-vNN`.

## 6. Scenario Drift Control

Before launching variants:
1. Validate scenario path references point to intended revision.
2. If scenario content changed, bump `scn-vNN`.
3. Never overwrite old scenario revision files.

## 7. Minimum Deliverables per Batch

1. Completed run registry entries
2. One report using `EXPERIMENT_REPORT_TEMPLATE.md`
3. Segment analysis for at least two attributes
4. Stability section (single-seed marked explicitly if no sweep)
