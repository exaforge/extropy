# Autonomous Workflows

## Workflow 1: New Study End-to-End

1. Create `runs/<study>/`
2. Initialize versioned structure and manifest (`RUN_VERSIONING.md`)
2. Execute full pipeline with seed 42
3. Run estimate before simulate
4. Generate baseline summary + 2 key segment cuts

## Workflow 2: Triage a Bad Run

1. Confirm run artifacts exist
2. Inspect `meta.json`, `by_timestep.json`, `timeline.jsonl`
3. Identify failure class (exposure, reasoning, outcomes, cost)
4. Apply smallest viable fix
5. Re-run and compare deltas vs prior run

## Workflow 3: Sensitivity Sweep

1. Pick one axis (e.g., threshold)
2. Create 3 variants around baseline
3. Run all variants with same seed set
4. Compare final outcomes + cost + stop behavior
5. Recommend next experiments

## Workflow 4: Confidence Sweep (Seed Stability)

1. Keep scenario + config fixed
2. Run 5-10 seeds
3. Aggregate key outcomes across seeds
4. Report mean + spread and identify unstable segments
5. Recommend whether more runs or model/config changes are needed

## Workflow 5: Batch Report and Registry Update

1. Fill report from `EXPERIMENT_REPORT_TEMPLATE.md`
2. Append all variant rows to `registry/runs.csv`
3. Mark canonical successful batch in `registry/latest.txt`
4. Record any config/scenario revision bumps
