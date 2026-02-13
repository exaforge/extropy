# Simulation Operations

Use this when running many experiments or managing simulation operations.

## Parameter Sweep Pattern

Vary one dimension at a time unless user requests full factorial.

Common sweep axes:
- `--seed`
- `--threshold` (multi-touch)
- `--chunk-size`
- `--model`, `--routine-model`
- provider/rate settings

## Batch Naming Convention

Use stable directory names:
- `runs/<study>/baseline/`
- `runs/<study>/thresh-2/`
- `runs/<study>/thresh-4/`
- `runs/<study>/seed-42/`

## Execution Discipline

1. Run `extropy estimate` before expensive variants.
2. Keep one canonical baseline run.
3. Record every variant’s command line in a short run log.
4. Summarize each run via `extropy results` and output JSON inspection.

## Resume/Recovery

Simulation supports checkpoint resume via `simulation.db` metadata.
If interrupted, re-run the same `extropy simulate ... -o <same results dir>` command.

## Resource Controls

Use when needed:
- `--rate-tier`
- `--rpm-override`
- `--tpm-override`
- `--chunk-size`

Prefer smaller chunk sizes if crash risk is high.
