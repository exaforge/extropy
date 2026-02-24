# Operations

Execution guide for current Extropy CLI behavior.

## Pipeline

```bash
extropy spec -> extropy scenario -> extropy persona -> extropy sample -> extropy network -> extropy simulate -> extropy results
```

All stages operate on a study folder containing `study.db`, `population.vN.yaml`, and `scenario/<name>/...`.

## Stage 1: Spec

```bash
extropy spec "5000 US adults" -o runs/us-national-study --use-defaults
cd runs/us-national-study
extropy validate population.v1.yaml
```

Notes:
- `-o` can be study folder, stem path, or explicit YAML file.
- In agent mode, clarifications return exit code `2` unless `--answers` or `--use-defaults` is provided.

## Stage 2: Scenario

```bash
extropy scenario "ASI capability shock over 6 months" -o asi-announcement -y
extropy validate scenario/asi-announcement/scenario.v1.yaml
```

Useful overrides:
- `--timeline auto|static|evolving`
- `--timestep-unit hour|day|week|month|year`
- `--max-timesteps N`

## Stage 3: Persona

```bash
extropy persona -s asi-announcement -y
extropy validate scenario/asi-announcement/persona.v1.yaml
```

## Stage 4: Sample

```bash
extropy sample -s asi-announcement -n 5000 --seed 42 --report --strict-gates
```

Core checks:
- sampled count equals requested `-n`
- strict gate violations are zero
- no critical impossible combinations in post-sample diagnostics

## Stage 5: Network

```bash
extropy network -s asi-announcement --seed 42 --quality-profile strict --validate
```

Common tuning flags:
- `--avg-degree`, `--rewire-prob`
- `--candidate-mode exact|blocked`
- `--candidate-pool-multiplier`
- `--block-attr` (repeatable)
- `--resume` for long calibration runs

## Stage 6: Simulate

```bash
extropy simulate -s asi-announcement --seed 42 --fidelity high --rpm-override 1000
```

Runtime controls:
- model overrides: `--strong`, `--fast`
- re-reasoning: `--threshold`
- performance: `--chunk-size`, `--writer-queue-size`, `--db-write-batch-size`
- checkpointing/resume: `--run-id`, `--resume`
- convergence override: `--early-convergence auto|on|off`

## Stage 7: Results and Queries

```bash
# high-level outputs
extropy results -s asi-announcement summary
extropy results -s asi-announcement timeline
extropy results -s asi-announcement segment income
extropy results -s asi-announcement agent agent_0042

# raw data
extropy query summary -s asi-announcement
extropy query agents -s asi-announcement --to agents.jsonl
extropy query edges -s asi-announcement --to edges.jsonl
extropy query states --to states.jsonl
extropy query sql "SELECT timestep, agents_reasoned FROM timestep_summaries ORDER BY timestep" --format table
```

## Reproducibility Rules

1. Set `--seed` on `sample`, `network`, `simulate`.
2. Keep one baseline run before sweeping knobs.
3. Change one axis per experiment (seed, fidelity, threshold, rate, etc.).
4. Record scenario version and run id for every comparison.

## Recommended Run Sequence for Expensive Studies

1. Gate-only dry pass: `sample` + `network --validate`.
2. One timestep smoke: set low `--max-timesteps` in scenario or stop after first checkpoint.
3. Full run only when gate and smoke checks pass.

## Config Commands

```bash
extropy config show
extropy config set cli.mode agent
extropy config set models.fast anthropic/claude-sonnet-4-6
extropy config set models.strong anthropic/claude-sonnet-4-6
extropy config set simulation.fast azure/gpt-5-mini
extropy config set simulation.strong azure/gpt-5-mini
extropy config set simulation.rate_tier 2
```
