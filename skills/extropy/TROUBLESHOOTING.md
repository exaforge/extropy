# Troubleshooting

Failure triage and escalation for current Extropy pipeline.

## Quick Triage

1. Confirm study context:
```bash
pwd
ls -la
extropy query summary
```

2. Validate YAML artifacts:
```bash
extropy validate population.v1.yaml
extropy validate scenario/<name>/scenario.v1.yaml
extropy validate scenario/<name>/persona.v1.yaml
```

3. Check config and provider readiness:
```bash
extropy config show
```

## Stage-Specific Failures

### Spec/Scenario/Persona

Symptoms:
- clarification loop in automation
- validation fail on generated files

Actions:
- use `--answers` or `--use-defaults`
- rerun command with `-y`
- inspect newest `.invalid` artifact and fix root constraints before retry

### Sample

Symptoms:
- coherence gate failures
- impossible combinations in sampled agents

Actions:
- rerun with explicit seed and `--strict-gates` for deterministic fail-fast
- run SQL checks via `query sql` for exact violation counts
- fix spec/scenario constraints upstream (not ad-hoc DB edits)

### Network

Symptoms:
- strict profile gate miss (modularity/clustering bands)
- disconnected graph or weak structural edges

Actions:
- run with `--validate` and capture metrics
- adjust `--avg-degree`, `--rewire-prob`, candidate settings
- if near-threshold float noise only, verify tolerance handling in code before changing profile

### Simulate

Symptoms:
- stalls, high error retries, runaway spend, no reasoning activity

Actions:
- confirm model/provider config for `simulation.strong` and `simulation.fast`
- tune `--rpm-override`, `--chunk-size`, writer queue / db batch sizes
- verify checkpointing with `--run-id` + `--resume`
- inspect `timestep_summaries` and `agent_states` with SQL, not only logs

## Useful SQL Checks

```bash
# latest runs
extropy query sql "SELECT run_id, scenario_name, status, started_at, completed_at FROM simulation_runs ORDER BY started_at DESC LIMIT 10" --format table

# timestep health
extropy query sql "SELECT timestep, new_exposures, agents_reasoned, shares_occurred, exposure_rate FROM timestep_summaries WHERE run_id='<run_id>' ORDER BY timestep" --format table

# awareness and positions
extropy query sql "SELECT aware, COUNT(*) FROM agent_states WHERE run_id='<run_id>' GROUP BY aware" --format table
extropy query sql "SELECT COALESCE(private_position, position), COUNT(*) FROM agent_states WHERE run_id='<run_id>' GROUP BY 1 ORDER BY 2 DESC" --format table
```

## Escalation Policy

Escalate immediately when:
1. same gate fails twice after a root-cause fix attempt,
2. fix requires changing scenario semantics or study assumptions,
3. runtime cost/risk exceeds approved budget,
4. required provider/key setup is unavailable.

Escalation payload should include:
- failing stage,
- exact command,
- exact error/metric evidence,
- two or three options with tradeoffs,
- recommended next move.
