# Analysis Playbook

Use this after runs complete.

## Primary Artifacts

- `results/meta.json`
- `results/by_timestep.json`
- `results/outcome_distributions.json`
- `results/agent_states.json`
- `results/timeline.jsonl`

## Analysis Sequence

1. Run-level summary
- population size
- timesteps completed
- stop reason
- total reasoning calls
- token/cost summary

2. Dynamics
- exposure curve shape over time
- when state changes plateau
- whether run stopped by condition or quiescence

3. Outcomes
- final distribution by outcome
- concentration/polarization patterns
- compare to baseline variants

4. Segments
Use:
```bash
extropy results <results_dir> --segment <attribute>
```
Evaluate heterogeneous effects by segment.

5. Agent-level deep dive
Use:
```bash
extropy results <results_dir> --agent <agent_id>
```
Look for representative or anomalous trajectories.

6. Convergence + uncertainty across runs
- Run the same scenario across multiple seeds.
- Report central tendency + spread for key outcomes (mean, min/max, std where possible).
- Flag unstable outcomes where between-seed variance is decision-relevant.
- Do not present one run as a definitive forecast.

## Comparative Analysis Template

When comparing runs, report:
1. What changed (single axis)
2. Delta in exposure speed
3. Delta in key outcomes
4. Delta in cost/runtime
5. Confidence assessment (needs more seeds or stable)
