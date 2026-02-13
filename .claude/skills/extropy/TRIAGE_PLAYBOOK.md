# Triage Playbook

Use this for deep debugging and root-cause analysis.

## A. Command Fails Immediately

1. Check file paths exist.
2. Check config/provider mismatch (`extropy config show`).
3. Check API key env vars for active provider.
4. Re-run with minimal command and smaller scope.

## B. Validation/Spec Issues

Use:
```bash
extropy validate <spec_or_scenario>
```

Typical causes:
- bad formula/condition references
- invalid distribution params
- dependency cycles
- scenario refs to unknown attributes/edge types

## C. Exposure Not Spreading

Inspect:
- scenario `seed_exposure.rules` probabilities and `when`
- spread `share_probability`, `share_modifiers`, `max_hops`, `decay_per_hop`
- network connectivity and edge types

Artifacts to inspect:
- `by_timestep.json` (exposure trajectory)
- `timeline.jsonl` (seed vs network exposure events)

## D. Weird Outcome Dynamics

Check:
- outcome definitions in scenario
- pass-2 classification viability (outcome schema too ambiguous)
- threshold too high or too low
- conviction/flip resistance effects

Inspect:
- `meta.json`
- `agent_states.json`
- `timeline.jsonl`

## E. Cost/Latency Blowups

1. Run `extropy estimate` and compare to actual in `meta.json` cost stats.
2. Reduce population size, max timesteps, or reasoning frequency.
3. Move pass-2 to cheaper routine model.
4. Tune rate/chunk settings.

## F. Triage Output Format

Always produce:
1. Symptom
2. Most likely root cause
3. Evidence file(s)
4. Minimal fix
5. Re-run command
