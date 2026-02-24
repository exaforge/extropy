# Analysis

Post-run analysis workflow using current `results` and `query` commands.

## First-Pass Readout

```bash
extropy results -s <scenario> summary
extropy results -s <scenario> timeline
```

Interpret immediately:
- awareness rate
- position distribution concentration
- reasoners per timestep trend
- spread velocity (new exposures / shares)

## Segment and Agent Deep Dives

```bash
extropy results -s <scenario> segment income
extropy results -s <scenario> segment age
extropy results -s <scenario> agent agent_0042
```

Use these to explain mechanism, not just headline outcomes.

## Raw Data Exports

```bash
extropy query agents -s <scenario> --to agents.jsonl
extropy query edges -s <scenario> --to edges.jsonl
extropy query states --to states.jsonl
```

## SQL Patterns (Read-only)

```bash
# positions
extropy query sql "SELECT COALESCE(private_position, position) AS p, COUNT(*) AS n FROM agent_states WHERE run_id='<run_id>' GROUP BY p ORDER BY n DESC" --format table

# conviction distribution
extropy query sql "SELECT ROUND(conviction,2) AS c, COUNT(*) AS n FROM agent_states WHERE run_id='<run_id>' GROUP BY c ORDER BY c" --format table

# timestep dynamics
extropy query sql "SELECT timestep, new_exposures, agents_reasoned, shares_occurred, exposure_rate FROM timestep_summaries WHERE run_id='<run_id>' ORDER BY timestep" --format table

# conversation volume
extropy query sql "SELECT timestep, COUNT(*) AS n FROM conversations WHERE run_id='<run_id>' GROUP BY timestep ORDER BY timestep" --format table
```

## Confidence Reporting

Single run guidance:
- treat as one draw, not final truth
- explicitly report seed and run id

Multi-seed guidance:
- run a seed sweep
- compare mean/range for key outcomes
- call out unstable segments separately

## Output Standards for Reports

Every report should include:
1. Decision question.
2. Scenario + population version references.
3. Primary distribution outcome.
4. Segment deltas.
5. Mechanism evidence (agent traces + timeline metrics).
6. Confidence statement (single-seed vs multi-seed).
7. Known caveats.

## Red Flags

Escalate when you see:
- flat timeline (`agents_reasoned` near zero after early timesteps)
- single-option collapse with no variance in plausible heterogeneous populations
- segment results contradicting known network/exposure flow
- huge run-to-run variance with minor seed changes
