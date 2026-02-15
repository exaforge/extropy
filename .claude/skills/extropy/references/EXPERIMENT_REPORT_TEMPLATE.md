# Experiment Report Template

Use this template for every completed experiment batch.

## 1. Decision Context

- Study name:
- Decision to support:
- Primary KPI/outcome:
- Constraints (budget, timeline, policy limits):

## 2. Experiment Setup

- Population description:
- Scenario description:
- Run set ID:
- Variants included:
- Seed policy (single seed or multi-seed):
- Model/provider config:

## 3. Headline Results

- Baseline outcome distribution:
- Most important segment deltas:
- Exposure dynamics summary:
- Stop condition and total timesteps:

## 4. Confidence and Stability

- Number of seeds:
- Between-seed variance for key outcomes:
- Stable findings (low variance):
- Unstable findings (high variance):
- Confidence statement (high/medium/low):

## 5. Why It Happened (Mechanism)

- Dominant drivers inferred from traces:
- Key peer influence patterns:
- Conviction/memory effects observed:
- Outlier trajectories and interpretation:

## 6. Cost and Operations

- Total token usage (pivotal/routine):
- Estimated cost:
- Runtime and bottlenecks:
- Any retries/errors/resume events:

## 7. Recommendations

1. Immediate decision recommendation
2. Risk caveats
3. Next experiment(s) to run
4. What would change your recommendation

## 8. Evidence Files

- `results/meta.json`
- `results/outcome_distributions.json`
- `results/by_timestep.json`
- `results/agent_states.json`
- `results/timeline.jsonl`
