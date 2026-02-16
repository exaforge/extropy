# Experiment Report Template

Use this template for every completed experiment.

---

## 1. Decision Context

- **Study**: [study name]
- **Decision to support**: [What decision will this simulation inform?]
- **Primary outcome**: [The key metric]
- **Constraints**: [Budget, timeline, policy limits]

## 2. Setup

- **Population**: [Description and size]
- **Scenario**: [Event/change being simulated]
- **Fidelity**: [low/medium/high]
- **Seed policy**: [Single seed N / Multi-seed N1-N5]
- **Model config**: [Provider, model, routine model]

## 3. Headline Results

### Outcome Distribution

```
[outcome_name]:
  option_1    XX%
  option_2    XX%
  option_3    XX%
```

### Key Segment Deltas

| Segment | Outcome | vs Baseline |
|---------|---------|-------------|
| [segment] | [value] | [delta] |

### Dynamics Summary

- **Exposure rate**: [final %]
- **Stop reason**: [why simulation ended]
- **Timesteps**: [N completed]

## 4. Confidence Assessment

- **Seeds run**: [N]
- **Between-seed variance**: [low/medium/high]
- **Stable findings**: [list]
- **Unstable findings**: [list]
- **Overall confidence**: [high/medium/low]

## 5. Mechanism

### Why This Happened

[Explain the dominant drivers inferred from traces]

### Key Patterns

- [Pattern 1: e.g., "Low-income agents without transit access defaulted to protest"]
- [Pattern 2: e.g., "Partner conversations reinforced existing positions"]
- [Pattern 3: e.g., "Coworker influence dominated in Week 2"]

### Outlier Notes

[Notable outlier trajectories and interpretation]

## 6. Operations

- **Total tokens**: [pivotal: X, routine: Y]
- **Estimated cost**: [$X.XX]
- **Actual cost**: [$X.XX]
- **Runtime**: [HH:MM]
- **Issues**: [retries, errors, resume events]

## 7. Recommendations

### Primary Recommendation

[One clear recommendation based on findings]

### Risk Caveats

- [Caveat 1]
- [Caveat 2]

### Next Experiments

1. [Recommended follow-up 1]
2. [Recommended follow-up 2]

### What Would Change This

[Conditions that would invalidate or change the recommendation]

## 8. Evidence

| Artifact | Path |
|----------|------|
| Study DB | `[path]/study.db` |
| Results | `[path]/results/` |
| Meta | `[path]/results/meta.json` |
| Timesteps | `[path]/results/by_timestep.json` |

---

*Generated from [N] simulation runs on [date]*
