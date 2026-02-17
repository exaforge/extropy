# Troubleshooting

Diagnosis, triage, quality gates, and escalation.

## Quick Diagnosis

### Command Fails Immediately

1. **Check file paths exist**
   ```bash
   test -f <file> && echo "exists" || echo "missing"
   ```

2. **Check provider config**
   ```bash
   extropy config show
   ```

3. **Check API keys**
   ```bash
   echo $OPENAI_API_KEY | head -c 10      # Should show sk-...
   echo $ANTHROPIC_API_KEY | head -c 10   # Should show sk-ant-...
   ```

4. **Re-run with minimal scope**
   - Reduce population size
   - Remove optional flags
   - Test with simpler description

### Validation Errors

```bash
extropy validate <spec.yaml>
```

Common causes:
| Error | Cause | Fix |
|-------|-------|-----|
| Formula reference error | Attribute name typo in formula | Check attribute names in `sampling_order` |
| Circular dependency | A depends on B depends on A | Refactor one to be independent |
| Invalid distribution params | Negative std, bad range | Check distribution parameters |
| Unknown edge type | Scenario refs edge type not in network | Add edge type to network config or remove from scenario |
| Condition syntax error | Bad expression in exposure rule | Fix Python-style condition syntax |

### Spec/Extend Hangs

These commands make LLM calls with web search. Expected time: 2-10 minutes.

If truly stuck:
- Check network connectivity
- Check API key validity
- Reduce attribute count with simpler description

### Sample Produces Wrong Count

Check:
- `--count` / `-n` flag value (this is the only way to set agent count)
- Household mode: count is households, not individuals

### Network Has Isolated Agents

Check:
- Population has enough attribute variance for similarity
- `--avg-degree` setting (default: 20)
- Structural edges require matching attributes (same occupation, same region)

### Exposure Not Spreading

Inspect scenario exposure rules:
```yaml
seed_exposure:
  channels:
    - name: mainstream_news
      reach: broadcast
      rules:
        - when: "timestep == 0"
          probability: 0.6
```

Check:
- `probability` values (0.0-1.0)
- `when` conditions evaluate correctly
- `reach` type matches intent

Inspect spread config:
```yaml
spread:
  share_probability: 0.3
  max_hops: 4
  decay_per_hop: 0.2
```

Evidence files:
- `by_timestep.json`: exposure trajectory
- Use `extropy results --timeline`

### Outcomes Look Wrong

**Central tendency** (everything clusters to middle):
- Check if using merged pass: try `--no-merged-pass` for 2-pass
- Outcome options may be too similar
- Schema may be anchoring the model

**Everyone picks same option**:
- Event may be unambiguous for this population
- Check conviction levels in results
- Review agent reasoning traces

**Classification fails**:
- Pass 2 model may be too weak
- Outcome descriptions may be ambiguous
- Try more distinct option names

### Cost/Latency Blowup

1. Compare estimate vs actual:
   ```bash
   extropy estimate scenario.yaml --study-db study.db --verbose
   # then check meta.json after run
   ```

2. Reduce scope:
   - Smaller population
   - Fewer timesteps (`simulation.max_timesteps` in scenario)
   - Lower fidelity (`--fidelity low`)
   - Cheaper Pass 2 model (`--routine-model gpt-5-mini`)

3. Tune rate settings:
   ```bash
   extropy config set simulation.rate_tier 2
   # or
   extropy simulate ... --rpm-override 500
   ```

## Stage Quality Gates

Run these checks after each pipeline stage.

### Gate 1: `spec` (base.yaml)

**FAIL if:**
- File missing/empty after command
- `extropy validate base.yaml` fails
- Attributes leak outcome information (e.g., `will_buy` as pre-event attribute)

**WARN if:**
- Distributions rely on weak sources (no citations)

### Gate 2: `extend` (population.yaml)

**FAIL if:**
- File missing/empty
- `extropy validate population.yaml` fails
- New attributes violate pre-event intent

**WARN if:**
- Modifier stacking creates boundary instability

### Gate 3: `sample` (study.db agents)

**FAIL if:**
- Requested count != actual count
- Null values in required fields
- Out-of-range values in bounded fields
- Impossible combinations (adult with `employment: child`, minor who is married)

**WARN if:**
- Marginal distributions drift >5% from expected
- Unexpected category collapse

Verify:
```bash
extropy sample ... --report
```

### Gate 4: `network` (study.db edges)

**FAIL if:**
- Orphan edge endpoints
- Zero edges generated

**WARN if:**
- Average degree far from target
- Large disconnected components
- Missing expected edge types

Verify:
```bash
extropy network ... --validate
```

### Gate 5: `scenario` (scenario.yaml)

**FAIL if:**
- File missing/empty
- Exposure logic produces no exposures
- Outcomes not measurable

**WARN if:**
- Contradictory stopping conditions
- Ambiguous outcome definitions

### Gate 6: `simulate` (results/)

**FAIL if:**
- Missing artifacts (`meta.json`, `by_timestep.json`)
- No valid stop reason and no max-timestep completion

**WARN if:**
- Flat dynamics (nothing happens)
- Cost significantly exceeds estimate
- High error rate in LLM calls

## Auto-Fix Loop

When a gate fails:

1. Identify the smallest upstream fix
2. Rerun only affected downstream stages
3. Re-check the same gate

If same gate fails twice: **stop and escalate**.

## Escalation Policy

Escalate to user before further autonomous action when:

1. **Same gate fails twice** after attempted fixes
2. **Fix changes core study assumptions** (population definition, outcome structure)
3. **Accuracy vs cost conflict** needs user decision
4. **Sensitive content** requires human judgment
5. **Missing required inputs** that user must provide

### Escalation Payload

Always provide:
1. Current stage
2. Exact blocker
3. Evidence (file paths, error messages)
4. Options (A/B/C) with tradeoffs
5. Recommended option

Example:
```
BLOCKED at Gate 3 (sample)

Problem: 12% of agents have invalid age/employment combinations
Evidence: agents with age < 18 and employment_status != "student" or "not_applicable"

Options:
A) Add hard constraint in population.yaml: employment_status derived from age
B) Accept and document as known limitation (warn in report)
C) Reduce age range to adults only

Recommendation: Option A - fixes root cause, 5 min to implement
```

## Common Failure Patterns

### Pattern: "Flat Exposure"

**Symptom**: Exposure rate stays at initial seed, doesn't spread.

**Check**:
- `share_probability` too low?
- `max_hops` too small?
- Network disconnected?
- Conviction too low to trigger sharing?

**Fix**: Increase share probability, check network connectivity.

### Pattern: "Instant Convergence"

**Symptom**: Everyone decides by timestep 1, no dynamics.

**Check**:
- Event unambiguous for population?
- Seed exposure probability too high?
- Population too homogeneous?

**Fix**: Lower initial exposure, add population variance.

### Pattern: "Stale Repetition"

**Symptom**: Agent reasoning identical across timesteps.

**Check**:
- High fidelity with repetition detection?
- New information arriving via timeline?
- Memory traces showing evolution?

**Fix**: Use `--fidelity high`, add timeline events, check memory config.

### Pattern: "Memory Blowup"

**Symptom**: Prompts getting too long, errors or truncation.

**Check**:
- Memory trace count
- Peer opinion count
- Persona length

**Fix**: Lower fidelity, reduce peer limit, simplify persona.

### Pattern: "Classification Mismatch"

**Symptom**: Pass 2 classifies incorrectly despite good Pass 1 reasoning.

**Check**:
- Outcome option names clear?
- Descriptions unambiguous?
- Pass 2 model capable enough?

**Fix**: Clarify outcome definitions, use stronger routine model.

## Triage Output Format

When diagnosing issues, always produce:

1. **Symptom**: What's wrong
2. **Evidence**: File paths, metrics, error messages
3. **Root cause**: Most likely explanation
4. **Fix**: Smallest viable change
5. **Rerun command**: Exact command to retry

Example:
```
Symptom: Network generation fails with "no edges created"
Evidence: network command output, population.yaml has only 10 agents
Root cause: Population too small for target avg_degree=20
Fix: Reduce --avg-degree to 5 or increase population size
Rerun: extropy network --study-db study.db -p pop.yaml --avg-degree 5 --seed 42
```

## Long-Running Commands

`spec`, `extend`, `scenario`, and `simulate` can take minutes to hours.

Do NOT assume failure because:
- File size unchanged mid-run
- stdout is quiet
- Process still active

Failure requires:
- Process exits with error
- User cancels
- Explicit timeout exceeded

Monitor non-destructively:
```bash
# Check if process running
ps aux | grep extropy

# Check recent output
tail -f results/meta.json  # Updates during simulate
```
