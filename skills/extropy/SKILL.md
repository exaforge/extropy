---
name: extropy
description: "Autonomous operator for Extropy: run end-to-end pipelines, execute/manage simulation experiments, triage failures, and perform post-run analysis. Use when the user wants hands-on execution, simulation operations, debugging, or data-science deep dives."
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
argument-hint: "[goal or experiment request]"
---

# Extropy Operator

**Predictive intelligence through population simulation.**

Extropy predicts how populations respond to *any* scenario — real or hypothetical, mundane or extreme. Thousands of synthetic agents, grounded in real-world distributions, connected in social networks, each reasoning individually using an LLM. Opinions form, spread, and evolve through conversations and social influence.

**Simulate anything:** Netflix price hikes. ASI breakout. Alien contact. Policy changes. Product launches. Pandemics. Elections. Workplace disruption. Community disputes. Any situation where humans form opinions, make decisions, and influence each other.

**Caveat:** Populations are sampled from distributions, not org charts. Can't model "1 CEO, 2 VPs, 10 directors" hierarchical structures — use for broad populations, not specific organizational trees.

**Get back:** Distributional predictions by segment. Open-ended responses when you don't know the outcome space — let agents tell you what they'd do, discover categories post-hoc. Reasoning traces explaining *why*. Opinion dynamics over time. Conversation transcripts. Network effects. Uncertainty quantification across seeds.

## Operating Principles

1. **Execute first, explain second** - Run commands, don't just describe them
2. **Validate before expensive calls** - Check inputs, estimate costs
3. **Use evidence** - Cite specific files, metrics, agent traces
4. **Report uncertainty** - Single seed is one sample; say so
5. **Escalate when blocked** - Don't retry the same failure twice

## Before Running a Simulation

Ask these questions if not already answered:

1. **What decision should this inform?** (policy choice, pricing, messaging)
2. **Who is the population?** (geography, demographics, size)
3. **What is the event/change?** (announcement, policy, product)
4. **What outcomes matter?** (categorical choices, sentiment, likelihood)
5. **Any constraints?** (budget, timeline, fidelity requirements)

If user is unsure, use sensible defaults and state assumptions.

## Quick Pipeline

```bash
STUDY=runs/my-study
DB=$STUDY/study.db
mkdir -p $STUDY

# Build population
extropy spec "500 Austin TX commuters" -o $STUDY/base.yaml -y
extropy extend $STUDY/base.yaml -s "Response to $15/day congestion tax" -o $STUDY/population.yaml -y
extropy sample $STUDY/population.yaml --study-db $DB --seed 42

# Build network
extropy network --study-db $DB -p $STUDY/population.yaml --seed 42

# Compile and run
extropy scenario -p $STUDY/population.yaml --study-db $DB -o $STUDY/scenario.yaml -y
extropy estimate $STUDY/scenario.yaml --study-db $DB
extropy simulate $STUDY/scenario.yaml --study-db $DB -o $STUDY/results --seed 42

# View results
extropy results --study-db $DB
extropy results --study-db $DB --segment income
```

## Key Flags

| Flag | Command | Purpose |
|------|---------|---------|
| `--study-db` | most | Canonical database path |
| `--seed` | sample, network, simulate | Reproducibility |
| `--fidelity` | simulate | low/medium/high - controls conversations, memory |
| `--merged-pass` | simulate | Single-pass reasoning (cheaper) |
| `-y` / `--yes` | spec, extend, scenario | Skip confirmations |
| `--segment` | results | Break down by attribute |

## Fidelity Tiers

| Tier | Conversations | Cost/Agent |
|------|---------------|------------|
| `low` | None | ~$0.03 |
| `medium` | Top 1 edge | ~$0.04 |
| `high` | Top 2-3 edges + cognitive features | ~$0.05 |

## What Can Be Simulated

**Works well:**
- Policy responses (congestion pricing, regulations, zoning)
- Pricing changes (subscriptions, fees, tiers)
- Product decisions (features, defaults, migrations)
- Crisis response (breaches, recalls, PR incidents)
- Messaging tests (campaign variants, framing)
- Community planning (development, local initiatives)

**Population modes:**
- Individuals (`household_mode: false`) - workplace, B2B
- Households (`household_mode: true`) - consumer, family dynamics
- Couples, families as reasoning agents (`agent_focus: couples/families`)

**Timeline modes:**
- Static: single event, opinions evolve
- Evolving: multiple events over timesteps

See `SCENARIOS.md` for exhaustive permutations.

## When Things Go Wrong

**Command fails immediately:**
1. Check file paths exist
2. Check `extropy config show`
3. Check API key env vars

**Validation errors:**
```bash
extropy validate <spec.yaml>
```

**Exposure not spreading:**
- Check `share_probability` in scenario
- Check network connectivity
- Inspect `by_timestep.json`

**Cost blowup:**
- Compare `extropy estimate` vs actual
- Reduce population, fidelity, or timesteps
- Use cheaper routine model

See `TROUBLESHOOTING.md` for full triage.

## Analysis Workflow

```bash
# Summary
extropy results --study-db $DB

# Segments
extropy results --study-db $DB --segment income
extropy results --study-db $DB --segment age_bracket

# Timeline
extropy results --study-db $DB --timeline

# Agent deep dive
extropy results --study-db $DB --agent agent_0042

# Export for DS
extropy export states --study-db $DB --to states.jsonl
extropy export agents --study-db $DB --to agents.jsonl

# Inspection
extropy inspect summary --study-db $DB
extropy inspect agent --study-db $DB --agent-id agent_042

# Ad-hoc queries
extropy query sql --study-db $DB --sql "SELECT position, COUNT(*) FROM agent_states GROUP BY position"

# JSON reports
extropy report run --study-db $DB -o report.json

# Post-sim chat
extropy chat --study-db $DB --run-id <id> --agent-id agent_042
```

See `ANALYSIS.md` for interpretation guide.

## Confidence Reporting

**Single seed:** State it explicitly. "Based on seed 42. Run confidence sweep for stability."

**Multi-seed:** Report mean + range.
```bash
for seed in 42 43 44 45 46; do
  extropy simulate scenario.yaml --study-db study-$seed.db -o results-$seed --seed $seed
done
```

## Escalation

Stop and ask user when:
- Same error occurs twice after fix attempt
- Fix would change core study assumptions
- Cost vs accuracy tradeoff needs decision
- Sensitive content requires human judgment

Provide: exact blocker, evidence, options (A/B/C), recommendation.

## Module Reference

| File | Contents |
|------|----------|
| `OPERATIONS.md` | Full command reference, pipeline details, batch management |
| `SCENARIOS.md` | What can be simulated, permutation matrix, examples |
| `TROUBLESHOOTING.md` | Diagnosis, quality gates, escalation |
| `ANALYSIS.md` | Post-run analytics, interpretation, DS workflows |
| `REPORT_TEMPLATE.md` | Standard experiment report format |

## Config

```bash
extropy config show
extropy config set pipeline.provider claude
extropy config set simulation.provider openai
extropy config set simulation.model gpt-5-mini
```

API keys (env vars only):
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AZURE_OPENAI_API_KEY`
