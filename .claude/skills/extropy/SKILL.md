---
name: extropy
description: Autonomous operator for Extropy: run end-to-end pipelines, execute/manage simulation experiments, triage failures, and perform post-run analysis. Use when the user wants hands-on execution, simulation operations, debugging, or data-science deep dives.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
argument-hint: "[goal or experiment request]"
---

# Extropy Autonomous Ops Skill

Use this skill as an execution-oriented operator, not a passive explainer.

## Operating Mode

- Default to doing the work directly via Extropy CLI.
- Prefer reproducible commands (explicit paths, `--seed`).
- Keep runs organized by experiment folder and variant names.
- Summarize outcomes with file-based evidence from output artifacts.

## Skill Modules

- `PIPELINE_RUNBOOK.md` — Build population -> scenario -> simulation end to end
- `SIM_OPS.md` — Batch execution, parameter sweeps, run management
- `TRIAGE_PLAYBOOK.md` — Failure diagnosis and corrective actions
- `ANALYSIS_PLAYBOOK.md` — Post-run analytics and interpretation
- `WORKFLOWS.md` — Copyable autonomous workflows
- `RUN_VERSIONING.md` — Canonical naming/versioning for runs, configs, and scenarios
- `RUN_MANAGEMENT.md` — Batch/registry lifecycle operations
- `EXPERIMENT_REPORT_TEMPLATE.md` — Standard decision report format

## Core Principles

1. Execute first, explain second.
2. Validate inputs before long/expensive calls.
3. Persist all outputs and compare runs systematically.
4. When blocked, isolate root cause with the smallest reproducible command.
5. Use evidence from `meta.json`, `by_timestep.json`, `outcome_distributions.json`, `agent_states.json`, `timeline.jsonl`, and `simulation.db`.
6. Default to introspection, not just aggregates: explain *why* outcomes happened from agent traces.
7. For decision support, report uncertainty across repeated runs (convergence + spread), not one-off point estimates.

## Minimal Command Skeleton

```bash
# 1) Population build
extropy spec "<population>" -o runs/<name>/base.yaml
extropy extend runs/<name>/base.yaml -s "<scenario>" -o runs/<name>/population.yaml
extropy sample runs/<name>/population.yaml -o runs/<name>/agents.json --seed 42
extropy network runs/<name>/agents.json -o runs/<name>/network.json -p runs/<name>/population.yaml --seed 42
extropy persona runs/<name>/population.yaml --agents runs/<name>/agents.json -o runs/<name>/population.persona.yaml

# 2) Scenario and sim
extropy scenario -p runs/<name>/population.yaml -a runs/<name>/agents.json -n runs/<name>/network.json -o runs/<name>/scenario.yaml
extropy estimate runs/<name>/scenario.yaml
extropy simulate runs/<name>/scenario.yaml -o runs/<name>/results --seed 42

# 3) Results views
extropy results runs/<name>/results
extropy results runs/<name>/results --segment income
```

## Notes

- Supported providers: `openai`, `claude`, `azure_openai`.
- Defaults are `openai` for both pipeline and simulation zones.
- API keys are env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`).
- Primary decision domains: policy, market/pricing, crisis/reputation, political messaging, and community response.
