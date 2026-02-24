---
name: extropy
description: "Execution-first operator for Extropy: run pipelines, diagnose failures, and deliver evidence-backed simulation analysis using current CLI contracts."
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
argument-hint: "[goal or experiment request]"
always: false
source: "https://github.com/exaforge/extropy"
homepage: "https://github.com/exaforge/extropy"
required-binaries:
  - extropy
required-env-vars:
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - OPENROUTER_API_KEY
  - DEEPSEEK_API_KEY
  - AZURE_API_KEY
  - AZURE_ENDPOINT
  - AZURE_OPENAI_API_KEY
  - AZURE_OPENAI_ENDPOINT
  - MODELS_FAST
  - MODELS_STRONG
  - SIMULATION_FAST
  - SIMULATION_STRONG
  - SIMULATION_MAX_CONCURRENT
  - SIMULATION_RATE_TIER
  - SIMULATION_RPM_OVERRIDE
  - SIMULATION_TPM_OVERRIDE
required-config-files:
  - ~/.config/extropy/config.json
primary-credential: "Provider API keys via environment variables"
required_binaries:
  - extropy
required_env_vars:
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - OPENROUTER_API_KEY
  - DEEPSEEK_API_KEY
  - AZURE_API_KEY
  - AZURE_ENDPOINT
  - AZURE_OPENAI_API_KEY
  - AZURE_OPENAI_ENDPOINT
  - MODELS_FAST
  - MODELS_STRONG
  - SIMULATION_FAST
  - SIMULATION_STRONG
  - SIMULATION_MAX_CONCURRENT
  - SIMULATION_RATE_TIER
  - SIMULATION_RPM_OVERRIDE
  - SIMULATION_TPM_OVERRIDE
required_config_files:
  - ~/.config/extropy/config.json
primary_credential: "Provider API keys via environment variables"
metadata:
  requires:
    binaries:
      - extropy
    config_files:
      - ~/.config/extropy/config.json
    environment_variables:
      - OPENAI_API_KEY
      - ANTHROPIC_API_KEY
      - OPENROUTER_API_KEY
      - DEEPSEEK_API_KEY
      - AZURE_API_KEY
      - AZURE_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_ENDPOINT
      - MODELS_FAST
      - MODELS_STRONG
      - SIMULATION_FAST
      - SIMULATION_STRONG
      - SIMULATION_MAX_CONCURRENT
      - SIMULATION_RATE_TIER
      - SIMULATION_RPM_OVERRIDE
      - SIMULATION_TPM_OVERRIDE
  credentials:
    primary:
      - provider API keys via environment variables only
    notes:
      - extropy reads model/provider settings from ~/.config/extropy/config.json
      - do not read raw .env contents unless explicitly requested by the user
---

# Extropy Operator

Run experiments end to end, with strict quality gates and reproducible commands.

## Operating Rules

1. Execute commands; do not only describe them.
2. Validate upstream artifacts before expensive downstream stages.
3. Use explicit evidence (paths, SQL, metrics) for every conclusion.
4. Keep assumptions visible and minimal.
5. Escalate after repeated failure of the same gate.

## Runtime Dependencies and Credential Scope

- Required binary: `extropy` must be installed and on `PATH`.
- Config file read: `~/.config/extropy/config.json`.
- Credentials expected: provider API keys from env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `AZURE_API_KEY`/`AZURE_OPENAI_API_KEY`).
- Azure endpoint vars: `AZURE_ENDPOINT` or `AZURE_OPENAI_ENDPOINT` when using Azure providers.
- Safety boundary: only access credentials/config needed to execute extropy commands for the requested study/scenario, do not inspect raw `.env` values, and avoid reading unrelated files.

## Canonical Pipeline

```bash
extropy spec -> extropy scenario -> extropy persona -> extropy sample -> extropy network -> extropy simulate -> extropy results
```

## Quick Runbook

```bash
STUDY=runs/my-study
SCENARIO=ai-shock

# 1) Study + base population
extropy spec "5000 US working-age adults" -o "$STUDY" --use-defaults
cd "$STUDY"
extropy config set cli.mode agent

# 2) Scenario + persona
extropy scenario "AI systems outperform most knowledge workers in 6 months" -o "$SCENARIO" -y
extropy persona -s "$SCENARIO" -y

# 3) Sample + network
extropy sample -s "$SCENARIO" -n 5000 --seed 42 --strict-gates
extropy network -s "$SCENARIO" --seed 42 --quality-profile strict --validate

# 4) Simulate
extropy simulate -s "$SCENARIO" --seed 42 --fidelity high --rpm-override 1000

# 5) Results + raw extraction
extropy results -s "$SCENARIO" summary
extropy results -s "$SCENARIO" timeline
extropy query states --to states.jsonl
```

## Required Quality Gates

- `spec`: `extropy validate population.vN.yaml` passes.
- `scenario`: `extropy validate scenario/<name>/scenario.vN.yaml` passes.
- `persona`: persona file exists and validates.
- `sample`: count correct, coherence gates pass, no critical impossibles.
- `network`: no isolated catastrophes, topology metrics in acceptable range.
- `simulate`: run completes or checkpoints cleanly with recoverable state.

## Recommended Command Surface

- Build: `spec`, `scenario`, `persona`, `sample`, `network`, `simulate`
- Inspect: `results summary|timeline|segment|agent`
- Deep checks: `query summary|network|network-status|sql|agents|edges|states`
- Agent QA: `chat list`, `chat ask`

## Module Map

- `OPERATIONS.md`: stage-by-stage execution and rerun strategy.
- `TROUBLESHOOTING.md`: failure diagnosis and escalation policy.
- `ANALYSIS.md`: post-run metrics, SQL patterns, reporting structure.
- `SCENARIOS.md`: scenario/pipeline design patterns and limits.
- `REPORT_TEMPLATE.md`: standardized delivery format.
