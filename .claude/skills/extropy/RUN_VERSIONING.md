# Run Versioning and Naming System

Use this as the canonical run/config management convention.

## Goals

1. Every run is uniquely identifiable.
2. Config and scenario revisions are explicit.
3. Cross-run comparisons are machine-readable.
4. Humans can quickly infer purpose from path/name.

## Directory Layout

```text
runs/
  <study_slug>/
    registry/
      runs.csv
      latest.txt
    specs/
      pop/
      persona/
      network-config/
      scenario/
    batches/
      <batch_id>/
        manifest.yaml
        variants/
          <variant_id>/
            inputs/
            results/
```

## ID Conventions

- `study_slug`: lowercase kebab-case, e.g. `austin-congestion-tax`
- `batch_id`: `bYYYYMMDD-HHMM-<intent>-vNN`, e.g. `b20260213-2140-baseline-v01`
- `variant_id`: `vr-<axis>-<value>-s<seed>`, e.g. `vr-threshold-3-s42`
- `scenario_rev`: `scn-vNN`, e.g. `scn-v03`
- `config_rev`: `cfg-vNN`, e.g. `cfg-v05`

## File Naming

- Population spec: `<study_slug>.pop-vNN.yaml`
- Persona config: `<study_slug>.persona-vNN.yaml`
- Network config: `<study_slug>.network-cfg-vNN.yaml`
- Scenario spec: `<study_slug>.scn-vNN.scenario.yaml`

Example:
- `austin-congestion-tax.scn-v03.scenario.yaml`

## Variant Manifest (`manifest.yaml`)

Each batch must include a manifest:

```yaml
study: austin-congestion-tax
batch_id: b20260213-2140-baseline-v01
scenario_rev: scn-v03
config_rev: cfg-v05
created_at: "2026-02-13T21:40:00Z"
objective: "Baseline plus threshold sensitivity"
base_paths:
  population_spec: runs/austin-congestion-tax/specs/pop/austin-congestion-tax.pop-v02.yaml
  persona_config: runs/austin-congestion-tax/specs/persona/austin-congestion-tax.persona-v01.yaml
  network_config: runs/austin-congestion-tax/specs/network-config/austin-congestion-tax.network-cfg-v05.yaml
  scenario_spec: runs/austin-congestion-tax/specs/scenario/austin-congestion-tax.scn-v03.scenario.yaml
variants:
  - id: vr-threshold-2-s42
    seed: 42
    overrides:
      threshold: 2
  - id: vr-threshold-3-s42
    seed: 42
    overrides:
      threshold: 3
```

## Run Registry (`runs.csv`)

Append one row per variant execution:

```csv
timestamp,study,batch_id,variant_id,scenario_rev,config_rev,seed,status,results_dir,notes
2026-02-13T22:05:00Z,austin-congestion-tax,b20260213-2140-baseline-v01,vr-threshold-3-s42,scn-v03,cfg-v05,42,success,runs/austin-congestion-tax/batches/b20260213-2140-baseline-v01/variants/vr-threshold-3-s42/results,baseline
```

## Revision Rules

Bump `scn-vNN` when:
- Event content/source changes
- Exposure rules/channels change
- Outcomes/stop conditions change

Bump `cfg-vNN` when:
- Provider/model/rate settings change
- Threshold/chunk-size defaults change
- Network/persona config logic changes

## Execution Pattern with Versioned Names

1. Create or choose scenario revision under `specs/scenario/`.
2. Create `batch_id` and `manifest.yaml`.
3. Run each variant to `batches/<batch_id>/variants/<variant_id>/results/`.
4. Append run registry row.
5. Update `registry/latest.txt` with most recent successful `batch_id`.

## Comparison Contract

Only compare variants directly when:
- Same `study_slug`
- Same `scenario_rev`
- Same `config_rev` (except intended override axis)
- Same seed policy
