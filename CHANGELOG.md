# Changelog

All notable changes to this project are documented in this file.

This project follows semantic versioning.

## [0.4.0] - 2026-02-24

### Summary

`v0.4.0` is a major operational release focused on pipeline coherence, runtime realism,
and reliability under production-scale studies. The release consolidates the CLI around
the scenario-first study model, hardens sampling and network quality gates, and upgrades
simulation behavior for evolving timelines with stronger re-reasoning and conversation
control.

Compared to `v0.3.0`, this release significantly changes how studies are authored and run:
the old base-plus-extend workflow is fully absorbed into `scenario`, command surfaces are
streamlined, and simulation/stopping semantics are now timeline-aware by default.

### Added

- Scenario-centric pipeline architecture:
  - scenario extension discovery/hydration integrated directly into `extropy scenario`.
  - versioned study artifacts for scenario and persona (`scenario.vN.yaml`, `persona.vN.yaml`).
- Agent-mode operational support:
  - structured clarifications and deterministic exit-code-driven automation flow.
  - richer machine-safe command behavior across pipeline and query surfaces.
- Advanced simulation realism controls:
  - timeline-safe stopping with future-event awareness and explicit early-convergence override.
  - epoch/provenance-based re-reasoning support for new information propagation.
  - conversation interleaving during timestep reasoning execution (instead of only terminal phase effects).
  - persisted and classified internal/external divergence (`THINK` vs `SAY`) with tracking support.
  - macro summary context and stronger action-intent accountability in reasoning prompts.
- Chat and analysis improvements:
  - LLM-backed post-simulation chat (`chat ask`) with DB-grounded context.
  - query-first data extraction and inspection flow for agents, edges, states, network status, and SQL.
- Model/provider capability expansion:
  - support for `claude-sonnet-4-6`.
  - improved long-response handling via streaming for Anthropic-backed calls.

### Changed

- CLI workflow and command contracts:
  - `extend` workflow removed from operational pipeline; scenario now owns extension generation.
  - default pipeline is now:
    `spec -> scenario -> persona -> sample -> network -> simulate -> results`.
  - `results` and `query` are now the canonical output/read path for analysis and export.
- Study and schema semantics:
  - scenario and downstream runtime behavior are keyed by `scenario_id`-centric flows.
  - household configuration ownership moved from base spec concerns into scenario-level semantics.
  - `agent_focus_mode` moved into scenario ownership and enforcement path.
- Sampling and household coherence:
  - stronger lifecycle normalization for household/member coherence.
  - exact target count fixes in sampling loops.
  - stricter handling of dependent naming and partner generation consistency.
- Network generation behavior:
  - default network config generation is enabled for meaningful topology (`--generate-config` true).
  - CLI no longer forcibly overrides LLM-proposed `avg_degree` and `rewire_prob` unless user overrides.
  - deterministic structural role resolution and stricter topology gate hardening.
  - similarity worker auto-detection and resource tuning improvements.
- Simulation execution/runtime tuning:
  - tighter re-reasoning and conversation budget guards.
  - improved async safety in conversation scheduling and runtime orchestration.
  - expanded runtime tunables and clearer fidelity-dependent behavior.
- Documentation and operator runbooks:
  - command docs aligned to current CLI surface.
  - automation docs tightened for reproducible, non-interactive operation.

### Fixed

- Timeline and event correctness:
  - fixed timeline timestep validation edge behavior.
  - fixed evolution handling around timeline/outcome generation and timestep units.
- Runtime and provider stability:
  - fixed token bucket internals usage bug with pydantic private attributes.
  - fixed async client reuse to prevent event-loop-close warnings.
  - fixed provider/runtime call paths for long-running reasoning calls.
- CLI and model correctness:
  - fixed multiple command consistency issues and stale flag/help mismatches.
  - fixed validation routing for versioned scenario/persona files.
  - fixed scenario population reference resolution paths in estimate/validate/network/sample/simulate flows.
- Persona/rendering quality:
  - fixed boolean/persona formatting edge cases.
  - fixed duplicate headers and empty population stats handling.
  - fixed currency/money rendering correctness in persona output.
- Data quality and coherence:
  - fixed partner/dependent naming preservation and generation behavior.
  - fixed cases where reasoning-agent age and household generation constraints could drift.

### Deprecated

- Direct usage of old docs/flags that referenced `extend`, `export`, `inspect`, `report`,
  or global `--json` usage patterns.

### Removed

- Legacy `extend` command operational path from the user-facing workflow.
- Root/global `--json` flag exposure from CLI surface.
- `estimate` exposure in active CLI command list (temporarily hidden/disabled while parity work continues).
- Legacy name-config/correlation contract pieces from spec/build paths where replaced by newer semantics.

### Breaking Changes

- Pipeline shape changed: if your automation scripts called `extend`, they must be migrated.
- JSON mode usage changed: configure `cli.mode=agent` for machine-oriented output behavior.
- Scenario ownership changed: household and agent-focus semantics are now scenario-owned.
- Some old help/documented flags are no longer valid and must be updated to current command forms.
- Legacy study scripts that assume direct `--study-db` command patterns should migrate to `--study` and study-folder auto-detection.

### Migration Guide (`0.3.x -> 0.4.0`)

1. Set/update agent mode in automation:
   - `extropy config set cli.mode agent`
2. Move to scenario-first pipeline:
   - old: `spec -> extend -> sample -> network -> scenario -> simulate`
   - new: `spec -> scenario -> persona -> sample -> network -> simulate`
3. Ensure scenario artifacts are versioned and colocated under:
   - `scenario/<name>/scenario.vN.yaml`
   - `scenario/<name>/persona.vN.yaml`
4. Update run scripts to use `results`/`query` for post-run extraction.
5. If you used estimate in CI/automation, remove hard dependencies until command parity is restored.

### Operational Notes

- This release increases realism and safety guards, but can increase prompt/context volume in high-fidelity,
  multi-timestep studies. Use smaller dry runs before full production scale.
- For evolving scenarios, timeline-aware stopping and epoch re-reasoning materially change long-run dynamics.
  Re-baseline historical comparisons when upgrading from `0.3.x`.

---

## [0.3.0] - 2026-02-16

Baseline for the pre-`0.4.0` command/runtime generation used by existing studies.

