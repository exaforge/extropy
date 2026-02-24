# What Extropy Can Simulate

This document is the consolidated capabilities reference. It merges prior showcase, use-case, and study notes into one current view of what the codebase supports.

---

## Core Capability Model

Extropy simulates how heterogeneous populations respond to events over time.

The engine combines:

1. Statistical grounding from sampled population distributions.
2. Scenario-specific extensions and exposure rules.
3. Social network propagation with structural edge types.
4. Multi-timestep LLM reasoning with memory, conviction, and sharing dynamics.

Output is distributional and segmentable, not a single headline score.

---

## Population Scope

### Geography

- Works for US and non-US populations.
- Name generation is **Faker-first with locale routing**, with bundled CSV fallback when needed.
- Geography is represented semantically (country/state/region/city attributes), not as a physics coordinate system.

### Unit of Simulation

- Individual-only and household-aware sampling are both supported.
- Household-aware runs support partner-linked adults and dependent NPC context.
- Scenario-level `agent_focus_mode` controls who is an active reasoning agent (`primary_only`, `couples`, `all`).

### Social Structure

Network generation supports structural edge families plus similarity fill:

- `partner`
- `household`
- `coworker`
- `neighbor`
- `congregation`
- `school_parent`
- `acquaintance` / `online_contact`

This supports real-world diffusion patterns beyond flat homophily.

---

## Scenario Scope

### Static Scenarios

Best for one-time shocks and settling dynamics:

- pricing changes,
- policy announcements,
- product launches,
- corporate decisions.

### Evolving Scenarios

Best for multi-stage world changes where new information arrives over time:

- geopolitical escalation,
- crisis response,
- economic cascade scenarios,
- long-horizon social transitions.

Current runtime behavior supports timeline-safe execution:

- future timeline events suppress premature convergence/quiescence auto-stop,
- event-level re-reasoning intensity (`normal` / `high` / `extreme`) is available,
- exposure provenance (`info_epoch`) is persisted and used by re-reasoning selection.

---

## Behavioral Depth

### Reasoning Pipeline

- Two-pass reasoning (default):
  - Pass 1 role-played reaction,
  - Pass 2 structured outcome extraction.
- Optional merged pass exists for cost/speed tradeoff experiments.

### Conviction + Memory

- Conviction is tracked and used for sharing/reconsideration behavior.
- Exposure history and memory traces inform later timesteps.
- Non-reasoning agents decay over time, preventing frozen states.

### Conversations

At medium/high fidelity, agents can trigger inter-agent conversations that are interleaved during timestep execution (budget and novelty gated).

### Public Discourse

- Public statements and social posts are tracked.
- Agents get local peer context and broader social feed context in prompts.

---

## Outcome Scope

Supported outcome types:

- categorical,
- boolean,
- float,
- open-ended.

This enables both hard decision tracking and qualitative signal capture in the same run.

---

## Consolidated Use-Case Families

These are the production-fit categories from prior docs.

1. Market research and consumer behavior.
2. Pricing response and elasticity.
3. Public policy and compliance response.
4. Product launch/adoption curves.
5. Political and message testing.
6. Reputation/crisis response.
7. Information spread and narrative resilience.
8. Community planning and development response.

---

## Showcase Study Patterns

These are the canonical study templates currently used by the team.

| Pattern | Why It Fits Extropy | Typical Config |
|---------|----------------------|----------------|
| ASI timeline shock | High uncertainty + staged societal changes | Evolving, month unit, ~6 timesteps, 5k agents |
| Iran-strikes geopolitical shock | Multi-domain cascade (military + economic + political) | Evolving, week unit, ~12 timesteps, 5k agents |
| Bitcoin extreme-rally social/economic shock | Wealth-effect + media contagion + fraud narratives | Evolving, week unit, 8-12 timesteps, 5k agents |
| Election projection (House/state race) | Segment-level turnout + issue salience + identity effects | Static or short evolving, 1k-5k agents |

For election studies:

- generic-ballot style questions are valid for House-control style studies,
- candidate-specific races need candidate-context scenarios and state-specific populations.

---

## Iran-Strike Study Lessons (Generalized)

The Iran-strikes study is a good stress test for evolving scenarios because it activates many dimensions at once:

- household financial margin,
- employment sector sensitivity,
- political identity,
- media ecosystem,
- institutional trust,
- military connection,
- geography-driven burden.

Generalizable design principles from that study:

1. Timeline events should describe **world facts**, not scripted emotional outcomes.
2. Include concrete, decision-relevant specifics (prices, casualties, disruptions) when possible.
3. Stage escalation so different subpopulations are activated at different timesteps.
4. Keep framing neutral to avoid baking conclusions into prompts.

---

## What Extropy Is Not For

Extropy is not a full replacement for:

1. Rigid org-chart workflow simulation (hard reporting-line process models).
2. Physical logistics optimization (route geometry, traffic flow physics, facility simulation).
3. High-frequency quantitative trading simulation.
4. Fully reactive game-master event injection loops (beyond authored timeline flow).

---

## Practical Interpretation Rule

Use Extropy when the question is:

"How will different groups of people process and propagate this event over time, and where do outcomes diverge by identity, incentives, and network position?"

Do not use Extropy when the primary bottleneck is physical mechanics, deterministic workflow constraints, or millisecond-level market microstructure.
