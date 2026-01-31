# Simulation Engine — Current Issues & Planned Changes

This document covers known issues with the Phase 3 simulation engine and concrete implementation plans for fixing them. Organized into three layers, each building on the previous.

---

## Layer 1: Two-Pass Reasoning (Central Tendency Fix)

### Problem

LLMs exhibit strong central tendency when picking from categorical options — the same way humans asked to "pick a number from 1-10" cluster around 7. With 4 outcome options, 83% of agents chose the safe middle (`wait_for_outcomes_review`). With 5 options (adding `pilot_first`), 99% chose `pilot_first`. The "safe middle" just shifts — the collapse is inherent to categorical forced-choice.

The current architecture asks the LLM to do two things in one call: role-play as a person AND pick from an enum. These are different cognitive tasks and the structured output constraint (JSON schema with `enum`) biases the model toward safe, non-committal options.

### Solution: Split Reasoning from Classification

**Pass 1 — Free-text reasoning (role-play call):**
Ask the agent to reason naturally with no schema constraints on outcomes. Just reasoning + will_share. This is where `gpt-5-mini` actually performs well — logged reasoning text already references specific persona traits ("as a digitally literate, guideline-focused surgeon who trusts medical AI...").

**Pass 2 — Classification (cheap extraction call):**
A separate call takes the free-text reasoning and maps it to the outcome categories. This is a classification task, not a role-play task — a different cognitive mode that doesn't suffer from central tendency. Can use the cheapest model available.

### What Changes

**`simulation/reasoning.py`:**

`build_response_schema()` — Split into two schemas:
- `build_reasoning_schema()`: Returns schema with `reasoning` (string) and `will_share` (boolean) only. No outcome enums.
- `build_classification_schema()`: Returns schema with just the outcome fields (categorical enums, floats, booleans).

`reason_agent()` — Two sequential calls:
1. `simple_call()` with reasoning schema → gets free-text reasoning + will_share
2. `simple_call()` with classification schema, prompt = "Given this person's reasoning: '{reasoning_text}', classify their response:" → gets outcome values

`_reason_agent_async()` — Same two-call pattern, both awaited sequentially within the same semaphore slot.

`batch_reason_agents()` — No structural change, each agent just does two calls internally.

**`simulation/engine.py`:**
No changes needed — it receives `ReasoningResponse` the same way.

**Cost impact:** Doubles the number of API calls per agent. Pass 2 is a short classification prompt (~200 tokens) so the actual cost increase is ~15-25%, not 2x. The classification call can use the cheapest available model.

---

## Layer 2: Behavioral Realism (State & Dynamics)

These are additive changes to the agent state model and reasoning loop. They don't change existing code paths — they extend them.

### 2a. Conviction / Belief Strength

**Problem:** Agents have a position (string) and sentiment (float) but no measure of how strongly they hold that position. An agent can flip from `adopt_now` to `not_adopt` between timesteps with no resistance.

**What changes:**

`core/models/simulation.py` — Add to `AgentState`:
```python
conviction: float = Field(default=0.5, description="How strongly held (0=uncertain, 1=firm)")
```

Add to `ReasoningResponse`:
```python
conviction: float = Field(default=0.5, description="How strongly they hold this view")
```

`simulation/reasoning.py` — Add `conviction` to the response schema:
```python
"conviction": {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "description": "How strongly do you hold this view? 0 = very uncertain, 1 = completely firm"
}
```

In `build_reasoning_prompt()`, when re-reasoning (current_state is not None):
```
"You currently believe: {position} (conviction: {high/moderate/low})"
```

`simulation/engine.py` — In `_state_changed()`, weight by conviction:
```python
# High-conviction agents resist change
if old.conviction and old.conviction > 0.7 and old.position != new.position:
    # Only accept position change if new conviction is also strong
    if new.conviction < 0.5:
        new.position = old.position  # Reject weak flip
```

### 2b. Prior Position Anchoring

**Problem:** When an agent re-reasons (multi-touch), their previous reasoning text is stored in `raw_reasoning` but never fed back to the LLM. The agent has no memory of what they previously concluded.

**What changes:**

`simulation/reasoning.py` — In `build_reasoning_prompt()`, when `context.current_state` is not None and has `raw_reasoning`:
```python
if context.current_state and context.current_state.raw_reasoning:
    prompt_parts.extend([
        "",
        "## What You Previously Thought",
        "",
        f"When you first heard about this, your reaction was:",
        f'"{context.current_state.raw_reasoning}"',
        f"You felt {format_sentiment(context.current_state.sentiment)} about it.",
        "",
        "You may or may not have changed your mind since then.",
    ])
```

No model changes needed — `ReasoningContext.current_state` already exists and is already passed by `engine.py`.

### 2c. Differential Source Credibility

**Problem:** Peer credibility is hardcoded at `0.85` in `propagation.py:259`. All peers have equal influence. A mentor has the same credibility as a weak tie. Edge weight and type exist in the network data but are ignored.

**What changes:**

`simulation/propagation.py` — Replace hardcoded `0.85` with edge-type lookup:
```python
PEER_CREDIBILITY = {
    "mentor_mentee": 0.95,
    "colleague": 0.85,
    "department": 0.80,
    "conference": 0.70,
    "regional": 0.65,
    "society": 0.60,
    "weak_tie": 0.50,
}

# In propagate_through_network():
edge_type = edge_data.get("type", "contact")
base_credibility = PEER_CREDIBILITY.get(edge_type, 0.70)
edge_weight = edge_data.get("weight", 0.5)
credibility = base_credibility * (0.7 + 0.3 * edge_weight)  # Weight modulates slightly
```

`simulation/reasoning.py` — In `build_reasoning_prompt()`, make exposure source more informative:
```python
# Instead of "Someone in your network told you about this"
if exp.source_agent_id:
    # credibility is already on the ExposureRecord
    if exp.credibility >= 0.9:
        prompt_parts.append(f"- A trusted {relationship} told you about this")
    elif exp.credibility >= 0.7:
        prompt_parts.append(f"- A {relationship} mentioned this to you")
    else:
        prompt_parts.append(f"- You heard about this through a loose connection")
```

### 2d. Conviction-Gated Sharing

**Problem:** `will_share` is determined purely by the LLM with no structural constraint. An agent with near-zero conviction can still decide to evangelize.

**What changes:**

`simulation/engine.py` — After processing the reasoning response, modulate sharing by conviction:
```python
# After extracting response into new_state:
if new_state.will_share and new_state.conviction is not None:
    if new_state.conviction < 0.3:
        new_state.will_share = False  # Too uncertain to share
```

Three lines, no architectural change.

---

## Layer 3: Prompt Differentiation (Trait Salience)

### Problem

The persona template lists 40+ traits in flat sections. `persona.py` already converts numbers to words (`0.24` → "Low", `0.94` → "Very high") via `format_value()`, so agents never see raw floats. But the prompt doesn't indicate which traits are *relevant* to the decision at hand. `commute_time_minutes` gets the same visual weight as `trust_in_ai_medical`.

The prompt can't explicitly suggest a direction ("you should be skeptical because...") — that would bias output. Differentiation has to come from making relevant traits more salient without prescribing conclusions.

### Solution: Scenario-Tagged Relevant Attributes

**Phase 2 (scenario compilation)** identifies which population attributes are decision-relevant. This information flows into the persona builder at simulation time.

**What changes:**

`core/models/scenario.py` — Add to `OutcomeConfig`:
```python
decision_relevant_attributes: list[str] | None = Field(
    default=None,
    description="Population attributes most relevant to this decision"
)
```

`scenario/outcomes.py` — The LLM that generates outcomes also identifies which population attributes are most relevant to the decision. It already sees the population spec. Add to the schema:
```python
"decision_relevant_attributes": {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 3,
    "maxItems": 10,
    "description": "Which population attributes most influence how agents respond to this event"
}
```

`simulation/persona.py` — `build_characteristics_list()` accepts a `relevant_attrs: set[str]` parameter. If provided, restructure the output:
```
## Factors Most Relevant to This Situation
- Trust In Ai Medical: Low
- Data Privacy Concern: Very high
- Prior Ai Tool Usage: None
- Digital Literacy Score: High

## Your Background
- Age: 46
- Marital Status: Married
...
```

The LLM sees the same traits but the relevant ones are first, under a heading that signals "these matter for what you're deciding."

`simulation/reasoning.py` — Pass `scenario.outcomes.decision_relevant_attributes` through to `generate_persona()`. Requires threading the scenario through the persona generation call in `engine.py`.

**What this does NOT do:** It doesn't tell the agent what to conclude. An agent with "Trust In AI Medical: Low" might still adopt if their institutional incentives are strong enough. The prompt just makes the relevant tradeoffs visible rather than buried in a flat list.

---

## Implementation Order

1. **Layer 2b (prior anchoring)** — Smallest change, highest immediate impact on dynamics. ~10 lines in `reasoning.py`. Already has all the data plumbing in place.
2. **Layer 2c (source credibility)** — Replace one hardcoded constant with a dict lookup, improve prompt text. ~20 lines across two files.
3. **Layer 2a (conviction)** — Model change + schema change + prompt change + gating logic. ~50 lines across 3 files but well-contained.
4. **Layer 2d (conviction-gated sharing)** — 3 lines, but depends on conviction being implemented first.
5. **Layer 1 (two-pass reasoning)** — Largest structural change, refactors `reason_agent()` flow. Most impactful for outcome quality but also most code to change and test.
6. **Layer 3 (trait salience)** — Requires scenario compiler changes and a new field on the model. Worth doing but needs testing to confirm it drives differentiation without introducing bias.

---

## Metrics to Validate Improvements

After implementing changes, measure these on the german-surgeons scenario:

- **Position entropy** — Shannon entropy of outcome distribution. Higher = better differentiation. Current: ~0.65 (83/17 split). Target: >1.5.
- **Trait-outcome correlation** — Do agents with high `trust_in_ai_medical` actually adopt more? Pearson correlation between key attributes and outcome. Current: unknown (not measured). Target: statistically significant correlations for decision-relevant attributes.
- **Re-reasoning stability** — When agents re-reason after multi-touch, how often do they flip positions? Current: unknown. With conviction, expect <20% flip rate for high-conviction agents.
- **Share rate variance** — Do different agent profiles share at different rates? Current: uniform (LLM decides freely). With conviction gating, expect correlation between conviction and sharing.
