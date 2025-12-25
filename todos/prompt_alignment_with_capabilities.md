# Prompt Alignment with Capabilities

**Priority:** High
**Component:** `entropy/scenario` prompts
**Related:** `docs/CAPABILITIES.md`

---

## Summary

The current LLM prompts in `entropy/scenario` reference outdated examples ("Company Policy", "Product Launch") that contradict the new architectural focus on "Targeted Synthetic Populations" and "General Population" simulations. They need to be updated to mirror the scenarios defined in `docs/CAPABILITIES.md`.

---

## Required Changes

### 1. `entropy/scenario/parser.py`

**Current State:**

- `announcement`: Lists "company policy changes, official statements, product updates".
- `policy_change`: Lists "company policy updates".
- `product_launch`: Examples are "new app feature".

**Required Update:**

- **announcement:** Change to "government mandates, public health guidance, platform-wide service changes".
- **policy_change:** Change to "legislation, regulatory shifts, professional board certification changes" (matching Scenario 4).
- **product_launch:** Reframe as "market-wide innovation" (e.g., "Med-tech firm launches AI tool") matching Scenario 1.

### 2. `entropy/scenario/exposure.py`

**Current State:**

- Generic "Exposure Strategy Guidance".

**Required Update:**

- **Rumors:** Explicitly mention "ambiguity" and "epistemic bubbles" (Scenario 3).
- **Guidance:** Add: "For rumors, consider how 'low trust' or 'high anxiety' groups might be exposed first."

### 3. `entropy/scenario/outcomes.py`

**Current State:**

- Focuses purely on predefined metrics.
- "Outcome Patterns" table lists generic "Price change".

**Required Update:**

- **Prompt Instruction:** Add: _"Ensure outcomes capture the **reasoning** behind decisions, not just the decision itself, to enable post-hoc discovery of emergent behaviors (e.g., 'rotating' instead of just 'cancelling')."_
- **Table:** Update to include patterns for:
  - **Professional Alignment:** `exit_intent`, `voice_intent`, `compliance_level` (Scenario 4).
  - **Information Warfare:** `belief_level`, `amplification_intent` (Scenario 3).

### 4. `entropy/scenario/interaction.py`

**Current State:**

- `share_modifiers` examples are generic.

**Required Update:**

- Add a "Psychographic" modifier example to reinforce the Targeted Population philosophy:
  - Example: `when: "institutional_trust < 0.3", multiply: 2.0`

---

## Files to Modify

| File                              | Change                                                                       |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `entropy/scenario/parser.py`      | Update `announcement`, `policy_change`, `product_launch` examples in prompt. |
| `entropy/scenario/exposure.py`    | Enrich `generate_seed_exposure` prompt with "epistemic bubble" guidance.     |
| `entropy/scenario/outcomes.py`    | Add "reasoning capture" instruction and update outcome patterns table.       |
| `entropy/scenario/interaction.py` | Add psychographic `share_modifier` example.                                  |

---

## Acceptance Criteria

- [ ] `parse_scenario` prompt no longer suggests internal corporate policies.
- [ ] `generate_seed_exposure` prompt encourages targeting based on trust/anxiety.
- [ ] `define_outcomes` prompt explicitly asks for reasoning-enabling outcomes.
- [ ] `determine_interaction_model` prompt shows a psychographic modifier example.
