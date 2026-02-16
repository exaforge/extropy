# Fixes and Improvements Log

## 2026-02-15: agent_focus prompt ambiguity

**Problem**: The `extropy spec` command was generating `agent_focus: "suburban homeowners with school-age children"` for a community scenario, which caused partners to become NPCs instead of active agents. For a book ban community reaction scenario, both partners should be simulated.

**Root cause**: The sufficiency check prompt in `extropy/population/spec_builder/sufficiency.py` didn't explain to the LLM that `agent_focus` controls household sampling modes:
- `"families"` / `"households"` → everyone simulated
- `"couples"` / `"partners"` → both adults simulated, kids are NPCs
- specific roles like `"surgeons"` → one person per household

The LLM just extracted a literal description without understanding the semantic implications.

**Fix**: Updated the prompt and schema description to explicitly document:
1. The three sampling modes and their trigger keywords
2. When to use each mode (communities → families, professional studies → specific role)
3. Concrete examples mapping scenario types to correct `agent_focus` values

**Files changed**:
- `extropy/population/spec_builder/sufficiency.py` — schema description and prompt text

**Verification**: `tests/test_agent_focus.py` (34 tests) all pass.

---

## 2026-02-15: Persona renderer duplicate sections and missing punctuation

**Problem**: The persona renderer was outputting duplicate sections. For example:
- `## Who I Am` rendered age, gender, state from `intro_template`
- `## About Me` rendered the same attributes again from the `basic_identity` group

Additionally, sentences were missing periods, causing run-on text like:
```
I'm 49 years old I'm a dad in this community My family lives in Kansas
```

**Root cause**: The `render_persona()` function in `extropy/population/persona/renderer.py`:
1. Rendered `intro_template` attributes, then also rendered them in their respective groups
2. Joined phrases with spaces but didn't ensure sentence-ending punctuation

**Fix**:
1. Added `extract_intro_attributes()` helper to parse `{attribute}` placeholders from `intro_template`
2. Modified `render_persona()` to track and exclude intro attributes from group rendering (same pattern used for `decision_relevant_attributes`)
3. Added `_ensure_period()` helper to ensure all phrases end with `.`, `!`, or `?`

**Files changed**:
- `extropy/population/persona/renderer.py` — added `extract_intro_attributes()`, `_ensure_period()`, updated `render_persona()`

**Verification**: `tests/test_persona*.py` (6 tests) all pass. Manual verification shows clean persona output without duplicates.

---

## 2026-02-15: Azure OpenAI schema rejection for actions array

**Problem**: Simulation failed with Azure OpenAI (gpt-5-mini) rejecting the reasoning schema:
```
Invalid schema for response_format 'agent_reasoning': In context=('properties', 'actions', 'items'), 'additionalProperties' is required to be supplied and to be false.
```

**Root cause**: Azure OpenAI has stricter JSON schema requirements than OpenAI direct. The `actions` array items (objects with `type`, `who`, `topic` properties) were missing `additionalProperties: false`.

**Fix**: Added `"additionalProperties": False` and included `"topic"` in `required` array for both `actions` item schemas in `extropy/simulation/reasoning.py`:
1. Line ~464 (two-pass schema)
2. Line ~640 (merged-pass schema)

Azure requires ALL properties to be listed in `required`, not just a subset.

**Files changed**:
- `extropy/simulation/reasoning.py` — added `additionalProperties: False` and `"topic"` to required array

**Verification**: Lint passes. Re-run simulation to confirm.

---

## 2026-02-15: Azure OpenAI schema rejection for conversation_turn

**Problem**: Simulation failed with Azure OpenAI rejecting the conversation_turn schema:
```
Invalid schema for response_format 'conversation_turn': In context=(), 'required' is required to be supplied and to be an array including every key in properties. Missing 'internal_reaction'
```

**Root cause**: Same Azure strictness issue. The conversation turn schema had `internal_reaction` in properties but not in the `required` array.

**Fix**: Updated `_build_conversation_schema()` in `extropy/simulation/conversation.py` to include all properties in the `required` array and added `additionalProperties: False`:
```python
"required": ["response", "internal_reaction", "updated_sentiment", "updated_conviction"],
"additionalProperties": False,
```

**Files changed**:
- `extropy/simulation/conversation.py` — line 390-391

**Verification**: Lint passes. Re-run simulation to confirm.
