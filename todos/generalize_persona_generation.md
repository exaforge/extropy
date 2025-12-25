# Generalize Persona Generation

**Priority:** Medium
**Component:** `entropy/simulation/persona.py`
**Goal:** Replace hardcoded "Surgeons" logic with a generalized, template-driven system.

---

## The Problem

The current `persona.py` module is brittle and overfitted to the initial "German Surgeons" example.
*   **Hardcoded Logic:** Functions like `_format_role`, `_format_employer`, and `_format_location` look for specific keys (`surgical_specialty`, `federal_state`) that may not exist in other populations (e.g., "Kyoto Homeowners").
*   **Rigid Templates:** It constructs sentences using a fixed logic (`"You are a {age}-year-old {gender}..."`) which feels robotic and repetitive.

## Proposed Solution: LLM-Generated Templates

Instead of hardcoding Python f-strings, we should use the LLM (during Phase 1 or just-in-time) to generate a **Persona Template** specific to the population spec.

### 1. New Workflow

1.  **Analyze Spec:** The system inspects the `PopulationSpec` attributes.
2.  **Generate Template (LLM):** We ask the LLM: *"Given these attributes (age, income, trust_level), write a Jinja2 template that converts a single agent's data into a natural language 1st-person bio."*
3.  **User Review (CLI):** Show the generated template to the user.
    *   *Example:* `"You are a {{ age }}-year-old {{ occupation }} living in {{ city }}. You deeply mistrust {{ trust_target }}..."`
    *   **Prompt:** `[Y/n]` to approve or regenerate.
4.  **Save Template:** Store this template in the `ScenarioSpec` or `PopulationSpec` metadata.
5.  **Runtime Generation:** `persona.py` simply renders this template for each agent, replacing the complex hardcoded logic.

### 2. Implementation Details

#### A. Interactive CLI Step
Add a step to `entropy scenario` (Phase 2) or `entropy spec` (Phase 1):

```python
# Pseudo-code
template = generate_persona_template(population_spec)
print(f"Proposed Persona Template:\n{template}")
if typer.confirm("Use this template?"):
    save_template(template)
else:
    template = refine_template_interactive()
```

#### B. Template Engine
Use `jinja2` for robust logic within the template (e.g., `{% if age > 60 %}...{% endif %}`).

#### C. Fallback
Keep a minimal generic fallback (`"You are an agent with attributes: {attrs}"`) for when the template fails or attributes are missing.

---

## Tasks

1.  [ ] **Create `PersonaTemplateGenerator`:** A new LLM chain that takes a `PopulationSpec` and outputs a Jinja2 string.
2.  [ ] **Update CLI:** Add the interactive "Review Persona Template" step to the workflow.
3.  [ ] **Refactor `persona.py`:** Delete the hardcoded formatting functions (`_format_role`, etc.) and replace them with a `render_persona(agent, template)` function.
4.  [ ] **Migration:** Ensure existing specs (Surgeons) work by generating a default template for them.

---

## Benefits
*   **Universality:** Works for ANY population (Voters, Gamers, Soldiers) without code changes.
*   **Quality:** LLMs write better natural language templates than Python f-strings.
*   **User Control:** Users can tweak the "voice" of their agents by editing the template.
