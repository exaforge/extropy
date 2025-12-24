# 1. Fix spec validation

Yes, I see the issue. Look at the modifier conditions — they use **different naming conventions** than the actual option values.

## The Problem

**In `employer_type` options:**

```yaml
options:
  - University_hospital
  - Public_non_university_hospital
  - Private_hospital
  - MVZ_or_private_practice
```

**But in modifier `when` clauses (e.g., `ai_tool_awareness`):**

```yaml
- when: employer_type == 'University hospital' # ❌ Space, no underscore
- when: employer_type in ['Private hospital','Medical care center/MVZ or private practice'] # ❌ Wrong
```

**In `professional_role` options:**

```yaml
options:
  - Resident_Assistenzarzt
  - Specialist_Facharzt
  - Senior_Oberarzt
  - Chief_Chefarzt
```

**But in modifier `when` clauses:**

```yaml
- when: professional_role in ['Senior/Oberarzt','Chief/Chefarzt'] # ❌ Slash, not underscore
- when: professional_role == 'Resident/Assistenzarzt' # ❌ Slash, not underscore
```

**In `research_activity_level` options:**

```yaml
options:
  - none
  - occasional
  - regular
  - lead_PI
```

**But in modifier `when` clauses:**

```yaml
- when: research_activity_level in ['regular','lead/PI'] # ❌ 'lead/PI' vs 'lead_PI'
```

---

## Root Cause

The LLM generated the base attributes with one naming convention (underscores: `Senior_Oberarzt`), then later generated modifiers using a different convention (slashes/spaces: `Senior/Oberarzt`).

---

## Quick Fix Options

### Option 1: Script to Auto-Fix

I can write a quick Python script that:

1. Reads the YAML
2. For each modifier `when` clause, finds option references
3. Fuzzy-matches to actual options
4. Replaces with correct values

### Option 2: Manual Search-Replace

Key replacements needed:

```
'University hospital' → 'University_hospital'
'Public non-university hospital' → 'Public_non_university_hospital'
'Private hospital' → 'Private_hospital'
'Medical care center/MVZ or private practice' → 'MVZ_or_private_practice'
'Resident/Assistenzarzt' → 'Resident_Assistenzarzt'
'Specialist/Facharzt' → 'Specialist_Facharzt'
'Senior/Oberarzt' → 'Senior_Oberarzt'
'Chief/Chefarzt' → 'Chief_Chefarzt'
'lead/PI' → 'lead_PI'
```

### Option 3: Re-run Hydration

With a stricter prompt that emphasizes: "Use EXACT option values from the base spec. Copy-paste them."

---

## Recommendation

**Option 1 (script)** is fastest and most reliable. Want me to write it?
