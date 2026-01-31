# Persona System Redesign

This document outlines the redesign of Entropy's persona generation system to achieve true agent embodiment while maintaining scalability and generalizability.

---

## The Core Problem

The simulation asks LLMs to reason **as** agents with specific attributes. But there's a fundamental difference between:

| Puppetry (current) | Embodiment (goal) |
|---|---|
| LLM reads attributes as facts about someone else | LLM internalizes attributes as its own worldview |
| "This person has Trust In Institutions: Low" | "I've learned not to trust what the city promises" |
| External observer applying traits like checkboxes | Internal perspective where attributes shape reasoning |
| Attributes are **data** | Attributes are **lived experience** |

The current prompt says "You ARE the person" but then presents attributes in third-person list format. The LLM is told to be someone while being handed a character sheet — it performs the role rather than inhabiting it.

---

## Current System Pitfalls

### 1. Minimal Narrative Intro

The current `persona_template` is just 2-3 sentences covering 5-6 attributes:

```
You are a {age}-year-old {gender} working as a {employment_status} in 
{occupation_industry}, living in {home_zip_code} in Austin. You have been 
in your current role for {years_in_current_role} years.
```

With 51 attributes, 45 are relegated to a flat bullet list.

### 2. Third-Person Attribute Lists

Remaining attributes are presented as external observations:

```
**Your Mindset & Values**
- Trust In Institutions: Low
- Price Sensitivity Transport: High
- Risk Tolerance: Moderate
```

This reads like a character sheet, not an internal perspective.

### 3. Loss of Numeric Precision

All 0-1 floats are bucketed into five labels (Very low, Low, Moderate, High, Very high). An agent at 0.20 and one at 0.35 both become "Low" — losing meaningful differentiation.

### 4. Absolute vs. Relative

"Low trust" in absolute terms is meaningless. What matters is: are you more or less trusting than others in your situation? The current system lacks population-relative framing.

### 5. No Grouping Logic

51 attributes in a flat list. `commute_distance_miles` has the same visual weight as `speeding_ticket_last_3y`. The LLM gets no signal about how attributes relate to each other.

### 6. Simulation Consequences

These issues contribute to documented simulation problems:

- **Central tendency**: 83%+ agents choose the "safe middle" option
- **Weak trait-outcome correlation**: Decisions don't correlate with relevant attributes
- **Generic reasoning**: LLM produces plausible but not persona-grounded responses

---

## Design Principles for the New Approach

### Principle 1: No Researcher Bias

We do NOT pre-select which attributes are "relevant" to the decision. That would impose our model of decision-making onto the agents. The simulation should discover which attributes drive decisions, not have us prescribe it.

**Implication**: All 51 attributes must be included. No filtering, no prioritization.

### Principle 2: Embodiment Through Voice

First-person throughout. Not "Your trust in institutions is low" but "I've learned not to trust what institutions promise."

**Implication**: Every attribute gets phrased as something the agent would say about themselves.

### Principle 3: Preserve True Values

Concrete facts keep their numbers. $22/day is $22/day, not "Low parking cost." 12 miles is 12 miles.

**Implication**: LLM-driven classification of which attributes are concrete (keep number) vs. psychological (use relative words).

### Principle 4: Relative Positioning for Psychological Traits

For latent constructs (trust, sensitivity, personality), position relative to the population: "more skeptical than most" rather than "Low."

**Implication**: Compute population mean/std from sampled agents, use for relative labeling.

### Principle 5: Template-Based Scalability

Generate templates/configuration once during setup. Apply to N agents via computation — no per-agent LLM calls for persona generation.

**Implication**: Persona rendering is string formatting + arithmetic, not LLM inference.

### Principle 6: Generalizability

No hardcoded rules like "0-1 floats are always psychological" or "attributes ending in _score need special treatment." The LLM reads attribute descriptions and decides how to handle each one.

**Implication**: Works for surgeons, commuters, retail workers, voters — any population.

---

## The Template Approach

### What Gets Generated Once (LLM Call During Setup)

The `entropy persona` command makes one LLM call that generates a **rendering configuration**:

1. **Attribute Classification**: For each attribute, is it a concrete fact (keep number) or psychological construct (use relative positioning)?

2. **Grouping Scheme**: Which attributes belong together? (identity, work, commute, finances, mindset, options, etc.)

3. **Phrasing Templates**: For each attribute, how would someone state this in first person?
   - Booleans: "I own a bike" / "I don't own a bike"
   - Categoricals: "My schedule is pretty fixed" / "I have some flexibility"
   - Concrete numbers: "I drive {value} miles"
   - Psychological traits: "I'm {relative_label} than most people when it comes to {trait}"

4. **Population Statistics**: Mean and standard deviation for each psychological attribute, computed from the sampled agents.

### What Gets Computed Per Agent (No LLM)

For each of 500 agents:

1. **Compute relative positions**: For psychological traits, calculate z-score and map to relative label ("more than most", "about average", "less than most")

2. **Fill templates**: String formatting with agent values

3. **Assemble sections**: Concatenate grouped sections into full persona

**Cost**: Zero LLM calls. Pure computation. ~1ms per agent.

---

## The LLM Prompt for Persona Configuration

```
You are configuring how agent personas will be rendered for a simulation.

POPULATION: {description}

ATTRIBUTES:
{for each attribute: name, type, description, distribution/options}

For each attribute, determine:

1. TREATMENT
   - "concrete": Keep the actual number/value. Use for measurable facts like 
     age, income, distance, time, counts, costs.
   - "relative": Express relative to population. Use for psychological traits,
     attitudes, preferences, tendencies.
   
   Base this on the attribute's DESCRIPTION and meaning, not its data type.

2. GROUPING
   Assign to a life domain. Common groups: identity, household, work, commute, 
   finances, transit_options, mindset, social. Create groups that make sense 
   for this specific population.

3. PHRASING
   Provide a first-person template for each attribute.
   
   For concrete attributes:
     "I drive {commute_distance_miles} miles to work"
     "Our household income is around ${annual_household_income_usd:,.0f}"
   
   For relative attributes, provide templates for five levels:
     much_below: "I trust institutions far less than most people"
     below: "I'm more skeptical of institutions than most"
     average: "I'm about as trusting of institutions as the average person"
     above: "I generally trust institutions more than most"
     much_above: "I have much more faith in institutions than most people"
   
   For booleans:
     true: "I own a bike"
     false: "I don't own a bike"
   
   For categoricals, provide phrasing for each option:
     cannot_shift: "My schedule is fixed — I can't easily shift my hours"
     shift_30m: "I have some flexibility to shift by about 30 minutes"
     ...

Output as structured YAML that can render any agent into a first-person persona.
```

---

## Sample Persona Output

For an Austin commuter with 51 attributes:

```
## Who I Am

I'm a 39-year-old Hispanic man living in the 78704 zip code in Austin. I'm 
married with kids — there are 4 of us in the household. I have a bachelor's 
degree. We own our home.

## My Work

I work full-time in professional and technical services at a medium-sized 
company (50-249 employees). I've been in my current role for about 6 years. 
I have some flexibility in my schedule, though not full flexibility. My 
workday typically starts around 8:30 AM.

## My Commute

I drive 12 miles to downtown, mostly via I-35. I leave around 7:15 in the 
morning and it usually takes about 42 minutes. I go into the office 4 days a 
week and work from home 1 day. My primary reason for going downtown is work.

My schedule is pretty fixed — with kids to get to school, I can't easily shift 
my departure time to avoid peak hours.

## My Vehicle

I drive a 6-year-old gasoline sedan. We have 2 vehicles in the household. I 
use toll roads sometimes — not regularly, but when traffic is bad. I haven't 
had a speeding ticket in the last 3 years. I do try to drive efficiently to 
save fuel.

## My Costs

I pay $22 a day for parking — my employer doesn't cover it. Our household 
income is around $95,000 a year. I spend about $148 a month on auto insurance. 
When I think about my time, I value it at around $24 an hour when making 
commute tradeoffs.

## My Transit Options

The nearest frequent bus stop is about a 9-minute walk from my home. Metro rail 
isn't really accessible from where I live. I don't have a transit pass. I own a 
bike, though I don't use it for commuting. I don't have any mobility limitations.

## How I Think

When it comes to trusting institutions — government, employers, that kind of 
thing — I'm more skeptical than most people. I've learned to take official 
announcements with a grain of salt.

I pay closer attention to transportation costs than most people do. When 
commute costs go up, I notice it and think about alternatives.

I'm about as open to trying new things as the average person. Not resistant to 
change, but not jumping at every new idea either.

I'm somewhat more organized and conscientious than average. I like having my 
routine.

I'm a bit less outgoing than most people — I'm not the first to strike up 
conversations with strangers.

I'm about as agreeable and cooperative as the average person.

I'm less prone to stress and worry than most people.

I'm less of a risk-taker than most. I prefer known quantities over gambles.

When it comes to following rules and paying what I owe — tolls, fees, that 
sort of thing — I'm about average. I generally comply but I'm not rigid 
about it.

## My Social Tendencies

I have some interest in carpooling if the right situation came up — I'm 
slightly more open to it than the average commuter.

## My Awareness

I wasn't aware of the details of this congestion tax before hearing about it 
now.
```

**Key properties:**

- All 51 attributes present
- All concrete values preserved ($22, 12 miles, $95k, 42 minutes, etc.)
- All psychological traits relative to population ("more skeptical than most")
- All in first person
- Grouped by life domain
- ~450 words total

---

## Updated Command Line Workflow

```
entropy spec      → base attributes (universal, population-specific, personality)
entropy extend    → scenario-specific attributes
entropy sample    → N agents with attribute values
entropy network   → social graph

entropy persona   → [NEW COMMAND]
    ├── Load population spec + sampled agents
    ├── Compute population statistics (mean, std per attribute)
    ├── LLM call: generate rendering configuration
    │   ├── Classify each attribute (concrete vs relative)
    │   ├── Generate groupings
    │   └── Generate first-person phrasing templates
    ├── Preview sample persona (render one agent)
    ├── Allow iteration (user feedback → regenerate)
    └── Save persona_config to population spec or separate file

entropy scenario  → event, exposures, outcomes
entropy simulate  → for each agent:
    ├── Render persona from config (no LLM, just templates)
    └── Call reasoning LLM with embodied persona
```

### The `entropy persona` Command

```bash
entropy persona austin/population.yaml \
  --agents austin/agents.json \
  --preview \
  --output austin/persona_config.yaml
```

**Flags:**
- `--agents`: Path to sampled agents (needed for population statistics)
- `--preview`: Show a rendered sample persona before saving
- `--output`: Where to save the persona configuration
- `--feedback "make it more conversational"`: Regenerate with guidance

**Interactive flow:**

```
✓ Loaded population spec: 51 attributes
✓ Loaded agents: 500 sampled
✓ Computed population statistics

✓ Generated persona configuration (23s)

┌──────────────────────────────────────────────────────────────┐
│                     PERSONA CONFIGURATION                    │
└──────────────────────────────────────────────────────────────┘

Attribute Classification:
  Concrete (keep values): 31 attributes
  Relative (use positioning): 20 attributes

Groupings:
  • Who I Am (7 attributes)
  • My Work (5 attributes)
  • My Commute (8 attributes)
  • My Vehicle (6 attributes)
  • My Costs (4 attributes)
  • My Transit Options (5 attributes)
  • How I Think (11 attributes)
  • My Social Tendencies (2 attributes)
  • My Awareness (1 attribute)

Sample Persona (Agent #42):

  [Full rendered persona shown here]

[Y] Save config  [r] Regenerate  [f] Give feedback  [n] Cancel: 
```

---

## How This Generalizes

The same approach works for any population because:

1. **No hardcoded rules**: The LLM reads attribute descriptions and decides treatment
2. **No domain-specific patterns**: We don't check for `*_score` or assume 0-1 floats are psychological
3. **Groupings are inferred**: The LLM creates groups that make sense for that population

### Example: German Surgeons

```
## Who I Am

I'm a 52-year-old woman, a senior attending surgeon specializing in orthopedics 
at a large academic medical center in Bavaria. I've been practicing for 24 years.

## My Practice

I perform about 180 procedures a year, mostly joint replacements. I spend 
roughly 12 hours a week on administrative documentation.

## My Technology Experience

I've used 2 AI-assisted diagnostic tools before. My department has a structured 
decision support system in place.

## How I Think

I'm more skeptical of AI in medical settings than most of my colleagues. I've 
seen too many promising technologies fail to deliver.

I'm more concerned about data privacy than the average surgeon. Patient 
confidentiality is paramount.

I follow clinical guidelines more closely than most — I believe in evidence-based 
practice.

...
```

### Example: Retail Workers

```
## Who I Am

I'm a 28-year-old woman working part-time as a sales associate at a big-box 
retailer in suburban Dallas. I've been here about 2 years.

## My Job

I work about 28 hours a week, mostly evening and weekend shifts. My hourly wage 
is $14.50. I don't get health benefits through work.

## How I Think

I trust management less than most of my coworkers. I've seen promises made and 
not kept.

I'm more worried about job security than the average employee here. Hours can 
get cut without warning.

...
```

The template generation prompt receives the population description and attributes — it figures out the rest.

---

## How This Solves Documented Issues

Referring to issues in `SIMULATION_IMPROVEMENTS.md`:

### Central Tendency (Layer 1)

**Issue**: 83%+ agents choose the "safe middle" option.

**How persona helps**: Richer embodiment means the LLM has more material to differentiate responses. An agent who says "I pay closer attention to transportation costs than most people" and "I pay $22/day out of pocket" has a concrete basis for a strong reaction to a $15/day tax.

**Note**: This doesn't replace the two-pass reasoning fix, but it improves the quality of the first pass.

### Trait-Outcome Correlation

**Issue**: Decisions don't correlate with relevant attributes.

**How persona helps**: Relative positioning makes trait differences salient. "I'm more skeptical than most" is a stronger signal than "Trust: Low" buried in a list. The LLM is more likely to ground reasoning in these self-descriptions.

### Prompt Differentiation (Layer 3)

**Issue**: 40+ traits in a flat list, no indication of relevance.

**How persona helps**: Grouping by life domain and first-person phrasing makes all attributes equally salient within their context. We don't pre-select relevance (that would bias), but we organize for comprehension.

### Prior Position Anchoring (Layer 2b)

**Issue**: Re-reasoning agents have no memory of previous conclusions.

**How persona helps**: The persona provides stable identity context. When combined with "What You Previously Thought" (per Layer 2b), the agent has both who they are and what they previously concluded.

---

## What This Doesn't Solve

The persona redesign improves input quality. It doesn't fix:

1. **Two-pass reasoning** (Layer 1): Still needed to separate role-play from classification
2. **Conviction mechanics** (Layer 2a): Still needed for state dynamics
3. **Source credibility** (Layer 2c): Still needed for network propagation
4. **Conviction-gated sharing** (Layer 2d): Still needed for realistic spread

These remain separate implementation items. The persona redesign is complementary.

---

## Cost Analysis

### Persona Configuration (One-Time)

- 1 LLM call during `entropy persona`
- ~30 seconds, similar cost to other pipeline steps
- Output is reusable for all simulations of that population

### Per-Agent Persona Rendering (Simulation Time)

- Zero LLM calls
- Pure computation: lookup population stats, compute z-scores, fill templates
- ~1ms per agent
- 500 agents = 0.5 seconds total

### Simulation Reasoning (Unchanged)

- Same number of reasoning calls as before
- Slightly longer prompts (~450 words vs ~150 words)
- Token cost increase: ~20-30% per reasoning call

---

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Attributes included | ~20 in narrative, rest in list | All 51 in first-person prose |
| Voice | Mixed first/third person | First person throughout |
| Numeric values | Bucketed to words | Concrete values preserved |
| Psychological traits | Absolute buckets | Relative to population |
| Organization | Flat list | Grouped by life domain |
| Scalability | Template + string.format | Same (template-based) |
| Generalizability | Hardcoded patterns | LLM-driven classification |
| LLM calls for persona | 0 per agent | 0 per agent (1 for config) |

The redesign achieves embodiment without sacrificing scalability or generalizability.
