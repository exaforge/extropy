# Extropy v2 — Full Pipeline Architecture

This document specifies every change across the entire Extropy pipeline, from `extropy spec` through `extropy results`. It supersedes `simulation-experience-gaps.md` (which identified the problems). This document defines the solutions.

---

## Design Principles

1. **Households, not individuals.** The atomic unit of sampling is a household. Partners are co-sampled. Kids are generated at sampling time. Economic fate is shared.
2. **Social roles, not similarity.** Network edges represent real relationships (partner, coworker, neighbor, friend) derived from agent attributes — not generic "acquaintance" edges from embedding distance.
3. **First person, always.** Agents think as "I" — not "You are Travis." The prompt reads like an internal monologue, not a briefing document.
4. **Names carry culture.** Every agent has a name. Names are demographically appropriate (SSA + census data). Partners, kids, and peers are named.
5. **Time is felt.** Agents know what day/week it is, how long ago they heard the news, whether things are getting better or worse.
6. **Conversations are real.** Agents talk to each other. Both sides are real agents (or NPC dependents). The conversation changes both participants.
7. **Outcomes emerge, not imposed.** For exploratory scenarios, outcomes are open-ended. Categories are discovered post-hoc through clustering, not pre-defined dropdowns.
8. **Scenarios evolve.** Events develop over time. New information arrives at specified timesteps. The world isn't frozen at t=0.
9. **Deterministic where possible.** Names, family, temporal framing, mood rendering, channel experience — all deterministic. LLM calls only for reasoning and conversations.
10. **Fidelity is tunable.** `--fidelity low/medium/high` controls prompt richness and conversation depth without changing the underlying data model.

---

## Pipeline Overview

```
extropy spec  →  extropy extend  →  extropy sample  →  extropy network  →  extropy persona  →  extropy scenario  →  extropy simulate  →  extropy results
                                         ↑                                                          ↑                      ↑
                                   HOUSEHOLD-BASED                                            SCENARIO TIMELINE      CONVERSATIONS
                                   SAMPLING (NEW)                                             + OUTCOME TRACKS       + COGNITIVE STATE
                                                                                              (NEW)                  (NEW)
```

---

## PHASE 1: POPULATION CREATION

### 1.1 `extropy spec` — No Changes

Spec building (sufficiency check → attribute selection → hydration → binding) stays the same. The LLM discovers attributes relevant to the scenario and builds distributions. No structural changes.

One addition: the spec builder should recognize household-level attributes and tag them appropriately. When the LLM discovers `household_income`, `marital_status`, `household_size` — these should be tagged `scope: household` in the attribute metadata, signaling to the sampler that they're shared within a household unit.

```yaml
# New attribute metadata field
- name: household_income
  scope: household  # shared across household members
  type: int
  category: universal
  ...

- name: neuroticism
  scope: individual  # default, independent per person
  type: float
  category: personality
  ...
```

### 1.2 `extropy extend` — Minor Changes

Same as spec, but with context from a base spec. The extension LLM should be aware that household-scoped attributes exist and avoid duplicating them across individual-level additions.

### 1.3 `extropy sample` — Major Rewrite

**Current:** Samples N individual agents sequentially. Each agent is independent. A married agent with `household_size: 5` exists alone — no partner, no kids.

**New:** Samples N/H households, where H is the average household size. Each household produces 1-2 adult agents plus NPC dependents.

#### Household Sampling Flow

```
1. Determine household composition
   - Sample household_type: single | couple | single_parent | couple_with_kids | multi_generational
   - Source: Census household composition rates by state, age bracket, race_ethnicity
   - This determines how many adults and dependents

2. Sample primary adult (Adult 1)
   - All individual attributes sampled as today (age, gender, personality, occupation, etc.)
   - Household-scoped attributes sampled once and shared

3. If couple household → Sample partner (Adult 2)
   - Correlated demographics:
     - Same state, urban_rural (shared household)
     - Same household_income, household_size (shared)
     - Age: sampled from joint distribution conditioned on Adult 1's age
       (mean = Adult 1 age ± 2 years, std = 3 years, constrained by gender norms for the cohort)
     - race_ethnicity: correlated via intermarriage rates from census
       (same-race probability ~85%, varies by group)
     - education_level: assortative mating correlation (~0.6 from research)
     - religious_affiliation: high correlation (~0.7)
     - political_orientation: moderate correlation (~0.5)
   - Independent attributes:
     - Personality traits (neuroticism, extraversion, etc.) — independent
     - occupation_sector — constrained by local job market (same state) but otherwise independent
     - All attitudes and context-specific attributes — independent
   - Same-sex couples:
     - Census same-sex household rates by state (~1-3% of couple households)
     - When sampled as same-sex, both adults share gender
     - Name generation respects this

4. Generate NPC dependents (children, elderly parents)
   - Number of kids = household_size - number_of_adults
   - Kid ages: sampled from plausible range given parent age + life_stage
     - If parent is 43 (middle_aged_adult), kids likely 5-20
     - If parent is 28 (young_adult), kids likely 0-8
     - Spacing: 2-4 years between siblings (sampled)
   - Kid attributes (minimal, for context only):
     - first_name (from same cultural pool as parents)
     - age
     - gender (50/50)
     - school_status: derived from age (0-4: home, 5-17: in school, 18+: college/working)
     - grade: derived from age
   - Elderly dependents: if household_labor_dependency implies it
     - first_name, age (65+), relationship ("mother", "father-in-law")
   - NPCs stored as structured metadata on the parent agent, NOT as agents

5. Generate names for all household members
   - Data source: SSA baby name frequency data (by birth decade + gender) + Census surname data (by ethnicity)
   - First name: filter by birth_decade (derived from age) + gender + race_ethnicity → weighted random pick
   - Last name: filter by race_ethnicity → weighted random pick → shared across household
   - Partner may have different last name (probability based on era/demographics)
   - ~50KB of bundled CSV data. Zero API calls.
```

#### Agent Record (New Schema)

```json
{
  "_id": "agent_0020",
  "first_name": "Travis",
  "last_name": "McAllister",
  "household_id": "household_0010",
  "household_role": "adult_primary",
  "partner_id": "agent_0021",
  "dependents": [
    {"name": "Tyler", "age": 17, "gender": "male", "relationship": "son", "school_status": "high_school_senior"},
    {"name": "Kayla", "age": 14, "gender": "female", "relationship": "daughter", "school_status": "high_school_freshman"},
    {"name": "Mason", "age": 9, "gender": "male", "relationship": "son", "school_status": "4th_grade"}
  ],
  "age": 43,
  "gender": "male",
  "race_ethnicity": "white",
  "state": "south_carolina",
  "marital_status": "cohabiting",
  "household_size": 5,
  "household_income": 52000,
  ...all other attributes...
}
```

#### What This Changes About Study DB Storage

The `agents` table stays the same (id → JSON attributes). But agents now have:
- `household_id` for grouping
- `partner_id` for cross-referencing
- `dependents` as structured metadata
- `first_name`, `last_name` as named attributes

The `populations` table in study DB gets a new `households` table:

```sql
CREATE TABLE households (
    id TEXT PRIMARY KEY,
    population_id TEXT,
    adult_ids JSON,        -- ["agent_0020", "agent_0021"]
    dependent_data JSON,   -- NPC details
    shared_attributes JSON -- household_income, state, etc.
);
```

### 1.4 `extropy network` — Major Rewrite

**Current:** Compute pairwise similarity from all attributes → threshold → create edges → assign edge types from similarity buckets → calibrate to target metrics.

**New:** Create edges from social role rules first, then fill with similarity-based edges.

#### Social Role Edge Generation

```
Phase 1: Structural edges (deterministic from attributes)
  - partner: from partner_id field (always created, weight = 1.0)
  - household: all adults in same household_id (always created, weight = 0.9)
  - coworker: same occupation_sector + same state (probabilistic, ~5-10 per agent)
  - neighbor: same state + same urban_rural + age within 15 years (probabilistic, ~3-5 per agent)
  - congregation: same religious_affiliation + same state (if religious, ~3-5 per agent)
  - school_parent: both have kids in school age + same state + same urban_rural (~2-3 per agent)

Phase 2: Similarity-based edges (fills remaining degree budget)
  - Compute similarity as today, but only for agents not yet connected
  - These become "acquaintance" or "online_contact" edges
  - Lower weight than structural edges

Phase 3: Calibration
  - Same calibration loop as today (hit target avg_degree, clustering, modularity)
  - But structural edges are PROTECTED — calibration can only add/remove similarity edges
  - Structural edges are never pruned
```

#### Edge Schema (Enhanced)

```json
{
  "source": "agent_0020",
  "target": "agent_0021",
  "edge_type": "partner",
  "weight": 1.0,
  "structural": true,
  "context": "household"
}
```

Edge types and their default weights:
| Edge Type | Weight | Source | Bidirectional |
|-----------|--------|--------|---------------|
| partner | 1.0 | household_id + partner_id | yes |
| household | 0.9 | household_id | yes |
| coworker | 0.6 | occupation_sector + state | yes |
| close_friend | 0.7 | similarity top-k + age proximity | yes |
| neighbor | 0.4 | state + urban_rural + age proximity | yes |
| congregation | 0.4 | religious_affiliation + state | yes |
| school_parent | 0.35 | kids in school + state + urban_rural | yes |
| acquaintance | 0.2 | similarity-based | yes |
| online_contact | 0.15 | similarity-based, different state OK | yes |

#### What The Scenario Can Override

The scenario YAML can declare **relationship priority** for this topic:

```yaml
relationship_weights:
  partner: 1.0       # always highest
  coworker: 0.9      # ASI: coworkers are critical
  close_friend: 0.7
  neighbor: 0.3      # ASI: neighbors less relevant
```

For ULEZ, the weights would be different:
```yaml
relationship_weights:
  partner: 1.0
  neighbor: 0.8      # ULEZ: neighbors directly affected
  coworker: 0.5      # ULEZ: less relevant unless both tradies
```

These weights are used in two places:
1. **Conversation conflict resolution** — when two people want to talk to the same agent, higher weight goes first
2. **Peer opinion ordering in the prompt** — higher weight peers appear first with full detail (name, demographics, statement), lower weight peers get summarized ("a few acquaintances also mentioned it")

The agent's OWN choice of who to talk to is LLM-driven — they see the list of available people and pick based on what they need. The weights handle tiebreaking and prompt ordering, not the agent's decision.

**Auto-generation:** `extropy scenario` generates these weights automatically — the compiler LLM knows that ASI is a workplace threat (coworker weight high), Netflix is a household product (family weight high), ULEZ is local policy (neighbor weight high). The user can override in the YAML after generation.

---

## PHASE 2: SCENARIO COMPILATION

### 2.1 `extropy scenario` — Significant Extensions

The scenario YAML gains three new top-level sections: `timeline`, `day_phases`, and enhanced `outcomes`.

#### Scenario Timeline (New)

**Current:** Single event at t=0. Nothing changes.

**New:** Sequence of events at specified timesteps.

```yaml
timeline:
  - timestep: 0
    event:
      type: news
      content: "AI systems demonstrate superhuman performance across all cognitive domains. Fortune 50 firms announce restructuring."
      source: "Major news outlets"
      credibility: 0.8
      ambiguity: 0.6
      emotional_valence: -0.7
  - timestep: 3
    event:
      type: news
      content: "Federal government announces emergency AI task force. No concrete policy yet. Unemployment claims spike 40%."
      source: "White House press briefing"
      credibility: 0.9
      ambiguity: 0.4
      emotional_valence: -0.5
  - timestep: 8
    event:
      type: news
      content: "First wave of layoffs hits. 200,000 jobs cut across tech and finance in the past month."
      source: "Bureau of Labor Statistics"
      credibility: 0.95
      ambiguity: 0.2
      emotional_valence: -0.8
  - timestep: 16
    event:
      type: policy_change
      content: "Government announces Emergency Workforce Transition Act: $500/person retraining voucher, extended unemployment to 52 weeks."
      source: "Congressional legislation"
      credibility: 0.9
      ambiguity: 0.3
      emotional_valence: -0.2
```

Each timeline event:
- Has its own exposure rules (which channels, which agents, what probability)
- Gets injected into agent prompts at the specified timestep as "new information this [week/day]"
- Accumulates — by timestep 16, agents have the full history of developments
- Can override or update the original event's parameters

For scenarios with no evolution (Netflix password sharing), the timeline is just the single t=0 event. No extra configuration needed.

#### Day Phases (New, Optional)

Templates that structure the agent's timestep experience into life contexts. Optional — if omitted, the engine uses an improved flat prompt (still much better than current).

```yaml
day_phases:
  defaults:
    - phase: morning
      condition: "true"
      template: "I wake up and check my phone."
      slots: [social_media_exposures, aggregate_mood]
    - phase: work
      condition: "employment_status in ['employed full-time', 'employed part-time', 'self-employed']"
      template: "I head to work at my {occupation_sector} job."
      slots: [workplace_exposures, coworker_opinions]
    - phase: school
      condition: "school_enrollment != 'not enrolled'"
      template: "I go to class."
      slots: [peer_opinions]
    - phase: evening
      condition: "true"
      template: "I'm home for the evening."
      slots: [family_context, partner_opinion, network_opinions, memory_trace]

  # Override for weekly timesteps
  weekly:
    - phase: this_week_at_work
      condition: "employment_status in ['employed full-time', 'employed part-time', 'self-employed']"
      template: "This week at work..."
      slots: [workplace_exposures, coworker_opinions]
    - phase: this_week_at_home
      condition: "true"
      template: "At home this week..."
      slots: [family_context, partner_opinion, aggregate_mood]
    - phase: this_week_online
      condition: "social_media_usage in ['heavy/multiple times daily', 'moderate/daily']"
      template: "Online this week..."
      slots: [social_media_exposures, public_discourse]
```

Phase selection adapts to `timestep_unit`:
- Daily → morning/work/evening
- Weekly → this_week_at_work/home/online
- Monthly → this_month overview
- Hourly → single phase per timestep

Agent attributes determine which phases render. A retiree skips `work`. A student gets `school` instead. Night shift workers get a different `work` time slot. This is evaluated at prompt build time using the condition expressions.

If no `day_phases` defined in scenario YAML → engine uses sensible defaults based on timestep_unit.

#### Outcome Tracks (New)

**Current:** All outcomes are pre-defined categorical/boolean/float with fixed options.

**New:** Two tracks, inferred from schema.

**Track 1: Known Outcomes** — Options are pre-defined. Used when you know the decision space.

```yaml
outcomes:
  suggested_outcomes:
    - name: action
      type: categorical
      options: [pay_own_account, stay_shared_with_fee, cancel, switch_service, workaround]
      description: "What the user does about Netflix account"
      required: true
    - name: sentiment
      type: float
      range: [-1.0, 1.0]
      description: "How they feel about the change"
      required: true
```

Engine behavior: Pass 2 extracts exact category from options. Results show distributions directly.

**Track 2: Exploratory Outcomes** — No pre-defined options. Used when the decision space is unknown.

```yaml
outcomes:
  suggested_outcomes:
    - name: primary_response
      type: open_ended
      description: "What are you actually going to do? Be specific about concrete steps, resources, timeline, and obstacles."
      required: true
    - name: sentiment
      type: float
      range: [-1.0, 1.0]
      description: "Overall sentiment"
      required: true
```

Engine behavior: The agent writes a free-form response. No classification pass needed for this outcome. `extropy results` clusters the responses post-hoc to discover categories.

**Track 3: Hybrid** — Categorical bucket + open elaboration.

```yaml
outcomes:
  suggested_outcomes:
    - name: primary_adaptation_strategy
      type: categorical
      options: [aggressive_upskilling, occupational_pivot, double_down, disengage, collective_resistance]
      description: "Broad strategy category"
      required: true
    - name: elaboration
      type: open_ended
      description: "Describe specifically what you're planning. Concrete steps, resources, timeline, obstacles."
      required: true
    - name: sentiment
      type: float
      range: [-1.0, 1.0]
      required: true
```

Engine behavior: Categorical extracted via classification. Elaboration captured as free text. Results show both: "54% aggressive upskilling" as the headline, then clustered elaborations as the story.

**Inference rule:** If `type: open_ended` and no `options` → exploratory track. If `type: categorical` with `options` → known track. If both exist → hybrid. No explicit `outcome_mode` field needed.

#### Channel Experience Templates (New, Optional)

```yaml
channel_experience:
  mainstream_news_media:
    default: "I read about this in the news."
    variants:
      - when: "primary_news_source == 'television'"
        template: "I saw a news segment about this on TV."
      - when: "primary_news_source == 'social media'"
        template: "A news article showed up in my feed."
      - when: "digital_literacy == 'basic'"
        template: "Someone at work showed me a news article about this."
  social_media_feeds:
    default: "I noticed some posts about this online."
    variants:
      - when: "social_media_usage in ['heavy/multiple times daily']"
        template: "My feed has been nonstop about this."
      - when: "social_media_usage == 'moderate/daily'"
        template: "I saw several posts about this."
```

If omitted → engine uses sensible default templates. If provided → scenario-specific experiential rendering.

### 2.2 ScenarioSpec Model Changes

```python
class TimelineEvent(BaseModel):
    """A single event in the scenario timeline."""
    timestep: int
    event: Event
    exposure_rules: list[ExposureRule] | None = None  # If None, reuse seed_exposure rules
    description: str | None = None  # Human-readable context for this development

class DayPhase(BaseModel):
    """A single phase in a day template."""
    phase: str
    condition: str
    template: str
    slots: list[str]

class DayPhaseConfig(BaseModel):
    """Day phase templates, optionally keyed by timestep unit."""
    defaults: list[DayPhase] | None = None
    hourly: list[DayPhase] | None = None
    daily: list[DayPhase] | None = None
    weekly: list[DayPhase] | None = None
    monthly: list[DayPhase] | None = None

class ChannelVariant(BaseModel):
    when: str
    template: str

class ChannelExperience(BaseModel):
    default: str
    variants: list[ChannelVariant] = []

class ScenarioSpec(BaseModel):
    meta: ScenarioMeta
    event: Event                              # Initial event (t=0)
    timeline: list[TimelineEvent] | None = None  # NEW: subsequent developments
    seed_exposure: SeedExposure
    interaction: InteractionConfig
    spread: SpreadConfig
    outcomes: OutcomeConfig
    simulation: SimulationConfig
    day_phases: DayPhaseConfig | None = None      # NEW
    channel_experience: dict[str, ChannelExperience] | None = None  # NEW
    relationship_weights: dict[str, float] | None = None  # NEW
```

---

## PHASE 3: SIMULATION

### 3.1 Persona Generation — Rewrite

**Current:** "You are a 43-year-old male..." + grouped attribute bullet points.

**New:** First-person identity with named family, household context, and economic reality.

```
I'm Travis McAllister. I'm 43, white, living in Greenville, South Carolina with my
partner Lisa (41, retail) and our three kids — Tyler (17, about to graduate high school),
Kayla (14), and Mason (9). I work full-time in the service industry. Our household income
is about $52,000. We've got maybe one month of savings.

My Mindset & Values
- Neuroticism: High
- Extraversion: Moderate
- Openness: Low
- Conscientiousness: Moderate
- Agreeableness: Moderate

My Attitudes & Concerns
- Institutional Trust: Low
- AI Threat Perception: High
- Economic Anxiety: Severe
- Technology Adoption: Low
...
```

Key changes:
- First person ("I'm Travis" not "You are a 43-year-old")
- Partner named and described (from partner agent's attributes)
- Kids named with ages and school status (from dependent metadata)
- Household economic context (shared income, savings)
- All remaining attributes still listed (structured format, nothing filtered)

### 3.2 Timestep Loop — Redesigned

**Current:** Expose → Reason (Pass 1 + Pass 2) → Propagate → Aggregate → Stop check.

**New:** Four-phase timestep with conversations.

```
Phase 1: EXPOSURE + CONTEXT BUILD
  - Apply seed exposures for this timestep
  - Apply timeline events if any scheduled for this timestep
  - Propagate network exposures from previous timestep's sharers
  - Build reasoning context for each agent (persona, exposures, memory, peers, mood, temporal)

Phase 2: REASONING (parallel, all agents)
  - All aware agents reason in parallel
  - Single merged LLM call (no separate Pass 1 + Pass 2)
  - Output: internal monologue, sentiment, conviction, public statement,
    position, outcomes, elaboration, actions (talk_to, post)
  - State from this phase is PROVISIONAL — conversations can override

Phase 3: CONVERSATIONS (parallel across pairs, sequential within conflicts)
  - Engine collects all talk_to actions from Phase 2
  - Builds conversation queue:
    - Priority by edge weight (partner > close_friend > coworker > acquaintance)
    - Non-conflicting pairs run in parallel
    - Conflicting pairs (share an agent) run sequentially by priority
  - Each conversation: 2-3 turns, both agents in character
    - Agent-agent: both sides are real agents, both states update
    - Agent-NPC: NPC side generated by LLM from NPC profile, only agent state updates
  - Conversation output: updated sentiment, conviction, position for BOTH participants
  - This OVERRIDES Phase 2 provisional state for agents who conversed

Phase 4: STATE UPDATE + AGGREGATION
  - Social posts collected and stored (feeds into next timestep's public discourse)
  - Final state written for all agents (Phase 3 output if conversed, Phase 2 output if not)
  - Sharing decisions computed (mechanical, based on will_share + spread config)
  - Timestep summary computed (exposure rate, position distribution, sentiment stats)
  - Stopping conditions evaluated
  - Checkpoint if needed
```

### 3.3 Prompt Structure — Complete Rewrite

**Current prompt:**
```
[System: You ARE this person]
[Persona: attribute bullet points]
[Event: content block]
[Exposures: "Someone told you" x17]
[Memory: 3 summaries]
[Peer opinions: "A acquaintance says..."]
[Instructions: respond as JSON]
```

**New prompt (medium fidelity, weekly timestep):**

```
You are going to think as Travis McAllister. Everything below is from Travis's
perspective. Respond as Travis — first person, honest, unfiltered.

---

I'm Travis McAllister. I'm 43, white, living in Greenville, SC with my partner
Lisa (41, retail) and our three kids — Tyler (17, about to graduate), Kayla (14),
and Mason (9). I work full-time in services. Household income ~$52K. About one
month of savings.

[Full characteristics list]

---

It's Week 3 since AI systems were announced to have superhuman performance across
all cognitive domains.

What's happened so far:
- Week 1: AI systems publicly demonstrated superhuman performance. Fortune 50 firms
  announced restructuring with major layoffs. Federal guidance unclear.
- Week 3: Government announced emergency task force but no concrete policy.
  Unemployment claims spiked 40%.

This week:
- I saw this on CNN and it was all over my X feed
- 23 people in my circle brought it up this week, including my coworker Darnell
  and my neighbor Marcus
- This is the 3rd week in a row everyone around me is talking about this

What people around me are saying:
- My partner Lisa (41, retail): "We need to be realistic about our savings. I'm
  scared." — she's very worried and firm about cutting costs
- My coworker Darnell (36, services): "I signed up for some free AI tutorials.
  Can't hurt." — he's anxious but doing something about it
- My neighbor Marcus (42, construction): "My job's physical, maybe I'm okay?"
  — he's uncertain, trying to convince himself
- Most people I know are anxious. The mood hasn't improved since last week.
  If anything, the layoff numbers made it worse.

What I've been thinking:
- Week 1: "I'm terrified. One month of savings. Five mouths to feed. I need to
  cut spending NOW and find backup work."  I was panicked and certain we were in trouble.
- Week 2: "Same as last week but heavier. Lisa's right — can't afford courses.
  Looking at gig apps. The anger is fading into something worse." Still anxious,
  getting more resigned.

I've been feeling panicked since this started. Last week the panic settled into
a heavy dread that hasn't lifted. I've been firm about survival-first since Week 1
and that hasn't wavered.

---

People in my life right now:
- Lisa (my partner, she's home with me)
- Tyler, Kayla, Mason (my kids, they're home)
- Darnell (my coworker, I'll see him at work)
- Marcus (my neighbor, I could text him)

Think honestly about how this week's developments land for someone in your exact
situation. What are you actually thinking? What do you feel? What are you going to do?

You can choose to:
- Talk to someone (pick 1-2 people from the list above)
- Post something on social media (or choose not to)

Respond as JSON:
{
  "internal_monologue": "Raw, honest stream of thought...",
  "sentiment": -0.8,
  "conviction": 75,
  "public_statement": "What I'd actually say out loud to people",
  "position": "your primary response to this situation",
  "elaboration": "Specifically what you're planning to do — concrete steps, resources, timeline, obstacles",
  "actions": [
    {"type": "talk_to", "who": "Lisa", "topic": "..."},
    {"type": "post", "platform": "x", "content": "..."}
  ]
}
```

### 3.4 Merged Pass 1 + Pass 2 (With Caveat)

**Why we originally split into 2 passes:** Single-pass role-play + classification caused 83% central tendency — agents hedged toward safe middle options because "be this person" and "pick from these categories" competed in the same generation.

**Why merging might work now:** The v2 merged pass is structured differently. The agent reasons freely FIRST (unconstrained monologue), then fills in structured fields AFTER — within the same generation. The classification is downstream of the reasoning, not competing with it. The reasoning comes first and informs the structured extraction.

**This is a hypothesis that needs testing.** A/B test: run 200 agents with merged pass vs 2-pass, compare outcome distributions. If merged produces the same spread as 2-pass, keep it (saves 1 round trip per agent, ~50% fewer LLM calls). If it collapses to center, revert to 2-pass. The architecture supports both — the engine just needs a flag.

**Note on Pass 2 context:** The current Pass 2 already receives the Pass 1 reasoning text (not completely disconnected). What it lacks is the full persona/exposures/peers — but the reasoning already reflects all of that. The real win from merging is latency reduction and cost savings, not fixing a context gap.

The single call produces:
- Free-form reasoning (internal_monologue, elaboration)
- Structured outputs (sentiment, conviction, position, outcomes)
- Actions (talk_to, post)

For **known outcome** scenarios, position must match one of the defined options. The prompt includes the options explicitly:

```
"position": one of ["pay_own_account", "cancel", "switch_service", "workaround"]
```

For **exploratory outcome** scenarios, position is free-form:

```
"position": "describe your primary response in a few words"
```

Classification into categories happens in `extropy results`, not during simulation.

### 3.5 Conversation System

When an agent's actions include `talk_to`, the engine resolves it:

**Who the agent talks to** is LLM-driven — the agent sees their available contacts and picks who's relevant based on what THEY need right now. Travis might choose Darnell over Lisa tonight because he needs field-specific intel about automation in services, not a budget argument. The LLM makes this choice in context, so it's naturally topic-aware.

**Conflict resolution** (when two people want to talk to the same person) uses scenario-defined `relationship_weights` × structural edge weight. These weights are auto-generated by `extropy scenario` (the compiler knows ASI is a workplace threat → coworker weight high, Netflix is a household product → family weight high) and can be overridden in the YAML.

```
1. Collect all talk_to requests across all agents
2. Build pairs: (initiator, target, topic, priority_score)
   priority_score = structural_edge_weight × scenario_relationship_weight
3. Sort by priority_score descending
4. Identify conflicts (same agent appears in multiple pairs)
5. Non-conflicting pairs → parallel execution
6. Conflicting pairs → sequential execution, highest priority first

For each conversation pair:
  Turn 1: Initiator speaks (LLM call with initiator's context + topic)
  Turn 2: Target responds (LLM call with target's context + initiator's statement)
  Turn 3: Initiator replies (LLM call with updated context)

  Output for EACH participant:
  {
    "response": "What they said",
    "updated_sentiment": float,
    "updated_conviction": int,
    "updated_position": "if changed",
    "internal_reaction": "What they actually thought during this conversation"
  }
```

**Agent-NPC conversations:** When Travis talks to Tyler (NPC kid), the engine generates Tyler's side using a system prompt built from Tyler's NPC profile:

```
You are Tyler McAllister, 17 years old, a high school senior living with your dad
Travis (43, services) and mom Lisa (41, retail) in Greenville, SC. You're about
to graduate. Respond naturally as a teenager would to your dad.
```

Tyler's response is LLM-generated but Tyler has no persistent state. Only Travis's state updates.

**Cost:** ~3 LLM calls per conversation (fast model). At medium fidelity, only the top-1 edge (partner/closest) gets a conversation per timestep. At high fidelity, top 2-3 edges.

**Conversation vs mechanical rules:** For agents who HAD a conversation this timestep, the conversation output IS their final state. Bounded confidence, flip resistance, conviction decay do NOT apply on top. These mechanical rules exist to approximate social influence when no conversation happens. For agents who did NOT converse, the mechanical rules still apply as the "passive influence" layer — the effect of scrolling past posts, overhearing things, seeing opinions without engaging.

### 3.6 Aggregate Mood Rendering

**Current:** TimestepSummary computed but never shown to agents.

**New:** Aggregate mood from the agent's LOCAL NETWORK rendered as fuzzy vibes.

```python
def render_local_mood(agent_id: str, adjacency: dict, agent_states: dict) -> str:
    """Render the mood of an agent's local network as natural language."""
    neighbors = adjacency[agent_id]
    sentiments = [agent_states[n].sentiment for n, _ in neighbors if agent_states[n].sentiment is not None]

    if not sentiments:
        return ""

    avg = sum(sentiments) / len(sentiments)
    variance = sum((s - avg) ** 2 for s in sentiments) / len(sentiments)

    # Mood label
    if avg > 0.6: mood = "optimistic"
    elif avg > 0.2: mood = "cautiously hopeful"
    elif avg > -0.2: mood = "uncertain and mixed"
    elif avg > -0.6: mood = "anxious and worried"
    else: mood = "deeply fearful"

    # Consensus
    if variance < 0.05: consensus = "Everyone seems to feel the same way."
    elif variance < 0.15: consensus = "Most people feel similarly."
    else: consensus = "Opinions are all over the place."

    return f"Most people I know seem {mood}. {consensus}"
```

No numbers. No pie charts. Just vibes — the way real humans sense the mood around them.

### 3.7 Temporal Awareness

Every prompt includes:
- **Current position in time:** "It's Week 3 since the announcement."
- **Timeline recap:** Bullet list of what's happened so far (from scenario timeline).
- **Memory timestamps:** "Week 1: I thought... Week 2: I thought..."
- **Exposure duration:** "This is the 3rd week in a row people are talking about this."
- **Emotional trajectory:** "I've been anxious since Week 1. Last week the panic settled into dread."

All deterministic string formatting from existing data. Zero LLM calls.

### 3.8 Memory System

**Current:** 3-entry sliding window of 1-sentence summaries.

**New:** Full reasoning history, timestamped, with emotional context.

- ALL reasoning entries kept (not capped at 3)
- Each entry shows: timestep label, truncated reasoning (first 3-4 sentences of raw_reasoning), emotional state at the time
- Conviction trajectory rendered: "I've been firm about this since Week 1" or "I started certain but I've been wavering"
- If token budget is a concern: at `--fidelity low`, show last 3 full traces. At `medium`, show all. At `high`, show all + consolidated beliefs.

### 3.9 Social Posts + Public Discourse

When an agent's actions include `post`:
- The post content is stored with the agent's ID, timestep, and platform
- Next timestep, the engine aggregates all posts into a public discourse summary:
  - "My X feed is a mix of panic and dark humor. Most posts are about job security."
  - "People on Reddit are sharing workaround guides."
- The summary is rendered into prompts as part of the `social_media_exposures` slot
- Individual posts are NOT shown to other agents (too many) — only the aggregate mood of the platform

For network contacts specifically: if Darnell posted on X AND Darnell is in Travis's network, Travis might see "Darnell posted on X: '...'" as a named peer exposure. This is higher influence than the anonymous aggregate.

### 3.10 Cognitive Architecture (Tiered by Fidelity)

Split into independent subsystems. Each assessed for actual value vs implementation cost.

**Tier 1: Build (trivial, high impact — string formatting from existing data)**

- **3.10a: Emotional trajectory rendering.** Map sentiment history to narrative: "I started panicked. By mid-week it settled into dread. It hasn't lifted." Deterministic lookup from sentiment values + trend. Gives the LLM emotional continuity between timesteps instead of starting fresh every time. Zero cost.

- **3.10b: Conviction self-awareness.** "I've been firm about this since Week 1" or "I started certain but I've been getting less sure." Deterministic from conviction history. Enables commitment bias (consistent agents resist change) and openness (wavering agents are more receptive). Zero cost.

**Tier 2: Build at high fidelity (medium effort, good value)**

- **3.10c: Internal monologue vs external action (THINK vs SAY).** Schema change — output includes both `internal_monologue` (raw, honest) and `public_statement` (socially filtered). Replaces the mechanical public/private split with agent-generated divergence. An agent with high agreeableness might have a large gap between what they think and what they say — that's interesting data. Schema change only, no new system.

- **3.10d: Repetition detection + forced deepening.** If string overlap between consecutive reasonings > 70%, inject a prompt nudge: "You've been thinking the same thing for several days. Has anything actually changed? Are you starting to doubt your plan? Have you actually done anything about it?" Simple string comparison, no embeddings needed. Prevents the stale convergence we saw in the ASI run ("No change — save, learn AI, backup income" × 5 timesteps). Without this, agents converge to identical outputs and the sim produces meaningless duplicate reasoning.

**Tier 3: Build at high fidelity (medium effort, marginal value over full traces)**

- **3.10e: Episodic vs semantic memory.** After N timesteps of consistent reasoning on a theme, engine extracts a belief statement and adds to persistent "beliefs" field. Shown as "Things I've come to believe:" separate from "What I thought recently." Extraction is rule-based: if the same keywords appear in 3+ consecutive reasonings, consolidate into a belief. Marginal value because the LLM already consolidates beliefs implicitly when reading its own full history — making it explicit is a nice-to-have.

**CUT: Not building**

- **~~3.10f: Attention/focus weighting.~~** ~~What the agent is currently focused on determines which inputs are foregrounded in the prompt.~~ **Cut.** The LLM already does this natively — if the memory trace says "budget first," the model naturally attends to budget-related inputs. Artificially weighting prompt sections is trying to replicate what attention heads already do. Unnecessary complexity.

**DEFERRED: Post-launch**

- **3.10g: Spontaneous memory recall.** Memories surface by RELEVANCE to current events, not recency. Requires embedding stored memories and comparing to current context. Small embedding model on short text, cheap per-call, but needs embedding infrastructure. **Deferred because:** for most scenarios with 10-15 timesteps, recency and relevance overlap heavily — the memory that's relevant IS usually recent. This matters more for long-running sims (50+ timesteps) where early memories might be contextually triggered. Build it when we actually have that use case.

**Fidelity tier mapping:**
| Feature | low | medium | high |
|---------|-----|--------|------|
| 3.10a emotional trajectory | Yes | Yes | Yes |
| 3.10b conviction self-awareness | Yes | Yes | Yes |
| 3.10c THINK vs SAY | No | No | Yes |
| 3.10d repetition detection | No | No | Yes |
| 3.10e episodic/semantic memory | No | Yes | Yes |
| ~~3.10f attention weighting~~ | — | — | **Cut** |
| 3.10g spontaneous recall | No | No | **Deferred** |

---

## PHASE 4: RESULTS + ANALYSIS

### 4.1 `extropy results` — Enhanced

**Current:** Outcome distributions, segment breakdowns, timeline visualization.

**New:** Same, plus elaboration clustering for exploratory outcomes.

#### Known Outcome Results (Same as Current)

```
Position Distribution:
  pay_own_account: 43.2%
  cancel: 12.1%
  switch_service: 18.7%
  workaround: 15.3%
  stay_shared_with_fee: 10.7%

Segment: age_bracket
  18-25: cancel 22%, workaround 31%
  26-35: pay_own 48%, switch 23%
  36-50: pay_own 52%, stay_shared 18%
  50+: pay_own 61%, cancel 8%
```

#### Exploratory Outcome Results (New)

For `type: open_ended` outcomes, `extropy results` runs post-hoc clustering:

```
1. Collect all elaboration texts (2000 agents × 1 elaboration each)
2. Embed all elaborations (small embedding model, batch)
3. Cluster (k-means or HDBSCAN on embeddings)
4. For each cluster:
   - LLM-generated label (1 fast-model call per cluster): "Survival-first: cut costs, gig work, no courses"
   - Representative quotes (closest to centroid)
   - Demographic breakdown of cluster members
   - Size as % of population
```

Output:
```
Emergent Response Patterns (from 2000 open-ended elaborations):

Cluster 1 (34%): "Immediate financial survival"
  Cut spending, gig work, hoard cash. No courses, no upskilling.
  Skews: lower income, service sector, older, dependents
  Representative: "I'm looking at gig apps. Maybe drive nights. Can't afford
  courses. Lisa's right — cash first."

Cluster 2 (28%): "Self-directed digital upskilling"
  Free online resources, YouTube tutorials, Google certificates.
  Skews: younger, moderate digital literacy, some savings buffer
  Representative: "Started a free Python course. Doing it after the kids
  sleep. 3 months to get something on my resume."

Cluster 3 (18%): "Career pivot to human-essential work"
  Trades, healthcare, care work — jobs requiring physical presence.
  Skews: male, manual labor adjacent, moderate age
  Representative: "My cousin does HVAC. Always needs people. AI can't fix
  your AC. I'm calling him Monday."

Cluster 4 (11%): "Political mobilization"
  Demand UBI, protest, organize, contact representatives.
  Skews: high political activation, strong liberal or strong conservative,
  younger, urban
  Representative: "We need to be in the streets. $500 retraining voucher is
  an insult. They're giving billions to the companies doing the firing."

Cluster 5 (9%): "Psychological withdrawal"
  Denial, disengagement, depression, fatalism.
  Skews: high neuroticism, low agency, already economically vulnerable
  Representative: "I don't know what to do. Nothing I learn matters if the
  machine is better. I just go to work and try not to think about it."
```

This is the real value of exploratory outcomes — categories nobody would have pre-defined. "Career pivot to human-essential work" as 18% of responses is a finding. "Psychological withdrawal" at 9% is a finding. You'd never put these in a dropdown.

#### Conversation Analysis (New)

```
Conversation Summary:
  Total conversations: 3,847 (across 12 timesteps)
  Average turns: 2.4
  State changes from conversations: 891 (23% of conversations changed someone's mind)

Most impactful conversations (by state change magnitude):
  1. Travis ↔ Lisa (Week 1): Travis shifted from "upskilling" to "survival-first" after budget discussion
  2. Graham ↔ Sandra (Week 1): Both solidified "raise prices 10%" after doing the math together
  ...

Conversation themes (clustered):
  - Financial planning (34%): Budget discussions, cost-cutting
  - Emotional support (28%): Reassurance, shared anxiety
  - Information sharing (22%): "Did you see...", "Have you heard..."
  - Disagreement (16%): Different coping strategies, arguments
```

---

## FIDELITY TIERS

`--fidelity low|medium|high` controls what goes into prompts and whether conversations happen.

| Feature | **low** | **medium** | **high** |
|---------|---------|-----------|----------|
| Names & household data | Yes | Yes | Yes |
| Temporal awareness | Yes | Yes | Yes |
| Aggregate mood (local) | Yes | Yes | Yes |
| Named peer opinions | Top 5 | Top 10 + consensus signal | All connected + demographics |
| Day phase templates | Yes (if defined) | Yes | Yes |
| Channel experience | Default templates | Scenario-defined variants | Full demographic adaptation |
| Memory | Last 3 full reasoning traces | All traces, timestamped | All + consolidated beliefs |
| Emotional trajectory | Label only | Label + trend | Full trajectory narrative |
| Conversations | None | Top 1 edge (partner/closest) | Top 2-3 edges |
| Internal monologue | Not separated | Not separated | Explicit THINK vs SAY |
| Repetition detection | No | No | Yes |
| Social posts | Stored, not rendered | Stored + aggregate | Stored + aggregate + named peer posts |
| Scenario timeline | Yes | Yes | Yes |
| Pass structure | Merged (test vs 2-pass) | Merged (test vs 2-pass) | Merged (test vs 2-pass) |

### Cost Estimates (Conservative)

**Per-call token budgets:**

| | Input tokens | Output tokens |
|---|---|---|
| **low** reasoning | ~2.5k | ~500 |
| **medium** reasoning | ~3.5k | ~600 |
| **medium** conversation turn | ~2k | ~200 |
| **high** reasoning | ~4.5k | ~700 |
| **high** conversation turn | ~2k | ~200 |

**Total cost (5-mini: $0.15/$0.60 per 1M tokens, 15 timesteps):**

| | 2k agents | 10k agents |
|---|---|---|
| **Current system** | ~$8 | ~$40 |
| **low** | ~$18 | ~$90 |
| **medium** | ~$30 | ~$150 |
| **high** | ~$44 | ~$220 |

**Total cost (Sonnet-class: ~$3/$15 per 1M tokens, 15 timesteps):**

| | 2k agents | 10k agents |
|---|---|---|
| **low** | ~$360 | ~$1,800 |
| **medium** | ~$600 | ~$3,000 |
| **high** | ~$900 | ~$4,500 |

**Wall time (1k RPM, conservative):**

| | 2k agents | 10k agents |
|---|---|---|
| **Current system** | ~30 min | ~2.5 hrs |
| **low** | ~45 min | ~4 hrs |
| **medium** | ~1.2 hrs | ~6 hrs |
| **high** | ~1.5 hrs | ~8 hrs |

**Default:** `medium`. Best cost/quality tradeoff. Names, narrative, temporal awareness, full memory, the one conversation that matters most (partner), aggregate mood. ~$150 for 10k agents on 5-mini. ~3.5x current cost for a fundamentally better simulation.

---

## VALIDATION & EVALUATION PLAN

### Output Quality

1. **A/B comparison:** Run same population + scenario through old system and new system (medium fidelity). Blind human eval on 50 agent outputs: "Which reads more like a real person's reasoning?" Measure win rate.

2. **Outcome distribution stability:** Run same scenario at low/medium/high fidelity. Outcome distributions should be SIMILAR — fidelity controls richness of reasoning, not WHAT people decide. If distributions diverge significantly across tiers, the richer prompts are causing systematic bias.

3. **Hallucination audit:** Sample 100 agent outputs. Check every factual claim against the prompt context. Agents should not invent information not in their exposures/memory/peers. Richer prompts = more grounding = less hallucination expected, but verify.

### Conversation Quality

4. **In-character consistency:** Sample 50 conversations. Both agents should stay in character (demographics, personality, relationship). Neither should suddenly become eloquent if their persona is low-education.

5. **State change plausibility:** For conversations that changed an agent's position, verify the change makes sense given what was said. "Lisa convinced Travis to focus on cash" should show Lisa making a compelling financial argument, not Travis randomly flipping.

### Elaboration Quality

6. **Scenario awareness:** Elaborations should be contextually appropriate. ASI scenario should NOT produce "I'll take an online course" if the premise is superhuman AI. Netflix scenario should NOT produce existential crisis responses.

7. **Demographic consistency:** An agent with `digital_literacy: basic` should not describe a plan involving "fine-tuning open-source models." Elaborations should reflect the agent's actual capabilities and constraints.

### Clustering Validation

8. **Cluster coherence:** For exploratory outcomes, verify clusters are semantically meaningful. Silhouette scores on embeddings. Human review of cluster labels vs representative samples. Bad clusters = too heterogeneous or too small.

---

## IMPLEMENTATION ORDER

### Phase A: MVP (Issues 2, 4, 8) — ~1 week

Changes: `persona.py`, `reasoning.py` only. Zero new subsystems.

- Agent names (SSA + census data, bundled CSV)
- Temporal awareness (timestep + unit in prompt, timestamped memory)
- Full memory (remove 3-entry cap, surface raw_reasoning)
- First-person prompt voice ("I'm Travis" not "You are a 43-year-old")

Ship this alone. Every simulation immediately feels more human.

### Phase B: Social Role Network + Named Peers (Issues 1, 6) — ~2 weeks

Changes: `sampler/core.py` (household sampling), `network/generator.py` (structural edges), `reasoning.py` (named peer rendering).

- Household-based sampling
- Structural edge generation (partner, coworker, neighbor, etc.)
- Named peer opinions with relationship context and conviction
- Dependent generation (NPC kids)

### Phase C: Scenario Timeline + Merged Pass (Issues timeline, 11) — ~1 week

Changes: `models/scenario.py`, `engine.py`, `reasoning.py`.

- Timeline events in scenario YAML
- Timeline injection at specified timesteps
- Merged Pass 1 + Pass 2 into single call
- Outcome track inference (known vs exploratory)

### Phase D: Conversations + Narrative Prompts (Issues 13, 10) — ~2 weeks

Changes: `engine.py` (conversation phase), new `conversation.py`, `reasoning.py` (day phase rendering).

- Conversation resolution system (priority queue, parallel/sequential)
- Agent-agent and agent-NPC conversations
- Day phase templates (optional)
- Channel experience rendering
- Social posts + public discourse

### Phase E: Cognitive Architecture (Issue 14) — ~2 weeks

Changes: `reasoning.py`, `engine.py`, `persona.py`.

- Tier 1 (easy): emotional trajectory, conviction self-awareness (all fidelity levels)
- Tier 2 (high fidelity): THINK vs SAY separation, repetition detection
- Tier 3 (high fidelity): episodic/semantic memory consolidation
- ~~Attention/focus weighting~~: **Cut** (LLM does this natively)
- Spontaneous memory recall: **Deferred** to post-launch

### Phase F: Aggregate Mood + Fidelity Tiers (Issues 7, fidelity) — ~1 week

Changes: `engine.py`, `reasoning.py`, CLI flag.

- Local network mood rendering
- `--fidelity` flag controlling prompt assembly
- Public discourse aggregation

### Phase G: Results Enhancement — ~1 week

Changes: `results/` module.

- Elaboration clustering for exploratory outcomes
- Conversation analysis
- Enhanced segment breakdowns

**Total estimated: ~10 weeks for full v2.**

Phase A alone (1 week) is a massive improvement with zero risk. Each subsequent phase is independently shippable and testable.
