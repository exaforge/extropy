# Simulation Agent Experience — Gap Analysis

## Overview

The current simulation treats agents as reasoning machines processing bullet-point data sheets. The goal is for agents to experience each timestep as a lived day — with named people, temporal awareness, social context, and narrative texture. This document catalogs every gap between what exists and what's needed.

---

## ISSUE 1: Population ↔ Network Disconnect

**What exists:**
- Population sampling produces agents with: `marital_status`, `household_size`, `household_labor_dependency`, `state`, `occupation_sector`, `urban_rural`, `age`, `gender`, `race_ethnicity`, `religious_affiliation`
- Network generation creates edges based on attribute similarity (embedding distance)
- Edge types are generic: "acquaintance", "neighbor" — derived from similarity buckets, not social roles

**What's broken:**
- Agent with `marital_status: cohabiting` and `household_size: 5` has NO partner edge in the network
- Two agents in the same `state` + same `occupation_sector` are not linked as coworkers
- Two agents in the same `state` + `urban_rural: suburban` with similar `age` children aren't linked as "school parent friends"
- Network edge types ("acquaintance", "neighbor") are assigned by similarity threshold, not by actual social relationship inference
- An agent's strongest influence (their partner) might not even be in their network at all

**What population attributes COULD drive network edges but don't:**

| Attribute | Could create edge type |
|-----------|----------------------|
| `marital_status` + `household_size` + `state` | partner (match two cohabiting/married agents in same state, compatible age/gender) |
| `occupation_sector` + `state` | coworker |
| `state` + `urban_rural` + similar `age` | neighbor |
| `religious_affiliation` + `state` | congregation member |
| kids' implied school age + `state` + `urban_rural` | school parent network |
| `political_orientation` + `state` | political community |
| `social_media_usage` + shared interests | online contact |

**What this means:** The network is currently a homophily graph (similar people connect). It should be a **social role graph** (people connect because they're partners, coworkers, neighbors, etc.). The attributes to do this already exist in the population. They're just ignored during network generation.

---

## ISSUE 2: Agents Have No Names

**What exists:**
- Agent ID is `agent_0020`
- No first name, last name, or any human identifier
- Nowhere in the pipeline (sampling, persona, network, simulation) does a name get generated
- The persona prompt says "You are a 43-year-old male" — no name

**What's broken:**
- Agent can't think of themselves as a person with a name
- Agent can't refer to peers by name — only "someone in your network" or "an acquaintance"
- The prompt feels like a survey, not a lived experience
- Names carry cultural signal — "Travis" vs "Darnell" vs "Alejandro" vs "Wei" tells you something about the person. The LLM uses this for contextual reasoning.

**What's needed:**
- Name generation based on demographics: `gender`, `race_ethnicity`, `age` (generational naming patterns), `religious_affiliation`, `state` (regional patterns)
- Names assigned at sampling time or persona generation time
- Stored as agent attribute (`first_name`) so it persists across the pipeline
- Partner/family member names also generated with demographic consistency (same cultural background)

**Complication:**
- Name generation needs to be statistically appropriate — not stereotyping, but reflecting actual name frequency distributions by demographic cohort
- Could use census/SSA baby name data filtered by decade of birth + ethnicity
- Or use an LLM call at persona generation time to pick a culturally appropriate name
- 2000 agents = 2000 names needed. LLM call per agent is expensive. Census data approach is better.

---

## ISSUE 3: Family Members Don't Exist

**What exists:**
- `household_size: 5` — a number
- `marital_status: cohabiting` — implies a partner
- `household_labor_dependency: one other` — implies partner works
- No children attributes, no parent attributes, no dependent details

**What's broken:**
- Travis has 3 kids but they have no names, ages, genders, or school status
- His partner is either another agent (not linked) or doesn't exist in the sim at all
- When Travis reasons about ASI, he can't think "Tyler's about to graduate into this" because Tyler doesn't exist
- The persona says "household_size: 5" but the LLM has to imagine who those 4 other people are every single time, inconsistently

**What's needed:**
- **Partner linking:** If `marital_status` is married/cohabiting AND another agent in the population matches (compatible demographics, same state), link them as partner edge. This partner IS a full agent with their own reasoning.
- **Dependent generation:** Based on `household_size` minus adults, generate NPC children/dependents:
  - Number of kids = `household_size` - (1 if single, 2 if partnered)
  - Ages inferred from agent's age + life_stage (middle_aged_adult with household_size 5 → kids probably 8-17)
  - Names from same cultural background as parent
  - School status from age (5-17 = in school, grade inferred)
  - These are NOT agents. They're persona context attached to the parent agent.
- **Elder dependents:** If `household_labor_dependency` implies elderly parent, generate that too

**Where this lives:** Generated during `extropy persona` or at simulation init (`generate_persona()`). Stored as part of the agent's persona text or as structured metadata.

---

## ISSUE 4: No Temporal Awareness

**What exists:**
- `timestep` integer tracked everywhere internally
- `timestep_unit` defined in scenario YAML (e.g., "hour", "day", "week")
- Memory traces have `timestep` field but it's never shown in prompt
- Exposures have `timestep` but never shown

**What's broken:**
- Agent doesn't know what day it is
- Agent doesn't know how long ago they first heard about the event
- Agent can't reason about urgency ("it's been a week and still no government response")
- Agent can't reason about fatigue ("I've been hearing about this for days")
- Agent can't reason about trajectory ("things are getting worse" vs "things are stabilizing")
- Memory traces say "Previously, you thought:" with no temporal anchor — could be 1 hour ago or 1 month ago

**What's needed:**
- Current timestep + unit injected into prompt: "It is Day 3 since the announcement"
- Memory traces timestamped: "2 days ago, you thought: ..."
- Exposure history timestamped: "On Day 1, you saw this on the news. On Day 2, 17 people in your network talked to you about it."
- Optional: scenario can define a "calendar" — e.g., "Day 1 = Monday March 3, 2026" so agents reason about weekdays, work schedules, etc.

---

## ISSUE 5: Exposures Are Meaningless Lines

**What exists:**
- ExposureRecord has: `channel`, `source_agent_id`, `credibility`, `timestep`
- Scenario defines channels with: `name`, `description`, `reach`, `credibility_modifier`
- Network exposures carry `source_agent_id` and edge relationship type

**What the prompt shows:**
```
- You heard about this via mainstream_news_media
- Someone in your network told you about this
- Someone in your network told you about this
(x17)
```

**What's broken:**
- "You heard about this via mainstream_news_media" — what does that mean experientially? Did you watch CNN? Read the NYT on your phone? See a push notification?
- "Someone in your network told you about this" x17 is noise. The agent can't distinguish 1 person from 100.
- Channel descriptions exist in the scenario but never reach the prompt
- Credibility scores exist but are invisible
- No aggregation — "37 people in your network discussed this with you today" vs 37 identical lines

**What's needed:**
- Channel descriptions rendered as experiential moments: "You saw a CNN segment about this" / "Your X feed was full of posts about this" / "A coworker brought it up at lunch"
- Network exposures aggregated and named: "12 people in your network brought this up today, including your coworker Marcus and your neighbor Denise"
- Credibility rendered as trust framing: "a highly credible news report" vs "a social media post" vs "a rumor from a friend of a friend"
- Exposure count surfaced as social pressure signal: "This is the 4th day in a row people around you are talking about this"

---

## ISSUE 6: Peer Opinions Are Anonymous and Shallow

**What exists:**
- PeerOpinion has: `agent_id`, `relationship`, `public_statement`, `sentiment`, `credibility`
- Engine gets top 5 neighbors by adjacency
- Edge type stored as `relationship` field

**What the prompt shows:**
```
- A acquaintance of yours says: "Don't wait — cut costs..."
- A acquaintance of yours seems strongly opposed
```

**What's broken:**
- "A acquaintance" — no name, no context, no demographic info
- Same relationship type for everyone — almost all edges are "acquaintance" because the network doesn't do social role assignment
- No peer conviction shown — agent can't tell if the peer is certain or wavering
- Only 5 peers shown — agent with 50+ connections sees a tiny slice
- No disagreement/consensus signal — if all 5 say the same thing, the agent doesn't know if that's representative
- No peer demographic context — "your 54-year-old coworker who's been in services for 30 years" hits different than "some acquaintance"

**What's needed:**
- Named peers with relationship context: "Your coworker Darnell (36, also in services) says: ..."
- Peer conviction visible: "... (he's absolutely certain about this)"
- More than 5 peers OR aggregated summary: "Most of your close contacts are anxious and talking about practical survival steps"
- Disagreement flagged: "Your contacts are split — some are panicking, others think it's overblown"
- Peer demographics matter: "Your neighbor Lisa, who works in healthcare, isn't as worried — she thinks her job is safe"

---

## ISSUE 7: No Aggregate / Public Sentiment

**What exists:**
- `TimestepSummary` stores: `average_sentiment`, `average_conviction`, `sentiment_variance`, `position_distribution`, `exposure_rate`
- Computed every timestep in `compute_timestep_summary()`
- Used for stopping conditions and post-hoc analysis only

**What the prompt shows:**
- Nothing. Zero aggregate data reaches the agent.

**What's broken:**
- Agent can't reason about social proof ("everyone's panicking" vs "I seem to be the only one worried")
- Agent can't reason about consensus ("the general mood is..." vs "opinions are divided")
- Agent can't reason about trend ("things are calming down" vs "panic is growing")
- In reality, people sense the aggregate mood through media, social feeds, workplace vibe

**What's needed:**
- Aggregate sentiment framed as vibes, not numbers. No agent walks around thinking "sentiment is -0.91." They think "everyone seems scared" or "people are weirdly calm about this" or "my feed is a mess." The rendering must be fuzzy and experiential:
  - NOT: "The general mood is -0.91"
  - YES: "Everyone around you seems deeply anxious. The mood at work is tense. Your social media feed is full of people panicking."
- Position distribution as vague social awareness, not pie charts:
  - NOT: "53% are upskilling, 17% doubling down"
  - YES: "Most people you talk to are scrambling to learn new skills. A few are stubbornly sticking to their current path."
- Trend as gut feeling, not data:
  - NOT: "Sentiment decreased 0.03 since yesterday"
  - YES: "The panic seems to be leveling off" or "If anything, people are getting more scared, not less"
- Source should be the agent's **local network neighborhood**, not global population — people sense mood from who they actually interact with, not from omniscient aggregate data

---

## ISSUE 8: Memory is Too Thin and Unlabeled

**What exists:**
- 3 memory entries max per agent (sliding window)
- Each entry: `timestep` (int), `sentiment` (float), `conviction` (float), `summary` (1 sentence)
- `raw_reasoning` stored in agent_states but never shown back to agent

**What the prompt shows:**
```
- Previously, you thought: "No change — save, learn AI, backup income." (you felt *firm* about this)
(x3, nearly identical)
```

**What's broken:**
- 3 entries is too few — by day 7, the agent has lost memory of days 1-4
- No timestamps — "Previously" could mean yesterday or last month
- Only summary shown, not the actual reasoning
- Sentiment not shown — agent can't see "I was more negative yesterday than today"
- No sense of consistency or change — "I've thought the same thing for 5 days" is meaningful but invisible
- The summaries converge to identical text because the agent keeps saying the same thing

**What's needed:**
- **Full reasoning history**, not summaries. The agent should see their actual reasoning trace from each timestep they were reasoned, not a compressed 1-sentence summary. This is already stored as `raw_reasoning` in agent_states — just surface it.
- **Entire history**, not sliding window of 3. If an agent reasoned on days 1, 3, 5, 6, 7 — they should see all 5 entries. The 3-entry cap loses critical early reactions.
- Timestamped: "On Day 2, you thought: ..." with actual elapsed time context
- Sentiment + conviction visible as felt experience: "You were angry and sure of yourself" not "sentiment: -0.8, conviction: 0.7"
- Consistency signal emergent from seeing the full trace — if you read 5 entries that all say the same thing, you know you've been consistent. No need for a separate "you've held this view for N days" signal.
- Token budget concern: full reasoning × many timesteps = lots of input tokens. Mitigation: truncate each entry to first 3-4 sentences if needed, but always show ALL timesteps, never drop history.

---

## ISSUE 9: Private vs Public is Imposed, Not Reasoned

**What exists:**
- Pass 1 asks for one reasoning + one public statement
- Engine then applies bounded confidence, flip resistance, and friction rules to mechanically split into public/private positions
- Agent never gets asked "what would you say publicly vs what do you actually think?"

**What's broken:**
- People consciously manage public vs private personas — especially on controversial topics
- Travis might tell his coworkers "yeah I'm upskilling" while privately thinking "I'm fucked and nothing will help"
- The current system imposes this gap mechanically instead of letting the agent reason about it

**What's needed:**
- Pass 1 should explicitly ask: "What would you tell people?" AND "What do you actually think privately?"
- Let the LLM generate the divergence based on the agent's personality (agreeableness, extraversion, social pressure)
- Remove or reduce the mechanical post-hoc private/public splitting
- More expensive (more output tokens) but more authentic

---

## ISSUE 10: The Prompt Structure is a Data Sheet, Not a Narrative

**What exists:**
```
[System instruction: You ARE this person]
[Persona: structured attribute list]
[Event: content block]
[Exposures: bullet list]
[Memory: bullet list]
[Peer opinions: bullet list]
[Instructions: respond as JSON]
```

**What's broken:**
- Reads like a form, not an experience
- No narrative arc — morning → work → evening
- No environmental texture — phone, TV, kitchen table conversation
- No emotional continuity — starts fresh each timestep

**What's needed:**
- The prompt should read like a day's narration
- Scenario-defined "day template" that structures exposures into life moments:
  ```
  Morning: You wake up and check your phone. [social media exposures here]
  Work: You get to [workplace]. [workplace exposures + coworker opinions here]
  Evening: You're home with [family]. [family/partner opinions + aggregate mood here]
  ```
- This template is generalizable — different scenarios define different day rhythms
- The agent's demographic drives which template applies (employed → has work section, retired → no work, student → school section)

---

## ISSUE 11: Pass 2 Classification is Disconnected

**What exists:**
- Pass 2 gets ONLY the reasoning text from Pass 1
- No persona, no exposures, no peer context

**What's broken:**
- Classification without context can misinterpret reasoning
- "I'm going to learn AI tools" could be "aggressive_upskilling" or "double_down_current_path" depending on whether the person is switching careers or enhancing their current one
- The agent's persona should inform classification

**What's needed:**
- Pass 2 should receive: reasoning text + agent's core demographic context + outcome descriptions
- Or: merge Pass 1 and Pass 2 into a single call — more tokens but eliminates disconnection

---

## ISSUE 12: Channel ≠ Experience

**What exists:**
- Channels defined in scenario: `mainstream_news_media`, `social_media_feeds`, `workplace_channels`
- Each has `description` and `credibility_modifier`
- Agent sees only the channel name label

**What's broken:**
- "mainstream_news_media" is not an experience. Opening CNN on your phone while eating breakfast IS.
- Different agents should experience the same channel differently based on demographics:
  - `primary_news_source: social media` + `social_media_usage: heavy` → sees it on X/TikTok first
  - `primary_news_source: television` → sees it on cable news
  - `digital_literacy: basic` → might hear about it from someone else before seeing it online
- Credibility should be experiential: "a well-sourced NYT article" vs "a viral tweet with no source"

**What's needed:**
- Channel rendering as a function of (channel_config × agent_demographics)
- Scenario defines channel templates with demographic variants
- Or: LLM-generated experiential rendering at prompt build time (more expensive but more natural)

---

## Priority Order

The biggest structural gaps are foundational — everything else builds on top:

1. **Issue 1** — Population ↔ Network (social role edges from attributes)
2. **Issue 2** — Agent names (culturally appropriate, demographic-aware)
3. **Issue 3** — Family members (partner linking + NPC dependents)
4. **Issue 13** — Agent interactions (conversations, social posts, actions)
5. **Issue 4** — Temporal awareness (timestep + unit in prompt)
6. **Issue 10** — Narrative prompt structure (day template)
7. **Issue 14** — Cognitive architecture (emotions, attention, memory types, self-awareness)
8. **Issue 5** — Rich exposures (experiential, aggregated, credibility-aware)
9. **Issue 6** — Named peer opinions (demographics, conviction, consensus signals)
10. **Issue 7** — Aggregate sentiment (public mood surfaced to agents)
11. **Issue 8** — Full memory (entire history, full reasoning traces, timestamped)
12. **Issue 12** — Channel × demographics rendering
13. **Issue 9** — Private vs public reasoning (agent-driven, not mechanical)
14. **Issue 11** — Pass 2 merged into Pass 1

---

## Implementation Approach: Deterministic Systems + Conversation Calls

Everything except conversations is deterministic — no extra LLM calls. Conversations add ~3 fast-model calls per pair per timestep.

### Names (Issue 2)
- **Data source:** SSA baby name data (public, by year + gender) + Census surname data (by ethnicity)
- **Method:** Agent's `age` → birth year → decade. Filter names by decade + `gender` + `race_ethnicity`. Weighted random pick from frequency distribution.
- **Cost:** One-time generation at sampling. ~50KB of bundled CSV data. Zero API calls.

### Family NPCs (Issue 3)
- **Partner:** Match two agents with compatible `marital_status` + `state` + age range + gender. Create "partner" edge in network. Partner is a real agent — already exists.
- **Kids:** `household_size` - adults = kid count. Ages sampled from plausible range given parent's age. Names from same demographic pool as parent. School grade from age. All deterministic rules.
- **Storage:** Structured metadata on agent (JSON field or persona context).

### Temporal Framing (Issue 4)
- **Method:** String formatting. `f"It is {timestep_unit.title()} {timestep + 1} since {event_summary}"`
- **Memory timestamps:** `f"{current_timestep - memory.timestep} {timestep_unit}s ago, you thought:"`

### Aggregate Sentiment (Issue 7)
- **Method:** Lookup tables mapping float ranges to natural language.
- **Sentiment → mood:** `> 0.6` = "optimistic", `> 0.2` = "cautiously hopeful", `> -0.2` = "uncertain", `> -0.6` = "anxious", else "deeply fearful"
- **Variance → consensus:** `< 0.05` = "everyone seems to agree", `< 0.15` = "most people feel similarly", else "opinions are all over the place"
- **Trend:** Compare last 2 timesteps. Diff > 0.1 = "getting worse/better". Diff < 0.1 = "about the same as yesterday."
- **Scope:** Computed from agent's local network neighborhood, not global population.

### Exposure Aggregation (Issue 5)
- **Method:** Group exposures by (channel, timestep). Count network sources per timestep. Render as:
  - Channels: "You saw this on the news and it was all over your social media feed"
  - Network: "14 people in your network brought it up today, including your coworker Darnell and your partner Lisa"
  - Cumulative: "This is the 4th day in a row people around you are talking about this"

### Channel → Experience (Issue 12)
- **Method:** Template lookup. Scenario defines channel templates. Agent's `primary_news_source` + `digital_literacy` + `social_media_usage` select the variant.
- **Example templates per channel:**
  ```
  mainstream_news_media:
    if primary_news_source == "television": "You saw a news segment about this on TV"
    if primary_news_source == "social media": "A news article showed up in your feed"
    if digital_literacy == "basic": "Someone at work showed you a news article"
    default: "You read about this in the news"
  social_media_feeds:
    if social_media_usage == "heavy": "Your feed has been nonstop about this"
    if social_media_usage == "moderate": "You saw several posts about this"
    default: "You noticed some posts about this online"
  ```
- ~10-20 templates per channel. Bundled in scenario YAML or in a defaults file.

### Peer Rendering (Issue 6)
- **Method:** String formatting with peer's name, age, relationship, sector, conviction label.
- `f"Your {relationship} {name} ({age}, {sector}) says: \"{statement}\" — {conviction_label}"`
- **Consensus signal:** Count peer sentiments. If >80% same direction = "Most of your contacts feel the same way." If mixed = "Your contacts are split on this."

### Narrative Day Template (Issue 10)
- **Method:** Scenario-defined phase templates. Engine slots data into phases:
  ```
  phases:
    morning:
      condition: "always"
      template: "You wake up and check your phone."
      slots: [social_media_exposures, aggregate_mood]
    work:
      condition: "employment_status in ['employed full-time', 'employed part-time']"
      template: "You head to work at your {occupation_sector} job."
      slots: [workplace_exposures, coworker_opinions]
    school:
      condition: "school_enrollment != 'not enrolled'"
      template: "You go to class."
      slots: [peer_opinions]
    evening:
      condition: "always"
      template: "You're home for the evening."
      slots: [family_context, partner_opinion, network_opinions, memory_trace]
  ```
- Agent demographic attributes determine which phases render. The engine fills slots with pre-formatted data.

### Full Memory (Issue 8)
- **Method:** Remove sliding window cap. Store all reasoning entries. Surface `raw_reasoning` (truncated to 3-4 sentences if needed) with timestep labels.
- **Token budget:** If 10 timesteps × 3 sentences = ~300 tokens of memory. Manageable.

### Private vs Public (Issue 9)
- **Method:** Modify Pass 1 schema to ask for both public and private responses explicitly. Remove or reduce post-hoc mechanical splitting.
- **Schema addition:** `"private_thought": "What you actually think but wouldn't say out loud"`

### Pass 2 Enrichment (Issue 11)
- **Method:** Append agent's core demographics (age, sector, education) to Pass 2 prompt alongside reasoning text.

---

## LLM Call Budget

- **Reasoning:** 1 call per agent per timestep (merged Pass 1 + Pass 2 into single call)
- **Conversations:** ~3 calls per conversation pair (fast model). ~30% of agents initiate per timestep.
- **Everything else** (names, family NPCs, temporal framing, aggregate mood, exposure rendering, channel templates, peer rendering, day narrative): zero LLM calls — all deterministic systems.
- **Net change vs current:** Current = 2 calls/agent (Pass 1 + Pass 2). New = 1 call/agent + ~0.9 conversation calls/agent on average. Roughly same total call volume, shifted from strong model to fast model for conversations.

---

## Generalization Across All Study Types

This isn't ASI-specific. Every study type has a fundamentally different social context, and the system must handle all of them without per-study custom code.

### How Different Studies Play Out

**Breaking crisis (ASI, bud-light-boycott):**
- Day 1 is a shock wave — everyone hears at once via news/social
- Network floods immediately — everyone's talking about it
- Strong opinions form fast, echo chambers harden
- Temporal awareness matters: "It's been 3 days and still no government response"
- Key relationships: coworkers, social media circles, politically aligned friends

**Policy/regulatory change (london-ulez, nyc-congestion-pricing, ma-sports-betting):**
- Slow rollout — people hear at different times depending on how directly it affects them
- Local geography matters enormously — inside vs outside the zone
- Practical impact matters — "how much will this cost ME"
- Key relationships: neighbors, local community, commuters, local government
- Day rhythm: commute, local news, council meetings, seeing the physical infrastructure change

**Corporate product change (netflix-password-sharing, netflix-ad-tier, apple-att-privacy):**
- Personal discovery — "I tried to log in and it didn't work"
- Family/household connections dominate — who shares the account, who pays
- Not emotional outrage but practical annoyance or adaptation
- Key relationships: family members, whoever's account you mooch off of
- Day rhythm: using the product, texting family, checking Reddit for workarounds

**Cultural/lifestyle shift (plant-based-meat, reddit-api-protest):**
- Slow burn — nobody panics, it comes up casually
- Identity and community matter — "am I the kind of person who eats this?"
- Social influence is peer-based, not media-driven
- Key relationships: friend groups, dining companions, online communities
- Day rhythm: grocery shopping, cooking dinner, casual conversation, seeing ads

### What Must Be Scenario-Configurable

| Element | Why it varies | Example |
|---------|--------------|---------|
| **Day phases** | A crisis has "check phone → panic → work → evening news." A product change has "try to use product → fail → text someone → search online." | ASI: morning/work/evening. Netflix: discover/react/adapt. |
| **Channel templates** | "Social media feed is blowing up" (crisis) vs "You saw an ad while browsing" (slow burn) | Same channel, different intensity based on scenario type |
| **Relationship weights** | Family dominates for household products. Coworkers dominate for workplace AI fears. Community dominates for local policy. | Netflix: family edges weighted highest. ASI: coworker edges weighted highest. |
| **Temporal framing** | "Day 1 since the announcement" (crisis) vs "It's been a few weeks since Netflix changed its policy" (gradual rollout) | Crisis = urgent clock. Policy = calendar. Product = "eventually you noticed." |
| **Aggregate mood source** | Breaking news → media/social sentiment visible. Local policy → neighborhood sentiment. Product change → your household. | ASI: global/media mood visible. ULEZ: local neighborhood mood. Netflix: household mood. |
| **Exposure rhythm** | Crisis = flood on day 1, taper. Policy = steady drip. Product = triggered by usage. | ASI: 99 network exposures day 2. Netflix: 2-3 family members over a week. |

### How The System Handles This

The scenario YAML already defines channels, exposure rules, and timestep configuration. Extending it to include:

1. **Day phase templates** — scenario declares what a "day" looks like for this study. Defaults provided for common patterns (crisis, policy, product, cultural).
2. **Channel experience templates** — scenario declares how each channel feels to different demographics. Defaults provided for common channels (news, social media, workplace, family).
3. **Relationship priority** — scenario declares which edge types carry the most weight for this topic. Network generation uses this to emphasize the right connections.
4. **Mood scope** — scenario declares whether agents sense global public mood, local network mood, or household mood.
5. **Temporal style** — scenario declares how time feels: urgent countdown, calendar dates, or vague "it's been a while."

All of these have sensible defaults. An existing scenario YAML with none of these new fields still works — it just gets the default day template, default channel rendering, default relationship weights. New scenarios can customize everything.

### No Backward Compatibility

This is a breaking change. The old prompt style (bullet-point data sheets, anonymous peers, no temporal awareness) is deprecated entirely. All studies get the new experience. Existing scenario YAMLs will need to be updated with day templates, channel experience configs, and relationship priorities — or sensible defaults kick in that are still vastly better than the current system.

---

## ISSUE 13: Agents Can't Interact With Each Other

**What exists:**
- Agents reason in isolation. No agent ever talks to another agent.
- Agent A produces a `public_statement`. Next timestep, Agent B sees it as a peer opinion. One-way broadcast.
- Agent B never responds to Agent A. Agent A never knows B heard them.
- 2000 people thinking alone in rooms, occasionally seeing post-it notes from each other.

**What's broken:**
- Real life is conversations, not monologues. Travis and Lisa argue about the budget. Darnell and Travis talk at break. Marcus texts Travis. These are back-and-forth exchanges that change both people.
- Social influence happens through dialogue, not broadcast. You push back, the other person doubles down or concedes, you update your view based on how the conversation went.
- Currently there's no way for an agent to choose who they engage with. The engine decides everything.

**What's needed:**

### Agent Actions: Post + Talk

The reasoning output expands to include actions the agent chooses to take:

```json
{
  "reasoning": "...",
  "sentiment": -0.9,
  "conviction": 85,
  "private_thought": "...",
  "actions": [
    {"type": "post", "platform": "x", "content": "This AI thing is terrifying..."},
    {"type": "talk_to", "who": "Lisa", "topic": "We need to look at our budget"},
    {"type": "talk_to", "who": "Darnell", "topic": "How's that AI course?"}
  ]
}
```

- **`post`** — agent posts on social media. Content feeds into aggregate public discourse. Other agents see it as part of their feed's mood next timestep. Zero extra LLM calls — it's just output text stored by the engine.
- **`talk_to`** — agent initiates a conversation with a named contact. Engine resolves this into an actual back-and-forth exchange.

The agent's prompt shows who's available to talk to, gated by network edges but presented as social reality:

```
People in my life right now:
- Lisa (my partner, she's home with me tonight)
- Tyler, Kayla, Mason (my kids, they're home)
- Darnell (my coworker, I'll see him tomorrow at work)
- Marcus (my neighbor, I could text him)
- Patricia (I know her online, I could DM her)
```

Availability is contextual to the day phase. During WORK, coworkers are reachable face-to-face. During EVENING, family. Online contacts are reachable anytime but lower influence (text vs in-person).

### Real Conversations (Not Fake Responses)

When Travis says `talk_to: Lisa`, the engine runs an actual multi-turn exchange. Both agents are real — Lisa has her own state, her own fears, her own reasoning. No faking her response.

A conversation is 2-3 turns:

```
Travis: "Lisa, we need to sit down and look at our budget. I've been
thinking about this all day."
Lisa: "I know. I've been looking at it. We have one month of savings.
That's it. I don't want to hear about AI courses right now — we need
cash."
Travis: "...yeah. Okay. Budget first. But I still think I need to
learn something."
```

Both agents' states update based on the conversation outcome. Lisa might shift Travis's priorities. Travis might make Lisa less scared. Or they might entrench further.

### Conflict Resolution: Priority Queue + Sequential

Multiple people can want to talk to the same person in the same timestep. All conversations happen within the SAME timestep — nothing gets bumped to the next day.

Resolution:
1. **Priority by edge weight.** Strongest relationship goes first. Partner > close friend > coworker > acquaintance.
2. **Sequential within conflicts.** Travis↔Lisa resolves first. Lisa's state updates. Then Darnell↔Lisa resolves with Lisa's updated state. Both happen in the same timestep.
3. **Parallel across non-conflicting pairs.** Travis↔Lisa and Marcus↔Keisha have no overlap — they run simultaneously.

This means conflicting conversations are slightly more expensive (sequential instead of parallel) but always resolve within the same timestep.

### Public Discourse: Social Posts

Agent posts on X/Instagram/Reddit feed into a public discourse layer:
- Posts are stored per timestep
- Other agents see aggregated public discourse as part of their feed next timestep
- The aggregate isn't individual posts — it's the mood/vibe: "Your X feed is a mix of panic and dark humor. Most posts are about job security."
- Agent's own posts can influence strangers beyond their network (low influence, high reach)
- This replaces the concept of "public_statement" — instead of one generic statement, agents choose what to post and where

### Timestep Flow (New)

```
Phase 1: REASONING (parallel)
  All agents reason in parallel.
  Output: internal state + actions (post, talk_to)
  Cost: 1 LLM call per agent (merged Pass 1 + Pass 2)

Phase 2: SOCIAL POSTS (no LLM calls)
  Engine collects all "post" actions.
  Stores them. Computes aggregate discourse metrics.
  Feeds into next timestep's mood rendering.
  Cost: zero

Phase 3: CONVERSATIONS (parallel across pairs, sequential within)
  Engine resolves talk_to requests into conversation pairs.
  Priority queue by edge weight for conflicts.
  All non-conflicting pairs run in parallel.
  Each conversation: 2-3 turns, both agents reasoning in character.
  Both agents' states update from conversation outcome.
  Cost: ~3 LLM calls per conversation pair (fast model)

Phase 4: STATE UPDATE
  Write all state changes (from reasoning + conversations).
  Compute timestep summary.
  Check stopping conditions.
```

### Cost Analysis

| Agents | RPM | Timestep unit | Timesteps | Calls/timestep | Wall time/timestep | Total |
|--------|-----|---------------|-----------|----------------|-------------------|-------|
| 1k | 1k | week | 12 | ~1.9k | ~2 min | ~24 min |
| 2k | 1k | week | 12 | ~3.8k | ~4 min | ~48 min |
| 5k | 1k | week | 12 | ~9.5k | ~10 min | ~2 hrs |
| 10k | 1k | week | 12 | ~19k | ~19 min | ~3.8 hrs |
| 10k | 2k | week | 12 | ~19k | ~10 min | ~2 hrs |

Calls/timestep breakdown (for 10k agents):
- 10k reasoning calls (strong model, parallel)
- ~3k conversation pairs × 3 turns = ~9k conversation calls (fast model, parallel across pairs)
- Weekly timesteps keep total timestep count low (12 for a 3-month scenario)

### Parallelism Impact

Reasoning phase: fully parallel, same as today. No change.

Conversation phase: parallel across non-conflicting pairs. Most pairs don't conflict (Travis↔Lisa and Darnell↔Marcus are independent). Conflicts are rare — maybe 5-10% of pairs share an agent. Those resolve sequentially within the timestep.

Worst case: one popular agent (e.g., a community leader) gets 10 conversation requests. Those 10 conversations run sequentially = 10 × 3 turns = 30 sequential calls. At ~3s per call on fast model = ~90s. Everything else runs in parallel around it. Not a bottleneck unless the scenario creates extreme hub agents.

---

## ISSUE 14: No Cognitive Architecture (Emotions, Attention, Self-Awareness)

Inspired by TinyTroupe's cognitive model — they track emotions, attention, goals, and memory consolidation as explicit agent state. We track sentiment as a float and conviction as a number. The agent has no inner life.

**What's missing:**

### 14a: Emotional state as felt experience
- Agent has `sentiment: -0.91` but never sees "I've been anxious for two weeks"
- Emotional trajectory matters — panic → dread → resignation is different from panic → panic → panic, even if the sentiment float is the same
- The agent should be aware of HOW they've been feeling, not just what they've been thinking
- Implementation: render emotional trajectory from sentiment history. "You started the week panicked. By mid-week it settled into a heavy dread. It hasn't lifted." Deterministic — map sentiment values + trend to narrative labels.

### 14b: Attention / focus
- What is the agent currently focused on? Travis might be focused on "immediate cash" while his feed pushes "learn AI skills." The gap between attention and incoming information creates tension.
- Currently the agent processes everything equally — exposures, peers, memory are all flat bullet points with no hierarchy of salience.
- Implementation: the agent's previous reasoning determines their focus. If Travis said "budget first" last timestep, budget-related inputs should be highlighted/foregrounded in the prompt this timestep. Deterministic — keyword extraction from previous reasoning to weight prompt sections.

### 14c: Repetition detection and forced deepening
- TinyTroupe detects when agents repeat themselves (>0.85 similarity) and forces variation
- Our agents say "No change — save, learn AI, backup income" for 5 straight timesteps. That's not human. Real people either deepen their thinking, get bored, take action, or shift focus.
- Implementation: if the engine detects high similarity between consecutive reasonings (cosine similarity or simple string overlap), inject a prompt nudge: "You've been thinking the same thing for several days. Has anything actually changed? Are you starting to doubt your plan? Have you actually done anything about it?" Forces the LLM to either deepen or evolve the response.

### 14d: Episodic vs semantic memory
- Currently: flat list of reasoning summaries. No distinction between "what happened to me" and "what I believe."
- Should have:
  - **Episodic:** "On Day 1, I saw the news and panicked. On Day 3, I talked to Lisa about the budget." — what happened, when.
  - **Semantic:** "I believe my job is at risk. I believe the government isn't going to help. I believe saving cash is more important than courses." — distilled beliefs that emerged from episodes.
- Semantic memories consolidate over time — after 3 timesteps of thinking the same thing, it becomes a belief, not a fresh reaction.
- Implementation: episodic = full reasoning traces (Issue 8). Semantic = after N timesteps of consistent reasoning on a theme, engine extracts a belief statement and adds it to a persistent "beliefs" field. Shown in prompt as "Things I've come to believe:" separate from "What I thought recently."

### 14e: Self-awareness of own conviction and consistency
- Agent outputs conviction but never sees it. Never sees "I've been firm about this for 4 days" or "I'm less sure than I was last week."
- Self-awareness of consistency affects behavior — knowing you've held a position for weeks makes you more resistant to change (commitment bias). Knowing you've been wavering makes you more open.
- Implementation: render conviction trajectory alongside memory. "You've been firm about this since Day 2. Your certainty hasn't wavered." Or: "You started certain but you've been getting less sure each day."

### 14f: Spontaneous memory recall (context-triggered)
- Currently: memory is a flat chronological list. The engine dumps the last N entries into the prompt regardless of what's happening now.
- Should be: memories surface BECAUSE they're relevant to what's happening. If Travis sees a headline about college costs, his memory of Tyler asking "is there a point in college?" should surface automatically — not because it was recent, but because it's contextually connected.
- This is how human memory works — you don't remember things in order, you remember things that are triggered by current experience.
- Implementation: at prompt build time, compute relevance between current timestep's exposures/events and all stored memories (semantic similarity). Surface the top N most RELEVANT memories, not the most RECENT. This requires embedding stored memories and comparing to current context. Cheap — small embedding model on short text, done locally or via a fast API call. Not an LLM reasoning call.

### 14g: Internal monologue vs external action
- Currently: the agent produces one reasoning output that serves as both their internal thought and their public behavior. "Reasoning" IS the monologue AND the action.
- Should be: explicit separation between THINK (what I'm actually processing internally) and ACT/SAY (what I do and tell people). Travis might internally think "I'm completely fucked, nothing is going to save my job, I'm terrified for my kids" but externally say "We'll figure it out, let's just be smart about this."
- This is deeper than Issue 9 (private vs public position). Issue 9 is about the engine mechanically splitting public/private. This is about the agent having a genuine internal life that diverges from their external behavior — and being AWARE of that gap.
- The internal monologue should be raw, unfiltered, honest. The external actions/statements are socially filtered through personality (agreeableness, extraversion, social pressure).
- Implementation: Pass 1 output schema includes both:
  ```json
  {
    "internal_monologue": "Raw, honest, unfiltered stream of thought",
    "external_actions": [...],
    "public_statement": "What I'd actually say out loud",
    "private_belief": "What I actually believe underneath"
  }
  ```
- The internal monologue feeds into memory (episodic). The public statement feeds into peer opinions for others. The gap between them is itself interesting data — agents with high agreeableness might have larger gaps (saying what others want to hear while privately disagreeing).

**Cost:** All of these are deterministic rendering from existing data (sentiment history, reasoning traces, conviction scores). Zero extra LLM calls except 14f which needs a small embedding computation per agent per timestep (cheap). The only other new computation is similarity detection for 14c (cosine similarity on embeddings or even simpler string overlap).

---

## What The New Simulation Feels Like: 4 Scenarios

These show the target agent experience across wildly different study types — crisis, product change, local policy, financial event. Same engine, same systems, completely different lives.

---

### Scenario 1: ASI Announced — Travis, 43, Services, South Carolina

**Week 1**

I'm Travis. I'm 43, I live in Greenville, SC with my partner Lisa (41, retail) and our three kids — Tyler (17, about to graduate), Kayla (14), and Mason (9). I work full-time in the service industry. I've got one month of savings. That's it.

Monday morning I wake up and grab my phone like I always do. My X feed is unrecognizable. Every post is about AI — not the usual tech drama, the real thing. "AI systems surpass humans in every cognitive domain." Fortune 50 companies announcing layoffs. I see a CNN link someone shared — "Mass workforce restructuring begins." I scroll Instagram, same thing filtered through memes. "We're cooked" with 50k likes.

I get to work. Darnell (36, works with me in services) catches me during break: "Bro did you see this? My cousin at Chase says they're already talking about cutting whole departments." My manager Carlos hasn't said a word. The break room TV is on CNN — some economist talking about how service jobs are "highly exposed to automation." I'm standing there watching this thinking — that's me. That's my job.

I come home. Lisa's been reading about it all day. She's quiet, which means she's scared. Tyler asks at dinner: "Dad, is there even a point in college if AI does everything?" Kayla's on her phone not paying attention. Mason's doing homework. I don't have an answer for Tyler. Marcus (42, my neighbor, works construction) texts me: "You seeing this? What the hell."

I check what people around me are thinking. Darnell's dead serious — he's already talking about cutting costs and learning AI tools. Lisa wants to look at our budget tonight. Marcus is uncertain — he thinks his hands-on work might be safe but he's not sure. Everyone I know online is panicking. My feed is a wall of anxiety. Nobody's calm about this.

*What I'm thinking:* I'm terrified. One month of savings. Five mouths to feed. My job is exactly the kind that gets automated. I can't go back to school — no money, no time. I need to cut spending NOW, learn something practical that could actually earn money, and line up backup work — gig driving, contract labor, whatever.

*What I do:*
- I post on X: "One month of savings and a family of 5. Nobody in Washington is saying a damn thing. We're on our own."
- I talk to Lisa tonight about the budget.
- I text Darnell asking if he knows any free AI courses.

**The conversation:**

Me: Lisa, we need to look at our budget. Tonight. I'm serious.

Lisa: I already did. Travis, we have $2,400 in savings. That's rent and groceries for one month. If either of us loses our job—

Me: I know. I've been thinking — maybe I should learn some AI tools, something I could use to—

Lisa: With what money? And when? You work 50 hours a week. Tyler needs shoes. Mason's got a field trip. I don't want to hear about courses right now. We need CASH.

Me: ...yeah. Okay. Budget first. But I'm not just gonna sit here and wait to get fired.

*After talking to Lisa, I'm less focused on upskilling and more focused on immediate survival. She's right — we can't afford courses. Cash first.*

**Week 2**

It's been two weeks. I've been thinking about this every single day.

Week 1 I was panicking. I said: "I'm terrified. One month of savings. I need to cut spending, learn something practical, line up backup work." I was angry and scared.

This week, 31 people in my circle brought it up. Same fears, same conversations. The mood hasn't changed — if anything people are more resigned than panicked now. Darnell signed up for some free YouTube tutorials on AI tools. My X feed has shifted from pure panic to "what to do" threads. Still no government response.

At work, two people quit this week — not because they got fired, but because they're scared and jumping to jobs they think are safer. Carlos still hasn't said anything. That silence is getting louder.

Lisa and I cut our streaming, cancelled some subscriptions, started cooking every meal at home. We're trying to stretch savings to two months. Tyler's been quiet — I think the college question is eating at him.

The general mood around me: everyone's still anxious. Most people I know are talking about practical survival — saving money, learning skills. A few are stubbornly saying "it'll blow over." Nobody I know is organizing or pushing back politically. It's all individual survival mode.

*What I'm thinking:* Same as last week but harder. Lisa's right — we can't afford courses. I'm looking at gig apps instead. Maybe I can drive nights. The anger is fading into something heavier — this feeling like the ground shifted and nobody's coming to help.

*What I do:*
- I text Marcus asking if he knows anyone hiring for side work.
- I don't post on X this week. Don't have anything new to say.

---

### Scenario 2: Netflix Kills Password Sharing — Jade, 26, Marketing, Chicago

**Week 1**

I'm Jade. I'm 26, Black, living in Logan Square, Chicago. I work in marketing at a mid-size agency. I live alone in a studio apartment. I've been using my mom Denise's Netflix account since college. So has my brother Kevin (29, accountant, lives in Bronzeville) and my cousin Aaliyah (23, still in grad school).

Tuesday evening. I get home from work, make dinner, open my laptop to watch something. Netflix pops up a screen: "This device isn't part of the account holder's household. Please sign in with your own account or ask the account holder to verify." I try refreshing. Same thing. I text the family group chat.

Me: yo Netflix just locked me out??

Kevin: Same. Just got the screen.

Aaliyah: WHAT. I'm in the middle of a show???

Mom (Denise, 54, nurse): I got an email saying they're "cracking down on sharing." It says I need to pay extra for each person outside my household or you all need your own accounts.

My coworker Priya (28, also in marketing) mentioned at lunch that this happened to her last week. She just caved and got her own account. My friend Terrell (27, teaches middle school) posted on X: "Netflix really charging me $15.49 to watch shows I already watched for free lmaooo." 47 likes.

The general vibe online: people are annoyed but not outraged. Lots of memes. Some people saying they'll cancel, most people saying they'll complain and then pay. A few Reddit threads about VPN workarounds.

*What I'm thinking:* I'm annoyed, not devastated. $15.49/month isn't going to break me but it's the principle. I've been on my mom's account for 6 years. This feels like Netflix squeezing us. But honestly? I'll probably just pay it. Or maybe I'll finally try that Criterion Channel trial Terrell's been telling me about.

*What I do:*
- I text the family group chat to figure out what we're doing.
- I check Reddit for workarounds.

**The conversation:**

Me: Mom, are you gonna pay for the extra people or should we all just get our own?

Denise: Baby, I'm already paying $22 a month. I'm not adding three more people at $8 each. That's $46 for Netflix. No.

Kevin: Yeah I'll just get my own. It's not that serious.

Me: Aaliyah, what about you? You're on a grad school budget.

Aaliyah: I'm NOT paying for Netflix. I'll just use someone else's Disney+ lmao.

Me: lol fair. Okay so Kevin gets his own, I'll probably get mine, Aaliyah's freeloading somewhere else. Mom keeps hers.

*After the conversation: resolved pretty quickly. Nobody's happy about it but nobody's cancelling in protest either. Just redistributing the cost.*

**Week 2**

It's been two weeks since the password thing.

Last week I was annoyed but figured I'd just pay. This week I still haven't signed up for my own account. I've been watching YouTube and reading instead. Honestly... I don't miss it that much?

Priya at work is already back to binge-watching on her own account. Terrell cancelled and went to Criterion Channel — he won't shut up about some French movie. Kevin got his own account on day 3. Mom's happier not paying for everyone.

The mood: nobody cares anymore. It was a two-day annoyance. People either paid or moved on. The Reddit workaround threads died. No one's boycotting.

*What I'm thinking:* I might just... not get Netflix. Which is weird because I've had it (well, my mom's) for 6 years. But $15.49 for something I can live without? In this economy?

---

### Scenario 3: London ULEZ Expands — Graham, 58, Plumber, Outer London

**Week 1**

I'm Graham. I'm 58, white, married to Sandra (55, part-time school receptionist). We live in Bromley, outer southeast London. Our daughter Charlotte (31) lives in Lewisham with her partner. Our son Jake (27) is still at home saving for a deposit. I run my own plumbing business — just me, my van, and 30 years of customers across South London.

Monday morning. I already knew this was coming but seeing the signs go up on the A205 makes it real. ULEZ expanded to cover all of London. My 2015 Transit van doesn't meet the emission standards. That's £12.50 a day, every day I drive into the zone. Which is every day, because that's where my customers are.

I'm in the van listening to LBC. Nick Ferrari's taking calls about it. Every other caller is a tradesman like me. Sandra texts me: "Did you see the council posted about ULEZ grants? Maybe we can get help for a new van." I look it up at lunch — the scrappage scheme gives you £2,000 toward a compliant vehicle. My van is worth more than that and a new Transit is £35,000. £2,000 is a joke.

At the pub after work, my mate Dave (61, electrician, same situation) is fuming: "Sadiq Khan doesn't give a toss about people like us. £12.50 a day, that's £250 a month, that's £3,000 a year just for doing my job." My other mate Phil (55, retired copper) says he voted against it in the consultation. My neighbour Janet (63, retired teacher) thinks it's good for air quality. I bite my tongue.

The general mood in my area: angry. Bromley voted overwhelmingly against the expansion. The signs are getting vandalised overnight. My customers are asking me if I'm going to raise my prices. Local Facebook group is wall-to-wall fury.

*What I'm thinking:* This is going to cost me £3,000 a year minimum. I can't buy a new van — I'm 58, I'm not taking out a £35,000 loan at this stage of my life. I can't stop driving — my customers are in the zone. I can either eat the cost, raise my prices and lose customers, or... I don't know. I feel like I'm being punished for working.

*What I do:*
- I post in the Bromley Community Facebook group: "Any other tradesmen working out how to deal with this? £12.50/day is no joke."
- I talk to Sandra about the numbers tonight.
- I text Dave about maybe going to that anti-ULEZ protest next Saturday.

**The conversation:**

Me: Sandra, I've done the maths. £12.50 a day, five days a week, that's sixty-two quid a week. Over three grand a year.

Sandra: Can we claim it as a business expense at least?

Me: Yeah, tax deductible, but I still have to pay it upfront. That's three grand I don't have just lying about. And the scrappage scheme is a joke — two thousand quid toward a thirty-five thousand pound van.

Sandra: What about going electric? Charlotte was saying—

Me: An electric Transit is forty-eight thousand pounds, Sandra. And where am I charging it? We haven't got a drive. I'm not running an extension lead out the kitchen window.

Sandra: So what do we do?

Me: I raise my prices ten percent and hope I don't lose customers. That's the only option I can see.

*After talking to Sandra: she's practical, which helps. But neither of us has a good answer. The ten percent price hike is what I'm going with.*

**Week 2**

It's been two weeks. I went to Dave's protest last Saturday — about 200 people, mostly tradesmen and outer London residents. Got some press coverage. Didn't change anything.

Last week I was angry and said I'd raise prices 10%. I did. Two customers have already asked why. One of them said he'd "shop around." That stung — I've done his boiler for 15 years.

Dave's considering retiring early. Phil thinks we should take legal action. Janet still thinks it's good for air quality — I had a proper row with her over the fence about it. The Bromley Facebook group is still angry but it's becoming repetitive. Some people are just paying it and getting on with life.

The mood: still bitter but people are accepting it. The anger's turning into resentment. Nobody thinks it's going to be reversed.

*What I'm thinking:* I'm resigned. I'm going to pay the charge, pass what I can to customers, and hope I don't lose too much business before I retire in 7 years. The protest was nice but pointless. This is just how it is now.

---

### Scenario 4: Bitcoin Hits $1M — Kenji, 34, Software Engineer, Austin

**Week 1**

I'm Kenji. I'm 34, Japanese-American, living in East Austin. I'm a senior software engineer at a mid-size SaaS company. I live with my wife Mara (32, UX designer) and our daughter Yuki (3). We own our house — bought in 2022, still owe $380K on the mortgage. I own 1.4 BTC that I bought between 2020-2022 at an average of around $35K. I also have a 401K and index funds. Mara thinks crypto is gambling.

Wednesday morning. I'm brushing my teeth and my phone is going insane. Bitcoin crossed $1,000,000 overnight. My 1.4 BTC is worth $1.4 million. I stare at the number on Coinbase for about thirty seconds. Then I check again because I don't believe it. Then I sit down on the edge of the bathtub.

Mara walks in: "What's wrong?" I show her my phone. She doesn't say anything for a moment, then: "Is that real? Can you actually take that money out?"

I get to work. My coworker Ryan (29, also an engineer, has like 0.3 BTC) is losing his mind in Slack. Our team lead Pradeep (41, no crypto) is joking about it but you can tell he's kicking himself. My college friend Marcus texts me from Brooklyn: "BRO. BRO. BRO." He's got like 3 BTC. He's a millionaire three times over this morning.

Crypto Twitter is euphoric. Mainstream news is covering it wall to wall. CNBC has some analyst saying it could go to $2M. Another one saying it's a bubble that'll crash to $200K by next month. My X feed is 50% celebration, 30% "told you so", 20% people who didn't buy being bitter.

My financial advisor Greg sent an email: "Given recent market developments, I'd recommend we schedule a call to discuss your portfolio allocation."

*What I'm thinking:* $1.4 million. That's our mortgage paid off with a million left over. That's Yuki's college fund. That's early retirement if I play it right. But I also remember 2022 when it crashed from $69K to $16K and I held through the whole thing. Do I sell? Some of it? All of it? Do I hold for $2M? Every fiber of my body wants to sell and lock in the gain but every crypto person I know is saying this is just the beginning.

And Mara. She's been tolerating the crypto thing for four years. She's going to want to sell everything immediately. That conversation tonight is going to be intense.

*What I do:*
- I text Marcus: "What are you doing? Selling?"
- I call Greg to schedule that portfolio meeting.
- I post nothing. I'm not telling the internet I have $1.4M in Bitcoin. But I do lurk in the r/Bitcoin thread reading everyone's exit strategies.
- I talk to Mara tonight. This is a big one.

**The conversation:**

Me: Okay so... we need to talk about the Bitcoin.

Mara: Yeah we do. Sell it. All of it. Tomorrow.

Me: Just — hear me out. I think we should sell some. Maybe half. Pay off the mortgage. That's $380K, we keep the rest—

Mara: Kenji. You said the same thing when it hit $69,000. "Just a little longer." Then it crashed to $16,000 and you didn't sleep for a month. We have a three-year-old. This isn't a game anymore.

Me: I know. But this is different — institutional adoption, ETFs, sovereign wealth funds are buying—

Mara: I don't care about ETFs. I care about Yuki having a college fund that doesn't depend on what Elon Musk tweets at 2am. Sell it. Pay the mortgage. Put the rest in index funds. Please.

Me: ...what if I sell 70%? Pay the mortgage, put $500K in index funds, keep 0.42 BTC as a moonshot?

Mara: *(long pause)* Fine. But you sell tomorrow. Not "this week." Tomorrow.

*After talking to Mara: I'm selling 70%. She's right about 2022. I didn't sleep for a month. But I'm keeping 0.42 BTC because if it goes to $2M that's another $840K and I'll never forgive myself if I sold everything.*

**Week 2**

It's been two weeks. I sold 0.98 BTC at $1,020,000 average. Paid off the mortgage. Put $520K into a brokerage account — index funds, like Mara wanted. Kept 0.42 BTC.

Last week I was euphoric and torn. This week I feel... weirdly empty? The mortgage payoff was anticlimactic — I just watched a number go to zero on a screen. The money in the brokerage doesn't feel real yet.

Ryan sold everything at $980K. He's already talking about quitting to start a company. Pradeep started buying Bitcoin the day after it hit $1M — textbook FOMO. Marcus hasn't sold a single sat. He's diamond-handing for $5M. Greg the financial advisor was professional about it but I could hear him sweating on the phone.

Bitcoin's at $1.1M now. My remaining 0.42 BTC is worth $462K. Part of me is sick that I sold. Part of me remembers Mara's face when she said "this isn't a game anymore."

The mood: crypto people are manic. Non-crypto people are a mix of FOMO and resentment. Mainstream discourse is shifting from "is Bitcoin real?" to "how do I buy Bitcoin?" which historically means it's close to a top. But nobody knows.

*What I'm thinking:* I made the right call. The mortgage is gone. Yuki's set. But I check the price 40 times a day and every time it goes up I feel a little sick. This is the psychological cost of selling — you're never at peace with the number you picked.

---

### What These Scenarios Demonstrate

Four completely different lives. Same simulation engine.

| | ASI (Travis) | Netflix (Jade) | ULEZ (Graham) | Bitcoin (Kenji) |
|---|---|---|---|---|
| **Emotional intensity** | Terror/survival | Mild annoyance | Slow-burn anger | Euphoric anxiety |
| **Key relationship** | Partner (budget) | Family (shared account) | Partner + mates (trade) | Partner (sell/hold) |
| **Conversation stakes** | Financial survival | Who pays $15/mo | £3K/year business cost | $1.4M life decision |
| **Social media behavior** | Vents publicly | Lurks memes | Posts in local FB group | Posts nothing (privacy) |
| **Day rhythm** | Phone → work panic → family fear | Discover → text family → check Reddit | Commute → pub → kitchen table | Check price → Slack → tense dinner |
| **Aggregate mood** | Universal anxiety | Brief annoyance → apathy | Local fury → resignation | Split euphoria/FOMO/resentment |
| **Temporal feel** | Urgent daily countdown | 2-day blip | Slow grinding weeks | Volatile hourly swings |
| **Convergence** | Everyone scared, same plan | Quick resolution, move on | Bitter acceptance | Depends on price action |
