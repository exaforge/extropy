# What Extropy Can Simulate

This document walks through Extropy's capabilities by example. Each section describes a type of scenario you can run today, with the underlying mechanics that make it work.

---

## Any Country, Any Culture

Extropy isn't locked to US demographics. You can simulate populations anywhere.

**US population responding to a Netflix price hike**: Out of the box. Names come from bundled SSA baby name data (1940-2010 birth decades) and Census surname data by ethnicity. A 45-year-old Black woman born in 1980 gets a name that reflects naming trends for Black girls in that era. Her white husband gets an appropriately correlated name. Their kids get names consistent with 2010s trends.

**Japanese employees reacting to a remote work policy**: Works. You provide a `NameConfig` with Japanese naming conventions (or let the LLM research it), and Extropy generates culturally appropriate names. The simulation mechanics are identical - the agents reason in first-person, form opinions, share with their network.

**Indian consumers across Mumbai, Delhi, and Bangalore responding to a fintech launch**: Works. Define city as an attribute with your desired distribution. Agents get sampled across cities, connected via network edges that respect geography, and exposed to marketing through channels you define.

**Brazilian families deciding whether to migrate for work**: Works. Household sampling gives you family units with correlated attributes. Partners share socioeconomic status, have age gaps that reflect real patterns, and kids are generated as dependents with appropriate ages.

The pattern: define your population's attributes and distributions, optionally provide country-specific name data, and run the pipeline. The simulation engine doesn't care about geography - it cares about attributes, networks, and reasoning.

---

## Individuals or Households

You control whether agents exist as isolated individuals or as family units.

**Individual professionals responding to an industry disruption**: Set `household_mode: false`. Each agent is independent. Good for workplace scenarios, B2B decisions, or any context where family structure doesn't matter.

**Households deciding whether to adopt rooftop solar**: Set `household_mode: true`. Now you get family units. A married couple shares a `household_id` and `last_name`. They have correlated attributes - similar education levels, aligned political views (with configurable assortative mating rates), compatible religious backgrounds.

The `agent_focus` field in your population spec controls who reasons vs. who exists as context:

**Primary only (default)**: Only the primary adult in each household is a reasoning agent. Partners and children exist as NPC data attached to that agent - named, with attributes, but not making decisions. Use this when you care about one decision-maker per household (e.g., "the subscriber", "the homeowner").

**Couples**: Both adults in a household are reasoning agents. Children are NPCs. Partners influence each other through network edges (weight 1.0). Use this when both adults' opinions matter (e.g., couples deciding on a major purchase, spouses with different political views).

**All (families)**: Everyone in the household is a reasoning agent, including children old enough to have opinions. Use this when family dynamics matter (e.g., teens influencing parents on tech adoption, multi-generational disagreements).

The mechanism: set `agent_focus` in your population spec metadata. Values like "families", "households", or "everyone" trigger the "all" mode. Values like "couples", "partners", or "spouses" trigger couples mode. Everything else defaults to primary-only.

**Household types are sampled by age bracket**:
- Singles (one adult, no kids)
- Couples (two adults, no kids)
- Single parents (one adult with kids)
- Couples with kids (two adults with kids)
- Multi-generational (extended family)

Each age bracket has configurable weights. Young adults skew toward singles and couples. Middle-aged adults skew toward families. Elderly skew toward couples and singles again.

**Single adults**: Sampled naturally from the "single" household type. No partner, no dependents. Network edges come from coworkers, neighbors, congregation - not family.

**Childless couples**: Sampled from the "couple" household type. Two adults linked by `partner_id`, sharing `household_id` and `last_name`. No dependents generated.

---

## Realistic Social Networks

Agents don't exist in isolation. They're connected in networks that reflect real social structures.

**Partners influence each other most strongly**: Partner edges have weight 1.0. When one spouse changes their opinion, the other is heavily exposed. This models the reality that intimate partners shape each other's views more than anyone else.

**Coworkers share industry-specific information**: Agents in the same occupation category get connected with `coworker` edges (weight 0.6, capped at 8 connections). An accountant hears about regulatory changes from other accountants, not from nurses.

**Neighbors observe each other's behavior**: Agents in the same region with similar ages get `neighbor` edges (weight 0.4, capped at 4). When your neighbor installs solar panels, you notice. This models the "keeping up with the Joneses" dynamic.

**Religious communities spread information through congregations**: Agents sharing religious affiliation and high religiosity get `congregation` edges (weight 0.4, capped at 4). Church announcements, mosque discussions, temple gatherings - information flows through these communities.

**Parents of school-age children form their own network**: Agents with kids in school, in the same region, get `school_parent` edges (weight 0.35, capped at 3). PTA meetings, school pickup conversations, parent group chats - this captures that social layer.

**The rest is similarity-based**: After structural edges are placed, remaining connections fill based on attribute similarity. People befriend others like themselves. The overall degree distribution follows a power law - most people have a handful of connections, a few have many.

---

## Static Events or Evolving Timelines

Some scenarios are a single shock. Others unfold over time.

**Static: Netflix announces a price increase**: One event, one moment. Netflix raises prices by $3/month. Agents hear about it through news, social media, or email. They form opinions - cancel, keep, or downgrade. They share with their network. Over a few timesteps, information propagates, opinions stabilize, and you see the final distribution. The event itself doesn't change; what evolves is awareness and social influence.

This is the right model when:
- The event is a discrete announcement or decision
- What matters is how the population responds and influences each other
- There's no new information after the initial shock

**Evolving: Netflix password crackdown unfolds over months**:
- Month 1: Netflix announces upcoming password-sharing restrictions
- Month 2: Enforcement begins in select markets
- Month 3: Full rollout, first reports of account lockouts
- Month 4: Netflix offers discounted "extra member" add-on
- Month 5: Competitor promotions target frustrated users

Each timestep, agents see what's happened so far. Their prompts include a recap: "Over the past few months, Netflix first announced the crackdown, then started enforcing it. Last month they offered a cheaper add-on option." The timeline creates a narrative arc where agent reasoning evolves with new information.

This is the right model when:
- The situation develops with new facts over time
- Agent responses to Week 1 should differ from Week 5
- You want to model how opinions shift as circumstances change

**Evolving: A crisis that develops and resolves**:
- Day 1: Initial reports of data breach, uncertainty about scope
- Day 2: Company confirms breach, announces investigation
- Day 3: Details emerge - 10 million accounts affected
- Day 5: Company offers free credit monitoring
- Day 7: CEO resigns
- Day 10: New security measures announced

Agents experience the crisis as it unfolds. Early timesteps have high uncertainty and speculation. Later timesteps have concrete information. Memory traces let agents reference their earlier reactions: "Last week I was panicking about my data. Now that they're offering monitoring, I'm less worried but still annoyed."

**Automatic detection**: If you define a single event with no timeline, Extropy treats it as static. If you provide multiple events or explicit timeline entries, it switches to evolving mode. You can override with `timeline_mode: static` or `timeline_mode: evolving`.

**Background context**: Scenarios can include ambient framing that appears in every agent's prompt. "The US economy is in a mild recession. Unemployment was at 4.5% before the AI announcement. It's early spring." This shapes reasoning without being the focal event.

**Timestep units are configurable**: Days, weeks, months - whatever fits your scenario. A crisis might unfold over days. A policy change might play out over months. A generational shift might span years.

---

## Multiple Exposure Channels

People hear about things through different channels, and the channel matters.

**Mainstream media reaches broadly but impersonally**: High reach probability, but agents process it as "something I saw on the news." Good for initial awareness, less effective for deep persuasion.

**Social media spreads through networks**: Reach follows network edges. If your connections are sharing something, you see it. The viral dynamic emerges naturally - well-connected agents amplify information.

**Word of mouth is personal and trusted**: Exposure happens through direct network edges only. Lower reach, higher impact. When your brother tells you something, it carries more weight than a headline.

**Official communication targets specific groups**: An employer announcement reaches employees. A utility notice reaches customers. Channel targeting filters by attributes - only relevant agents get exposed.

**Observation lets agents notice behavior**: Agents can witness what others do, not just hear what they say. When a neighbor buys an electric car, agents on that network edge might get exposed through observation. This models the "seeing is believing" dynamic.

Each channel has:
- Reach probability (what fraction of eligible agents get exposed)
- Targeting rules (which agents are eligible)
- Experience template (how the agent encounters the information)

You can mix channels. A scenario might start with mainstream media coverage, then spread through social media, then deepen through word of mouth as people discuss it with family.

---

## Any Kind of Outcome

You decide what you're measuring.

**Categorical choices**: "Will you support, oppose, or remain neutral?" "Will you buy, wait, or skip?" Any discrete set of options. The first required categorical outcome becomes the agent's "position" - the headline metric for aggregation.

**Boolean decisions**: "Will you share this with others?" "Will you attend the event?" Yes or no.

**Continuous scales**: "How price-sensitive are you on a scale of 0 to 1?" "What's your trust level from 0 to 100?" Useful for measuring intensity, not just direction.

**Open-ended responses**: "What are your main concerns?" Free text. The agent reasons naturally without being forced into categories. These skip the classification pass entirely - the reasoning itself is the outcome.

You can mix outcome types. A scenario might have:
- A categorical position (support/oppose/neutral)
- A boolean share intention
- A continuous intensity score
- An open-ended elaboration

All get captured in the same simulation run.

---

## Two-Pass or Merged Reasoning

You control the tradeoff between cost and reasoning quality.

**Two-pass (default)**:
1. Pass 1 asks the agent to reason freely in first-person. No outcome categories in sight. Just "You're this person, this happened, how do you feel?"
2. Pass 2 takes that reasoning and classifies it into your defined outcomes using a faster, cheaper model.

This separation prevents the central tendency problem where agents gravitate to safe middle options when they see the categories upfront. Reasoning quality is higher because the agent isn't gaming the schema.

**Merged pass** (`--merged-pass`):
Single call with both reasoning and outcomes in one schema. Cheaper - one API call instead of two. Faster - no round-trip between passes. But the agent sees the outcome categories while reasoning, which can bias responses toward the middle.

Use merged pass for:
- Cost-sensitive runs with many agents
- Quick exploratory simulations
- Scenarios where you trust the model to reason past the schema

Use two-pass for:
- Final production runs where quality matters
- Scenarios with polarizing topics where central tendency is a real risk
- Research where reasoning traces need to be unbiased

---

## Memory and Temporal Awareness

Agents aren't goldfish. They remember.

**Full memory traces**: Every timestep, agents get their complete history. "In Week 1, I was skeptical. By Week 3, I was coming around. Last week I committed to trying it." Memories include the reasoning summary, sentiment, and conviction at each point.

**Temporal labeling**: Prompts explicitly state the current timestep. "It's now Week 5 of this situation." Agents can reason about time - how long something has been going on, whether their views have been stable or shifting.

**Emotional trajectory**: The system detects sentiment trends and renders them as narrative. "I've been feeling increasingly negative since this started" or "My mood has been fairly steady." This gives agents emotional continuity between timesteps instead of starting fresh every time.

**Conviction self-awareness**: Agents know how firm they've been. "I've been firm about this since Week 1" or "I started certain but my certainty has been slipping." This enables commitment bias (consistent agents resist change) and openness (wavering agents are more receptive).

**Intent accountability**: If an agent said they'd do something, they get reminded. "Last week you said you were going to look into alternatives. Has anything changed?" This prevents agents from making bold claims they never follow through on.

**Repetition detection**: When agents keep thinking the same thing for multiple timesteps, they get nudged: "You've been thinking the same things for a while now. Has anything actually changed? Are you starting to doubt yourself? Have you done anything about it, or just thought about it?" This prevents stale convergence where agents repeat identical reasoning.

**Conviction decay**: Strong opinions fade without reinforcement. A conviction score of 0.9 doesn't stay at 0.9 forever. Configurable decay rate means you can model how quickly certainty erodes.

**Flip resistance**: High-conviction agents are harder to move. If someone is absolutely certain, new information needs to be compelling to shift them. This prevents unrealistic opinion swings.

---

## Social Dynamics That Emerge

You don't program social behavior explicitly. It emerges from the mechanics.

**Peer pressure**: Agents see what their network neighbors think. "My coworker Darnell is strongly opposed. My neighbor Maria is on board." This named, specific peer pressure is more realistic than abstract statistics.

**Conformity variation**: Agents have a `conformity` attribute (0-1). High-conformity agents get prompted with "I tend to go along with what most people around me are doing." Low-conformity agents get "I tend to form my own opinion regardless of what others think." This shapes how they weight peer opinions.

**Local mood**: Agents sense the aggregate sentiment of their network. "Most people around me seem worried." This is vibes, not statistics - realistic ambient social pressure.

**Macro trends as context**: Population-level shifts get injected as background. "The general mood is shifting toward acceptance." "More and more people are taking action." Agents sense the zeitgeist without knowing exact numbers.

**Viral sharing**: Agents with high conviction are more likely to share. When they share, their network neighbors get exposed. Popular opinions spread; unpopular ones don't. Network structure determines what goes viral - well-connected agents amplify.

---

## Agent Conversations

Agents can talk to each other. When reasoning, an agent can choose to initiate a conversation with someone in their network.

**Talk-to actions**: During reasoning, agents can request to talk to someone - a coworker, neighbor, partner, or family member. The request includes who they want to talk to and what topic they want to discuss.

**Multi-turn exchanges**: Conversations are full back-and-forth dialogues. The initiator speaks, the target responds, they go back and forth for 2-3 turns depending on fidelity settings. Each turn is a complete LLM call with the speaker's persona and context.

**State updates from conversations**: Conversations change minds. After talking, both participants get updated sentiment, conviction, and potentially position. These conversation-driven state changes override provisional reasoning state - a compelling conversation can shift someone more than passive exposure.

**Partner and NPC conversations**: Agents can talk to their partner (whether another full agent or an NPC spouse) or household dependents. NPC conversations work the same way, but only the initiator's state updates - NPCs don't have persistent state.

**Conflict resolution**: When multiple agents want to talk to the same target, priority determines who wins. Higher relationship weight wins - a partner request beats a coworker request. Deferred requests can execute in later timesteps.

**Fidelity control**: The `--fidelity` flag controls conversation depth and prompt richness:
- `low`: No conversations, last 5 memory traces, basic prompts
- `medium` (default): 2 turns (4 messages), 1 conversation per agent, full memory traces
- `high`: 3 turns (6 messages), up to 2 conversations per agent, explicit THINK vs SAY separation, repetition detection

---

## Social Feed and Public Discourse

Agents don't just see their direct network neighbors. They perceive broader public discourse.

**Public statements become posts**: When agents share (set `will_share=True`), their public statement gets recorded as a social post. These accumulate over timesteps, creating a timeline of what people are saying.

**Social feed beyond network**: In subsequent timesteps, agents see a "What People Are Saying Online" section in their prompts. This shows recent posts from the broader population - not just direct neighbors. It's like seeing trending takes on social media from strangers.

**Network vs. public**: Peer opinions come from direct network neighbors. The social feed comes from everyone else who's sharing. Agents experience both - personal network influence and ambient public discourse.

**Configurable lookback**: The feed pulls from recent timesteps (default: 3 timesteps back, 5 posts shown). This models the recency bias of social media - you see what's trending now, not old takes.

---

## Scenarios You Can Run Today

To make it concrete, here are scenarios that work right now with no additional development:

- US households responding to a streaming service price increase
- Japanese employees adapting to return-to-office mandates
- Indian consumers in multiple cities evaluating a new fintech app
- Brazilian families weighing migration decisions
- UK residents responding to congestion pricing expansion
- German citizens reacting to energy policy changes
- Mixed urban/rural populations facing a natural disaster
- Multi-generational households navigating technology adoption
- Professional networks processing industry disruption news
- Religious communities responding to doctrinal changes
- Parent networks reacting to school policy updates
- Couples having conversations that shift their positions
- Workplace discussions that change minds
- Social media dynamics where public discourse influences individuals
- Any population, any country, any event, any outcome structure

---

## Cognitive Depth at High Fidelity

At `--fidelity high`, agents get additional cognitive architecture features:

**THINK vs SAY separation**: Prompts explicitly distinguish between internal monologue (raw, honest thoughts) and public statement (socially filtered). Agents with high agreeableness might have a large gap between what they think and what they say - that's interesting data.

**Repetition detection**: If an agent's reasoning is too similar to their previous timestep (>70% trigram overlap), they get a prompt nudge forcing them to go deeper. This prevents the stale convergence where agents just repeat "save money, learn AI, find backup work" verbatim for 5 timesteps.

These features are always-on at high fidelity, adding cognitive realism without additional configuration.

---

## Scenarios You Can Run Today

To make it concrete, here are scenarios that work right now with no additional development:

- US households responding to a streaming service price increase
- Japanese employees adapting to return-to-office mandates
- Indian consumers in multiple cities evaluating a new fintech app
- Brazilian families weighing migration decisions
- UK residents responding to congestion pricing expansion
- German citizens reacting to energy policy changes
- Mixed urban/rural populations facing a natural disaster
- Multi-generational households navigating technology adoption
- Professional networks processing industry disruption news
- Religious communities responding to doctrinal changes
- Parent networks reacting to school policy updates
- Couples having conversations that shift their positions
- Workplace discussions that change minds
- Social media dynamics where public discourse influences individuals
- Crisis scenarios that evolve over days/weeks with new developments
- Any population, any country, any event, any outcome structure

The core simulation engine is complete through Phase F - households, networks, timelines, conversations, social posts, cognitive architecture, fidelity tiers, and results export.

---

## What You Get Out

After simulation runs, you have:

**Position distributions**: What fraction of the population supports, opposes, or remains neutral? Segmented by any attribute - how do young people differ from old? Urban from rural? High-income from low-income?

**Sentiment trajectories**: How did emotional response evolve over time? Did initial negativity soften? Did enthusiasm fade?

**Conviction patterns**: Where are the true believers vs. the persuadable middle? How does certainty correlate with position?

**Sharing behavior**: Who's talking about this? Which demographics amplify vs. stay silent?

**Reasoning traces**: The actual first-person reasoning each agent produced. Qualitative insight into why people think what they think.

**Network effects**: How did information flow? Which communities adopted early? Where did resistance cluster?

**Conversation impact**: Which conversations changed minds most? Ranked by sentiment and conviction delta. See exactly what was said that moved people.

**Elaborations export**: Flattened CSV with every agent's demographics, outcomes, and reasoning - ready for downstream analysis, clustering, or visualization.

**Social posts timeline**: Every public statement made during the simulation, with agent name, position, and sentiment.
