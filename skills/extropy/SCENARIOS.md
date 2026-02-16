# Scenarios

What Extropy can simulate, with exhaustive permutations.

## Core Capability

Extropy simulates **how populations respond to events**. It creates synthetic agents grounded in real-world distributions, connects them in social networks, and has each agent reason individually using an LLM. Opinions form, spread through the network, and evolve.

Not a survey. Not a poll. A simulation of collective human behavior.

## Population Dimensions

### Geography

Any country or region. Examples:
- US (bundled SSA + Census name data)
- Japan, India, Brazil, UK, Germany (provide NameConfig or let LLM research)
- Multi-region (agents distributed across cities, states, countries)

Extropy doesn't care about geography - it cares about attributes and distributions. Define `state`, `city`, `region` as attributes with your desired distribution.

### Household Mode

| Mode | What You Get | Use When |
|------|--------------|----------|
| `household_mode: false` | Independent individuals | Workplace scenarios, B2B, professional networks |
| `household_mode: true` | Family units with correlated partners and NPC dependents | Consumer decisions, household economics, family dynamics |

### Agent Focus (when household mode is on)

| Focus | Who Reasons | Use When |
|-------|-------------|----------|
| `primary` (default) | Primary adult only; partner/kids are NPCs | One decision-maker per household ("the subscriber") |
| `couples` | Both adults; kids are NPCs | Both partners' opinions matter (major purchases, political views) |
| `families` / `all` | All household members | Teen influence on parents, multi-generational dynamics |

### Household Types

Sampled by age bracket:
- **Singles**: One adult, no kids
- **Couples**: Two adults, no kids
- **Single parents**: One adult with kids
- **Couples with kids**: Two adults with kids
- **Multi-generational**: Extended family

Young adults skew toward singles/couples. Middle-aged toward families. Elderly toward couples/singles.

## Scenario Dimensions

### Event Types

| Type | Examples |
|------|----------|
| `announcement` | Company policy, product feature, organizational change |
| `news` | Breaking news, industry report, research findings |
| `policy_change` | Government regulation, tax change, zoning decision |
| `product_launch` | New product, service update, feature rollout |
| `rumor` | Unconfirmed reports, speculation, leaked information |
| `emergency` | Crisis, natural disaster, security incident |
| `observation` | Behavioral change noticed in environment |

### Timeline Mode

| Mode | When to Use |
|------|-------------|
| **Static** (single event) | Discrete announcement, one-time decision. Event doesn't change; what evolves is awareness and opinion. |
| **Evolving** (timeline) | Situation develops over time. New information arrives at specified timesteps. Agent reasoning evolves with context. |

Static example: Netflix price hike announced → population responds over 2-3 weeks.

Evolving example:
- Week 1: AI announcement, uncertainty about scope
- Week 3: First layoffs reported
- Week 5: Government response announced
- Week 8: Retraining programs launched

Each timeline event gets injected into agent prompts as "what's happened since last time."

### Timestep Units

| Unit | Typical Use |
|------|-------------|
| `days` | Crisis response, breaking news, urgent decisions |
| `weeks` | Policy rollout, product adoption, attitude shifts |
| `months` | Market dynamics, long-term behavior change |
| `years` | Generational shifts (rare) |

### Exposure Channels

| Channel Type | Reach | Trust | Examples |
|--------------|-------|-------|----------|
| `broadcast` | High reach, many agents | Varies | News media, social feeds, public announcements |
| `targeted` | Filtered by attributes | Higher | Email notifications, employer HR, official notices |
| `organic` | Network-dependent | High | Word of mouth, peer conversations, neighbor observation |

Channels have:
- `reach`: Probability of exposure per eligible agent
- `credibility_modifier`: Affects how seriously agents take the information
- `experience_template`: How the agent experiences it ("I saw a news segment about this" vs "My coworker mentioned it")

## Outcome Types

| Type | Schema | Use When |
|------|--------|----------|
| `categorical` | Enum options | Known decision space (buy/wait/skip, support/oppose/neutral) |
| `boolean` | Yes/No | Binary decisions (will share, will attend, will switch) |
| `float` | Range [-1,1] or [0,1] | Intensity measures (sentiment, likelihood, trust level) |
| `open_ended` | Free text | Unknown decision space; categories discovered post-hoc |

Mix outcome types in the same scenario:
```yaml
outcomes:
  suggested_outcomes:
    - name: action
      type: categorical
      options: [pay_premium, cancel, switch_service, workaround]
    - name: sentiment
      type: float
      range: [-1.0, 1.0]
    - name: will_share
      type: boolean
    - name: concerns
      type: open_ended
```

## Fidelity Tiers

| Tier | Conversations | Memory | Cognitive Features | Cost |
|------|---------------|--------|-------------------|------|
| `low` | None | Last 5 traces | Basic | ~$0.03/agent |
| `medium` | Top 1 edge (partner/closest) | All traces | Standard | ~$0.04/agent |
| `high` | Top 2-3 edges | All + beliefs | THINK vs SAY, repetition detection | ~$0.05/agent |

Use `low` for quick exploration. Use `medium` (default) for production runs. Use `high` when conversation dynamics or cognitive depth matter.

## Social Network Features

### Structural Edges (from attributes)

| Edge Type | Source | Weight |
|-----------|--------|--------|
| `partner` | `partner_id` field | 1.0 |
| `household` | Same `household_id` | 0.9 |
| `coworker` | Same `occupation_category` + region | 0.6 |
| `neighbor` | Same region + similar age | 0.4 |
| `congregation` | Same `religious_affiliation` + high religiosity | 0.4 |
| `school_parent` | Both have school-age kids + same region | 0.35 |

### Similarity Edges (fill remaining degree)

Based on attribute similarity. Become `acquaintance` or `online_contact` edges.

### Scenario Relationship Weights

Scenarios can override relationship importance:
- ASI/workplace threat: coworker weight high
- Household product: partner weight high
- Local policy: neighbor weight high

These affect conversation priority and peer opinion ordering.

## Agent Cognitive Features

### Memory

- Full reasoning history, timestamped
- Emotional trajectory rendering ("I've been anxious since Week 1")
- Conviction self-awareness ("I've been firm about this")
- Intent accountability ("Last week I said I'd look into alternatives...")

### Conversations (medium/high fidelity)

- Agents request `talk_to` actions during reasoning
- Multi-turn exchanges (2 turns at medium, 3 at high)
- Both participants update state independently
- Agent-NPC conversations (kids, elderly parents)

### Repetition Detection (high fidelity)

If reasoning is >70% similar to previous timestep, agent gets nudged:
"You've been thinking the same things for a while. Has anything changed? Have you actually done anything?"

### THINK vs SAY (high fidelity)

Explicit separation between internal monologue (raw, honest) and public statement (socially filtered).

## Decision Domains

Extropy works well for:

| Domain | Example Scenarios |
|--------|------------------|
| **Policy** | Congestion pricing, zoning changes, tax proposals, regulatory shifts |
| **Market/Pricing** | Price increases, subscription tiers, feature paywalls, competitive response |
| **Product** | Feature launches, default changes, UX modifications, platform migrations |
| **Crisis** | Data breaches, recalls, PR incidents, leadership changes |
| **Messaging** | Campaign variants, framing tests, communication strategies |
| **Community** | Development proposals, service changes, local initiatives |
| **Healthcare** | Treatment adoption, behavior change, policy compliance |
| **Enterprise** | Policy rollouts, tool adoption, organizational change |

## What Extropy Does NOT Do

- **Hierarchical/org-chart populations**: Populations are sampled from distributions, not org charts. Can't model "1 CEO, 2 VPs, 10 directors" — use for broad populations, not specific organizational trees
- **Physics/logistics optimization**: Use OR tools, not population simulation
- **Real-time prediction**: Simulations take minutes to hours, not milliseconds
- **Guaranteed outcomes**: Results are simulation-informed forecasts with uncertainty
- **Individual prediction**: Works at population level, not "what will John specifically do"
- **Multi-event cascades**: Better modeled as staged runs with timeline events

## Example Scenario Permutations

### 1. US Household Product Decision

```yaml
population: "5000 US Netflix subscribers, household representation"
household_mode: true
agent_focus: couples  # Both partners decide
timeline_mode: static
event_type: announcement
outcomes: [action (categorical), sentiment (float), will_cancel (boolean)]
fidelity: medium
```

### 2. Japanese Workplace Policy

```yaml
population: "1000 Tokyo office workers in finance"
household_mode: false
timeline_mode: static
event_type: policy_change
outcomes: [compliance_intent (categorical), sentiment (float)]
fidelity: low  # Large population, cost-conscious
```

### 3. Evolving Crisis Response

```yaml
population: "2000 customers of affected company"
household_mode: false
timeline_mode: evolving
timestep_unit: days
timeline:
  - day 1: Initial breach reports
  - day 2: Company confirms
  - day 3: Scope revealed
  - day 5: Compensation offered
  - day 7: CEO resigns
outcomes: [trust_level (float), will_churn (boolean), concerns (open_ended)]
fidelity: medium
```

### 4. Multi-City Product Launch

```yaml
population: "3000 consumers across Mumbai, Delhi, Bangalore"
household_mode: true
agent_focus: primary
distribution:
  city: categorical [mumbai: 0.4, delhi: 0.35, bangalore: 0.25]
timeline_mode: static
outcomes: [adoption_intent (categorical), price_sensitivity (float)]
fidelity: medium
```

### 5. Political Messaging Test

```yaml
population: "2000 registered voters in swing state"
household_mode: true
agent_focus: couples  # Household political dynamics
timeline_mode: static
event_type: announcement
# Run 3-5 message variants as separate scenarios, same population
outcomes: [support_level (categorical), enthusiasm (float), will_share (boolean)]
fidelity: high  # Conversation dynamics matter
```

### 6. Community Planning

```yaml
population: "500 residents near proposed development"
household_mode: true
agent_focus: families  # Kids might care too
timeline_mode: evolving
timeline:
  - week 1: Proposal announced
  - week 3: Public hearing
  - week 5: Environmental report
outcomes: [position (categorical: support/oppose/neutral), concerns (open_ended)]
fidelity: medium
```

## Trigger Phrases

Use this skill when users ask:
- "simulate how people will respond to..."
- "what happens if we raise price by..."
- "which segments will churn/adopt/protest"
- "test these message variants"
- "run scenario analysis with uncertainty"
- "why did this segment flip"
- "compare policy alternatives"
- "model household decision dynamics"
- "how does this spread through the network"

## Capability Quick Check

| Question | Answer |
|----------|--------|
| Can I simulate non-US populations? | Yes, any country with appropriate NameConfig |
| Can I model families? | Yes, `household_mode: true` with configurable `agent_focus` |
| Can I have evolving events? | Yes, use timeline with multiple events |
| Can I get open-ended responses? | Yes, `type: open_ended` outcomes |
| Can agents talk to each other? | Yes, at medium/high fidelity |
| Can I run parameter sweeps? | Yes, vary seed/threshold/fidelity/model |
| Can I compare message variants? | Yes, same population with different scenarios |
| Can I see why agents decided? | Yes, reasoning traces in results |
