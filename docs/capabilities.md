# Extropy Capabilities Reference

This document describes what Extropy can simulate as of Phase C completion. Use this as a reference for designing scenarios and understanding system constraints.

---

## Population Capabilities

### Geographic Scope

| Region | Status | Notes |
|--------|--------|-------|
| United States | Full support | Bundled SSA baby names (1940-2010), Census surnames by ethnicity, state/region distributions |
| Japan | Supported | Requires custom `NameConfig` via LLM research or manual CSV |
| India | Supported | Requires custom `NameConfig` via LLM research or manual CSV |
| Any country | Supported | Provide country-specific name data or use LLM-researched `NameConfig` |

**Name generation** adapts to:
- Gender
- Ethnicity/cultural background
- Birth decade (for first names)
- Country of origin

### Population Structures

| Structure | Status | How to Use |
|-----------|--------|------------|
| Individual sampling | Supported | Default mode, `household_mode: false` |
| Household sampling | Supported | Set `household_mode: true` in population spec |
| Single adults | Supported | Sampled as households of size 1 |
| Couples (no children) | Supported | Two adults with `partner_id` linking them |
| Families with children | Supported | Adults + NPC dependents with age, school status |
| Multi-generational | Supported | Configure `household_size_distribution` accordingly |

### Household Features

| Feature | Description |
|---------|-------------|
| Partner matching | Age-correlated (configurable gap mean/std), ethnicity assortative mating rates |
| Shared attributes | Partners share `last_name`, `household_id`, correlated education/religion/politics |
| Dependent generation | Children as NPCs with `first_name`, `age`, `gender`, `relationship`, `school_status` |
| Household roles | `adult_primary`, `adult_secondary`, `dependent_child`, `dependent_teenager`, etc. |

### Demographic Attributes

Extropy can sample any attribute with a defined distribution. Common built-ins:

| Category | Attributes |
|----------|------------|
| Demographics | `age`, `gender`, `race_ethnicity`, `state`, `region`, `urban_rural` |
| Socioeconomic | `education`, `income_bracket`, `employment_status`, `occupation_category` |
| Psychographics | `political_orientation`, `religious_affiliation`, `religiosity` |
| Personality | Big Five (`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`) |
| Behavioral | `risk_tolerance`, `institutional_trust`, `digital_literacy`, `conformity` |

**Custom attributes**: Define any attribute with `type: categorical | continuous | ordinal` and a distribution.

---

## Network Capabilities

### Edge Types (Role-First)

Structural edges are generated automatically from agent attributes:

| Edge Type | Weight | Source | Cap |
|-----------|--------|--------|-----|
| `partner` | 1.0 | `partner_id` field | 1 |
| `household` | 0.9 | Same `household_id` | Household size |
| `coworker` | 0.6 | Same `occupation_category` + employed | 8 |
| `neighbor` | 0.4 | Same `region` + age within 15 years | 4 |
| `congregation` | 0.4 | Same `religious_affiliation` + religiosity > 0.5 | 4 |
| `school_parent` | 0.35 | Has school-age children in same region | 3 |

### Similarity Edges

After structural edges, remaining degree budget filled by similarity:

| Feature | Description |
|---------|-------------|
| Attribute weighting | Configure which attributes matter for similarity |
| Degree distribution | Power-law (configurable exponent) or uniform |
| Average degree target | Configurable, default ~8 |

### Network Properties

| Property | Supported |
|----------|-----------|
| Weighted edges | Yes, weight 0-1 |
| Bidirectional traversal | Yes |
| Disconnected components | Possible (isolated agents) |
| Hub/influencer generation | Emergent from degree distribution |

---

## Scenario Capabilities

### Event Types

| Type | Description | Use Case |
|------|-------------|----------|
| `announcement` | Official statement from authority | Company policy, government decree |
| `news` | Media coverage of development | Breaking news, investigative report |
| `rumor` | Unverified information spreading | Workplace gossip, social media speculation |
| `policy_change` | Rule/regulation modification | Price change, new law, service update |
| `product_launch` | New offering introduction | Tech release, service expansion |
| `emergency` | Crisis requiring response | Natural disaster, security incident |
| `observation` | Agent witnesses something | Seeing neighbor's behavior, noticing trend |

### Timeline Modes

| Mode | Description | Auto-Detection |
|------|-------------|----------------|
| `static` | Single event, no evolution | Events with no temporal dependencies |
| `evolving` | Multi-timestep narrative | Multiple events or explicit `timeline` field |

**Timeline features**:
- Events at specific timesteps
- Automatic recap in agent prompts ("What's happened so far")
- Background context injection (economic conditions, cultural moment)

### Exposure Channels

| Channel | Description | Targeting |
|---------|-------------|-----------|
| `mainstream_media` | TV, newspapers, major outlets | Broad reach, demographic filtering |
| `social_media` | Platforms, feeds, viral content | Network-based spread |
| `word_of_mouth` | Direct interpersonal | Network edges only |
| `official_communication` | Direct from source | Attribute-based (e.g., employees only) |
| `observation` | Witnessing behavior | Location/proximity based |

**Channel features**:
- Reach probability per channel
- Demographic targeting rules
- Experience templates (how the agent encounters the information)

### Outcome Types

| Type | Description | Example |
|------|-------------|---------|
| `categorical` | Discrete choice from options | `position: [support, oppose, neutral]` |
| `boolean` | Yes/no decision | `will_purchase: true/false` |
| `float` | Continuous value in range | `price_sensitivity: [0, 1]` |
| `open_ended` | Free-text response | `concerns: <any text>` |

**Outcome features**:
- Primary position outcome (first required categorical)
- Required vs optional outcomes
- Extraction instructions for classification

---

## Simulation Capabilities

### Reasoning Modes

| Mode | LLM Calls | Cost | Quality | Flag |
|------|-----------|------|---------|------|
| Two-pass (default) | 2 per agent per timestep | Higher | Better reasoning | (default) |
| Merged pass | 1 per agent per timestep | Lower | Adequate | `--merged-pass` |

**Two-pass**:
1. Pass 1: Free-text role-play reasoning (strong model)
2. Pass 2: Structured classification (fast model)

**Merged pass**:
- Single call with combined schema (reasoning + outcomes)
- Uses strong model only

### Temporal Dynamics

| Feature | Description |
|---------|-------------|
| Timestep labeling | "Week 3", "Day 5", etc. (configurable unit) |
| Memory traces | Full history with timestamps, sentiment, conviction |
| Conviction decay | Configurable decay rate over time |
| Flip resistance | High conviction agents harder to change |
| Intent accountability | "Last week you said X. Has anything changed?" |

### Social Dynamics

| Feature | Description |
|---------|-------------|
| Network propagation | Exposure spreads via edges |
| Peer opinions | Named peers with positions ("My coworker Darnell thinks...") |
| Local mood | Aggregate sentiment of neighbors |
| Macro summary | Population-level trends as ambient context |
| Conformity phrasing | High/low conformity self-awareness in prompts |
| Share behavior | Conviction-weighted sharing decisions |

### Agent Cognition

| Feature | Status | Description |
|---------|--------|-------------|
| First-person voice | Implemented | "I'm a 34-year-old electrician..." |
| Emotional trajectory | Implemented | Sentiment trend detection |
| Memory system | Implemented | Full trace, timestamped, emotional context |
| Conformity awareness | Implemented | Prompt phrasing based on conformity score |
| Repetition detection | Not yet | Trigram similarity to avoid loops |
| Episodic/semantic split | Not yet | Belief consolidation over time |

---

## Example Scenario Configurations

### US Consumer Response to Price Change

```yaml
population:
  size: 500
  household_mode: true
  country: US

scenario:
  event_type: policy_change
  title: "Streaming Service Price Increase"
  timeline_mode: static

  outcomes:
    - name: subscription_decision
      type: categorical
      options: [keep, cancel, downgrade]
      required: true
    - name: price_sensitivity
      type: float
      range: [0, 1]
```

### Japanese Workplace Policy Change

```yaml
population:
  size: 300
  household_mode: false  # Individual employees
  country: JP
  name_config:
    researched: true  # LLM generates culturally appropriate names

scenario:
  event_type: announcement
  title: "Remote Work Policy Update"
  channels:
    - official_communication
    - word_of_mouth
```

### Indian Multi-City Product Launch

```yaml
population:
  size: 1000
  household_mode: true
  country: IN
  attributes:
    - name: city
      type: categorical
      distribution:
        Mumbai: 0.3
        Delhi: 0.25
        Bangalore: 0.2
        Chennai: 0.15
        Kolkata: 0.1

scenario:
  event_type: product_launch
  timeline_mode: evolving
  timeline:
    - timestep: 1
      event: "Product announced in Mumbai"
    - timestep: 3
      event: "Expansion to Delhi and Bangalore"
    - timestep: 5
      event: "Nationwide availability"
```

### Evolving Crisis Scenario

```yaml
scenario:
  event_type: emergency
  timeline_mode: evolving
  max_timesteps: 10
  timestep_unit: day

  timeline:
    - timestep: 1
      event: "Initial reports of incident"
      exposure_rules:
        - channel: social_media
          reach: 0.2
    - timestep: 2
      event: "Official confirmation and safety guidance"
      exposure_rules:
        - channel: mainstream_media
          reach: 0.8
    - timestep: 5
      event: "Situation stabilizing, restrictions lifted"
    - timestep: 8
      event: "Post-incident review published"

  background_context: |
    The region has experienced similar incidents before.
    Trust in local authorities varies by demographic.
```

---

## Current Limitations

| Limitation | Phase | Description |
|------------|-------|-------------|
| No agent-agent conversations | Phase D | Agents don't talk to each other directly |
| No social posts | Phase D | Agents don't create public content |
| No fidelity tiers | Phase F | Can't trade off cost vs quality at runtime |
| No backtesting | Phase G | No validation against historical outcomes |
| Partner/kids not in persona | Phase A gap | Household data exists but not rendered in prompts |

---

## CLI Quick Reference

```bash
# Full pipeline
extropy spec "US adults responding to AI announcement" > spec.yaml
extropy extend spec.yaml > extended.yaml
extropy sample extended.yaml > agents.json
extropy network agents.json > network.json
extropy persona agents.json > personas.json
extropy scenario "AI job displacement fears" --population extended.yaml > scenario.yaml
extropy simulate scenario.yaml --agents agents.json --network network.json

# Simulation options
extropy simulate scenario.yaml \
  --merged-pass           # Single-call reasoning (cheaper)
  --max-timesteps 5       # Limit duration
  --chunk-size 50         # Agents per batch
  --checkpoint-every 5    # Checkpoint frequency

# Results
extropy results <run-id>
```
