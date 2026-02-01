# CeraVe Retinol Serum Launch — Consumer Research Study

**Prepared for**: Adi, Anthony  
**Date**: January 31, 2026  
**Status**: Ready for simulation (pending approval)  
**Meeting**: Thursday planning session

---

## Executive Summary

Synthetic consumer study to understand how women aged 20-45 would respond to CeraVe's new Retinol Resurfacing Serum. The study measures:

1. **Purchase intent** — Will they buy?
2. **Price sensitivity** — At what price points?
3. **Channel friction** — How much does DTC-only hurt adoption?

---

## Population Overview

| Metric | Value |
|--------|-------|
| **Population size** | 500 agents |
| **Geography** | US (state-weighted by Census) |
| **Demographics** | Women aged 20-45 |
| **Grounding level** | Medium (124 sources) |
| **Attributes** | 33 total |

### Demographic Breakdown

| Segment | Distribution |
|---------|-------------|
| **Age** | Uniform 20-45 (μ = 32.6) |
| **Education** | Bachelor's 34%, Some college 29%, HS 23%, Graduate 14% |
| **Ethnicity** | White 58%, Hispanic 19%, Black 13%, Asian 7%, Other 3% |
| **Location** | Urban 47%, Suburban 43%, Rural 10% |
| **Income (HH)** | μ = $92,465 (σ = $53,895) |
| **Employment** | Full-time 64%, Part-time 12%, Self-employed 11% |

### Skincare Behavior

| Segment | Distribution |
|---------|-------------|
| **Routine complexity** | Moderate (4-6 products) 48%, Minimal 35%, Extensive 17% |
| **Retinol experience** | Never used 33%, Regular 27%, Occasional 23%, Former 17% |
| **Monthly skincare spend** | μ = $77 (σ = $74) |
| **Skin type** | Combination 30%, Dry 24%, Oily 21%, Normal 16%, Sensitive 9% |
| **Primary concerns** | Aging/wrinkles 41%, Acne 22%, Hyperpigmentation 19% |

### Shopping & Influence Patterns

| Segment | Distribution |
|---------|-------------|
| **Brand loyalty** | Somewhat loyal 45%, Switches frequently 39%, Very loyal 16% |
| **Reviews influence** | Heavily 48%, Somewhat 29%, Only reviewed 13% |
| **Clean beauty preference** | Somewhat important 37%, Very important 33%, Not important 19% |
| **Social media daily** | μ = 2.6 hours |
| **Subscription service user** | 32% |

---

## Product Scenario

### The Launch

| Element | Details |
|---------|---------|
| **Product** | CeraVe Retinol Resurfacing Serum |
| **Price** | $22 |
| **Channels** | DTC only — cerave.com, Instagram Shop, TikTok Shop |
| **NOT available at** | Ulta, Sephora, Target, CVS, Walgreens, Walmart |

### Product Claims

- Dermatologist-developed
- Encapsulated 0.3% retinol
- MVE (MultiVesicular Emulsion) delivery technology
- Slow release for minimal irritation
- Fragrance-free

### Why DTC-Only Matters

CeraVe's core positioning is "drugstore dermatologist brand" — accessible, affordable, everywhere. DTC-only is a significant departure that will test:

1. Whether brand trust transfers to a new channel
2. Price tolerance outside retail context
3. Friction tolerance among different shopping preference segments

---

## Exposure Channels

The simulation exposes agents through 5 channels, mimicking realistic discovery patterns:

| Channel | Type | Credibility | Primary Audiences |
|---------|------|-------------|-------------------|
| **Brand email/app** | Targeted | 1.2x | Existing customers, subscribers, retinol users |
| **Social media ads** | Targeted | 0.95x | Heavy social users, younger demo, online shoppers |
| **Beauty influencers** | Organic | 0.85x | Influencer followers, social media engaged |
| **Beauty news sites** | Broadcast | 1.1x | Ingredient-aware, heavy researchers |
| **Word of mouth** | Organic | 0.9x | Extroverts, social skincare community |

---

## Outcomes Measured

### Primary: Purchase Intent (Categorical)

- `purchase_immediately` — Will buy at launch
- `purchase_if_cheaper` — Price is the blocker
- `consider_later` — Interested but not now
- `unlikely_due_to_dtc` — Would buy if retail available
- `not_interested` — Not for them

### Secondary: Sentiment (Float)

- Scale: -1.0 (very negative) to +1.0 (very positive)
- Captures overall reaction to the launch

### Derived Analysis (Post-simulation)

- Price sensitivity curves at $18 / $22 / $28 / $35
- Channel friction by segment (shopping preference × purchase intent)
- Segment-level conversion funnels

---

## Sample Persona

Below is **Agent 0** from the population, showing how the LLM will "think" as this consumer:

---

> ### Who I Am
> 
> I'm a 36-year-old woman living in Louisiana. I'd describe my skin as combination, and my main concern is hyperpigmentation. My current skincare routine is moderate, and I spend about $189 per month on products. When it comes to retinol, I've never used it.
>
> ### My Household
> 
> I'm single. There are 2 people living in my household.
>
> ### Economic Profile
> 
> My household income is $122,270 per year. I personally earn $67,828 annually. I work full-time. I have basic health insurance coverage.
>
> ### Shopping Preferences
> 
> I prefer to buy skincare at beauty specialty stores like Sephora or Ulta. I switch between skincare brands frequently. I'm very price-sensitive when buying skincare. I don't currently use any subscription-based beauty or skincare services.
>
> ### What Influences My Purchases
> 
> Online reviews heavily influence my purchase decisions. Clean or natural beauty products are very important to me. I have high knowledge about skincare ingredients.
>
> ### Digital & Social Media
> 
> I spend around 3.4 hours on social media each day. I actively follow beauty influencers.
>
> ### Personality
> 
> I have a moderate tolerance for risk with new products. I'm much more impulsive with purchases than most—I often buy on a whim. I'm much less open to new experiences than most people. I'm more outgoing and energetic than most people. I experience more anxiety and emotional sensitivity than most people.

---

**Prediction for this agent**: Likely `unlikely_due_to_dtc` or `consider_later` — she's price-sensitive, prefers Ulta/Sephora, has never used retinol, but is influenced by reviews and follows beauty influencers. The DTC-only channel is a significant friction point for her shopping pattern.

---

## Simulation Config

| Parameter | Value |
|-----------|-------|
| **Max timesteps** | 100 hours |
| **Stop condition** | >95% exposure + 10 timesteps of stability |
| **Share probability** | 45% base (modified by personality/behavior) |
| **Network** | 500 nodes, 4,977 edges, avg degree 19.9 |
| **Pipeline LLM** | Claude (Anthropic) |
| **Simulation LLM** | OpenAI |

---

## Files Generated

```
studies/cerave-retinol-launch/
├── population.yaml          # 33 attributes, 124 sources
├── population.persona.yaml  # Persona rendering config
├── agents.json              # 500 sampled agents
├── network.json             # Social network (4,977 edges)
├── scenario.yaml            # Full scenario spec
└── STUDY_BRIEF.md           # This document
```

---

## Next Steps

1. **Review this brief** — Any adjustments to scenario or population?
2. **Approve simulation run** — `entropy simulate studies/cerave-retinol-launch/scenario.yaml`
3. **Results analysis** — Segment breakdowns, price curves, channel friction quantification

---

## Questions for Thursday

1. Do we want to add price threshold questions directly into the scenario? (Currently inferring from purchase intent)
2. Should we compare against a hypothetical retail-available variant?
3. Any specific segments to over-sample for deeper analysis?

---

*Generated by Entropy v0.1.0 | Pipeline: Claude | Simulation: OpenAI (pending)*
