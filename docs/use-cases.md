# Use Cases

What Extropy can simulate — and what it can't.

---

## What Extropy Is Built For

Extropy simulates **targeted synthetic populations** — statistically grounded yet semantically enriched groups of heterogeneous agents. It models how real populations respond to events by combining:

1. **Statistical grounding** — Real-world distributions (census, research) with source citations for every attribute
2. **Semantic extrapolation** — LLM-inferred psychographic and behavioral attributes derived from grounded demographics
3. **Social propagation** — Network-based information spread where agents influence each other through semantic arguments, not just labels
4. **Temporal dynamics** — Multi-timestep belief evolution with conviction, memory, and flip resistance

The output is distributional predictions segmented by any attribute — not a single number, but a picture of who does what and why.

---

## Scenario Categories

### Market Research & Consumer Behavior

**Synthetic survey replacement.** Traditional surveys take weeks, cost $50k+, and capture stated preferences rather than likely behavior. Extropy simulates the population and measures behavioral intent.

```bash
extropy spec "2,000 US women aged 25-45 who regularly purchase skincare products" -o skincare/base.yaml
extropy extend skincare/base.yaml \
  -s "Launch of a $65 retinol serum positioned as clean beauty, sold DTC only" \
  -o skincare/population.yaml
```

The attribute discovery layer finds what actually matters for *this* population — ingredient sensitivity, brand loyalty patterns, price anchoring from current routine spend, clean beauty ideology, social media influence susceptibility — rather than forcing generic demographic buckets.

**Outputs:** Purchase intent distribution, price sensitivity curves by segment, channel resistance (DTC-only vs. retail preference), word-of-mouth likelihood through the network.

**Why simulation beats surveys:** People say one thing in surveys and do another. Self-reported willingness-to-pay runs 2-3x higher than actual behavior. Agents reason through real trade-offs rather than performing for an interviewer.

---

### Pricing Strategy & Elasticity

**Use case:** Predicting behavioral response to price changes in context — with competing budget pressures, inertia, and social influence.

```bash
extropy spec "1,500 US households currently subscribing to 2+ streaming services" -o streaming/base.yaml
extropy extend streaming/base.yaml \
  -s "Netflix announces a $5/month price increase across all tiers" \
  -o streaming/population.yaml
```

**Discovered attributes:** Current monthly entertainment spend, service stacking behavior, content consumption patterns, churn history, password-sharing arrangements, free alternative awareness.

**Outputs:** Churn probability by segment, downgrade-to-ad-tier likelihood, substitution patterns (which competitor captures defectors), and time-to-churn curves. Network propagation shows how "everyone's canceling" sentiment spreads through social clusters.

---

### Public Policy & Compliance

**Use case:** Simulating compliance with mandates in heterogeneous, culturally distinct populations to identify friction points before implementation.

```bash
extropy spec "500 Austin TX commuters who drive into downtown for work" -o austin/base.yaml
extropy extend austin/base.yaml \
  -s "Response to a $15/day downtown congestion tax during peak hours" \
  -o austin/population.yaml
```

**Outputs:** Mode shift predictions, revenue projections based on actual likely compliance, protest/opposition intensity by segment, equity analysis showing disproportionate burden on populations without transit access.

**Why this matters:** New York's congestion pricing launched after years of debate with minimal behavioral modeling. Stockholm's success vs. Manchester's referendum failure shows the same policy produces radically different outcomes depending on the population.

---

### Product Launch & Adoption Curves

**Use case:** Predicting adoption patterns for complex products where utility isn't the only factor.

```bash
extropy spec "2,000 practicing physicians who currently prescribe GLP-1 agonists" -o glp1/base.yaml
extropy extend glp1/base.yaml \
  -s "FDA approves an oral GLP-1 alternative at 40% lower cost with comparable efficacy" \
  -o glp1/population.yaml
```

**Discovered attributes:** Formulary access patterns, pharma rep relationships, patient population severity, prior authorization fatigue, comfort with newer molecules, peer network influence from KOLs, insurance mix.

**Outputs:** Adoption curve by physician segment, switching triggers and barriers, time-to-adoption distribution. Network propagation shows how endorsements cascade through academic affiliations.

---

### Political & Message Testing

**Use case:** Testing messaging on granular voter segments to understand resonance before deployment.

```bash
extropy spec "5,000 registered voters in Pennsylvania Congressional District 7" -o pa7/base.yaml
extropy extend pa7/base.yaml \
  -s "Candidate proposes eliminating the carried interest tax loophole" \
  -o pa7/population.yaml
```

**Discovered attributes:** Local economic exposure, tax policy awareness, partisan lean at precinct level, media diet composition, issue salience rankings.

**Outputs:** Vote shift probability, enthusiasm impact, and — critically — how the message propagates differently through different social clusters within the same district.

---

### Crisis Response & Reputation

**Use case:** Pre-testing crisis responses against a synthetic population of actual stakeholders.

```bash
extropy spec "3,000 active customers of a mid-tier US airline" -o airline/base.yaml
extropy extend airline/base.yaml \
  -s "Viral video of passenger being forcibly removed from overbooked flight" \
  -o airline/population.yaml
```

**Outputs:** Customer defection probability by loyalty tier, social media amplification likelihood, boycott participation rates. Test multiple response strategies (apology, deflection, policy change) against the same population.

---

### Information Spread & Narrative Resilience

**Use case:** Modeling how misinformation spreads and mutates within specific demographic bubbles.

```bash
extropy spec "1,000 residents of a coastal town" -o coastal/base.yaml
extropy extend coastal/base.yaml \
  -s "Rumor spreads that the water supply is contaminated, contradicting official reports" \
  -o coastal/population.yaml
```

**Discovered attributes:** Institutional trust, media literacy, information source preference (social media vs. official news), anxiety level.

**Outputs:** Which demographic groups are most susceptible, how the rumor propagates through trust-based clusters, and where targeted communication interventions would be most effective.

---

### Real Estate & Community Response

**Use case:** Predicting community response to development proposals beyond the vocal minority.

```bash
extropy spec "1,000 residents within 2 miles of a proposed development site in East Austin" -o dev/base.yaml
extropy extend dev/base.yaml \
  -s "Proposal for a 400-unit mixed-use development with 15% affordable units" \
  -o dev/population.yaml
```

**Outputs:** Support/opposition distribution, the specific concerns driving opposition (traffic vs. character vs. displacement), identification of persuadable segments, and how opposition organizes through the social network.

---

## Cross-Cutting Strengths

**Discovered schemas beat templates.** Skincare buyers, Pennsylvania voters, and GLP-1 prescribing physicians require completely different attribute schemas. The attribute discovery layer finds what matters for each population rather than forcing generic demographic templates.

**Network propagation is the differentiator.** Most synthetic survey tools generate independent agents. Extropy's social network layer captures how opinions spread, cluster, and cascade through real social structures — the thing surveys fundamentally cannot measure.

**Two-pass reasoning eliminates central tendency.** Pass 1 (freeform reasoning) -> Pass 2 (classification) prevents the LLM from collapsing everything to safe middle responses.

**The spec-as-IR pattern enables continuous intelligence.** Once a population spec is built, the same base can be re-extended for new scenarios without rebuilding — ongoing monitoring rather than one-off studies.

---

## What Extropy Does NOT Do

Extropy simulates **populations** (social graphs), not **organizations** (hierarchies) or **physics** (spatial systems).

### Rigid Organizational Hierarchy

Agents are independent nodes in a social network, not positions in an org chart. Extropy cannot model reporting lines, workflow dependencies, or strict cardinalities ("there can be only one CEO").

**Not supported:** "How will the approval chain break down if the VP quits?"

### Physical & Spatial Logistics

Geography in Extropy is a semantic label ("Austin"), not a coordinate system. No distance, velocity, collision, or capacity.

**Not supported:** "Optimize warehouse foot traffic" or "Evacuation route bottlenecks."

### High-Frequency Quantitative Models

LLM reasoning is semantic and probabilistic, not mathematically precise or real-time.

**Not supported:** "Predict millisecond price fluctuations after a rate hike."

### Multi-Event Cascades (Current Limitation)

The current system supports single-event scenarios. Sequential, reactive event chains ("Netflix raises price, then CEO tweets, then competitor announces promo") require a game-master loop not yet implemented. You can simulate events individually or bundle them into one description.
