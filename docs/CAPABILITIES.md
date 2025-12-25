# Entropy Capabilities: Targeted Synthetic Populations & Predictive Scenarios

This document defines the scope of **predictive intelligence scenarios** supported by the Entropy framework. It analyzes the specific technical capabilities implemented in Phases 1, 2, and 3 (as detailed in `WORKFLOW.md`) to clarify exactly what kinds of problems Entropy can solve.

---

## Table of Contents

1. [Core Philosophy: Targeted Synthetic Populations](#1-core-philosophy-targeted-synthetic-populations)
   - [The "Grounding + Extrapolation" Engine](#the-grounding--extrapolation-engine-phase-1)
   - [Temporal & Emergent Flexibility](#2-temporal--emergent-flexibility)
2. [Supported Predictive Scenarios](#2-supported-predictive-scenarios)
   - [Scenario 1: Diffusion of Innovation](#scenario-1-diffusion-of-innovation-targeted-market-fit)
   - [Scenario 2: Public Policy & Compliance](#scenario-2-public-policy--compliance-stress-testing)
   - [Scenario 3: Information Warfare](#scenario-3-information-warfare--narrative-resilience)
   - [Scenario 4: Professional Community Alignment](#scenario-4-professional-community-alignment-unionguild-dynamics)
   - [Scenario 5: Electorate "Message Market" Fit](#scenario-5-electorate-message-market-fit)
3. ["Anti-Patterns" (What Entropy Does NOT Do)](#3-anti-patterns-what-entropy-does-not-do)
4. [Summary of Distinction](#summary-of-distinction)

---

## 1. Core Philosophy: Targeted Synthetic Populations

Entropy is designed to simulate **Targeted Synthetic Populations**—statistically grounded yet semantically enriched groups of heterogeneous agents.

### The "Grounding + Extrapolation" Engine (Phase 1)

Unlike static census data, Entropy builds agents that have "interior lives" capable of reasoning. This is achieved through a two-step process in **Phase 1: Population Creation**:

1.  **Statistical Grounding (The "Who"):**

    - **Implementation:** `hydrate_independent()` (Step 2a) fetches real-world distributions (e.g., "Age: Normal(47, 10)", "Income: LogNormal(...)").
    - **Value:** Ensures the population is demographically realistic and representative of the target group (e.g., "German Surgeons," "Kyoto Homeowners").

2.  **Semantic Extrapolation (The "Why"):**

    - **Implementation:** `hydrate_derived()` and `hydrate_conditional_modifiers()` (Steps 2b-2d) use LLMs to infer unmeasured psychological/behavioral attributes based on the grounded data.
    - **Value:** It fills in the blanks. Census data tells you a person is "45, Rural, Unemployed." Entropy infers their likely "Risk Tolerance," "Trust in Authority," or "Economic Anxiety" based on those facts.

    * **Why this matters:** You cannot download a census of "Surgeons' Trust in AI." Entropy _synthesizes_ this layer, creating agents that are statistically plausible but possess the rich interior life needed for behavioral prediction.

### 2. Temporal & Emergent Flexibility

- **Configurable Time Scales:** Simulations can run in minutes, hours, days, or weeks (`timestep_unit`), allowing modeling of fast-moving crises (hours) or slow policy adoption (months).
- **Outcome Discovery:** Unlike traditional models that only track predefined metrics (e.g., "infected/not infected"), Entropy captures **full agent reasoning**. This allows for post-hoc discovery of _emergent behaviors_—e.g., discovering that users aren't "cancelling" Netflix, but "rotating subscriptions"—a nuance that wasn't in the original outcome list.

**Key Distinction:**

- **Census Data:** "500 people, avg age 45, 60% employed." (Static, shallow).
- **Entropy Population:** "500 people, avg age 45, where _unemployed rural agents have 3x higher distrust in federal institutions than urban employed agents_." (Dynamic, predictive).

---

## 2. Supported Predictive Scenarios

These scenarios are directly enabled by the architecture described in `WORKFLOW.md`.

### Scenario 1: Diffusion of Innovation (Targeted Market Fit)

**Use Case:** Predicting adoption curves for complex/niche products where "utility" isn't the only factor.

- **Scenario:** _"A med-tech firm launches a new AI diagnostic tool for Neurosurgeons that improves accuracy but increases documentation time."_
- **Target Population:** Neurosurgeons (grounded by `specialty`, `years_experience`).
- **Key Attributes (Extrapolated):** `technological_openness` (derived from age/specialty), `professional_autonomy` (derived from employer_type), `administrative_burden_tolerance`.
- **Mechanism:**
  - **Phase 2 (Interaction):** `passive_observation` + `direct_conversation`.
  - **Phase 3 (Reasoning):** Agents weigh "accuracy" vs. "documentation time" based on their `administrative_burden_tolerance`.
- **Outcome:** Entropy predicts _who_ adopts early (young, academic hospitals) vs. _who_ resists (private practice, high autonomy), and _why_ (the documentation time is the dealbreaker, not the AI).

### Scenario 2: Public Policy & Compliance Stress-Testing

**Use Case:** Simulating compliance with mandates in heterogeneous, culturally distinct groups to identify friction points.

- **Scenario:** _"A city introduces a 'congestion tax' for downtown driving to reduce traffic."_
- **Target Population:** Local Commuters (grounded by `zip_code`, `commute_method`).
- **Key Attributes (Extrapolated):** `economic_price_sensitivity` (derived from income), `environmental_values`, `trust_in_local_gov`.
- **Mechanism:**
  - **Phase 2 (Event):** "Policy Change" event with specific cost details.
  - **Phase 3 (Reasoning):** Agents calculate personal ROI. High-income green voters support it; low-income drivers feel targeted.
- **Outcome:** Reveals second-order effects: e.g., "Wealthy suburbanites pay the tax and complain; gig workers are forced to quit or protest." It identifies _inequity_ before implementation.

### Scenario 3: Information Warfare & Narrative Resilience

**Use Case:** Modeling how misinformation spreads and mutates within specific demographic bubbles.

- **Scenario:** _"A rumor spreads in a coastal town that the water supply is contaminated, contradicting official reports."_
- **Target Population:** Residents (grounded by location).
- **Key Attributes (Extrapolated):** `institutional_trust`, `media_literacy`, `information_source_preference` (social media vs. official news).
- **Mechanism:**
  - **Phase 2 (Exposure):** `word_of_mouth` channel with high ambiguity.
  - **Phase 2 (Interaction):** `share_modifiers` increase sharing probability for agents with low `institutional_trust` or high `anxiety`.
  - **Phase 3 (Network):** The rumor propagates through "low trust" clusters faster than official corrections.

* **Outcome:** Data identifying "epistemic vulnerability"—revealing which demographic groups are most susceptible to the rumor, allowing for targeted communication interventions.

### Scenario 4: Professional Community Alignment (Union/Guild Dynamics)

**Use Case:** Predicting collective action (strikes, adoption, boycott) within a specialized workforce.

- **Scenario:** _"The AMA changes certification requirements, requiring 50 extra hours of CME credits annually."_
- **Target Population:** Board-certified doctors.
- **Key Attributes (Extrapolated):** `career_ambition` vs. `retirement_proximity`, `burnout_level`.
- **Mechanism:**
  - **Phase 3 (Reasoning):** Older doctors with high `retirement_proximity` decide to retire early (Exit). Younger doctors with high `career_ambition` comply (Loyalty). Burned-out mid-career doctors protest (Voice).
- **Outcome:** Predicts a "silent brain drain" (mass early retirement) that simple polling ("Do you like this policy?") would miss.

### Scenario 5: Electorate "Message Market" Fit

**Use Case:** Testing political or corporate messaging on granular voter segments to optimize resonance.

- **Scenario:** _"A candidate pivots their stance on fracking to win rural votes, framing it as 'Energy Independence'."_
- **Target Population:** Rural voters in swing states.
- **Key Attributes (Extrapolated):** `economic_dependence_on_energy`, `local_environmental_concern`, `identity_alignment`.
- **Mechanism:**
  - **Phase 3 (Reasoning):** Agents parse the semantic framing ("Energy Independence").
  - **Nuance:** Agent A (Oil worker) likes the job security. Agent B (Farmer nearby) worries about groundwater. Agent C (Green voter) sees it as a betrayal.
- **Outcome:** Functions as a **Synthetic Focus Group** at scale. It reveals the specific _trade-offs_ voters make ("I like the jobs, but I'm worried about my water") rather than just a binary "Yes/No."

---

## 3. "Anti-Patterns" (What Entropy Does NOT Do)

Entropy simulates **populations** (social graphs), not **organizations** (hierarchies) or **physics** (spatial maps).

### ❌ 1. Rigid Organizational Hierarchy

- **The Constraint:** Entropy agents are independent nodes in a social network (Phase 1/Network Gen). They do not adhere to a rigid Org Chart (CEO -> VP -> Director -> IC).

* **Unsupported Scenario:** _"How will the approval chain for Project X break down if the VP of Engineering quits?"_
* **Reason:** The system does not model "reporting lines," "workflow dependencies," or strict cardinalities (e.g., "There can be only one CEO").

### ❌ 2. Physical & Spatial Logistics

- **The Constraint:** "Geography" in Entropy is a semantic label (e.g., "Berlin"), not a coordinate system.
- **Unsupported Scenario:** _"Optimizing warehouse foot traffic during Black Friday."_ or _"Evacuation route bottlenecks."_
- **Reason:** Agents have no concept of distance, velocity, collision, or spatial capacity.

### ❌ 3. High-Frequency Quantitative Trading

- **The Constraint:** LLM reasoning is semantic and probabilistic (Phase 3), not mathematically precise or instantaneous.
- **Unsupported Scenario:** _"Predicting the exact millisecond price fluctuation of AAPL after a rate hike."_
- **Reason:** This requires mathematical equilibrium models and sub-second latency, not behavioral simulation.

### ❌ 4. Multi-Event Cascades (MVP Limitation)

- **The Constraint:** The current MVP supports **single-event scenarios**.
- **Unsupported Scenario:** _"Netflix raises price, then CEO tweets justification, then competitor announces promo."_
- **Reason:** Sequential, reactive event chains require a "Game Master" loop that is planned for post-MVP. Currently, you must simulate events individually or bundle them into one description.

---

## Summary of Distinction

| Feature         | **Census / Market Data** | **Entropy Synthetic Population** |
| :-------------- | :----------------------- | :------------------------------- |
| **Data Source** | Historical, Survey-based | Grounded + **Extrapolated**      |
| **Attributes**  | Demographics (Who)       | **Psychographics (Why)**         |
| **Granularity** | Aggregate Stats          | **Individual Agents**            |
| **Interaction** | None (Static)            | **Social Propagation (Dynamic)** |
| **Use Case**    | Retrospective Analysis   | **Predictive Simulation**        |
