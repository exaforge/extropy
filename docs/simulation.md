# Entropy Simulation Engine

The simulation engine is the core of Entropy's Phase 3. It takes a population of agents with attributes, a social network, and a scenario specification, then simulates how information spreads and how agents react over time.

## Core Philosophy

Entropy simulations are **emergent**, not prescriptive:

- No formula says "40% of surgeons will adopt the AI tool"
- That outcome **emerges** from thousands of individual agent decisions
- Each agent reasons independently using LLM-powered cognition
- Social influence propagates through the network
- The simulation discovers what happens

---

## The Simulation Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SIMULATION TIMESTEP LOOP                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for timestep in 0..max_timesteps:                                     │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │ 1. SEED EXPOSURE                                                │ │
│    │    Apply scenario exposure rules                                │ │
│    │    - Who sees the event this timestep?                          │ │
│    │    - Via which channel? (media, announcement, peer)             │ │
│    │    - With what credibility?                                     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │ 2. NETWORK PROPAGATION                                          │ │
│    │    Agents who will_share spread to neighbors                    │ │
│    │    - Edge weights affect share probability                      │ │
│    │    - Relationship type affects credibility                      │ │
│    │    - Echo chamber effects considered                            │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │ 3. AGENT REASONING (LLM-powered)                                │ │
│    │    For each agent needing to reason:                            │ │
│    │    - Build persona from attributes                              │ │
│    │    - Include exposure history                                   │ │
│    │    - Include peer opinions from neighbors                       │ │
│    │    - LLM generates: position, sentiment, will_share, outcomes   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │ 4. STATE UPDATE                                                 │ │
│    │    Update agent states in database                              │ │
│    │    Log events to timeline                                       │ │
│    │    Compute timestep summary                                     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                              ↓                                          │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │ 5. STOPPING CONDITIONS                                          │ │
│    │    Check if simulation should end:                              │ │
│    │    - Max timesteps reached                                      │ │
│    │    - Exposure rate > threshold AND no state changes             │ │
│    │    - Custom conditions from scenario                            │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Phenomena Modeled

### 1. Information Asymmetry

Not everyone knows the same thing at the same time.

**How it works:**

- Exposure is gated by channels (media reaches some, not others)
- Timing matters (early adopters vs. late majority)
- Agent attributes determine exposure probability
- Some agents never hear the news; some hear it first

**Code:** `propagation.py:apply_seed_exposures()`, `evaluate_exposure_rule()`

### 2. Agent–Event Interaction

When an agent receives information, they reason about it based on who they are.

**How it works:**

- Agent persona built from attributes (age, role, personality)
- LLM generates response: position, sentiment, action intent
- No hardcoded rules like "surgeons reject AI"
- Response depends on individual agent context

**Code:** `reasoning.py:reason_agent()`, `build_reasoning_prompt()`

### 3. Agent–Agent Interaction

Agents influence each other through the network.

**How it works:**

- Peer opinions included in reasoning context
- "My department head adopted it" affects junior surgeon
- Relationship type matters (mentor vs. acquaintance)
- Social proof and argumentation modeled

**Code:** `engine.py:_get_peer_opinions()`, `reasoning.py:create_reasoning_context()`

### 4. Agent–Network Interaction

Network topology shapes information flow.

**How it works:**

- Information flows along edges
- Edge weights affect share probability
- Clusters can form echo chambers
- Bridges (weak ties) spread info across groups
- Influence weighted by PageRank/betweenness

**Code:** `propagation.py:propagate_through_network()`, `calculate_share_probability()`

### 5. Temporal Dynamics

Opinions evolve over time.

**How it works:**

- Agents can re-reason when receiving new information
- `multi_touch_threshold` controls re-reasoning frequency
- Early reactions differ from settled views
- Cascades and tipping points can emerge
- Simulation runs forward, not to a snapshot

**Code:** `engine.py:run()`, `state_manager.get_agents_to_reason()`

### 6. Emergent Outcomes

Final results are not prescribed.

**How it works:**

- No formula dictates adoption rates
- Outcome emerges from individual decisions
- Same scenario, different seed → different trajectory
- Aggregate behavior from micro-level reasoning

**Code:** `aggregation.py:compute_outcome_distributions()`

---

## When Do Agents Reason?

**NOT every agent, every timestep.** That would be prohibitively expensive.

Agents reason when:

1. **First exposure** — First time hearing about the event
2. **Multi-touch threshold** — After N new exposures since last reasoning
3. **Significant update** — New info worth reconsidering

**Typical per-agent reasoning calls:** 1-5 over full simulation

---

## LLM-Powered Reasoning

Each reasoning call sends a structured prompt:

```
You are {persona}.

You have learned about: {event_content}
You heard about it from: {exposure_sources}

Your peers think:
- Dr. Schmidt (department head): Supportive, will adopt
- Dr. Müller (colleague): Skeptical, concerned about liability

Based on who you are and what you've learned, how do you respond?

Output:
- position: Your stance
- sentiment: How you feel (-1 to 1)
- will_share: Will you discuss this with colleagues?
- {scenario-specific outcomes}
```

**The LLM generates:**

- Position (e.g., "supportive", "opposed", "undecided")
- Sentiment (-1 to 1)
- Whether they'll share with peers
- Scenario-defined outcomes (adoption_intent, etc.)

---

## Reproducibility

### What IS reproducible (same seed → same result):

- Agent attributes
- Network structure
- Exposure order
- Random draws for probabilities

### What is NOT fully reproducible:

- LLM response variance (temperature > 0, model updates)

**Mitigations:**

- Run multiple times, report confidence intervals
- Use temperature=0 (reduces variance)
- Cache LLM responses for exact replay (expensive)

**Honest framing:** Structurally reproducible, behaviorally stochastic.

---

## Output Files

```
results/
├── meta.json                 # Run metadata (seed, model, timestamps)
├── by_timestep.json          # Timeline of aggregate metrics
├── agent_states.json         # Final state of each agent
├── outcome_distributions.json # Distribution of outcomes
└── timeline.jsonl            # Event log (exposures, reasoning, shares)
```

---

## Module Structure

```
simulation/
├── engine.py        # Main orchestrator, simulation loop
├── state.py         # Agent state management (SQLite backend)
├── reasoning.py     # LLM-powered agent cognition
├── propagation.py   # Exposure and network spread
├── persona.py       # Persona generation from attributes
├── stopping.py      # Stop condition evaluation
├── timeline.py      # Event logging
└── aggregation.py   # Outcome aggregation and summaries
```

---

## CLI Usage

```bash
# Run simulation
entropy simulate surgeons_ai.scenario.yaml -o results/

# With options
entropy simulate surgeons_ai.scenario.yaml \
  -o results/ \
  --model gpt-5 \
  --seed 42 \
  --multi-touch 3

# View results
entropy results results/ --format summary
entropy results results/ --format report
```

---

## Cost Estimation

| Population    | Typical LLM Calls | Approx Cost (GPT-4) |
| ------------- | ----------------- | ------------------- |
| 500 agents    | 1,000–2,500       | $5–15               |
| 2,000 agents  | 4,000–10,000      | $20–60              |
| 10,000 agents | 20,000–50,000     | $100–300            |

_Costs depend on scenario complexity, spread dynamics, and reasoning frequency._

---

## Design Decisions

### Why LLM Reasoning?

Traditional ABM: `if (seniority > 3 and openness > 0.6) then adopt`

Entropy: Ask the LLM what a 45-year-old skeptical chief surgeon would actually think.

**Benefits:**

- No need to hardcode decision rules
- Handles nuance and edge cases
- Generates believable reasoning
- Adapts to novel scenarios

### Why SQLite State?

Agent states stored in SQLite, not memory:

- Scales to large populations
- Survives interruption
- Queryable for analysis
- Efficient updates

### Why Event-Driven Reasoning?

Agents don't think every timestep:

- Realistic (humans don't constantly reconsider)
- Cost-efficient
- Captures "stimulus → response" pattern
