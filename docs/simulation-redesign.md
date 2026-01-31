# Entropy Simulation Engine Redesign

## Problem Statement

The current simulation is a **signal-passing model**, not a social simulation. Agents see 5 neighbors' position labels ("Supportive"), never exchange arguments, have no ambient awareness of public opinion, no memory between reasoning calls, and all peer influence uses a hardcoded credibility of 0.85. The `InteractionConfig` types (PASSIVE_OBSERVATION, DIRECT_CONVERSATION, BROADCAST_RESPONSE, DELIBERATIVE) are defined in the scenario spec but completely ignored by the engine.

Additionally, the reasoning architecture has a **central tendency problem**: asking an LLM to role-play as a persona AND pick from a categorical enum in the same call biases toward safe middle options (83% chose the safe middle with 4 options; 99% chose it with 5).

## Design Philosophy

Move to a **semantic-discursive model** where:
- Agents argue with *text*, not labels
- Agents perceive a *distorted, bubble-filtered* public feed (not ground-truth stats)
- High-stakes relationships escalate to short threaded conversations
- Agents remember their own reasoning history
- Peer credibility is dynamic, based on attributes
- Network weights drift based on agreement/disagreement
- Cost is controlled via tiered models and population sampling

Everything must be **generalized** — no scenario-specific logic in the engine.

### Position as Output, Not Simulation Mechanic

A key architectural change: **position (categorical outcome) is removed from the peer influence channel**. In the current system, agents see "Dr. Schmidt: Supportive" — a discrete label that anchors the LLM's interpretation and creates artificial clustering. This is problematic because:

1. Not every scenario has clean positions (behavioral intents like `drive_and_pay` aren't stances)
2. Labels are reductive — they collapse nuanced views into buckets before peers can engage with them
3. Position-based convergence is brittle — massive sentiment shifts within "considering" go undetected

**New approach**: Agents influence each other through **arguments** (public_statement) and **emotional tone** (sentiment). Position is extracted in a separate classification pass and used only for aggregation/output — the researcher sees "23% will adopt" but no agent ever sees that label about a peer.

**Peer influence channel**: `public_statement` + `sentiment` + `credibility`
**Aggregation/output**: categorical positions (extracted via classification pass)
**Convergence detection**: sentiment variance + conviction stability (continuous, not discrete)

---

## Phase 0: Foundation Fixes (Prerequisites)

**Goal**: Fix the central tendency problem and add belief resistance. These improve every LLM call made by all subsequent phases.

### 0a. Two-Pass Reasoning (from SIMULATION_IMPROVEMENTS Layer 1)

Split the single reasoning call into two cognitively distinct calls:

**Pass 1 — Free-text reasoning (role-play):**
The agent reasons naturally with NO categorical constraints. Schema:
```python
{
    "reasoning": str,       # Free-text internal monologue
    "will_share": bool,     # Will they discuss with others
    "public_statement": str, # 1-2 sentence argument for peers
    "reasoning_summary": str, # 1-sentence summary for memory
    "sentiment": float,     # -1 to 1
    "conviction": float,    # 0 to 1, how firmly held
}
```
This is a role-play task — the LLM is good at this without schema bias.

**Pass 2 — Classification (cheap extraction):**
A separate call takes the free-text reasoning and maps it to outcome categories:
```python
{
    "adoption_intent": "will_adopt" | "considering" | "unlikely" | ...,
    "primary_concern": "liability" | "accuracy" | ...,
    # ... all scenario-defined categorical/boolean/float outcomes
}
```
This is a classification task — different cognitive mode, doesn't suffer from central tendency. Uses the cheapest model available.

**Impact**: Pass 1 produces authentic reasoning. Pass 2 extracts structured outcomes without biasing the role-play. Cost increase ~15-25% (Pass 2 is ~200 tokens).

### 0b. Conviction / Belief Strength (from SIMULATION_IMPROVEMENTS Layer 2a)

Add conviction to `AgentState` and `ReasoningResponse`. Stored as a float internally but **the agent never sees a number**. The LLM picks from categorical levels that map to numeric values for thresholds:

| Level | Meaning | Stored as | The agent's self-description |
|-------|---------|-----------|------------------------------|
| `very_uncertain` | No real opinion formed | 0.1 | "I really don't know what to think" |
| `leaning` | Gut feeling, easily swayed | 0.3 | "I have an instinct but could easily change my mind" |
| `moderate` | Considered view, open to argument | 0.5 | "I've thought about it, but good arguments could shift me" |
| `firm` | Strong view, needs strong evidence | 0.7 | "I'm quite sure. It would take serious evidence to change my mind" |
| `absolute` | Deeply held, not changing | 0.9 | "This is a core belief. I'm not budging" |

**In Pass 1 schema** (what the LLM outputs):
```python
"conviction": {
    "type": "string",
    "enum": ["very_uncertain", "leaning", "moderate", "firm", "absolute"],
    "description": "How firmly do you hold this view?"
}
```

**In storage** (AgentState): mapped to float (0.1/0.3/0.5/0.7/0.9) for threshold math.

**In re-reasoning prompts** (what the agent sees about their past self): "Last time you felt *firm* about this" — the categorical word, never a number.

**Mechanics**:
- High-conviction agents resist position changes: if `old.conviction >= firm` and new position differs, reject the flip unless `new.conviction >= moderate`
- Gates sharing: agents with `conviction = very_uncertain` don't share (too uncertain to evangelize)
- Conviction-weighted edge dynamics: uncertain agents don't move edge weights much (Phase 4)

### 0c. Trait Salience (from SIMULATION_IMPROVEMENTS Layer 3)

Add `decision_relevant_attributes: list[str]` to `OutcomeConfig` (generated by scenario compiler). The persona renderer groups these attributes first under "Factors Most Relevant to This Situation" instead of burying them in a flat 40-attribute list.

Does NOT tell the agent what to conclude — just makes the relevant tradeoffs visible. Works alongside `entropy persona`'s relative positioning (which handles *how* traits are described, not *which* traits are prominent).

### Files Modified
- `entropy/simulation/reasoning.py` — split into two-pass, add conviction to schema
- `entropy/core/models/simulation.py` — add conviction to AgentState, ReasoningResponse
- `entropy/simulation/engine.py` — conviction-gated sharing, conviction-based flip resistance
- `entropy/core/models/scenario.py` — add decision_relevant_attributes to OutcomeConfig
- `entropy/scenario/outcomes.py` — extend LLM prompt to generate decision_relevant_attributes
- `entropy/simulation/persona.py` — group relevant attributes first

---

## Phase 1: Semantic Signals + Memory

**Goal**: Agents share *arguments*, not position labels. Agents remember their own reasoning.

### Data Model Changes (`entropy/core/models/simulation.py`)

Add `MemoryEntry` model:
```python
class MemoryEntry(BaseModel):
    timestep: int
    sentiment: float | None
    conviction: float | None
    summary: str  # 1-sentence summary of reasoning
```

Extend `AgentState` with:
- `public_statement: str | None = None` — 1-2 sentence public argument
- `memory_trace: list[MemoryEntry] = []` — sliding window, max 3 entries

Extend `PeerOpinion` with:
- `public_statement: str | None = None` — the peer's argument text
- `credibility: float = 0.5` — dynamic credibility (Phase 2, but field added now)

Extend `ReasoningContext` with:
- `memory_trace: list[MemoryEntry] = []` — agent's own reasoning history

Note: `public_statement` and `reasoning_summary` are already in the Pass 1 schema from Phase 0. This phase wires them into the propagation and memory systems.

### Prompt Changes (`entropy/simulation/reasoning.py`)

New prompt structure for `build_reasoning_prompt` (Pass 1):
```
You ARE the person described below...

## Who You Are
{persona — with relevant traits grouped first per Phase 0c}

## Your Recent Thinking                          ← NEW
- "3 timesteps ago: I was cautiously optimistic because the data looked promising."
- "Last time: After hearing colleagues' concerns about liability, I grew more cautious."
  (conviction was moderate)

## What You Just Learned
{event content + source}

## How This Reached You
{exposure history — unchanged}

## What People Around You Think                  ← ENHANCED
- A senior colleague (highly credible):
  "The efficiency gains alone justify the transition costs."   ← argument, not label
- A peer in your field:
  "I've seen three implementations fail at similar organizations." ← argument, not label

## Your Authentic Response                       ← ENHANCED
Given YOUR background, YOUR previous thinking, and what you're hearing:
1. How has your thinking EVOLVED from before?
2. What is your current genuine stance and how firmly do you hold it?
3. What would you SAY to colleagues about this? (1-2 sentences)
```

**Key change**: Peer opinions show `public_statement` and credibility level, NOT position labels. The agent reacts to arguments, not categories.

### Engine Changes (`entropy/simulation/engine.py`)

- After reasoning: store `MemoryEntry` from response (summary + sentiment + conviction), cap at 3 per agent
- Pass `memory_trace` into `create_reasoning_context()`
- Store `public_statement` on `AgentState`

### Propagation Changes (`entropy/simulation/propagation.py`)

- When building `PeerOpinion` in `_get_peer_opinions()`: include neighbor's `public_statement` and `sentiment`, NOT position
- In network propagation: include sender's `public_statement` in ExposureRecord content

### State Changes (`entropy/simulation/state.py`)

- Add `public_statement`, `conviction` columns to `agent_states`
- Add `memory_traces` table: `(id, agent_id, timestep, sentiment, conviction, summary)`
- New methods: `save_memory_entry()`, `get_memory_trace(agent_id, limit=3)`

### Files Modified
- `entropy/core/models/simulation.py` — MemoryEntry, extend AgentState/PeerOpinion/ReasoningContext
- `entropy/simulation/reasoning.py` — prompt restructure, remove position from peer display
- `entropy/simulation/engine.py` — memory storage, context building
- `entropy/simulation/propagation.py` — public statement in peer opinions, no position labels
- `entropy/simulation/state.py` — new table, new columns, new methods

---

## Phase 2: Dynamic Peer Credibility + Temporal Decay

**Goal**: Influence is realistic (senior people carry more weight). Old news fades.

### Dynamic Credibility

New model in `entropy/core/models/scenario.py`:
```python
class CredibilityFactor(BaseModel):
    attribute: str          # e.g., "years_experience"
    comparison: str         # "sender_higher", "same_value", "sender_has"
    weight: float           # 0.05-0.20

# Add to ScenarioSpec:
credibility_factors: list[CredibilityFactor] = []
```

New function in `entropy/simulation/propagation.py`:
```
calculate_peer_credibility(sender, receiver, edge, scenario) -> float:
    base = 0.40
    + structural_authority (from credibility_factors, capped at 0.25)
    + relationship_strength (edge_weight * 0.25)
    + influence_score (network centrality * 0.10)
    = clamped [0.1, 0.99]
```

Replaces the hardcoded `credibility=0.85` in `propagate_through_network()`.

### Temporal Decay

New method in `state.py`: `compute_effective_exposure_weight(agent_id, current_timestep, decay_rate)`:
```
weight = sum(exp.credibility * (1 - decay_rate)^(current_ts - exp.timestep) for exp in exposures)
```

Add `effective_exposure_weight: float = 0.0` to `AgentState`.

Modify multi-touch threshold logic: instead of raw `exposure_count >= threshold`, use `effective_exposure_weight` crossing a threshold. This means recent high-credibility exposures trigger re-reasoning faster than stale low-credibility ones.

### Implement SpreadConfig Fields

`decay_per_hop` and `max_hops` are already defined in `ScenarioSpec.spread` but never implemented. Implement them in `propagate_through_network()`:
- Track hop count per propagation chain
- Apply `credibility *= (1 - decay_per_hop)` per hop
- Stop propagating beyond `max_hops`

### Scenario Compiler Changes (`entropy/scenario/compiler.py`)

New step "Generate Credibility Factors" — LLM examines population attributes and event to determine which attributes confer authority. Generates 3-5 `CredibilityFactor` entries. Falls back to empty list for old scenarios.

### Files Modified
- `entropy/core/models/scenario.py` — CredibilityFactor, extend ScenarioSpec
- `entropy/core/models/simulation.py` — extend AgentState
- `entropy/simulation/propagation.py` — credibility formula, hop decay, max_hops
- `entropy/simulation/state.py` — effective_exposure_weight, temporal decay methods
- `entropy/simulation/engine.py` — call temporal decay, pass credibility to peer opinions
- `entropy/scenario/compiler.py` — credibility factor generation step

---

## Phase 3: Perceptual Public Feed

**Goal**: Agents have ambient awareness of broader opinion — distorted by their social bubble, not ground-truth stats.

### New File: `entropy/simulation/feed.py`

Core function: `compute_ambient_feeds(state_manager, network, agents, bubble_strength) -> dict[str, str]`

Algorithm:
1. Get global sentiment distribution from state_manager (average sentiment, sentiment spread)
2. For each network cluster (from `node_metrics.cluster_id` in network JSON):
   a. Compute cluster-local sentiment average and spread
   b. Blend: `perceived = bubble_strength * local + (1 - bubble_strength) * global`
   c. Add Gaussian noise (sigma=0.05) — nobody perceives exact numbers
   d. Sample 2-3 `public_statement`s from agents OUTSIDE this cluster (weighted by network influence score — prominent voices travel further)
   e. Render natural language:
      ```
      From what you can gather, people seem [mostly concerned / quite divided / generally optimistic / ...].
      A few things you've heard outside your immediate circle:
      - "[statement_1]" (someone in your field)
      - "[statement_2]" (a widely-shared take)
      ```
   f. Cache per cluster (not per agent) for efficiency

Key properties:
- **No exact stats**: Natural language like "people seem..." not "43% support"
- **No position labels**: Feed describes sentiment direction and argument themes, not categorical buckets
- **Bubble-filtered**: `bubble_strength` (default 0.6) controls distortion. Agent in a negative-sentiment cluster perceives majority concern even if the real average is neutral
- **Real voices from outside**: Sampled statements provide concrete arguments from beyond the bubble
- **Vague attribution**: "someone in your field", not agent IDs — mimics social media
- **Zero LLM cost**: Computed locally from existing state

### Prompt Changes

Add new section between "What People Around You Think" and "Your Authentic Response":
```
## The Broader Buzz
{ambient_feed text from feed.py}
```

### Context Changes

Add `ambient_feed: str = ""` to `ReasoningContext`.

### State Changes

Add `public_statements` table for efficient sampling:
`(id, agent_id, timestep, statement, sentiment)`
- Indexed on `(agent_id)` and `(timestep)`
- Populated when agent reasons and generates a public_statement

### Files Modified/Created
- `entropy/simulation/feed.py` — **NEW FILE**
- `entropy/core/models/simulation.py` — extend ReasoningContext
- `entropy/simulation/reasoning.py` — add "Broader Buzz" prompt section
- `entropy/simulation/engine.py` — call compute_ambient_feeds(), inject into contexts
- `entropy/simulation/state.py` — public_statements table, sampling queries

---

## Phase 4: Soft Dynamic Network

**Goal**: Edge weights drift during simulation. Agreement strengthens ties, disagreement weakens them.

### New File: `entropy/simulation/network_dynamics.py`

New model in `entropy/core/models/simulation.py`:
```python
class EdgeState(BaseModel):
    source: str
    target: str
    initial_weight: float
    current_weight: float
    agreement_history: list[float] = []  # sliding window of 5
    severed: bool = False
```

Core function: `update_edge_weights(network_state, state_manager, config)`:
- For each edge where both endpoints have sentiment:
  - Compute agreement: `1.0 - abs(a.sentiment - b.sentiment) / 2.0` (sentiment similarity on [-1,1])
  - Weight by conviction: `agreement *= min(a.conviction, b.conviction)` (uncertain agents don't move edges much)
  - If agreement > 0.5: strengthen by `+0.10 * (agreement - 0.5)`
  - If agreement < 0.5: weaken by `-0.05 * (0.5 - agreement)`
  - Clamp to [0.0, 1.0]
  - If weight < 0.05: mark `severed=True`
- Asymmetric rates: strengthening (0.10) is faster than weakening (0.05) — takes ~10 timesteps of persistent disagreement to halve an edge

Note: Uses sentiment (continuous) not position (categorical) for agreement calculation. Two agents can hold the same position label but with wildly different sentiment — the edge weight reflects their actual alignment, not their bucket.

### Engine Integration

- At engine init: create `EdgeState` for every edge, copying static weight
- After reasoning batch: call `update_edge_weights()`
- In propagation: use `current_weight` instead of static weight; skip severed edges
- In peer opinion gathering: use `current_weight` for credibility calculation

### State Changes

New `edge_states` table:
```sql
(edge_key TEXT PRIMARY KEY, source, target, initial_weight, current_weight, agreement_history_json, severed)
```

### Files Modified/Created
- `entropy/simulation/network_dynamics.py` — **NEW FILE**
- `entropy/core/models/simulation.py` — EdgeState model
- `entropy/simulation/engine.py` — init edge states, call updates
- `entropy/simulation/propagation.py` — use current_weight, skip severed
- `entropy/simulation/state.py` — edge_states table and methods

---

## Phase 5: Escalated Conversations

**Goal**: High-stakes relationships produce short threaded debates (2-3 turns), not just signal passing.

### New File: `entropy/simulation/escalation.py`

New models in `entropy/core/models/simulation.py`:
```python
class ConversationTurn(BaseModel):
    speaker_id: str
    timestep: int
    content: str  # 2-3 sentences
    sentiment_after: float | None = None
    conviction_after: float | None = None

class ConversationThread(BaseModel):
    peer_id: str
    peer_relationship: str
    turns: list[ConversationTurn] = []
    trigger_reason: str
```

New model in `entropy/core/models/scenario.py`:
```python
class EscalationConfig(BaseModel):
    edge_weight_threshold: float = 0.7
    sentiment_divergence_threshold: float = 0.6
    relationship_types: list[str] = ["mentor_mentee"]
    max_conversations_per_timestep: int = 20
```

Add optional `escalation_config: EscalationConfig | None = None` to `InteractionConfig`.

### Escalation Triggers (generalized)

`identify_escalations(agents_to_reason, state_manager, network_state, config)`:
- For each agent about to reason, check their already-opinionated neighbors:
  1. High-weight edge + sentiment divergence: `edge.current_weight > threshold AND |A.sentiment - B.sentiment| > divergence_threshold`
  2. Privileged relationship type (in `escalation_config.relationship_types`)
- Sort by `edge_weight * sentiment_divergence` (highest-stakes first)
- Cap at `max_conversations_per_timestep`

### Conversation Execution

Separate conversation prompt (shorter, focused):
```
You are having a direct conversation with a {relationship} about {event_summary}.

Your current feeling: {sentiment_description} (conviction: {high/moderate/low})
Your argument: "{public_statement}"

The exchange so far:
- They said: "..."
- You said: "..."

Their latest: "{peer_latest}"

Respond in 2-3 sentences. You may hold firm, partially agree, or be genuinely persuaded.
```

Note: No position labels in conversation prompts either. Agents exchange arguments and emotional stances.

Execution per timestep:
- All Turn 1s in parallel (agent A responds to B's public_statement)
- All Turn 2s in parallel (agent B responds to A's Turn 1)
- Optional Turn 3 if sentiments still diverge
- Each turn: max_tokens=150, uses pivotal model tier
- Both agents' states updated after conversation completes
- After all conversations: run Pass 2 classification on updated reasoning to extract positions for output

### Backwards Compatibility

If `escalation_config` is None (old scenarios): use defaults. If `enable_conversations=False` in SimulationRunConfig: skip entirely.

### Scenario Compiler Changes

Extend `determine_interaction_model()` prompt to also generate `EscalationConfig` — which relationship types warrant direct conversation, what divergence threshold makes sense for this scenario.

### Files Modified/Created
- `entropy/simulation/escalation.py` — **NEW FILE**
- `entropy/core/models/simulation.py` — ConversationThread, ConversationTurn, extend ReasoningContext
- `entropy/core/models/scenario.py` — EscalationConfig, extend InteractionConfig
- `entropy/simulation/reasoning.py` — conversation prompt builder
- `entropy/simulation/engine.py` — escalation detection, conversation execution in loop
- `entropy/simulation/state.py` — conversations table
- `entropy/scenario/interaction.py` — extend LLM prompt for escalation config

---

## Phase 6: Tiered Reasoning + Population Sampling

**Goal**: Cost control. Cheap models for routine reactions, expensive for pivotal moments. Large populations use sampling.

### Tier Classification

```
classify_reasoning_tier(agent_id, state, timestep, active_conversations) -> "pivotal" | "routine":
    Pivotal if:
    - First-ever reasoning (never reasoned before)
    - Involved in escalated conversation this timestep
    - Conviction changed significantly in most recent reasoning (unstable agent)
    Everything else: routine
```

### Model Routing

**Supported models (no thinking/extended models — standard inference only):**

| Tier | OpenAI | Anthropic |
|------|--------|-----------|
| Pivotal (Pass 1) | gpt-5 | claude-sonnet-4.5 |
| Routine (Pass 1) | gpt-5-mini | claude-haiku-4.5 |
| Classification (all Pass 2) | gpt-5-mini | claude-haiku-4.5 |
| Conversations | gpt-5 | claude-sonnet-4.5 |

"Expensive" = standard gpt-5 / claude-sonnet-4.5. NOT o1, o3, extended thinking, or any reasoning-trace models. These are straightforward structured-output calls — thinking models add latency and cost with no benefit for persona role-play.

Extend `SimulationRunConfig`:
- `pivotal_model: str = ""` — standard model (default: gpt-5 / claude-sonnet-4.5)
- `routine_model: str = ""` — cheap model (default: gpt-5-mini / claude-haiku-4.5)
- `enable_conversations: bool = True`
- `sampling_threshold: int = 2000` — population size above which sampling kicks in
- `sampling_ratio: float = 0.3` — fraction of routine agents to sample

The existing `simple_call_async()` already accepts a `model` param. Route based on tier.

Note: The cheap model is used for Pass 1 (routine) and the cheapest model for all Pass 2 (classification) calls regardless of tier.

### Population Sampling (for populations > `sampling_threshold`)

- All pivotal agents: always reason (100%)
- Routine agents: stratified sample by cluster (30% default), ensuring every cluster represented
- Skipped agents: heuristic update — no position flip, only gradual sentiment drift toward cluster-local average. Conservative by design.

### Extend Config (`entropy/config.py`)

Add CLI options for `--pivotal-model` and `--routine-model` on the simulate command.

### Files Modified
- `entropy/core/models/simulation.py` — extend SimulationRunConfig
- `entropy/simulation/engine.py` — tier classification, model routing, sampling logic, heuristic updates
- `entropy/config.py` — new config fields
- `entropy/cli/commands/simulate.py` — new CLI options

---

## Phase 7: Output + Observability

**Goal**: New features are visible and analyzable in simulation results.

### New Metrics in Aggregation (`entropy/simulation/aggregation.py`)

- **Network polarization**: average edge weight change, count of severed edges, cluster-level sentiment homogeneity
- **Conversation summaries**: count per timestep, sentiment-change rate from conversations
- **Feed accuracy**: how distorted the perceived feed was vs ground truth (for researcher analysis)
- **Tier breakdown**: pivotal vs routine agent counts, sampling rate
- **Conviction distribution**: mean conviction over time, conviction-by-segment

### New Timeline Events (`entropy/simulation/timeline.py`)

- `CONVERSATION_STARTED` — escalated conversation initiated
- `CONVERSATION_TURN` — individual turn in thread
- `EDGE_WEIGHT_CHANGED` — significant weight shift on an edge
- `AGENT_MEMORY_UPDATED` — memory trace entry stored

### Extended Output Files

- `edge_dynamics.json` — edge weight trajectories over time
- `conversations.json` — all conversation threads with full turns
- Existing files (`agent_states.json`, `by_timestep.json`, etc.) gain new fields but retain all existing ones
- Position distributions in output are derived from Pass 2 classification (same format as today for the researcher)

### Files Modified
- `entropy/simulation/aggregation.py` — new metric computations
- `entropy/simulation/timeline.py` — new event types
- `entropy/simulation/engine.py` — export new result files

---

## Revised Simulation Loop (Final)

```
for timestep in 0..max_timesteps:

    1. SEED EXPOSURE (unchanged)
       Apply scenario exposure rules for this timestep

    2. TEMPORAL DECAY                              ← NEW
       Update effective_exposure_weight for all agents

    3. NETWORK PROPAGATION (enhanced)
       - Use dynamic credibility (not 0.85)
       - Include public_statements in exposures
       - Use current_weight from EdgeState (not static)
       - Skip severed edges
       - Implement decay_per_hop and max_hops

    4. COMPUTE AMBIENT FEEDS                       ← NEW
       Per-cluster perceptual feed with bubble distortion
       Based on sentiment, not position labels

    5. SELECT AGENTS TO REASON
       First exposure OR effective_exposure_weight crossed threshold

    6. CLASSIFY TIERS                              ← NEW
       Pivotal (first-time, unstable, in-conversation) vs routine

    7. SAMPLE (large populations only)             ← NEW
       Stratified sample of routine agents

    8. BUILD CONTEXTS (enhanced)
       Include: memory_trace, ambient_feed, peer public_statements + credibility
       NO position labels in any peer-facing context

    9. IDENTIFY ESCALATIONS                        ← NEW
       High-weight + high-divergence edges → conversation threads

   10. BATCH LLM REASONING — PASS 1 (tiered)
       Pivotal agents → expensive model
       Routine agents → cheap model
       Returns: reasoning, sentiment, conviction, public_statement, will_share

   11. EXECUTE CONVERSATIONS                       ← NEW
       2-3 turn threaded exchanges for escalated pairs
       Updates sentiment + conviction for both parties

   12. BATCH CLASSIFICATION — PASS 2 (cheapest model)
       All agents who reasoned or conversed this timestep
       Maps free-text reasoning → categorical outcomes for output

   13. UPDATE STATES (enhanced)
       Store public_statement, memory_entry, conviction, sentiment
       Apply conviction-gated sharing filter
       Apply conviction-based flip resistance

   14. UPDATE EDGE WEIGHTS                         ← NEW
       Sentiment agreement strengthens, disagreement weakens
       Weighted by conviction (uncertain agents don't move edges)

   15. HEURISTIC UPDATES (large populations)       ← NEW
       Skipped agents get conservative sentiment drift

   16. COMPUTE SUMMARY (extended metrics)

   17. CHECK STOPPING CONDITIONS (enhanced)
       Based on sentiment variance + conviction stability
       Not position distribution variance
```

---

## Rate Limiting & Throughput Architecture

### The Problem

The current engine has a hardcoded `asyncio.Semaphore(50)` in `batch_reason_agents()` and zero actual rate limiting. The `SimZoneConfig.max_concurrent = 50` field exists but is **never wired** into the batch function. There's no token tracking, no RPM awareness, no retry-after parsing. The SDKs (OpenAI/Anthropic) handle 429 retries internally with backoff, but the engine has no awareness of its own throughput capacity.

At Anthropic Tier 1 (50 RPM), firing 50 concurrent requests burns your entire minute budget in one burst, then you're throttled for ~60 seconds. At OpenAI Tier 1, the RPM/TPM limits vary by model but a naive blast of 50 concurrent calls will still hit ceilings.

### Provider Rate Limits (Tier 1 — Lowest)

**Anthropic (4.5 series):**
| Model | RPM | ITPM | OTPM |
|-------|-----|------|------|
| Claude Sonnet 4.x | 50 | 30,000 | 8,000 |
| Claude Haiku 4.5 | 50 | 50,000 | 10,000 |

**OpenAI (5 series):**
| Model | RPM | TPM |
|-------|-----|-----|
| gpt-5 / gpt-5.2 | 500 | 500,000 |
| gpt-5-mini | 500 | 500,000 |

At higher tiers limits increase dramatically:

| Tier | Anthropic Sonnet RPM / OTPM | Anthropic Haiku RPM / OTPM | OpenAI gpt-5 RPM / TPM |
|------|----------------------------|---------------------------|------------------------|
| Tier 1 | 50 / 8k | 50 / 10k | 500 / 500k |
| Tier 2 | 1,000 / 90k | 1,000 / 90k | 5,000 / 1M |
| Tier 3 | 2,000 / 160k | 2,000 / 200k | 5,000 / 2M |
| Tier 4 | 4,000 / 400k | 4,000 / 800k | 10,000 / 4M |

### Solution: `entropy/core/rate_limiter.py` (NEW FILE)

A provider-aware rate limiter that sits between the simulation engine and the LLM providers. Not hardcoded — configurable per provider, per model, and overridable from CLI.

```python
class RateLimiter:
    """Token bucket rate limiter aware of RPM, ITPM, and OTPM constraints."""

    def __init__(self, provider: str, model: str, tier: int | None = None):
        # Load limits from provider defaults or config override
        self.rpm_bucket = TokenBucket(capacity=limits.rpm, refill_rate=limits.rpm / 60)
        self.tpm_bucket = TokenBucket(capacity=limits.tpm, refill_rate=limits.tpm / 60)

    async def acquire(self, estimated_tokens: int) -> float:
        """Wait until we have capacity. Returns wait time in seconds."""
        # Check both RPM and TPM buckets
        # Return immediately if capacity available
        # Sleep if needed, return actual wait time

    def update_from_headers(self, headers: dict):
        """Adjust limits based on response headers (retry-after, remaining, etc.)."""
        # Parse anthropic-ratelimit-* or x-ratelimit-* headers
        # Dynamically adjust bucket capacities

    @classmethod
    def for_provider(cls, provider: str, model: str, overrides: dict | None = None):
        """Factory with sensible defaults per provider/model."""
```

Key design:
- **Token bucket algorithm** (same as Anthropic uses internally) — capacity continuously refills
- **Dual bucket**: tracks both RPM and TPM simultaneously, blocks on whichever is tighter
- **Header-adaptive**: reads response headers to self-correct if default limits are wrong
- **Provider-aware defaults**: knows Anthropic Tier 1 Sonnet = 50 RPM / 30k ITPM, etc.
- **Overridable**: user can set `--rate-tier 3` or `--rpm-limit 2000` if they know they're on a higher tier
- **Zero-config by default**: the user just runs the command and it goes as fast as their tier allows without 429s

### Zero-Config Default Behavior

The user should never think about concurrency numbers. The rate limiter auto-calculates `max_safe_concurrent` from the model's profile:

```
max_safe_concurrent = min(
    rpm / 60 * avg_call_duration_sec,   # how many can be in-flight given RPM
    tpm / avg_tokens_per_call           # how many can be in-flight given TPM
)
```

Examples at Tier 1:
- **gpt-5-mini**: `min(500/60 * 5, 500000/800)` = `min(41, 625)` = **41 concurrent** — RPM-bound
- **gpt-5**: `min(500/60 * 5, 500000/800)` = `min(41, 625)` = **41 concurrent** — RPM-bound
- **Claude Sonnet 4.x**: `min(50/60 * 5, 30000/800)` = `min(4, 37)` = **4 concurrent** — RPM-bound, very tight
- **Claude Haiku 4.5**: `min(50/60 * 5, 50000/200)` = `min(4, 250)` = **4 concurrent** — RPM-bound

Anthropic Tier 1 is extremely restrictive (50 RPM shared). The rate limiter will pace requests at ~1/sec for Anthropic, ~8/sec for OpenAI, automatically.

```bash
# Just works — rate limiter auto-detects model and applies Tier 1 defaults:
entropy simulate scenario.yaml -o results/ --seed 42

# If you know you're on a higher tier:
entropy simulate scenario.yaml -o results/ --seed 42 --rate-tier 3

# Manual override if the defaults are wrong:
entropy simulate scenario.yaml -o results/ --seed 42 --rpm-limit 2000 --tpm-limit 800000
```

### Default Rate Limit Profiles

```python
# entropy/core/rate_limits.py — provider/model defaults (Tier 1)

RATE_LIMIT_PROFILES = {
    "anthropic": {
        "default": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},  # Sonnet
        "claude-sonnet-4": {"rpm": 50, "itpm": 30_000, "otpm": 8_000},
        "claude-haiku-4": {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
        "tiers": {
            2: {"rpm": 1_000, "itpm": 450_000, "otpm": 90_000},
            3: {"rpm": 2_000, "itpm": 800_000, "otpm": 160_000},
            4: {"rpm": 4_000, "itpm": 2_000_000, "otpm": 400_000},
        },
    },
    "openai": {
        "default": {"rpm": 500, "tpm": 500_000},  # gpt-5 series
        "gpt-5": {"rpm": 500, "tpm": 500_000},
        "gpt-5-mini": {"rpm": 500, "tpm": 500_000},
        "gpt-5.2": {"rpm": 500, "tpm": 500_000},
        "tiers": {
            2: {"rpm": 5_000, "tpm": 1_000_000},
            3: {"rpm": 5_000, "tpm": 2_000_000},
            4: {"rpm": 10_000, "tpm": 4_000_000},
        },
    },
}
```

### Integration Points

**1. Replace hardcoded semaphore** in `batch_reason_agents()`:
```python
# BEFORE (reasoning.py:374):
semaphore = asyncio.Semaphore(max_concurrency)

# AFTER:
rate_limiter = RateLimiter.for_provider(provider, model, overrides=config.rate_limits)

async def reason_with_rate_limit(ctx):
    estimated_tokens = estimate_tokens(ctx)  # rough prompt + response estimate
    await rate_limiter.acquire(estimated_tokens)
    result = await _reason_agent_async(ctx, scenario, config)
    rate_limiter.update_from_headers(result.response_headers)  # if available
    return (ctx.agent_id, result)
```

**2. Wire `SimZoneConfig.max_concurrent`** — actually use it:
```python
# In engine.py, pass config to batch_reason_agents:
results = await batch_reason_agents(contexts, scenario, config,
    max_concurrency=self.sim_config.max_concurrent,
    rate_limiter=self.rate_limiter)
```

**3. Multi-model support** — the tiered reasoning system (Phase 6) uses different models per tier. The rate limiter handles this by maintaining separate buckets per model:
```python
pivotal_limiter = RateLimiter.for_provider(provider, pivotal_model)
routine_limiter = RateLimiter.for_provider(provider, routine_model)
classify_limiter = RateLimiter.for_provider(provider, classify_model)
```

**4. CLI overrides** on `entropy simulate`:
```
--rate-tier 3              # Use tier 3 limits for the configured provider
--rpm-limit 2000           # Override RPM directly
--tpm-limit 800000         # Override TPM directly
```

**5. Provider response headers** — both OpenAI and Anthropic return rate limit headers. The providers (`openai.py`, `claude.py`) should expose these in the response so the rate limiter can self-correct:
- Anthropic: `anthropic-ratelimit-requests-remaining`, `anthropic-ratelimit-input-tokens-remaining`, `retry-after`
- OpenAI: `x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`, `retry-after`

### Runtime Estimates (1000 agents, ~100 reasoning per timestep)

**Anthropic Tier 1 — Sonnet 4.x + Haiku 4.5 (50 RPM each, separate limits):**
| Step | Calls | Model | Bottleneck | Time |
|------|-------|-------|-----------|------|
| Pass 1 (100 agents) | 100 | Sonnet | 50 RPM → paced at ~1/sec. OTPM (8k) is tight: 100×200 output = 20k → ~2.5 min | ~2.5 min |
| Pass 2 (100 classify) | 100 | Haiku | Separate 50 RPM. 100×50 output = 5k vs 10k OTPM → fine | ~2 min |
| Conversations (20 turns) | 20 | Sonnet | Shares Sonnet RPM with Pass 1. Paced. | ~24 sec |
| **Per timestep** | | | | **~5 min** |
| **50 timesteps** | | | | **~4 hours** |

**OpenAI Tier 1 — gpt-5 + gpt-5-mini (500 RPM, 500k TPM):**
| Step | Calls | Model | Bottleneck | Time |
|------|-------|-------|-----------|------|
| Pass 1 (100 agents) | 100 | gpt-5 | 500 RPM → fine. 100×800=80k vs 500k TPM → fine | ~15 sec |
| Pass 2 (100 classify) | 100 | gpt-5-mini | 100×200=20k vs 500k TPM → fine | ~15 sec |
| Conversations (20 turns) | 20 | gpt-5 | fits easily | ~5 sec |
| **Per timestep** | | | | **~35 sec** |
| **50 timesteps** | | | | **~29 min** |

**Anthropic Tier 3 — Sonnet 4.x + Haiku 4.5 (2000 RPM, 160k/200k OTPM):**
| Step | Calls | Model | Bottleneck | Time |
|------|-------|-------|-----------|------|
| Pass 1 + Conversations | ~120 | Sonnet | 2000 RPM, 160k OTPM → all fit | ~8 sec |
| Pass 2 | ~100 | Haiku | 2000 RPM, 200k OTPM → all fit | ~4 sec |
| **Per timestep** | | | | **~12 sec** |
| **50 timesteps** | | | | **~10 min** |

**OpenAI Tier 3 — gpt-5 + gpt-5-mini (5000 RPM, 2M TPM):**
| Step | Calls | Bottleneck | Time |
|------|-------|-----------|------|
| Everything | ~220 | 5000 RPM, 2M TPM → all fit in seconds | ~5 sec |
| **Per timestep** | | | **~5 sec** |
| **50 timesteps** | | | **~4 min** |

**Bottom line**: OpenAI gpt-5 series is dramatically faster than Anthropic at Tier 1 due to 10x higher RPM (500 vs 50). At Tier 3+, both are fast enough that simulation is compute-bound, not rate-limited.

### Files Modified/Created
- `entropy/core/rate_limiter.py` — **NEW FILE**: TokenBucket, RateLimiter, provider defaults
- `entropy/core/rate_limits.py` — **NEW FILE**: Default rate limit profiles per provider/model/tier
- `entropy/simulation/reasoning.py` — Replace semaphore with rate limiter
- `entropy/core/providers/openai.py` — Expose response headers
- `entropy/core/providers/claude.py` — Expose response headers
- `entropy/config.py` — Add rate limit config fields (tier, rpm_override, tpm_override)
- `entropy/cli/commands/simulate.py` — Add `--rate-tier`, `--rpm-limit`, `--tpm-limit` CLI options

### Implementation Note

This should be implemented as **Phase 0** alongside the foundation fixes, since it affects every LLM call the simulation makes. The rate limiter is independent of the semantic redesign — it's infrastructure that the current engine already needs.

---

## Cost Estimate (1000-agent population)

| Component | Calls/timestep | Model | Tokens/call | Cost/timestep |
|-----------|---------------|-------|-------------|---------------|
| Pass 1 pivotal | ~15 | gpt-5 / sonnet-4.5 | ~800 | ~$0.10 |
| Pass 1 routine | ~85 | gpt-5-mini / haiku-4.5 | ~700 | ~$0.05 |
| Pass 2 classification | ~100 | gpt-5-mini / haiku-4.5 | ~200 | ~$0.01 |
| Conversations | ~20 turns total | gpt-5 / sonnet-4.5 | ~300 | ~$0.05 |
| Public feed | 0 (local compute) | — | — | $0.00 |
| **Total** | | | | **~$0.21/timestep** |

**Per simulation** (50-100 timesteps): **$10-$21** (vs $1.50-$3.00 current — ~7x increase)

Cost levers: disable conversations, reduce sampling_ratio, use cheaper models for all tiers.

---

## Backwards Compatibility

- All new model fields have defaults — old scenario YAMLs load without changes
- All new SQLite tables use `CREATE IF NOT EXISTS`
- All new AgentState columns have defaults
- Existing output files retain all current fields (new fields added, nothing removed)
- Position distributions in output look identical to today (derived from Pass 2)
- If `credibility_factors` is empty: formula gives ~0.55-0.75 range (reasonable fallback)
- If `escalation_config` is None: sensible defaults used
- If `enable_conversations=False`: conversations skipped entirely

---

## What SIMULATION_IMPROVEMENTS.md Items Are Covered

| Item | Status | How |
|------|--------|-----|
| Layer 1: Two-Pass Reasoning | **Phase 0a** | Split role-play from classification |
| Layer 2a: Conviction | **Phase 0b** | Added to Pass 1 schema + state |
| Layer 2b: Prior Anchoring | **Superseded by Phase 1** | Memory traces are a richer version |
| Layer 2c: Source Credibility | **Superseded by Phase 2** | Generalized formula > hardcoded dict |
| Layer 2d: Conviction-Gated Sharing | **Phase 0b** | 3 lines, part of conviction impl |
| Layer 3: Trait Salience | **Phase 0c** | decision_relevant_attributes grouping |
