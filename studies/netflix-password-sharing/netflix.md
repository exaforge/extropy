# Netflix Password Sharing Crackdown Study

Simulating how 1000 US Netflix subscribers who currently share passwords respond to Netflix's password-sharing crackdown policy.

## Background

In 2023, Netflix began enforcing its long-standing rule against password sharing outside households. Users sharing accounts had to either:
- Pay an extra $7.99/month per additional member
- Stop sharing entirely

This study simulates how different types of password-sharers respond: do they pay up, cancel, comply, or find workarounds?

---

## Pipeline Commands

Run these in order. Each step produces a file consumed by the next.

### Prerequisites

```bash
# Set your API key (from environment or .env file)
export ANTHROPIC_API_KEY="your-api-key-here"

# Navigate to project root
cd /path/to/entropy
```

### Step 1: Generate Base Population Spec

```bash
entropy spec "1000 US Netflix subscribers who currently share their password with people outside their household" \
  -o studies/netflix-password-sharing/base.yaml \
  -y
```

**Output:** `base.yaml` - Population spec with demographics, subscription details, sharing patterns
**Verify:** Check that attributes include income, age, subscription tier, sharing relationship type, usage patterns

### Step 2: Extend with Scenario Attributes

```bash
entropy extend studies/netflix-password-sharing/base.yaml \
  -s "Netflix announces enforcement of password-sharing rules: users must pay $7.99/month extra per shared member outside household or stop sharing" \
  -o studies/netflix-password-sharing/population.yaml \
  -y
```

**Output:** `population.yaml` - Extended spec with behavioral attributes
**Verify:** Check for attributes like `price_sensitivity`, `netflix_attachment`, `sharing_dependency`, `alternative_awareness`

### Step 3: Sample Concrete Agents

```bash
entropy sample studies/netflix-password-sharing/population.yaml \
  -o studies/netflix-password-sharing/agents.json \
  -n 1000 \
  --seed 42 \
  --report
```

**Output:** `agents.json` - 1000 sampled agents with concrete attribute values
**Verify:** Check `--report` output for reasonable distributions (age spread, income variety, etc.)

### Step 4: Build Social Network

```bash
entropy network studies/netflix-password-sharing/agents.json \
  -o studies/netflix-password-sharing/network.json \
  -p studies/netflix-password-sharing/population.yaml \
  --seed 42 \
  --validate
```

**Output:** `network.json` - Social graph connecting agents
**Verify:** Check `--validate` output for reasonable clustering coefficient and path lengths

### Step 5: Generate Persona Config

```bash
entropy persona studies/netflix-password-sharing/population.yaml \
  --agents studies/netflix-password-sharing/agents.json \
  -o studies/netflix-password-sharing/population.persona.yaml \
  --preview
```

**Output:** `population.persona.yaml` - Persona rendering configuration
**Verify:** Check `--preview` output for coherent, plausible persona narratives

### Step 6: Compile Scenario

```bash
entropy scenario \
  -p studies/netflix-password-sharing/population.yaml \
  -a studies/netflix-password-sharing/agents.json \
  -n studies/netflix-password-sharing/network.json \
  -o studies/netflix-password-sharing/scenario.yaml \
  -y
```

**Output:** `scenario.yaml` - Complete simulation specification
**Verify:** Check outcomes include options like: pay_extra, cancel_subscription, comply_stop_sharing, find_workaround, ignore_rules

---

## Verification Checklist

After each step, verify the generated files make sense:

### base.yaml / population.yaml
- [ ] Has 25-40 attributes across categories (universal, population_specific, context_specific, personality)
- [ ] Income distributions are realistic for US (median ~$75k)
- [ ] Age ranges make sense (Netflix skews younger but has all ages)
- [ ] Netflix-specific attributes present: subscription_tier, account_holder_status, sharing_relationship
- [ ] Sources cited for key distributions

### agents.json
- [ ] Exactly 1000 agents
- [ ] No impossible combinations (e.g., student + retired)
- [ ] Attribute correlations seem reasonable (higher income → premium tier)

### network.json
- [ ] Reasonable average degree (15-25 connections)
- [ ] Edge types make sense (family, friend, coworker)
- [ ] Clustering coefficient > 0.1 (not random graph)

### population.persona.yaml
- [ ] Persona preview reads naturally in first person
- [ ] Relative attributes positioned correctly ("more price-sensitive than most")
- [ ] No contradictions in persona text

### scenario.yaml
- [ ] Event captures the policy change accurately
- [ ] Outcomes are distinct behavioral choices (not hedging options)
- [ ] Exposure channels make sense (email notification, news, social media)

---

## Running the Simulation (DO NOT RUN YET)

Once all files are verified, simulation command would be:

```bash
entropy simulate studies/netflix-password-sharing/scenario.yaml \
  -o studies/netflix-password-sharing/results/ \
  --seed 42
```

**Wait for explicit approval before running simulation.**

---

## Estimate Cost First

Before running simulation, estimate the cost:

```bash
entropy estimate studies/netflix-password-sharing/scenario.yaml --verbose
```

This shows predicted LLM calls, tokens, and USD cost without making any API calls.
