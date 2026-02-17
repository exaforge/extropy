# Plan: Rewrite Extropy Skills

## Execution Timing

**WAIT for pending CLI changes before executing this plan.**

Known pending changes:
- `network --generate-config` becoming default behavior
- [Any other CLI changes in progress]

Once CLI is stable, execute this plan against the final CLI state.

---

## Context

The current Extropy skills (6 files, ~1,667 lines) are outdated:
1. **Wrong CLI patterns**: Reference old commands (`extend`, `--study-db` everywhere) that no longer exist
2. **Missing the craft**: Don't teach HOW to write good spec/scenario prompts - just list commands
3. **No real examples**: Generic descriptions instead of concrete, annotated examples
4. **Over-fragmented**: 6 files is too many; user wants 2 focused files

The user provided excellent guidance on what makes good vs bad prompts (see conversation above), plus real examples (ASI announcement, crypto airdrop, inflation shock, Netflix password sharing).

## Goals

1. Teach the **craft** of writing good spec and scenario prompts
2. Use **current CLI** patterns (`spec → scenario → persona → sample → network → simulate`)
3. Include **annotated real examples** showing why they work
4. Consolidate to **2 files**: SKILL.md (the craft) + REFERENCE.md (commands/troubleshooting)

---

## File Structure

### SKILL.md (~500 lines)
The main skill entry point. Focuses on **understanding Extropy** and **writing effective prompts**.

**Sections:**
1. **What Extropy Is** (~50 lines)
   - The pitch: **Predictive intelligence through population simulation** (not surveys, not polls)
   - Synthetic agents grounded in real-world distributions, connected in social networks, each reasoning individually via LLM
   - What you get back:
     - Distributional predictions by segment
     - **Open-ended responses** when you don't know the outcome space (let agents tell you what they'd do, discover categories post-hoc)
     - Reasoning traces explaining *why*
     - Opinion dynamics over time
     - Network effects and conversation transcripts

2. **What It Can and Can't Do** (~30 lines)
   - Works well: policy, pricing, product, crisis, messaging, community
   - Limitations: no org charts, no real-time, no individual prediction

3. **The Craft of Spec Prompts** (~100 lines)
   - Mental model: "Constrain WHO, let LLM discover WHAT"
   - Spec creates reusable base population; scenario extends it
   - **Bad patterns** with examples:
     - Too vague: "Americans"
     - Too prescriptive: dictating exact brackets
     - Too narrow: over-specialized for one scenario
   - **Good patterns** with examples:
     - Clear population boundary
     - General attitudinal/psychological dimensions (trust, risk, tech adoption)
     - Reusable across scenarios
   - **Annotated example**: The US adults spec from user's chat

4. **The Craft of Scenario Prompts** (~150 lines)
   - Mental model: Scenario extends base with event-specific attributes
   - **The evolving test**: "Does the network change the outcome?"
     - Non-evolving: concrete personal decisions, population heterogeneity is the story (Netflix)
     - Evolving: information/credibility dynamics, social influence matters (ASI)
   - **Bad patterns**:
     - Too abstract: no concrete details
     - Predetermined framing: telling agents how to feel
     - Wrong timeline choice: evolving when should be static, or vice versa
     - Too many timeline events: drowns out network propagation
     - Wrong timestep granularity
   - **Good patterns**:
     - Concrete and specific: named entities, exact numbers
     - Neutral framing: present facts, let agents disagree
     - Strategic timeline events: 2-4 max, spaced for propagation
   - **Timestep table**: hours/days/weeks/months with use cases
   - **Annotated examples**:
     - Non-evolving: Netflix password sharing (from extropy.run)
     - Evolving: ASI announcement (from user's chat)
     - Edge case: Inflation shock (hybrid dynamics)

5. **Quick Pipeline** (~50 lines)
   - Current CLI flow with correct commands
   - Study folder structure
   - Common flags

6. **Fidelity and Cost** (~30 lines)
   - Low/medium/high tiers
   - When to use each
   - Cost estimates

7. **Operating Mode** (~40 lines)
   - **Always set `cli.mode agent`** at start of session for JSON output
   - Model configuration: `extropy config set simulation.strong`, etc.
   - **Escalation policy**: When to stop and ask user vs proceed autonomously
     - Escalate: same error twice, core assumptions change, cost vs accuracy tradeoff, sensitive content
     - Don't escalate: normal pipeline execution, validation passes, expected behavior
   - **Non-interactive flags**: `-y` to skip confirmations, `--use-defaults` for spec

8. **Module Reference** (~20 lines)
   - Points to REFERENCE.md for commands, troubleshooting, analysis

### REFERENCE.md (~400 lines)
Condensed operational reference. Everything you need to run simulations.

**Sections:**
1. **Command Reference** (~150 lines)
   - Condensed from docs/commands.md
   - Each command: purpose, key flags, example
   - Focus on most-used options, not exhaustive

2. **Study Folder Structure** (~30 lines)
   - Directory layout
   - File naming conventions (v1, v2, etc.)

3. **Results & Analysis** (~80 lines)
   - Condensed from ANALYSIS.md
   - How to read results
   - Segment analysis
   - Agent deep dives
   - Export commands

4. **Troubleshooting** (~100 lines)
   - Condensed from TROUBLESHOOTING.md
   - Common failure patterns
   - Quality gates (quick checklist)
   - When to escalate

5. **Configuration** (~60 lines)
   - **Agent mode setup**: `extropy config set cli.mode agent`
   - Model configuration:
     - `models.strong` / `models.fast` for pipeline (spec, scenario, persona)
     - `simulation.strong` / `simulation.fast` for simulation passes
   - API keys (env vars): OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY, AZURE_API_KEY
   - Rate limiting: `simulation.rate_tier`, RPM/TPM overrides
   - Typical config for agent use:
     ```bash
     extropy config set cli.mode agent
     extropy config set models.strong openai/gpt-5
     extropy config set simulation.strong anthropic/claude-sonnet-4.5
     ```

---

## Files to Delete

After creating new SKILL.md and REFERENCE.md, delete:
- `skills/extropy/OPERATIONS.md` (absorbed into REFERENCE.md)
- `skills/extropy/SCENARIOS.md` (absorbed into SKILL.md)
- `skills/extropy/ANALYSIS.md` (absorbed into REFERENCE.md)
- `skills/extropy/TROUBLESHOOTING.md` (absorbed into REFERENCE.md)
- `skills/extropy/REPORT_TEMPLATE.md` (delete - too rigid, Claude can generate reports on demand)

---

## Key Content to Include

### Spec Example (Good)
```bash
extropy spec "Nationally representative US adult population (18-80). Must capture the demographic, economic, and attitudinal fault lines that drive divergent responses to major national events — especially technology disruption, economic shocks, and cultural controversies. Beyond standard demographics, prioritize attributes that determine HOW people process and react to news: technology adoption posture, media ecosystem and information sources, institutional trust level, financial margin and economic anxiety, consumer identity and brand relationship patterns, social media behavior and influence. Geographic distribution should span urban/suburban/rural across all major US regions." -o population.v1.yaml -y
```

**Why it works**: Defines boundary (US adults 18-80), hints at behavioral dimensions without dictating exact categories, includes attitudinal layer (trust, tech adoption, media habits), reusable across many scenarios.

### Scenario Example - Non-Evolving (Netflix)
```bash
extropy scenario "Netflix announces enforcement of password sharing policy. Subscribers who share their password with people outside their household must either: pay $8/month for an extra member slot, remove shared users, or accept that shared users will be blocked. The policy takes effect in 30 days. Coverage is widespread across tech news, social media, and mainstream outlets." -o netflix-password -y
```

**Why non-evolving**: The decision is concrete and personal. Population heterogeneity (income, household composition, alternatives awareness) drives the interesting splits. My neighbor's opinion doesn't change whether I can afford $8/month.

### Scenario Example - Evolving (ASI)
```bash
extropy scenario "OpenAI holds a press conference announcing they have achieved artificial superintelligence — a system that demonstrably exceeds human cognitive ability across every domain. Sam Altman presents benchmark results showing the system outperforms top experts in science, law, medicine, strategy, and creative reasoning. The announcement is covered live on all major networks. Timeline events: Day 1 - OpenAI's announcement and initial reactions. Day 3 - Anthropic and Google DeepMind release statements confirming they have independently reached similar capabilities. Day 5 - xAI and Meta confirm the same; Congress announces emergency hearings. Day 7 - Deloitte announces it will replace 30% of its workforce with AI systems within 12 months." -o asi-announcement --timeline evolving -y
```

**Why evolving**: Information/credibility dynamics matter. Day 1 people can dismiss as hype. By Day 5 when every lab confirms, denial breaks. Day 7 makes it personal. Network propagation happens in the gaps between events.

### The Evolving Test
> "Does the network change the outcome? If agents talking to each other would meaningfully shift results compared to running independent LLM calls, use evolving. If not, non-evolving is cleaner and more honest."

### Timestep Table
| Unit | Use When |
|------|----------|
| Hours | Active crisis, market crash, disaster response |
| Days | Viral news, product launches, policy announcements |
| Weeks | Adoption curves, behavior change, campaign effects |
| Months | Cultural shifts, market trends, policy adaptation |

---

## Verification

After implementation:
1. Run `extropy --help` and verify all referenced commands exist
2. Run through quick pipeline example to ensure commands work
3. Check that SKILL.md frontmatter is valid YAML
4. Verify cross-references between SKILL.md and REFERENCE.md are correct
5. Test that skill loads in Claude Code: `/extropy`

