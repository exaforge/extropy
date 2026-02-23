# Study 2: US Strikes on Iran — Evolving Behavioral Simulation

## Overview

An evolving 12-week simulation of how the American population responds to a US military strike on Iran, modeled from the moment of the strike through economic cascade, military escalation, fragile ceasefire, and into the 2026 midterm election cycle.

This study replaces the previously planned Taiwan scenario. The Iran crisis is not hypothetical — as of February 22, 2026, the US has amassed its largest Middle Eastern naval deployment since 2003, reports indicate strikes could come within days, and the situation is on a knife's edge between military action and a last-minute deal. Simulating the behavioral aftermath of strikes landing is immediately relevant, maximally viral, and activates nearly every demographic fault line in the national spec.

**Why Iran is a stronger study than Taiwan:**

- **Immediate relevance.** People are already anxious about this. The simulation lands in an active news cycle.
- **Richer fault lines.** Taiwan was primarily an economic/semiconductor story. Iran hits simultaneously: oil prices and gas at the pump, military families facing deployment, anti-war vs. hawkish splits within BOTH parties, Muslim-American communities, veterans who remember Iraq, energy sector workers, defense contractors, the protest/regime-change moral dimension.
- **Cross-study coherence.** The Iran crisis feeds directly into Study 4 (2026 midterms). The same 5,000 agents experience the Iran crisis AND then vote in November. The midterm simulation gains a concrete backdrop.
- **Every dimension in the existing spec is activated.** Military connection, political identity, media ecosystem, institutional trust, employment sector, financial margin, religious engagement, consumer behavior — all directly relevant.

---

## Spec: National US Population (Spec A — Shared)

This study uses the same national US base spec shared across Studies 1, 2, and 3. No dedicated spec is needed — the Iran scenario activates the existing dimensions comprehensively.

### Spec Prompt

```bash
extropy spec "Nationally representative US adult population (18-80). Must capture the
demographic, economic, and attitudinal fault lines that drive divergent responses to
major national events — especially technology disruption, economic shocks, and cultural
controversies.

Beyond standard demographics (age, gender, race/ethnicity, education, income, geography),
prioritize attributes that determine HOW people process and react to news and make
decisions under uncertainty:

- Technology adoption posture and digital literacy (from technophobe to early adopter)
- Media ecosystem and primary information sources (cable news, social media, podcasts,
  local news, none)
- Institutional trust level (government, corporations, media, science, financial system)
- Financial margin and economic anxiety (savings buffer, debt load, paycheck-to-paycheck
  status)
- Employment sector and job security perception (federal/state government, private sector,
  gig, self-employed, retired)
- Consumer identity and brand relationship patterns (brand loyal, price-driven,
  values-driven)
- Social media behavior and influence (passive consumer, active poster, influencer,
  non-user)
- Investment and financial literacy (no investments, 401k only, active trader, crypto
  holder)
- Religious engagement and worldview (secular, moderate, devout, specific denomination)
- Political identity and engagement level (strong partisan, moderate, disengaged,
  single-issue)

Geographic distribution should span urban/suburban/rural across all major US regions
(Northeast, Southeast, Midwest, Southwest, West Coast), with realistic correlation
structure between attributes (e.g., rural + lower digital literacy + higher religious
engagement)." \
  -o specs/us-national-v1 -y
```

### Why This Spec Works for Iran

The spec was designed to be scenario-agnostic — capturing general human fault lines rather than topic-specific attributes. For the Iran strike scenario, the key activated dimensions are:

| Spec Dimension | How Iran Activates It |
|---|---|
| **Financial margin** | Gas at $6/gallon hits paycheck-to-paycheck agents catastrophically. Comfortable agents notice but adapt. |
| **Employment sector** | Federal/military employees face deployment risk. Defense contractors see mixed signals (contracts surge but risk increases). Energy sector workers potentially benefit. Service/gig workers get crushed by fuel costs. |
| **Political identity** | Hawks vs. doves is NOT a clean party split. Some Republicans oppose foreign intervention, some Democrats support strikes to prevent nuclear Iran. This is where the simulation gets interesting. |
| **Media ecosystem** | Fox News viewers and MSNBC viewers will be living in different realities about what happened and why. Social media users will encounter different narratives than cable news viewers. |
| **Institutional trust** | High-trust agents believe the Pentagon's justification. Low-trust agents assume ulterior motives (oil, distraction, defense industry profits). |
| **Religious engagement** | Evangelical communities may frame this through eschatological lenses. Muslim-American agents face a completely different experience. Secular agents process it through economic/political lenses. |
| **Geography** | Military base communities (Fayetteville, San Diego, Norfolk) react differently than college towns. Rural communities dependent on driving are hit harder by gas prices than dense urban areas with transit. |
| **Investment literacy** | Active traders react to market chaos differently than 401k-only agents who may not check their balance for months. |

### What to Check After `extropy spec`

1. **Completeness**: All 10+ attribute dimensions present
2. **Distribution realism**: Income, age, geography match Census data (~20% rural, ~30% bachelor's+, median income ~$75K)
3. **Correlation structure**: Encoded realistically (rural + lower digital literacy + higher religious engagement, etc.)
4. **No degenerate categories**: Every attribute has 3-4+ meaningful values, no "Other" bucket >20%
5. **Behavioral dimensions present**: Trust, media ecosystem, financial margin — not just a census table
6. **No hallucinated data**: Spot-check 3-4 cited percentages against known sources

**If any fail:** Fix before proceeding. The spec is the foundation for Studies 1, 2, 3, and 4a. Getting it right matters for everything downstream.

---

## Scenario

### Design Principles Applied

- **Milestones, not reactions.** Every timeline event describes what happens in the world — strike targets, missile counts, oil prices, troop deployments, ceasefire terms. Never scripts how people feel or behave. "Protests erupt" is factual (they will). "People are terrified" would be a violation.
- **Concrete specifics.** Not "gas prices go up" but "$3.40 → $4.80 overnight, $6.20 by Week 3, California above $8." Agents need specific numbers to reason against their own financial circumstances.
- **Escalation that hits different demographics differently.** Week 1 is a military/political event. Week 3 is an economic event (Strait of Hormuz). Week 5 hits employment and food prices. Week 8 brings the political dimension (midterms). Week 12 introduces a ceasefire that doesn't undo the damage. Each stage activates new population segments.
- **Neutral framing.** The scenario presents facts about what both sides do and say. It does not editorialize about whether the strikes were justified or unjustified.

### Context Grounding (as of February 22, 2026)

The scenario is anchored in real events. Key facts injected into the scenario:

- The US deployed two carrier strike groups (Abraham Lincoln, Gerald R. Ford) plus B-2 bombers, F-22s, F-35s, and submarines
- Iran fortified nuclear sites at Natanz and Parchin with concrete bunkers
- The US and Israel previously struck Iran's nuclear facilities in June 2025 during a 12-day conflict
- Iran retaliated by striking Al Udeid Air Base in Qatar during the June 2025 conflict
- Indirect talks were held in Muscat (Feb 6) and Geneva (Feb 18) without resolution
- Trump set a 10-15 day deadline for a deal
- Iran's IRGC Navy attempted to seize a US tanker in the Strait of Hormuz on Feb 3
- A US F-35 shot down an Iranian drone approaching the USS Abraham Lincoln
- Oil markets were already jittery; pre-strike gasoline averaged ~$3.40/gallon nationally
- 2026 midterm elections are on November 3

### Scenario Prompt

```bash
extropy scenario -s "On February 23, 2026, the United States launched airstrikes against
Iran, targeting nuclear facilities at Natanz and Parchin, IRGC command infrastructure,
ballistic missile production sites, and air defense systems across multiple provinces.
President Trump addressed the nation from the Oval Office, stating that Iran's refusal
to reach a diplomatic agreement and its continued pursuit of nuclear weapons capability
left the United States 'no choice but to act decisively.' The strikes involved B-2
stealth bombers operating from Diego Garcia, carrier-based F/A-18s from the USS Abraham
Lincoln and USS Gerald R. Ford carrier strike groups in the Arabian Sea, and Tomahawk
cruise missiles launched from submarines in the Persian Gulf. The Pentagon confirmed
strikes on 47 targets across Iran. No US ground forces were deployed.

Iran's Supreme Leader Khamenei issued a statement calling the strikes 'an act of war
against the Iranian nation' and vowed retaliation. Iran's foreign minister recalled all
diplomatic contacts with the United States. The UN Security Council convened an emergency
session. Oil futures spiked 35% within hours of the first reports. European allies
issued mixed responses — the UK declined to participate, France called for immediate
ceasefire, Germany urged restraint from both sides. Israel expressed support but did not
participate directly in the strikes.

Timeline events:

Week 1: The immediate aftermath. The Pentagon releases battle damage assessments showing
significant destruction of Iran's nuclear enrichment infrastructure and missile
production facilities. Iran fires 30+ ballistic missiles at US military installations
in Qatar (Al Udeid Air Base) and Bahrain (Naval Support Activity). US missile defense
systems intercept most but not all — 3 US service members are killed and 47 wounded at
Al Udeid. Gasoline prices in the US jump from $3.40/gallon national average to $4.80
overnight. The Dow drops 1,400 points. Protests occur in major US cities — both anti-war
demonstrations and pro-strike rallies. Congress is divided: Senate passes a resolution
supporting the strikes 54-46, House fails to pass any resolution.

Week 3: Iran announces a naval blockade of the Strait of Hormuz. Two commercial tankers
are struck by IRGC anti-ship missiles, killing 14 crew members. Lloyd's of London
suspends insurance coverage for all vessels transiting the Persian Gulf. Oil reaches
$140/barrel. US gasoline hits $6.20/gallon nationally, with California stations above
$8.00. The Federal Reserve issues an emergency statement warning of inflationary
pressures. Major US airlines announce fuel surcharges of $75-150 per ticket.
Iran-backed Houthi forces intensify attacks on Red Sea shipping, effectively closing
a second major trade route. Walmart, Target, and Amazon warn of supply chain disruptions
and potential price increases on imported goods within 30-60 days.

Week 5: The economic cascade deepens. US unemployment claims rise 40% as energy-dependent
industries begin layoffs. Trucking companies impose 25% fuel surcharges on all freight.
Food prices rise 15% nationally as transportation and fertilizer costs spike. The Federal
Reserve holds an emergency meeting but declines to cut rates, citing inflation risk. Iran
launches a second wave of missile strikes targeting Saudi oil infrastructure at Abqaiq
and Ras Tanura, temporarily knocking 4 million barrels/day of Saudi production offline.
Oil briefly touches $180/barrel. The US announces deployment of 15,000 additional troops
to the Middle East. The Pentagon calls up 30,000 reservists.

Week 8: The Strait of Hormuz remains contested. US Navy minesweepers and escort convoys
have partially reopened shipping lanes, but transit times have doubled and insurance
premiums remain 10x pre-crisis levels. Oil has stabilized around $130/barrel. US gasoline
averages $5.80/gallon. Cumulative US job losses attributed to the energy shock reach
400,000. Iran's internal protest movement has fractured — some factions rally behind the
regime against foreign attack, others blame the regime for provoking the strikes. Russia
and China issue a joint statement condemning the US strikes and announcing expanded
economic support for Iran. The 2026 midterm elections are 3 months away. Both parties
are attempting to frame the conflict — Republicans argue the strikes prevented a nuclear
Iran, Democrats argue the administration stumbled into an unnecessary war that is
destroying the economy.

Week 12: A fragile ceasefire is brokered through Omani and Qatari mediation. Iran agrees
to halt Strait of Hormuz operations in exchange for a US commitment to cease strikes and
begin indirect negotiations on a new nuclear framework. Oil drops to $105/barrel but
remains well above pre-crisis levels. US gasoline averages $4.60/gallon. Total estimated
US economic cost of the 12-week crisis exceeds $800 billion. 23 US service members have
been killed across all theaters. The political landscape has shifted — the crisis
dominates every 2026 midterm campaign. Inflation is running at 8.5% annualized. Consumer
confidence is at its lowest level since 2008." \
  --type evolving \
  --timestep weeks \
  --timesteps 12 \
  -o scenario/iran-strikes -y
```

### What to Check After `extropy scenario`

1. **Milestones only**: No milestone scripts human behavioral responses. "Protests occur" is factual (observable). "People are panicking" would be a violation.
2. **Evolving type confirmed**: Weekly timesteps, 12 total
3. **Neutral framing**: Both the US justification and Iran's response are presented factually
4. **Concrete specifics present**: Dollar amounts, barrel prices, casualty numbers, troop counts — agents need these to reason against their own circumstances
5. **Escalation hits different demographics at different stages**: Military families (Week 1), all drivers (Week 1-3), energy-dependent workers (Week 5), reservists (Week 5), voters (Week 8), everyone (Week 12)
6. **No future-event contamination risk**: Each week's milestones are self-contained and don't reference later events

---

## Outcomes

### Design Rationale

Every outcome follows the best practices:

- **No floats.** Every outcome is categorical or open-ended.
- **Categorical options are concrete and distinguishable.** Each describes an observable state or action, not a point on a vague scale.
- **Two open-ended fields** surface emergent behavior beyond the categorical framework.
- **No lazy booleans.** Every question where the "how" matters more than the "whether" uses categorical.

### Outcome Definitions

```
--outcomes "war_support: categorical (strongly support the strikes — Iran had to be
stopped, support but concerned about escalation and consequences, conflicted — see
arguments on both sides, oppose but understand the rationale for acting, strongly
oppose the strikes — this was unnecessary and reckless)"

--outcomes "personal_economic_impact: categorical (severely affected — can't afford
basic needs like gas groceries or bills, noticeably affected — cutting back significantly
on spending, somewhat affected — adjusting budget and delaying purchases, minimally
affected — noticed price increases but managing fine, not meaningfully affected)"

--outcomes "political_shift: categorical (more supportive of the current administration,
no change in political views, less supportive of the current administration, completely
changed my view of this administration, disengaged from politics entirely)"

--outcomes "behavioral_change: categorical (no changes to daily life, minor adjustments
— driving less or combining errands, significant changes — canceled travel or delayed
major purchases, major life disruption — job affected or drawing down savings,
preparing for worse — stockpiling supplies or making contingency plans)"

--outcomes "midterm_vote_impact: categorical (more motivated to vote and support the
administration party, more motivated to vote and oppose the administration party, no
change in my voting plans, less likely to vote — disillusioned with both parties, was
not planning to vote regardless)"

--outcomes "threat_perception: categorical (Iran is the primary threat — strikes were
necessary to prevent a nuclear-armed Iran, the economic fallout is the primary threat
— the strikes made everything worse, wider regional war is the primary threat — this
could spiral out of control, domestic instability is the primary threat — the country
is tearing itself apart, do not feel personally threatened by any of this)"

--outcomes "personal_response: open-ended (describe how this crisis is affecting your
daily life, your family, your community, and your view of the country's direction)"

--outcomes "unexpected_factor: open-ended (what is happening in your life or community
because of this crisis that you think the news is not covering?)"
```

### Why These Outcomes

| Outcome | What It Measures | Key Spec Dimensions Activated |
|---|---|---|
| `war_support` | Core political question. NOT a clean hawk/dove binary — the middle three options capture the messy ambivalence where most Americans live. | Political identity, media ecosystem, institutional trust, military connection |
| `personal_economic_impact` | Observable material consequences. $6/gallon doesn't hit a remote worker in Manhattan the same way it hits a trucker in rural Texas. | Financial margin, employment sector, geography (urban/rural), consumer identity |
| `political_shift` | Does the crisis move people politically? Rally-round-the-flag vs. economic pain backlash. | Political identity, media ecosystem, institutional trust |
| `behavioral_change` | Actions, not feelings. Are people actually changing behavior? | Financial margin, geography, employment sector |
| `midterm_vote_impact` | Directly connects this study to Study 4 (midterms). Same population, same spec. | Political identity, political engagement level |
| `threat_perception` | What are people actually afraid of? Segments beautifully by media ecosystem — Fox viewers and MSNBC viewers will identify different primary threats. | Media ecosystem, institutional trust, political identity, military connection |
| `personal_response` | Open-ended narrative. Captures the subjective experience and reasoning that categoricals can't. | All dimensions |
| `unexpected_factor` | Explicitly invites responses outside the categorical framework. This is where emergent second-order effects surface. | All dimensions |

---

## What to Check After Simulation

### Temporal Evolution (Week 1 → 12)

- [ ] `war_support` should shift over time. Expect rally-round-the-flag in Week 1 that erodes as economic pain mounts through Weeks 3-8. The ceasefire at Week 12 should NOT fully restore Week 1 support levels.
- [ ] `personal_economic_impact` should escalate sharply Week 1→3 (gas price shock) and continue deepening through Week 5 (food prices, job losses). Should NOT snap back at Week 12 — agents who lost jobs or depleted savings don't recover because oil dropped $25.
- [ ] `political_shift` should show divergent trajectories: initially supportive agents may turn as costs mount; initially opposed agents likely harden. The middle ("conflicted") should shrink over 12 weeks as people are forced to take a position.
- [ ] `midterm_vote_impact` should intensify over time as the election approaches. By Week 8, nearly every agent should have SOME midterm impact — "no change" should be a shrinking category.

### Demographic Divergence

- [ ] **Military-connected agents** should have distinct trajectories. Support for strikes but deep anxiety about deployment/escalation. Casualty reports (Week 1, ongoing) hit this group personally. Reservist callup at Week 5 is a concrete life disruption for this segment.
- [ ] **Low financial margin agents** should show the sharpest economic impact escalation. By Week 3, paycheck-to-paycheck agents at $6/gallon gas should be in genuine crisis. High-income agents should still be at "adjusting budget."
- [ ] **Rural agents** should be hit harder by gas prices than urban agents (longer commutes, no transit alternatives, more driving-dependent).
- [ ] **Federal/defense sector employees** should have a complex response — the strike may benefit their industry long-term but the crisis creates immediate uncertainty.
- [ ] **Media ecosystem should drive threat perception.** Conservative media consumers more likely to identify Iran as the primary threat. Liberal media consumers more likely to identify economic fallout or wider war. Social-media-primary agents may have the most fragmented threat perception.
- [ ] **Religious engagement** should produce distinct framing. Evangelical agents may invoke eschatological or moral frameworks. Muslim-American agents face a fundamentally different experience (community anxiety, potential backlash). Secular agents process through economic/political lenses.

### Quality Checks

- [ ] Reasoning traces are first-person, differentiated, and reference specific scenario details (gas prices, casualty numbers, not vague "the situation")
- [ ] No more than 20% of agents give near-identical responses at any timestep
- [ ] No future-event contamination: Week 3 agents must not reference the ceasefire brokered at Week 12
- [ ] Open-ended `unexpected_factor` responses surface genuinely novel insights beyond the categorical outcomes — if they just restate the categorical selection, central tendency bias is dominating
- [ ] Intent-to-action sanity check: if 80% of agents report "major life disruption" by Week 3, that's LLM enthusiasm bias. Real populations are stickier than that.

---

## Cross-Study Connections

### Link to Study 4 (2026 Midterms)

The Iran crisis is the defining context for the November 2026 midterm elections. The same 5,000 Spec A agents who experience the Iran strike scenario in Study 2 are the same national electorate being modeled in Study 4a (House control prediction).

This creates a powerful cross-study narrative:

- Study 2 shows HOW the Iran crisis reshapes political attitudes over 12 weeks
- Study 4a shows WHETHER those reshaped attitudes translate into actual midterm voting behavior
- The `midterm_vote_impact` outcome in Study 2 directly previews what Study 4a will model in detail

### Link to Study 1 (ASI Announcement)

Both studies use Spec A and can be compared side-by-side: "Here's how the same 5,000 Americans respond to a technological black swan vs. a geopolitical black swan." The demographic segmentation patterns should be dramatically different — the ASI scenario primarily activates technology posture and employment sector, while Iran primarily activates political identity, financial margin, and military connection.

---

## Pipeline Commands

```bash
# Spec A should already be generated and validated from Study 1
# If not:
extropy spec [spec prompt above] -o specs/us-national-v1 -y

# >>> CHECKPOINT: Gate 1 — validate spec (see checklist above)

# Generate personas for Iran study
extropy persona -s iran-strikes -y

# >>> CHECKPOINT: Gate 3 — verify persona range
# >>> Must include: military family in base town, paycheck-to-paycheck rural commuter,
#     wealthy urban professional, Muslim-American, evangelical veteran, anti-war college
#     student, defense contractor employee, energy sector worker, retired fixed-income,
#     politically disengaged single parent
# >>> Must NOT be all politically engaged coast-dwellers

# Sample agents
extropy sample -s iran-strikes -n 5000 --seed 42

# >>> CHECKPOINT: Gate 4 — pull 30 random agents, verify distribution
# >>> Verify: geographic spread (not all coastal cities)
# >>> Verify: financial margin distribution (at least 25-30% paycheck-to-paycheck)
# >>> Verify: military connection exists but isn't overrepresented (~7-10% veteran, ~1% active duty)
# >>> Verify: political identity spans full range (not all moderates)

# Build network
extropy network -s iran-strikes --seed 42

# >>> CHECKPOINT: Gate 5 — verify network topology
# >>> LLM REVIEW MANDATORY

# Run simulation
extropy simulate -s iran-strikes --seed 42

# >>> CHECKPOINT: Gate 6 — full results review
extropy results
extropy results --segment political_identity
extropy results --segment financial_margin
extropy results --segment employment_sector
extropy results --segment geography
extropy results --segment media_ecosystem
```

---

## What Extropy Surfaces That Other Methods Can't

1. **The rally-to-resentment trajectory is not uniform.** It depends on which specific demographic you are, how the economic pain hits you, and what information ecosystem you're in. A poll at Week 3 tells you "58% oppose." Extropy tells you which 58%, why, and how they got there from Week 1.

2. **Economic pain is geographically and demographically clustered.** $6 gas in Manhattan (where most people take the subway) is a different event than $6 gas in rural Oklahoma (where everything requires driving 30+ miles). Extropy models this because the spec encodes geography, financial margin, and employment sector jointly.

3. **Military families are the swing constituency.** They're the one demographic with both strong patriotic framing (support the mission) AND direct personal cost (deployment risk, casualty anxiety). How this group moves over 12 weeks likely predicts the midterm impact better than any national poll.

4. **The ceasefire doesn't reset the damage.** An agent who lost their job in Week 5 doesn't get un-fired in Week 12. An agent who depleted their savings on $6 gas doesn't get a refund. Extropy tracks cumulative impact, not point-in-time sentiment.

5. **Media ecosystem creates parallel realities.** By Week 8, Fox viewers and MSNBC viewers may have fundamentally incompatible understandings of what happened and why. This fragmentation is invisible in polls but Extropy can measure it because the spec encodes media ecosystem as an agent attribute.
