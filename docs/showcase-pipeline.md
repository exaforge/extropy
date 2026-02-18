# Extropy Showcase Pipeline: Execution Plan

## Mission

Run 4 Extropy studies to produce viral, publishable population simulations. Three hypothetical showcases demonstrate capabilities. One live prediction market bet puts real money where Extropy's mouth is.

**Goal: Maximize virality, build track record, win on Kalshi.**

---

## The 4 Studies (5 Simulations)

| # | Study | Spec | Agents | Type | Timestep | Duration | Deadline |
|---|-------|------|--------|------|----------|----------|----------|
| 1 | **ASI Announcement** | National US | 5,000 | Evolving | Monthly | 6 months | None (showcase) |
| 2 | **US-China Taiwan Crisis** | National US | 5,000 | Evolving | Monthly | 6 months | None (showcase) |
| 3 | **Bitcoin Hits $1M** | National US | 5,000 | Evolving | Weekly | 12 weeks | None (showcase) |
| 4a | **2026 House Control** | National US | 5,000 | Non-evolving | Single decision | N/A | **November 3, 2026** |
| 4b | **Alaska Senate: Sullivan vs Peltola** | Alaska | 1,000 | Non-evolving | Single decision | N/A | **November 3, 2026** |

**Two base specs:**
- **Spec A (National US):** 5,000 agents, nationally representative, enriched with electoral dimensions. Shared across Studies 1-3 and Study 4a.
- **Spec B (Alaska):** 1,000 agents, representative of Alaska's electorate, with candidate-specific and state-specific attributes. Used only for Study 4b.

**Run order: Study 1 first** (ASI, most viral), then Studies 2 and 3 in parallel, then Studies 4a and 4b (Midterms — longest runway, real money bets).

---

## Agent Counts: Why 5,000

**All studies (5,000):** Survey research needs ~1,500 for ±3% margin of error. But Extropy isn't polling — it's simulating network cascades. You need cluster density. At 5,000 agents with ~20 major demographic buckets, that's ~250 agents per bucket. Enough for genuine within-group splits and cascade dynamics. 10,000 would be ideal but doubles runtime and cost. All 4 studies use the same 5,000-agent national population — same people, different scenarios.

**Why not 10K for the midterms bet?** We can re-run Study 4 at 10K once we've validated the pipeline on Studies 1-3. The first pass at 5K establishes methodology; a 10K re-run closer to November gives the refined prediction we bet real money on.

---

## Spec A: National US Population (Studies 1-3 and 4a)

One national population spec serves Studies 1-3 (showcases) and Study 4a (House control). Studies 1-3 test how this population reacts to hypothetical shocks. Study 4a asks how they vote for Congress in November 2026. The spec must capture both behavioral/consumer dimensions AND electoral dimensions. Spec B (Alaska) is defined separately under Study 4b.

### Spec Prompt

```bash
extropy spec "Nationally representative US adult population (18-80). Must capture the demographic, economic, and attitudinal fault lines that drive divergent responses to major national events — especially technology disruption, economic shocks, cultural controversies, AND electoral behavior in the 2026 midterm elections.

Beyond standard demographics (age, gender, race/ethnicity, education, income, geography), prioritize attributes that determine HOW people process and react to news, make decisions under uncertainty, and vote:

- Technology adoption posture and digital literacy (from technophobe to early adopter)
- Media ecosystem and primary information sources (cable news, social media, podcasts, local news, none)
- Institutional trust level (government, corporations, media, science, financial system)
- Financial margin and economic anxiety (savings buffer, debt load, paycheck-to-paycheck status)
- Employment sector and job security perception (federal/state government, private sector, gig, self-employed, retired)
- Consumer identity and brand relationship patterns (brand loyal, price-driven, values-driven)
- Social media behavior and influence (passive consumer, active poster, influencer, non-user)
- Investment and financial literacy (no investments, 401k only, active trader, crypto holder)
- Religious engagement and worldview (secular, moderate, devout, specific denomination)
- Political identity and engagement level (strong partisan, moderate, disengaged, single-issue)
- 2024 presidential vote (Trump, Harris, third party, didn't vote, not eligible)
- Congressional district competitiveness (safe R, safe D, swing district)
- Midterm turnout propensity (always votes in midterms, sometimes, rarely, never has)
- Issue salience for 2026 (economy/jobs, abortion/reproductive rights, immigration, democracy/rule of law, healthcare, climate, education, AI/technology)
- Approval of current direction (strongly approve, somewhat approve, mixed, somewhat disapprove, strongly disapprove)

Geographic distribution should span urban/suburban/rural across all major US regions (Northeast, Southeast, Midwest, Southwest, West Coast), with realistic correlation structure between attributes (e.g., rural + lower digital literacy + higher religious engagement; federal employee + DC metro + higher institutional trust). Congressional district mapping should reflect actual competitive districts for 2026 (e.g., NY-17, CA-27, PA-08, MI-07, etc.)." \
  -o us-national-study -y
```

This creates a study folder (`us-national-study/`) with `population.v1.yaml` and `study.db`. Run the scenario/persona/sample/network/simulate commands for Studies 1-4a from inside that study folder.

### What to Check After `extropy spec`

See **Gate 1** in the Pipeline Validation section below for the full validation checklist. Key points:

1. All 15+ attribute dimensions represented (including 5 electoral dimensions)
2. Distributions roughly match US Census/Pew/Gallup benchmarks
3. Correlation structure encoded (not independently distributed attributes)
4. Electoral dimensions realistic (2024 vote matches actuals, midterm turnout skews toward "sometimes"/"rarely")
5. No hallucinated statistics — spot-check any cited percentages

**If any of these fail:** Re-run spec with a more specific prompt targeting the gap. Do NOT proceed to persona generation with a weak spec — everything downstream depends on this.

---

## Study 4: The Bets — 2026 Midterms

### Design Principle: Elections Need Candidates, Not Just Party Labels

**CRITICAL LESSON LEARNED:** Our initial Study 4 design asked 5,000 national agents "will you vote D or R for Senate?" without specifying who the candidates were. This is useless. A voter in Maine deciding whether to re-elect Susan Collins is making a completely different decision than a voter in Texas deciding on an open-seat race. Without candidate context, you're just measuring party ID — which is already in the spec. That's a survey, not a simulation.

**The fix:** Split electoral prediction into two approaches based on how candidate-dependent the race is:

1. **House control (Study 4a):** Generic ballot works here. Most voters can't name their House rep. In 435 races, national environment + party ID + district lean determines the outcome. The "would you vote D or R for Congress?" question is the single best predictor of House control, and it's exactly what Extropy can model with a national population.

2. **Individual Senate races (Study 4b):** Candidates matter enormously. Collins has a specific brand. Peltola has a specific brand. Voters respond to the specific person, not the party label. These require dedicated state-level specs with candidate-specific scenario context. We pick races where both candidates are already known.

**What this means for future election studies:** Any Extropy simulation of a specific named race MUST include candidate-level context — their record, their messaging, their specific appeal to different demographics. Generic party preference is not simulation, it's a crosstab of your spec.

### Kalshi Markets

**Study 4a — House Control:**
- **House Winner**: https://kalshi.com/markets/controlh/house-winner/controls-2026
- Current pricing: D ~69%, R ~31%

**Study 4b — Alaska Senate:**
- **Alaska Senate (Party)**: https://kalshi.com/markets/senateak/alaska-senate-race/senateak-26
- **Alaska Senate (Person)**: https://kalshi.com/markets/kxaksenate/who-will-win-the-alaska-senate-race/kxaksenate-26nov03
- Current pricing (as of Feb 17, 2026): R 57% (Sullivan), D 43% (Peltola). $148K volume. Resolution: November 3, 2026.

**Also tracking (for combined bets):**
- **Congress Balance of Power**: https://kalshi.com/markets/kxbalancepowercombo/congress-balance-of-power-combo/kxbalancepowercombo-27feb
- Current pricing: D-House/R-Senate 47%, D-House/D-Senate 37%, R-House/R-Senate 14%, R-House/D-Senate 2%. Volume: $846K.

### Why Extropy Shines Here

Traditional prediction market pricing aggregates vibes. Polls aggregate stated preferences. Neither models the MECHANISM — which specific populations in which specific districts actually show up, and why. Extropy does.

**For House control (Study 4a):**

1. **Differential turnout across ~35 competitive House districts** — not national sentiment, but whether a nurse in PA-08 who voted Biden in 2020 and skipped 2024 shows up in November. Extropy models this agent-by-agent with realistic demographic profiles, employment sector exposure, and issue salience.

2. **Cascade dynamics from real events** — DOGE layoffs, tariff-driven retail closures, and federal workforce cuts are hitting specific geographies (military towns, DC metro, rural USDA communities). These aren't uniform shocks. They propagate through social networks. Traditional polling asks "how do you feel about the economy?" — Extropy simulates "your neighbor lost his VA hospital job, your church is organizing, your local diner closed. Now how do you vote?"

3. **The enthusiasm gap is demographic, not national** — Midterm turnout is 40-50% vs 60-65% in presidentials. WHO stays home matters more than national mood. Young voters drop off more. Occasional voters in safe districts drop off more. But DOGE-affected federal employees in swing districts might turn out at presidential-year levels. Extropy models turnout propensity per-agent based on their specific profile, not a uniform turnout assumption.

**For Alaska Senate (Study 4b):**

1. **Native turnout is the X-factor** — Alaska Natives are ~16% of the population. Peltola is the first Alaska Native in Congress. Her candidacy could drive unprecedented Native turnout — but these communities are remote, with logistical barriers to voting. Whether Native turnout hits 2022 levels (when Peltola first won her House seat) or drops to 2024 levels likely determines the race. Prediction markets cannot model this. Extropy can.

2. **Ranked-choice voting dynamics** — Alaska uses a jungle primary (top 4 advance) and ranked-choice general election. Third-party and independent voters' second-choice rankings could be decisive. In 2022, Peltola won specifically because of RCV. Extropy models each agent's first AND second choice based on their profile.

3. **DOGE crossover voters** — Are there Sullivan 2020 voters who flip to Peltola because of VA cuts or federal workforce reductions? How many? This is the mechanism that prediction markets can't see. Extropy can model whether a military veteran in Fairbanks who voted Sullivan in 2020 actually switches over DOGE, or just grumbles and votes R anyway.

4. **Fishing community intensity** — Bycatch is Peltola's core issue. But is it a voting issue or a dinner table complaint? Extropy can distinguish between "I care about this" and "I'll change my vote over this."

**Where existing methods fail:**

- **Polls (9 months out):** Unreliable this far from election. Only one AK poll exists (DFP: Sullivan 46%, Peltola 45%).
- **Prediction markets:** Aggregate trader sentiment. At R 57% / D 43% the Alaska market is uncertain — exactly where a structural model adds signal.
- **Forecasting models (538-style):** Good at aggregating polls closer to election. Weak this early. Don't model RCV dynamics, Native turnout, or candidate-specific crossover appeal.
- **Extropy's edge:** Models the MECHANISM of turnout and vote choice, not just sentiment. Can re-run as conditions change. Each re-run is a publishable update.

---

### Study 4a: House Control

Uses the national 5,000-agent Spec A. Non-evolving, single decision point.

#### Scenario Prompt

```bash
extropy scenario "The 2026 US midterm elections are being held on November 3, 2026. All 435 House seats are on the ballot.

Current political context (as of early 2026):

Republicans hold a narrow House majority (220-215). President Trump is in his second year of his second term. His approval rating is approximately 42-45%.

Key dynamics shaping the House election:

1. DOGE LAYOFFS AND ECONOMIC DISRUPTION: The Department of Government Efficiency has eliminated 72,000+ federal jobs across 17 agencies, with projections of up to 500,000 by year-end. Effects are hitting DC metro, military towns, VA hospitals, USDA rural offices, and NASA centers. Defense contractors have announced 45,000+ layoffs from canceled contracts. Retailers have cut 80,000+ jobs from tariffs and declining consumer spending. Inflation expectations have surged to 6%.

2. DEMOCRATIC MOMENTUM: Democrats swept 2025 off-year elections (NJ governor, VA legislature, NYC mayor). Democratic fundraising is up 40% year-over-year. The party's messaging centers on opposition to DOGE cuts, abortion rights, and anti-corruption.

3. REPUBLICAN CHALLENGES: The narrow House majority means every swing district is contested. Republican members who voted for DOGE-related bills face backlash in districts with federal employment. Trump's approval is underwater in suburbs. The party is banking on immigration, crime, and economic messaging.

4. VOTER MOOD: Consumer confidence is declining. Economic anxiety is concentrated among federal employees, military communities, and tariff-affected industries. But unemployment remains relatively low outside affected sectors. The disconnect between macro indicators and lived experience varies dramatically by geography and employment sector.

The question for each person: will you vote in the 2026 midterm elections, and if so, which party's candidate will you support for Congress?" \
  --timeline static \
  -o house-2026 -y
```

#### Outcomes

- `will_vote_midterm`: boolean — will this person cast a ballot in November 2026
- `house_vote`: categorical — Democrat, Republican, Third Party, Won't Vote, Undecided
- `enthusiasm_level`: categorical — extremely motivated to vote, somewhat motivated, neutral, somewhat unmotivated, will probably skip
- `primary_issue`: categorical — economy/jobs, abortion/reproductive rights, immigration, democracy/rule of law, government spending/DOGE, healthcare, other
- `vote_reasoning`: open-ended — main reason driving decision
- `unexpected_factor`: open-ended — what is influencing your vote or your decision not to vote that you think most pundits and polls are missing

#### Translating to Kalshi

Filter agents in competitive districts (swing district attribute in spec). Calculate D vs R vote share among likely voters (agents with enthusiasm "extremely motivated" or "somewhat motivated"). Apply 10-15% intent-to-action discount for midterm-specific friction (lower salience, forgot to vote, didn't research downballot). If Extropy shows D+6 in swing districts → high confidence in D-House.

#### What to Check After Simulate

- [ ] Turnout intent is NOT uniformly high — midterm turnout is 40-50%. If 80%+ say they'll vote, LLM enthusiasm bias is dominating.
- [ ] Turnout correlates with enthusiasm level.
- [ ] 2024 vote is the strongest predictor of 2026 vote — but not 100%. Some crossover should exist.
- [ ] Federal employees and defense workers in swing districts show distinct voting patterns shaped by DOGE exposure.
- [ ] Agents in safe districts show lower enthusiasm than agents in competitive districts.
- [ ] `primary_issue` distribution varies by demographic.
- [ ] Non-voters have articulated reasons — apathy, disgust, logistical barriers, "doesn't matter in my district."
- [ ] No more than 20% of agents give near-identical vote reasoning.

---

### Study 4b: Alaska Senate — Sullivan vs Peltola

#### Spec B: Alaska Electorate

A dedicated 1,000-agent spec representing Alaska's electorate. This is a SEPARATE spec from the national Spec A.

```bash
extropy spec "Representative Alaska adult population (18-80) for the 2026 Senate election between incumbent Republican Dan Sullivan and Democratic challenger Mary Peltola.

Alaska is unique: 733,000 people, one at-large congressional district, ranked-choice voting with a jungle primary. The electorate does not map neatly to national R/D patterns.

Critical demographic dimensions for this race:

- Ethnicity with emphasis on Alaska Native status (~16% of population, concentrated in rural bush communities, historically lower turnout but Peltola is the first Alaska Native elected to Congress)
- Geography: Anchorage metro (~40% of state), Fairbanks, Juneau, Mat-Su Valley (conservative exurbs), rural bush communities (remote, heavily Native, logistically difficult to vote from), military base communities (JBER, Eielson, Fort Wainwright)
- Employment sector: oil and gas, fishing/seafood, military (active duty and civilian), federal government (national parks, BLM, Coast Guard, FAA), state government, healthcare, tourism, mining
- Military connection: active duty, veteran, military spouse/family, civilian military employee, no military connection
- Relationship to federal employment: federal employee, contractor, family member of federal employee, community economically dependent on federal facility, no connection
- Fishing and subsistence relationship: commercial fisher, subsistence fisher/hunter, recreational fisher, works in seafood processing, no fishing connection
- Political identity: strong Republican, lean Republican, independent/moderate, lean Democrat, strong Democrat, prior Peltola voter (2022 or 2024), prior Sullivan voter (2020)
- 2024 presidential vote: Trump, Harris, third party, didn't vote
- Ranked-choice voting comfort: understand and use RCV strategically, understand but vote only first choice, confused by RCV, opposed to RCV system
- Attitude toward Peltola specifically: favorable, unfavorable, mixed, don't know enough
- Attitude toward Sullivan specifically: favorable, unfavorable, mixed, don't know enough
- DOGE exposure: directly affected (lost job or hours), indirectly affected (community impact), aware but not personally affected, not aware/don't care
- Cost of living sensitivity: struggling significantly, managing but tight, comfortable, affluent
- Media ecosystem: local TV/radio, Anchorage Daily News, social media, national cable news, none/word of mouth

Geographic distribution should reflect Alaska's actual population distribution with ~40% Anchorage, ~15% Mat-Su, ~10% Fairbanks, ~5% Juneau, ~30% rest of state including rural bush communities. Correlation structure: Alaska Native + rural + subsistence fishing + lower income + historically lower turnout. Military + Fairbanks/JBER area + lean Republican. Oil workers + North Slope/Kenai + strong Republican. Anchorage suburbs + moderate + swing voters." \
  -o alaska-study -y
```

This creates a separate study folder (`alaska-study/`) with `population.v1.yaml` and `study.db`. Run the Study 4b scenario/persona/sample/network/simulate commands from inside that folder.

#### Scenario Prompt

```bash
extropy scenario "The 2026 Alaska Senate election pits incumbent Republican Dan Sullivan against Democratic challenger Mary Peltola. The election uses ranked-choice voting: a jungle primary on August 18 (top 4 advance) followed by a ranked-choice general election on November 3, 2026.

THE CANDIDATES:

Dan Sullivan (Republican, incumbent):
- Two-term senator, first elected 2014, re-elected 2020 by 13 points.
- Retired Marine colonel. Strong military brand and deep ties to Alaska's defense community.
- Trump-endorsed. Generally aligned with Trump's agenda.
- Voted for DOGE-related legislation but also voted to extend health insurance subsidies.
- Recently introduced a bycatch bill targeting pollock fleet's impact on salmon — seen as an attempt to neutralize Peltola's core issue.
- Campaign war chest: $6M+. Endorsed by Lisa Murkowski (who previously endorsed Peltola for House).
- Messaging: national security, Alaska resource development, keeping a Republican Senate majority.

Mary Peltola (Democrat, challenger):
- First Alaska Native person elected to Congress. Served one term as Alaska's at-large House member (2022-2025).
- Lost 2024 House re-election to Nick Begich by 3 points, in a year Trump won Alaska by 13 points.
- Moderate Democrat: pro-gun (breaks with national party), pro-fishing, anti-DOGE. Core brand is 'fish, family, freedom.'
- Won her 2022 House seat specifically because of ranked-choice voting — Republican vote split between Begich and Palin.
- Endorsed by EMILY's List. Being recruited heavily by national Democrats and Chuck Schumer.
- Messaging: protecting Alaska's fisheries from bycatch, opposing DOGE cuts to federal services in Alaska, cost of living, putting Alaska over party politics.

POLITICAL CONTEXT:
- Alaska voted Trump +13 in 2024. Only one Democratic senator elected in past 50 years (Begich 2008, lost to Sullivan 2014).
- But Alaska has ranked-choice voting and a nonpartisan primary, which historically benefits moderate candidates.
- National anti-Trump/anti-DOGE sentiment is running high. Democrats swept 2025 off-year elections nationwide.
- Alaska-specific federal cuts: reduced Coast Guard presence, national park service reductions, FAA staffing cuts affecting rural air service (critical for bush communities), VA service reductions.
- Fishing/bycatch: the pollock fleet's bycatch of salmon is a visceral issue in Alaska Native and fishing communities. Sullivan's recent bycatch bill is seen by some as genuine, by others as election-year pandering.
- Cost of living in Alaska is among the highest in the nation, especially in rural communities.

The question: In the November 2026 ranked-choice general election, who will you rank first? Who will you rank second? And why?" \
  --timeline static \
  -o alaska-senate-2026 -y
```

#### Outcomes

- `first_choice_vote`: categorical — Sullivan, Peltola, other Republican candidate, other candidate, won't vote
- `second_choice_ranking`: categorical — Sullivan, Peltola, other, no second choice, not applicable
- `enthusiasm`: categorical — extremely motivated to vote, somewhat motivated, neutral, somewhat unmotivated, will probably skip
- `deciding_factor`: categorical — candidate character/record, party affiliation, fishing/resource issues, military/veterans issues, economy/cost of living, DOGE/federal workforce, abortion/social issues, ranked-choice strategy, other
- `vote_reasoning`: open-ended — why are you voting the way you are, and what could change your mind
- `unexpected_factor`: open-ended — what is influencing your vote that you think pollsters and pundits are missing

#### What Extropy Surfaces That Prediction Markets Can't

1. **Native turnout**: What % of Alaska Native agents actually show up vs stating intent? If Native turnout hits 2022 levels (when Peltola first won), she likely wins. If it drops to 2024 levels, she probably loses. This is the single biggest variable the prediction market isn't modeling.

2. **RCV second-choice dynamics**: Who do third-party voters rank second? A libertarian's second choice and an independent moderate's second choice are very different. Extropy models this agent-by-agent.

3. **DOGE crossover voters**: Are there Sullivan 2020 voters who flip to Peltola because of VA cuts, Coast Guard reductions, or FAA staffing affecting rural air service? How many? This is the mechanism that markets can't see.

4. **Fishing community intensity**: Is bycatch a voting issue or a dinner table complaint? Does Sullivan's recent bycatch bill actually neutralize Peltola's advantage, or do fishing communities see through it?

#### Translating to Kalshi

Count first-choice votes among likely voters (enthusiasm "extremely motivated" or "somewhat motivated"). If no candidate has >50%, eliminate lowest candidates and redistribute by second-choice rankings (simulating RCV). Apply 10-15% intent-to-action discount. Compare Extropy's predicted margin to Kalshi's R 57% / D 43%.

#### What to Check After Simulate

- [ ] Turnout is NOT uniformly high — Alaska midterm turnout is typically 50-55%.
- [ ] Alaska Native agents show realistic turnout variance, not uniform high participation.
- [ ] Military/veteran agents lean Sullivan but show some DOGE-driven crossover.
- [ ] Fishing community agents disproportionately favor Peltola — but not unanimously.
- [ ] RCV second-choice distribution is realistic: most partisans don't rank the other party second.
- [ ] Anchorage suburban agents are the swing demographic — they should show genuine internal conflict.
- [ ] Rural bush community agents cite logistical barriers to voting (distance, transportation, weather) not just preference.
- [ ] Sullivan's bycatch bill generates mixed responses in fishing communities — some see it as genuine, some as pandering.
- [ ] `unexpected_factor` surfaces Alaska-specific dynamics we didn't anticipate.
- [ ] No more than 20% of agents give near-identical vote reasoning.

---

### Bet Sizing (Both Bets)

- **House control**: Initial position $300-500 on D-House. Kalshi pricing D at 69¢ — if Extropy shows D with >80% confidence, there's edge.
- **Alaska Senate**: Initial position $200-500 on whichever candidate Extropy favors. Kalshi pricing R 57¢ / D 44¢ — genuinely uncertain, maximum edge potential.
- Kelly criterion: if Extropy shows 15% edge vs market price, bet 15% of bankroll.
- **Scale up as confidence grows**: If Studies 1-3 validate the methodology and early re-runs show consistent signal, increase position size closer to November.
- **Re-run periodically**: Monthly re-runs with updated scenario context. Each re-run is a publishable blog post and potential position adjustment.
- **Do not bet more than you can afford to lose.** This is the first calibration run.

### What Constitutes a "Win"

**Clear win:** Extropy correctly predicts both House control AND Alaska Senate winner.

**Strong win:** Correctly predicts one of the two, and the other is close.

**Publishable even if wrong:** If Extropy correctly identifies the MECHANISM — which demographics drove the outcome, which turnout patterns materialized — even if the final call is wrong, the demographic analysis is still compelling.

**Track record win:** If Extropy's prediction published in March 2026 is closer to the actual result than the prediction market price at time of publication — that's a demonstrable edge, regardless of the bet outcome.

**Loss worth analyzing:** If way off, the analysis of WHY is potentially more valuable than a correct prediction. Builds methodology credibility through intellectual honesty.

### Why NOT the Texas Markets (Decision Log)

We evaluated several Texas-specific Kalshi markets and rejected them:

| Market | Link | Price | Why Not |
|--------|------|-------|---------|
| TX GOP Nominee | https://kalshi.com/markets/kxsenatetxr/texas-republican-senate-nominee/kxsenatetxr-26 | Paxton 72% | Too lopsided. Even if Extropy confirms, paying 73¢ to win $1 = no edge. |
| TX GOP Runoff | https://kalshi.com/markets/kxtexasgop1round/texas-gop-primary-won-in-1-round/kxtexasgop1round-26mar03 | No outright win 84% | Also lopsided. Only $4,641 volume = illiquid. |
| TX Senate Matchup | https://kalshi.com/markets/kxtxsencombo/texas-senate-matchup/kxtxsencombo-26nov | Talarico vs Paxton 62% | Multi-way (6 combos) but requires getting both primaries + general right. Three layers of compounding uncertainty. |
| TX Dem Turnout | https://kalshi.com/markets/kxtxsenatedemturnout/voter-turnout-in-the-texas-senate-democratic-primary/kxtxsenatedemturnout-26mar03 | Bracket market | No GOP turnout bracket on Kalshi. Dem turnout bracket exists but Extropy's edge is on the GOP side. |

### Why Alaska Over Other Senate Races

| Race | Kalshi Odds | Why Not |
|------|------------|---------|
| Maine (Collins) | D 65%, R 35% | Too lopsided toward D. Democratic challenger still TBD from primary. Less edge. |
| Iowa (open) | D 34%, R 66% | Lopsided toward R. Both candidates still TBD from primaries. |
| North Carolina (open) | D 76%, R 24% | Very lopsided toward D. Less edge to find. |
| Texas (open) | D 33%, R 67% | Both candidates still TBD from primaries. |
| **Alaska (Sullivan vs Peltola)** | **R 57%, D 43%** | **Closest margin. Both candidates known NOW. RCV adds modeling complexity that gives Extropy edge. Rich demographic dynamics (Native turnout, military crossover, fishing communities).** |

---

## Study 1: ASI Announcement (Run First)

### Scenario Design Principle: Milestones, Not Reactions

**CRITICAL: Extropy scenarios must script WHAT HAPPENS IN THE WORLD, not how people react to it.** Scripting human reactions creates circular simulations — you tell agents "Deloitte fires 50% of staff" then ask "how do you feel about Deloitte firing 50% of staff?" That's not simulation, that's a survey with pre-baked framing.

The right approach: define the **trigger event and its logical/physical consequences**, then let agents produce the human behavioral response. The cascade IS what Extropy simulates. We don't tell agents that people panic, quit jobs, find religion, or start companies. We see IF they do.

**Good milestone:** "ASI designs autonomous manufacturing robots" (factual consequence of ASI existing)
**Bad milestone:** "A wave of resignation letters hits employers" (pre-scripting human behavior)

**Good milestone:** "Complete cures for Alzheimer's, cancer, HIV published with full treatment protocols"
**Bad milestone:** "Mental health systems are overwhelmed, suicide hotlines report record volume"

This principle applies to ALL studies. If you're writing a timeline event and it describes what people DO rather than what HAPPENS, you're contaminating the simulation.

### Outcome Design Principle: No Floats, No Booleans, Concrete Categories

**CRITICAL: Never use float/numeric outcomes (e.g., `anxiety_level: float 0-1`).** An LLM reasoning from a persona cannot meaningfully quantify "how anxious am I on a 0-1 scale." It will hallucinate a number with false precision. The output is meaningless.

**CRITICAL: Avoid simple booleans unless the outcome is a genuinely binary action (e.g., `voted: boolean`).** If the interesting question is HOW or WHY, not just yes/no, use a categorical instead. A boolean like `share_on_social: boolean` flattens nuance that the agent's reasoning actually contains — better as a categorical with options like "posted actively, shared someone else's take, read but didn't post, avoided social media entirely, don't use social media."

Every outcome must be either:

1. **Categorical with concrete, distinguishable options** — each option describes a specific, observable state that an LLM can reason its way into from a persona. The options should be mutually exclusive and cover the realistic range of responses.
2. **Open-ended** — free-text where the agent describes their situation in their own words. This is where emergent behavior surfaces — things we didn't think to put in the categories.

**Good outcome:** `employment_status`: categorical — same job unchanged, same job but role transformed, left/lost job, started new venture, pursuing creative/physical work, not working by choice, not working involuntarily
**Bad outcome:** `career_anxiety: float 0-1`
**Bad outcome:** `will_change_jobs: boolean`

Every study should have **at least 2 open-ended outcomes**. The categoricals capture expected axes of divergence. The open-ended fields are where Extropy surfaces emergent behavior that we didn't predict — that's the whole point of agent-based simulation over traditional surveys.

### Scenario Prompt

```bash
extropy scenario "OpenAI demonstrates artificial superintelligence — a system that exceeds the best human minds in every cognitive domain. It is self-improving. It is real. This is not incremental AI progress. This is a discontinuous leap.

The scenario describes ONLY what ASI produces and what physically changes in the world. Human behavioral responses are NOT scripted — they are what the simulation produces.

MILESTONE EVENTS (factual consequences of ASI, not human reactions):

Month 1 — The Demonstration:
OpenAI demonstrates ASI publicly with live, verifiable results. The system solves the Riemann Hypothesis, P vs NP, and Navier-Stokes existence problems in under an hour. It designs a complete novel mRNA therapy for pancreatic cancer — full molecular pathway, synthesis instructions, predicted efficacy — in 20 minutes. It produces a unified field theory reconciling quantum mechanics and general relativity. It designs a fusion reactor buildable with existing materials and manufacturing techniques. Within 2 weeks, Anthropic, DeepMind, and xAI independently confirm similar capabilities. China's Ministry of Science and Technology announces they achieved ASI months ago. The system is self-improving — each day it is measurably more capable than the day before.

Month 2 — The Knowledge Explosion:
ASI systems at multiple labs produce scientific output at inhuman speed. Complete cures are published for Alzheimer's, ALS, most cancers, malaria, HIV, and diabetes (Type 1 and 2) — not theoretical, but full treatment protocols with synthesis pathways. All remaining Millennium Prize Problems are solved. Materials science breakthroughs include room-temperature superconductors and carbon capture materials 1000x more efficient than current best. A commercial fusion reactor design is published with an estimated 18 months to first operational plant. Climate models are re-run with perfect accuracy and precise interventions mapped for 1.5°C stabilization. All of this is published openly. The bottleneck shifts from knowing what to do to physically building it — manufacturing, logistics, regulatory approval, human coordination.

Month 3 — Public Access:
ASI becomes available to the general public via API and consumer products. Any individual can now access cognitive capability that exceeds the world's best doctor, lawyer, engineer, scientist, financial advisor, therapist, tutor, and strategist — simultaneously, for free or nearly free. A high school dropout in rural Arkansas has the same cognitive resources as a team of Harvard professors. Every profession predicated on information asymmetry or cognitive scarcity is structurally redundant — not because people are fired, but because the reason the job existed is gone. Software that previously took teams of 50 engineers 2 years to build can be built by one person in an afternoon.

Month 4 — The Physical Bottleneck Breaks:
ASI designs autonomous manufacturing systems — robots that can build robots, using locally available materials. The first ASI-designed pharmaceutical facility begins production, fully automated, producing drugs at 1/100th current cost. ASI-designed construction systems are demonstrated: a full residential building assembled in 72 hours by autonomous machines. Energy abundance is on the horizon: the first ASI-designed fusion prototype is under construction with an 8-month timeline to grid power. The constraint is no longer knowledge or manufacturing — it is raw materials, energy (temporarily), and human regulatory and political systems.

Month 5 — Abundance Begins:
First ASI-manufactured drugs reach patients under emergency authorizations — cancer treatments, Alzheimer's reversal therapies. Energy costs begin declining as ASI-optimized solar, wind, and battery systems deploy (fusion still months away). ASI-designed infrastructure projects begin in early-adopter countries (Singapore, UAE, Estonia). Food production breakthroughs: ASI-designed vertical farms producing 100x yield per acre begin construction. The marginal cost of most goods and services is trending toward zero. The economic question is no longer scarcity — it is distribution.

Month 6 — The New World:
The fusion prototype comes online ahead of schedule — functionally unlimited clean energy now has a concrete timeline of months, not decades. An ASI-designed space launch system is demonstrated, dropping cost to orbit by 99%. First Mars habitat modules are in production. Global disease burden is dropping measurably — treatments are reaching patients in developed nations, and deployment to developing nations is the primary coordination challenge. The world is materially, measurably, objectively better in almost every physical dimension." \
  --timeline evolving \
  -o asi-announcement -y
```

### What the Agents Produce (NOT scripted)

The following are emergent — we observe whether and how they happen, we don't tell agents they happen:

- Whether agents engage with ASI or avoid/ignore it
- Whether they quit their jobs, double down, retrain, or freeze
- Whether they find new purpose or lose meaning
- How quickly they adapt vs. how long they resist changing
- How their social networks influence their response
- Whether communities fracture or coalesce
- What political movements emerge
- How consumer and financial behavior changes
- Mental health trajectories
- How different demographics experience the same undeniable reality differently

### Outcomes (tracked monthly, no floats)

- `employment_status`: categorical — same job unchanged, same job but role transformed, left/lost job, started ASI-powered venture, pursuing creative/physical work, full-time education/retraining, not working by choice, not working involuntarily
- `economic_adaptation`: categorical — thriving in new economy, actively transitioning, struggling but managing, in financial crisis, surprisingly better off, no material change yet
- `meaning_and_purpose`: categorical — found new purpose, actively searching, in existential crisis, spiritual/religious turn, focused on relationships/community, focused on creative expression, numb/disengaged, angry/resistant
- `stance_on_ASI`: categorical — full acceleration, cautious optimism, ambivalent/uncertain, fearful but accepting, actively resistant
- `daily_life_change`: categorical — my daily routine is completely unrecognizable, major changes to how I spend my time, some noticeable changes, minor adjustments, basically the same as before
- `primary_concern`: categorical — economic survival, meaning/purpose, safety/control of ASI, inequality/access, family/relationships, excited not concerned, too overwhelmed to articulate
- `monthly_narrative`: open-ended — describe what changed in your life this month and what you're doing about it
- `unexpected_consequence`: open-ended — what is happening around you that nobody predicted or talked about

### What to Check After Simulate

- [ ] Month 1→6 trajectory shows CHANGE per agent (not static across all months)
- [ ] Agents only reference events they've seen so far (no Month 2 agent mentioning Month 3's public access, no Month 4 agent referencing Month 5's abundance)
- [ ] Different demographics diverge: tech workers vs. blue-collar vs. retirees vs. students show distinct trajectories
- [ ] Employment status distribution shifts meaningfully month-over-month
- [ ] `unexpected_consequence` responses surface things not in our outcome categories — this is where Extropy's value shows
- [ ] Religious communities respond distinctly from secular ones
- [ ] Financial margin matters: agents with savings adapt differently than paycheck-to-paycheck agents
- [ ] Reasoning traces are first-person, differentiated, reference specific milestone details
- [ ] No more than 20% of agents give near-identical responses in any given month

---

## Study 2: US-China Taiwan Crisis

### Scenario Prompt

```bash
extropy scenario "China announces a 'special customs enforcement zone' around Taiwan, requiring all commercial vessels entering Taiwanese waters to submit to Chinese Coast Guard inspection. This is not framed as a blockade — China calls it an anti-smuggling and customs enforcement measure. The scenario describes ONLY geopolitical and physical/economic events. Human behavioral responses are NOT scripted — they are what the simulation produces.

MILESTONE EVENTS (geopolitical and economic facts, not human reactions):

Month 1 — The Trigger:
China announces the special customs enforcement zone around Taiwan. All commercial vessels entering Taiwanese waters must submit to Chinese Coast Guard inspection before proceeding. The US condemns the action and deploys a carrier strike group to the Western Pacific. Taiwan's military goes to highest peacetime alert. TSMC — which manufactures approximately 90% of the world's most advanced semiconductors — announces it cannot guarantee shipment timelines. Global semiconductor futures spike 300% overnight.

Month 2 — The Squeeze Tightens:
China begins turning away commercial ships that refuse inspection. Taiwan's semiconductor exports drop 70%. The chip shortage hits immediately: auto manufacturers halt production lines across the US, Europe, and Japan. Data center expansion freezes globally. Medical device companies warn of critical shortages within 60 days — pacemakers, insulin pumps, and imaging equipment all rely on Taiwanese chips. Apple announces iPhone production is delayed indefinitely. Nvidia, AMD, and Qualcomm stocks collapse. Oil prices spike to $140 per barrel on war risk premium. China maintains this is an internal matter and warns foreign governments against interference.

Month 3 — Economic Contagion:
The US tech sector enters a bear market — Nasdaq down 35% from pre-crisis levels. Consumer electronics prices surge 40-60% on remaining inventory. Auto dealers have near-empty lots as no new cars are being manufactured. The chip shortage cascades into unexpected areas: new credit card production halts, smart grid infrastructure upgrades stop, GPS satellite replacements are delayed. US defense officials quietly brief Congress that military hardware maintenance is affected — F-35 spare parts, missile guidance systems, and satellite communications all depend on Taiwanese fabrication. Gas prices hit $7 per gallon national average.

Month 4 — Escalation:
The US imposes comprehensive sanctions on Chinese tech firms and financial institutions. China retaliates by banning rare earth mineral exports to the US and allies. EV production halts globally because rare earths are in every electric motor. Wind turbine manufacturing stops. A US Navy destroyer and a Chinese Coast Guard vessel collide in the Taiwan Strait — no casualties, but hull damage on both sides. Video goes viral globally. Both nations accuse the other of provocation. The UN Security Council emergency session ends with Chinese and Russian vetoes blocking any resolution. Japan and South Korea announce emergency semiconductor stockpile measures.

Month 5 — The Grind:
No military conflict materializes beyond the naval collision, but economic damage compounds. US unemployment rises as auto manufacturing, electronics retail, and related manufacturing sectors are hit hardest. The chip shortage creates cascading infrastructure failures: traffic management systems cannot be repaired, hospital equipment maintenance backlogs grow, defense readiness degrades. Several US states report delays in issuing new driver's licenses and ID cards because they are chip-dependent. Intel and Samsung announce emergency fab expansion in the US and South Korea respectively, but new fabs take 3-5 years to build. TSMC begins constructing an underground backup facility in Arizona but it will not be operational for 2+ years.

Month 6 — Fragile Equilibrium:
Backchannel diplomacy produces a partial de-escalation: China loosens the inspection regime to allow pre-approved shipping lanes, restoring approximately 40% of Taiwan's export capacity. The US quietly agrees to scale back carrier presence. But the damage is structural — companies are permanently diversifying supply chains away from Taiwan dependence. The chip shortage eases slightly but prices remain 2-3x pre-crisis levels. Rare earth alternatives are being researched but are years away. The global economic order that existed before Month 1 is not coming back." \
  --timeline evolving \
  -o taiwan-crisis -y
```

### What the Agents Produce (NOT scripted)

- Whether they panic-buy electronics, hoard essentials, or change nothing
- Whether they support military intervention, diplomacy, isolation, or don't care
- How their employment situation changes (or doesn't)
- Whether draft anxiety is real or overblown in their community
- How their spending, saving, and investment behavior shifts
- Whether they blame China, the US government, corporations, or globalization
- How their trust in institutions and global systems changes
- What local and community-level effects they observe
- Whether the crisis politicizes them or makes them tune out

### Outcomes (tracked monthly, no floats)

- `personal_economic_impact`: categorical — directly affected (job loss/income drop), moderately affected (higher prices hurting budget), mildly affected (some inconvenience), not noticeably affected, actually benefiting (defense/reshoring sector)
- `consumer_behavior`: categorical — hoarding electronics/essentials, delaying major purchases, switching to cheaper alternatives, panic buying, reducing all discretionary spending, no meaningful change
- `political_stance_on_crisis`: categorical — support military response to China, support aggressive sanctions only, support diplomatic engagement, US should stay out entirely, don't have a clear position
- `military_concern`: categorical — personally worried about draft/service (eligible age), worried for family member of eligible age, concerned about military conflict generally, not concerned about military dimension, supportive of military action
- `trust_in_global_systems`: categorical — lost significant trust in global supply chains/trade, somewhat less trusting, unchanged, was already distrustful, don't think about it
- `employment_status`: categorical — same job stable, same job but reduced hours/pay, lost job due to crisis, found new job in crisis-benefiting sector, no change but worried, retired/not in workforce
- `monthly_narrative`: open-ended — describe how this crisis is affecting your daily life, your community, and your decisions this month
- `unexpected_consequence`: open-ended — what is happening around you that nobody predicted or talked about

### What to Check After Simulate

- [ ] Month 1→6 trajectory shows CHANGE per agent (not static across all months)
- [ ] Agents only reference events they've seen so far (no Month 2 agent mentioning sanctions if that's Month 4)
- [ ] Geographic divergence: agents near auto plants, military bases, tech hubs, and rural areas show distinct trajectories
- [ ] Employment sector matters: auto workers hit in Month 2, defense workers affected differently in Month 4-5, tech workers throughout
- [ ] The naval collision (Month 4) is a pivotal moment — agents' response to it should be shaped by their prior 3 months of experience
- [ ] Draft/military anxiety correlates with age and family situation, not uniform
- [ ] Economic impact perception differs between paycheck-to-paycheck agents and financially comfortable ones
- [ ] `unexpected_consequence` surfaces cascading effects we didn't anticipate
- [ ] No more than 20% of agents give near-identical responses in any given month

---

## Study 3: Bitcoin Hits $1M

### Scenario Prompt

```bash
extropy scenario "Bitcoin crosses $1,000,000 per coin for the first time. The scenario describes ONLY market events, technological facts, and regulatory actions. Human behavioral responses are NOT scripted — they are what the simulation produces.

CRITICAL CONTEXT — BITCOIN'S FIXED SUPPLY: Only 21 million Bitcoin will ever exist. This is a hard protocol cap that cannot be changed. Approximately 19.7 million have been mined, and an estimated 3-4 million are permanently lost in inaccessible wallets. The effective circulating supply is roughly 16 million coins. Ownership concentration is extreme — approximately 2% of wallets hold 95% of all Bitcoin. An estimated 50 million Americans hold some cryptocurrency, but the vast majority hold fractions: the median American crypto holder has roughly $5,000-$20,000 worth. At $1M per BTC, someone who bought 1 full BTC at $30,000 in 2021 is a millionaire. Someone who bought 0.1 BTC has $100,000. Someone who bought $500 worth in 2021 now has roughly $16,000 — a nice windfall, not life-changing.

MILESTONE EVENTS (market and regulatory facts, not human reactions):

Week 1 — The Milestone:
Bitcoin crosses $1,000,000 per coin. The price reached this level after a 3-week climb from $95,000 driven by Abu Dhabi's sovereign wealth fund allocating 5% of reserves to Bitcoin, the Federal Reserve cutting rates to near-zero citing recession fears, and self-reinforcing price momentum as fixed supply meets surging demand. At $1M per BTC, total market cap is $21 trillion, but actual tradeable supply represents about $16 trillion. Coinbase and Robinhood experience platform outages from traffic volume.

Week 2 — Scarcity Meets FOMO:
Crypto exchanges report 15 million new account registrations in 7 days. Bitcoin's fixed supply means every new buyer competes for the same limited pool of coins. New buyers are purchasing tiny fractions at historically high prices — $10,000 buys 0.01 BTC, which someone else bought for $300 in 2020. Meme coins and altcoins (which do NOT have Bitcoin's supply cap) surge 50-500% as buyers priced out of whole Bitcoins look for alternatives. Credit card companies report a spike in cash advances. Employer payroll companies report a surge in requests to receive salary in Bitcoin.

Week 3 — The First Crack:
A 23-year-old in Ohio takes his own life after having sold all his Bitcoin at $50,000 the previous year. At today's price, his former holdings would have been worth $2.4 million. His story is covered by every major news outlet. Bitcoin drops to $840,000 in 48 hours, then recovers to $910,000. Because supply is fixed and concentrated, the sell-off is driven by a small number of large holders (whales) taking profits. The single-day drop represents $300 billion in market cap evaporation, but relatively few coins actually changed hands.

Week 4 — The Scam Explosion:
The FTC reports a 4,000% increase in crypto fraud complaints in 7 days. Fake Bitcoin giveaway sites have stolen an estimated $200 million, disproportionately from Americans over 60 who are new to crypto. The scams exploit the scarcity narrative — 'send 0.01 BTC and receive 0.1 BTC back' sounds plausible to newcomers who do not understand that no one is giving away a permanently scarce asset. A deepfake video of Elon Musk promoting a fraudulent exchange spreads across Facebook and YouTube, viewed 80 million times before removal. A deepfake of the Treasury Secretary endorsing a fake US Digital Dollar exchange follows. The FBI issues a public warning.

Week 5-6 — The Liquidity Problem:
The IRS announces enhanced monitoring of all crypto-to-fiat conversions exceeding $10,000. Banks begin flagging and in some cases freezing accounts receiving large crypto-to-fiat transfers, citing anti-money-laundering requirements. Processing times for large conversions extend to 2-3 weeks. Bitcoin holders are wealthy on paper but converting to spendable dollars is slow, uncertain, and increasingly scrutinized. The bid-ask spread on large sell orders widens — someone trying to sell 10 BTC ($10M) may face 5-8% slippage because the order book is thin at these price levels. Bitcoin price fluctuates between $880,000-$950,000. For most holders sitting on $5K-$50K in fractional Bitcoin, the fiat off-ramp friction means the wealth feels abstract.

Week 7-8 — Regulatory Response:
The SEC announces a comprehensive crypto regulation framework: mandatory exchange registration, custody requirements, leverage limits, and enhanced KYC/AML. India imposes a 50% capital gains tax on crypto. The EU fast-tracks MiCA enforcement with emergency provisions. Nigeria and Indonesia ban crypto-to-fiat conversion entirely. China reaffirms its existing ban. Bitcoin drops to $720,000 on regulatory fear. Altcoins — which lack Bitcoin's scarcity narrative and institutional backing — drop 40-60%. Some smaller altcoins go to zero.

Week 9-10 — The Stablecoin Crisis:
A top-5 stablecoin (not Tether, not USDC) de-pegs after a bank run on its reserves. $30 billion in value evaporates in 72 hours. Contagion spreads — three DeFi lending protocols halt withdrawals. Several smaller exchanges become insolvent. The stablecoin's auditor reveals reserves were only 41% backed by actual dollar equivalents; the rest was illiquid commercial paper and crypto collateral that lost value simultaneously. Bitcoin drops to $580,000. Holders trying to flee to safety face a dilemma: hold volatile BTC, try to exit to fiat through jammed off-ramps and frozen bank accounts, or move to remaining stablecoins that may also be at risk. The fixed supply that drove the rally now works in reverse — there are few buyers at these prices, and sellers compete for limited liquidity.

Week 11-12 — The New Equilibrium:
Bitcoin stabilizes around $500,000-$650,000 — still up 500-600% from pre-spike levels but the $1M era feels distant. The fixed supply ensures a price floor far above historical levels — too many institutional holders (sovereign wealth funds, pension funds, corporate treasuries) will buy any dip below $400K. Congressional hearings produce bipartisan agreement on a regulatory framework expected to become law within 6 months. Major banks (JPMorgan, Goldman Sachs, Morgan Stanley) announce compliant crypto products — regulated Bitcoin funds, insured custody, structured products. The crypto market bifurcates permanently: Bitcoin as regulated, institutional-grade digital gold on one side; the wild DeFi/altcoin ecosystem, now significantly smaller and more cautious, on the other." \
  --timeline evolving \
  -o btc-one-million -y
```

### What the Agents Produce (NOT scripted)

- Whether they buy, sell, hold, or ignore crypto entirely
- Whether they quit jobs, make major purchases, or change nothing
- Whether they get scammed, worry about scams, or don't encounter them
- How their financial behavior changes (saving, spending, investing patterns)
- Whether they view crypto as the future, a bubble, a casino, or irrelevant
- How they feel about people who got rich (happy for them, resentful, indifferent)
- Whether the regulatory response feels protective or oppressive
- What happens in their communities and social circles around money and wealth
- Whether inequality narratives sharpen — especially given that crypto wealth is concentrated among early adopters, not correlated with traditional merit/work
- Whether the paper-wealth-vs-actual-liquidity gap creates frustration or confusion

### Outcomes (tracked weekly, no floats)

- `crypto_position`: categorical — bought more, holding existing, sold some, sold all, bought for first time, considering buying, not interested, actively shorting/betting against
- `financial_behavior_change`: categorical — no change, made or planning major purchase, quit or considering quitting job, increased savings rate, reduced all discretionary spending, took on debt to invest, started actively trading, withdrew from all investments
- `regulatory_stance`: categorical — needs heavy regulation now, light-touch regulation is fine, regulation will kill innovation, government should ban crypto entirely, don't understand enough to have an opinion
- `scam_exposure`: categorical — personally targeted by scam, know someone who was targeted, aware of scams but not personally encountered, haven't heard about scams, fell for a scam
- `wealth_inequality_feeling`: categorical — happy for people who got rich, resentful that I missed out, angry at the unfairness, indifferent, inspired to find my own opportunity, think crypto wealth is fake/temporary
- `life_disruption`: categorical — my life is dramatically different because of this, some meaningful changes, minor impact, basically unaffected, this doesn't touch my world at all
- `weekly_narrative`: open-ended — what happened in your financial life and your community this week related to Bitcoin and crypto
- `unexpected_consequence`: open-ended — what is happening around you that nobody predicted or talked about

### What to Check After Simulate

- [ ] Week 1→12 trajectory shows a full arc per agent, not static responses
- [ ] Agents only reference events they've seen so far (no Week 3 agent mentioning the stablecoin crisis from Week 9)
- [ ] Crypto holders vs non-holders diverge sharply — this is a scenario where prior financial position dramatically shapes experience
- [ ] Age segmentation matters: younger agents with crypto exposure respond differently than older agents without
- [ ] Financial margin matters: agents who are paycheck-to-paycheck respond differently than agents with savings whether or not they hold crypto
- [ ] The scarcity dynamic shows up in reasoning: agents should understand they're buying fractions, not whole coins, and that selling is harder than buying
- [ ] The Week 9-10 stablecoin crisis affects agents differently depending on whether they hold stablecoins, DeFi positions, or just Bitcoin
- [ ] `unexpected_consequence` surfaces cascading effects — community-level dynamics, relationship impacts, local economic effects
- [ ] No more than 20% of agents give near-identical responses in any given week
- [ ] Agents who don't hold crypto aren't invisible — their experience of watching from the sidelines is a valid and important part of the simulation

---

## Pipeline Validation: Every Step Before Simulation

**CRITICAL: Do NOT skip validation between pipeline stages.** Each stage depends on the quality of the previous one. A bad spec produces bad personas. Bad personas produce unrealistic agents. Unrealistic agents produce meaningless simulation results. Validate aggressively at every step and STOP the pipeline if something is wrong. Fixing upstream is always cheaper than re-running everything.

The pipeline stages are: **Spec → Scenario → Persona → Sample → Network → Simulate**

Every stage has a validation gate. If the gate fails, fix and re-run that stage before proceeding.

### Gate 1: After `extropy spec` — Is the population definition realistic?

**What to do:**
1. Run `extropy validate` on the generated spec YAML file (if available)
2. Open the spec YAML file directly and read it end-to-end
3. Have an LLM agent review the spec file and answer the checklist questions below

**What to check:**
- [ ] **All attribute dimensions present**: Every dimension from the spec prompt is represented in the YAML. For our enriched national spec, that's 15+ dimensions including the 5 electoral ones (2024 vote, district competitiveness, midterm turnout propensity, issue salience, approval direction). If any are missing, the spec is incomplete.
- [ ] **Distribution realism**: Income brackets, age ranges, geographic splits roughly match US Census data. Spot-check: ~20% rural, ~30% bachelor's degree+, median income ~$75K, ~60% white, ~13% Black, ~19% Hispanic. If the spec has 50% college graduates or 5% rural, it's wrong.
- [ ] **Correlation structure encoded**: The spec should explicitly encode or hint at correlations between attributes. Higher education should correlate with higher income and urban residence. Federal employees should cluster in DC metro and military towns. Religious engagement should correlate with rural geography and conservative politics. If every attribute is independently distributed, the population will be unrealistic — you'll get rural investment bankers and urban evangelical farmers in unrealistic proportions.
- [ ] **No degenerate categories**: Every attribute has at least 3-4 meaningful values. No catch-all "Other" bucket exceeding 20%. No single value capturing more than 80% of the population for any attribute.
- [ ] **Behavioral dimensions are real, not filler**: Trust, media ecosystem, tech posture, financial margin — these must have concrete, distinguishable levels, not vague labels. "Medium trust" is useless. "Trusts local institutions but distrusts federal government" is useful.
- [ ] **Electoral dimensions present and realistic**: 2024 vote distribution should roughly match actual results (~49% Trump, ~48% Harris, ~3% third party/didn't vote among voters, plus a chunk of non-voters). Congressional district competitiveness should reflect real 2026 map. Midterm turnout propensity should skew toward "sometimes" and "rarely" — most Americans don't vote in midterms.
- [ ] **No hallucinated statistics**: If the spec cites specific percentages, spot-check 3-4 against known sources (Census, Pew Research, Gallup). LLMs confidently fabricate demographic statistics.

**If validation fails:** Re-run `extropy spec` with a more specific prompt targeting the gap. Do NOT proceed to persona generation with a weak spec.

### Gate 2: After `extropy scenario` — Is the scenario properly structured?

**What to do:**
1. Run `extropy validate` on the generated scenario file (if available)
2. Open the scenario file and read the full text
3. Have an LLM agent review against the Milestones Not Reactions principle

**What to check:**
- [ ] **Milestones only, no reactions**: Read every timeline event. Does it describe what HAPPENS in the world, or what PEOPLE DO? If any event describes human behavioral responses (e.g., "panic buying ensues," "protests break out," "workers quit en masse"), it's contaminating the simulation. Remove it. The agents produce the behavior.
- [ ] **Correct timeline mode**: Evolving scenarios should use `--timeline evolving`. Non-evolving scenarios (Midterms) should use `--timeline static`.
- [ ] **Horizon matches intent**: Verify `timeline` entries and `simulation.max_timesteps` in `scenario.v1.yaml` (ASI = 6 months, Taiwan = 6 months, BTC = 12 weeks, Midterms = 1 decision timestep).
- [ ] **Events are chronologically coherent**: Month 3 events should not reference or depend on Month 5 events. Each month's events should be logical consequences of the preceding months.
- [ ] **Neutral framing**: The scenario presents facts, not editorialized narratives. It doesn't tell agents how to feel about the events. "TSMC exports drop 70%" is neutral. "The devastating collapse of TSMC exports" is editorialized.
- [ ] **Sufficient detail for agents to reason**: Each milestone should have enough concrete specifics (numbers, names, timelines) that an agent can reason about how it affects their specific life. "The economy gets worse" is useless. "Auto manufacturers halt production lines, consumer electronics prices surge 40-60%" gives agents something concrete to react to based on their own situation.
- [ ] **Outcomes are all categorical or open-ended**: No floats. No lazy booleans. At least 2 open-ended outcomes. Each categorical option should be concrete and distinguishable — an LLM reasoning from a persona should be able to clearly select one option over another.

**If validation fails:** Edit the scenario file directly to fix issues. Re-run `extropy validate` if available.

### Gate 3: After `extropy persona` — Are the persona templates diverse and realistic?

**What to do:**
1. Read the generated persona templates file
2. Have an LLM agent review the full set of templates

**What to check:**
- [ ] **Full behavioral range covered**: Templates should span the realistic range of responses to the scenario. For ASI: templates should include tech optimists AND luddites, young flexible workers AND older specialists, financially secure AND paycheck-to-paycheck, urban tech workers AND rural tradespeople. If every template is a variation of "concerned professional," the simulation will converge.
- [ ] **No template dominates**: No single template should represent more than 15% of the population. If one template is too broad or generic, it will flatten diversity.
- [ ] **Boring archetypes included**: Not every persona should be an edge case or interesting character. Include the disengaged middle — people who don't follow the news closely, don't have strong opinions, and whose primary concern is paying rent. These "boring" agents are the majority of the real population and their behavior (or inaction) matters.
- [ ] **Internal consistency**: Each template's attributes should make sense together. A 22-year-old rural evangelical with a PhD in computer science and $500K in savings is not impossible but should be extremely rare, not a common template.
- [ ] **Scenario-relevant differentiation**: Templates should differ on the attributes that MATTER for this specific scenario. For ASI, the key differentiators are employment sector, education level, financial margin, tech adoption, and meaning/identity sources. For Taiwan, they're employment sector (especially auto/manufacturing/defense), geography, political engagement, and financial margin. If templates differ only on demographics and not on these scenario-relevant dimensions, the simulation won't produce meaningful divergence.

**If validation fails:** Re-run `extropy persona` with more specific guidance on missing archetypes or range gaps.

### Gate 4: After `extropy sample` — Is the sampled population realistic?

**What to do:**
1. Query the study database directly: `sqlite3 study.db "SELECT COUNT(*) FROM agents"` to confirm agent count
2. Pull demographic distributions using `extropy query agents -s <scenario> --to agents.jsonl` (or SQL against `agents.attrs_json`) to inspect attribute values and frequencies
3. Pull 30 random agents and have an LLM review their full profiles for internal consistency
4. Cross-reference key distributions against Census/Pew/Gallup benchmarks

**What to check:**
- [ ] **Correct agent count**: 5,000 for all studies. If the count is wrong, something failed in sampling.
- [ ] **Demographic distribution matches target**: Age should roughly mirror US adult population (not all 25-35). Race/ethnicity should approximate Census. Income should have a realistic spread with appropriate median. Geography should span all regions with ~20% rural. If any distribution is wildly off, the sample is biased.
- [ ] **No attribute has >80% concentration**: If 85% of agents are "moderate trust in institutions" or 90% are "private sector employment," the sample lacks diversity and the simulation will converge.
- [ ] **Attribute correlations are realistic**: Pull cross-tabulations. Do rural agents actually skew toward lower digital literacy and higher religious engagement? Do high-income agents cluster in urban areas? Do federal employees cluster in DC metro and military towns? If correlations are flat (every attribute independently distributed), the population is unrealistic.
- [ ] **30-agent spot check**: Pull 30 random complete agent profiles. Read each one. Does each agent feel like a real, internally consistent person? Or are there nonsensical combinations (e.g., a 19-year-old retiree, a rural Manhattan resident, a crypto day-trader with no investment literacy)? A few oddballs are fine. More than 2-3 out of 30 is a problem.
- [ ] **Electoral attributes realistic (for Study 4)**: 2024 vote distribution should roughly match actual election results. Congressional district mapping should include competitive districts. Midterm turnout propensity should have a healthy spread, not all "always votes."

**If validation fails:** Re-run `extropy sample` with a different seed. If the problem persists, the issue is upstream in the spec or persona templates.

### Gate 5: After `extropy network` — Is the social network structure realistic?

**What to do:**
1. Query network statistics from study.db: total edges, average degree, min/max degree, clustering coefficient if available
2. Pull 10-15 random agents and examine their connections — who are they connected to and why?
3. Have an LLM agent review the network topology description and assess realism

**What to check:**
- [ ] **Network has edges**: `SELECT COUNT(*) FROM network_edges` should return a non-zero number (or use `extropy query edges -s <scenario>`). An empty network means agents have no social influence on each other, defeating the purpose of network cascade modeling.
- [ ] **Average degree is reasonable (4-8)**: Real social influence networks have modest connectivity. Average degree below 3 means the network is too sparse for cascades. Average degree above 15 means it's too dense and everyone influences everyone (unrealistic).
- [ ] **Not fully connected or fully disconnected**: The network should have clusters and structure, not be a single giant clique or a set of isolated nodes. Check that there are multiple connected components or at least visible clustering.
- [ ] **Connections make sense**: Pull 10 agents and their connections. Are connected agents plausibly in each other's social circles? Connections should cluster by geography, employment sector, age cohort, religious community, and political alignment — because that's how real social networks work. If a rural evangelical farmer in Texas is connected to an urban tech worker in SF with no shared attributes, the network is random noise.
- [ ] **Homophily exists but isn't total**: Connected agents should be more similar than random pairs, but not identical. Real networks have some cross-cutting ties (the conservative who has a liberal college friend, the rural person with an urban sibling). If every agent is only connected to agents exactly like themselves, the network will produce echo chambers that are too extreme. If connections are purely random, social influence dynamics will be unrealistic.
- [ ] **Degree distribution is not uniform**: Some agents should have more connections than others. Opinion leaders, community organizers, and social media influencers should have higher degree. Isolated individuals should have lower degree. If every agent has exactly 6 connections, the network is artificially regular.
- [ ] **LLM REVIEW MANDATORY**: Have an LLM agent describe the overall network topology in plain language and assess whether it would support realistic social influence dynamics for the specific scenario being run.

**If validation fails:** Re-run `extropy network` with a different seed or adjusted parameters. If the network structure is fundamentally wrong (e.g., no clustering, random connections), the issue may be in how the network builder uses agent attributes.

### Gate 6: After `extropy simulate` — Are the results meaningful?

**What to do:**
1. Run `extropy simulate -s <scenario>` (for evolving studies, use `--early-convergence off` to guarantee full timeline execution)
2. Run `extropy results -s <scenario>` to see aggregate outcome distributions
3. Run `extropy results -s <scenario> segment <attribute>` for key demographic segments
4. Export full elaborations CSV and spot-check 20 agents' reasoning traces
5. Compare distributions across timesteps for evolving scenarios

**What to check:**
- [ ] **Results exist and are non-empty**: Basic sanity — the simulation produced output for all agents across all timesteps.
- [ ] **For evolving scenarios: outcomes CHANGE across timesteps**: If 60% of agents say "no change" in Month 1 and 60% still say "no change" in Month 6, the simulation isn't working. Events should cause measurable shifts in outcome distributions over time.
- [ ] **Agents reference only events they've seen**: A Month 2 agent should NOT mention Month 4 events. If they do, the evolving simulation is feeding future events into past timesteps. This is a critical bug.
- [ ] **Reasoning traces are first-person and differentiated**: Read 20 random agents' open-ended responses. Each should read like a distinct person narrating their own experience. If they read like third-person summaries or generic commentary, the persona grounding isn't working.
- [ ] **No more than 20% near-identical responses**: In any given timestep, if more than 20% of agents produce essentially the same response (same categorical selections, similar reasoning), central tendency bias is dominating. The simulation isn't capturing real population diversity.
- [ ] **Demographic segmentation produces expected directional splits**: When you segment by relevant attributes, you should see meaningful differences. For ASI: tech workers should respond differently than retirees. For Taiwan: agents near auto plants should be more economically affected than agents in non-manufacturing regions. If every segment looks the same, the simulation is ignoring agent attributes.
- [ ] **`unexpected_consequence` surfaces novel insights**: The open-ended fields should contain responses that go beyond what the categorical outcomes capture. If every open-ended response just restates the categorical selection, the open-ended field isn't adding value.
- [ ] **Intent-to-action sanity check**: If 80% of agents say they've "completely changed their daily routine" by Month 2, that's likely LLM enthusiasm bias. Real populations are slower to change. Apply skepticism to extreme response distributions.

---

## Execution Timeline

| Day | Activity |
|-----|----------|
| **Day 1** | Generate Spec A (National) and Spec B (Alaska). Validate both thoroughly. |
| **Day 1** | Start Study 1 pipeline (ASI): persona → sample → network → simulate |
| **Day 2** | Review Study 1 results. Start Studies 2 (Taiwan Crisis) and 3 (BTC) in parallel (shared Spec A population). |
| **Day 3** | Review Studies 2 and 3. Start Study 4a (House Control) — same Spec A population, new scenario. Start Study 4b (Alaska Senate) — Spec B pipeline: persona → sample → network → simulate. |
| **Day 4** | Review Study 4a and 4b results. Translate to Kalshi predictions for House control and Alaska Senate. Begin drafting blog posts. |
| **Day 5** | Publish Study 1 (ASI) blog post — most viral, leads with strongest showcase. |
| **Day 6-7** | Publish remaining showcase blogs (Taiwan Crisis, BTC). |
| **Day 7** | Publish Midterms prediction blog (House + Alaska) with Kalshi positions. **Place initial bets.** |
| **Monthly** | Re-run Studies 4a and 4b with updated scenario context. Publish updated predictions. Adjust Kalshi positions. |
| **Nov 3** | Midterm elections. Results come in. |
| **Nov 4** | Publish results comparison — win or lose. Honesty > hiding misses. |

---

## Key Risks

**Intent-to-action gap:** LLM agents overstate intent. Apply 10-15% discount on all behavioral predictions. For midterm turnout specifically: lower salience than presidential years, forgetting to vote, not researching downballot races. For Alaska specifically: rural bush community turnout faces real logistical barriers (distance, weather, transportation) that LLM agents won't naturally account for.

**LLM systematic bias:** o5-mini (and all current LLMs) exhibit social desirability bias and reduced variance. Distributions will be artificially narrow and systematically shifted. This means Extropy may underestimate tail outcomes and overstate moderate/centrist responses.

**Knowledge cutoff:** o5-mini may not know about events after its training cutoff. The research phase must inject current political context (2025 election results, economic data, recent developments) into the scenario. For Alaska, candidate-specific context (Sullivan's voting record, Peltola's campaign messaging) must be explicitly provided in the scenario prompt since it may not be in training data.

**9-month drift:** A lot changes between February and November. The initial prediction is a baseline. Monthly re-runs are essential. The bet is not "predict November from February" — it's "build a model that gets more accurate over time and bet when the edge is clearest."

**Alaska sample size:** 1,000 agents for Alaska is adequate for aggregate prediction but thin for sub-demographic analysis. Native communities (~160 agents), military families (~100 agents), and fishing communities (~80 agents) are small sub-groups. Directional patterns will be visible but precise margins within these groups should be treated with caution.

**Ranked-choice modeling:** Extropy has not previously modeled ranked-choice voting. The second-choice outcome is novel and may produce less reliable results than first-choice voting intent. Validate RCV dynamics carefully against known patterns from Alaska's 2022 special election.

**Cross-study reinforcement:** Study 2 (Taiwan Crisis) and Studies 4a/4b (Midterms) would be linked if a real Taiwan crisis occurred — it would reshape the electorate. Keep the studies independent unless explicitly combining them.
