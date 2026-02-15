# Capabilities and Examples

Use this file to map user intent to what Extropy can model and how to execute it.

## 1) Core Capability Classes

1. Population synthesis
- Build statistically grounded synthetic populations from natural-language scope.
- Add scenario-specific behavioral/psychographic attributes.

2. Social graph simulation
- Generate network structures and influence pathways.
- Model diffusion and exposure propagation.

3. Scenario compilation
- Translate events/policies/product changes into executable exposure + outcome logic.

4. Agent reasoning dynamics
- Simulate iterative belief updates, memory effects, and classification outcomes.

5. Outcome analytics
- Produce timeline dynamics, final distributions, segment deltas, and agent-level traces.

6. Experiment operations
- Support estimation, batching, sweeps, versioning, triage, and reporting.

## 2) Decision Domains

- Public policy and governance
- Market/pricing strategy
- Product launch and diffusion
- Crisis and reputation response
- Messaging and political strategy
- Community and urban planning
- Healthcare behavior change
- B2B and enterprise transformation

## 3) Advanced Study Patterns

1. Counterfactual suites
- Baseline vs alternatives under fixed population/config.

2. Sensitivity analysis
- Sweep one axis at a time around baseline assumptions.

3. Confidence sweeps
- Multi-seed reruns for stability/variance analysis.

4. Segment stress tests
- Identify cohorts with fragile or highly parameter-sensitive outcomes.

5. Mechanism-first analysis
- Explain outcomes from exposure paths and agent state traces.

## 4) Practical Boundaries

- Best for social-behavioral dynamics, not physics/logistics optimization.
- Multi-event cascades are better modeled as staged runs.
- Outputs are simulation-informed forecasts, not guaranteed outcomes.

## 5) Trigger Phrases

Use this skill when users ask things like:
- "simulate how people will respond to..."
- "what happens if we raise price by..."
- "which segments will churn/adopt/protest"
- "test these message variants before launch"
- "run scenario analysis with uncertainty"
- "why did this segment flip in simulation"

## 6) Example Requests (Illustrative Only)

These are examples, not defaults.

1. Policy: congestion pricing alternatives
- Ask: estimate compliance/backlash across income and commute-access segments.
- Shape: baseline + alternatives + equity cuts.

2. Public health messaging
- Ask: find least responsive groups and best message frame.
- Shape: same population, multiple message scenarios, compare adoption/sentiment.

3. SaaS pricing
- Ask: estimate churn/downgrade/stay under +10/+20/+30 price shifts.
- Shape: counterfactual suite + revenue-risk tradeoff.

4. Product launch
- Ask: predict enable/disable behavior for default-on AI feature.
- Shape: adoption outcomes + trust/privacy sensitivity sweep.

5. Crisis response
- Ask: compare apology-only vs refund vs policy-change response.
- Shape: trust recovery and negative WOM dynamics by segment.

6. Political messaging
- Ask: compare message resonance/backlash by ideology/economic exposure.
- Shape: frame variants + propagation differences.

7. Community planning
- Ask: simulate support/neutral/oppose response to development proposal.
- Shape: concern taxonomy + coalition risk.

8. Healthcare adoption
- Ask: model clinician switching under reimbursement changes.
- Shape: policy variants + adoption friction analysis.

9. Enterprise change
- Ask: simulate compliance/disengagement/attrition intent under policy shift.
- Shape: role/commute/trust segment breakdown.

10. Deep triage
- Ask: debug flat exposure curve or unstable seed outcomes.
- Shape: evidence-led root cause, minimal fix, rerun command.

## 7) Quick Execution Templates

1. Baseline + sensitivity
- 1 baseline
- 3 variants
- 3 seeds each
- 2 to 3 key segment cuts

2. Message shootout
- 1 population
- 3 to 5 message scenarios
- fixed config + seeds
- rank by primary KPI + stability

3. Decision brief inputs
- decision objective
- top findings
- segment impacts
- confidence/stability
- recommendation + caveats

## 8) Capability to File Map

- Can Extropy model this? -> this file
- How to run it? -> `OPERATIONS.md`
- How to validate/fix/escalate? -> `QUALITY_TRIAGE_ESCALATION.md`
- How to analyze outcomes? -> `ANALYSIS_PLAYBOOK.md`
- How to write decision report? -> `EXPERIMENT_REPORT_TEMPLATE.md`
