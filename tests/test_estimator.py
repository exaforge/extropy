"""Tests for simulation cost estimator and pricing modules."""

from datetime import datetime

import pytest

from extropy.core.pricing import (
    ModelPricing,
    get_pricing,
    resolve_default_model,
)
from extropy.simulation.estimator import (
    estimate_simulation_cost,
    _compute_avg_degree,
    _evaluate_rule_reach,
    _estimate_token_counts,
)
from extropy.core.models.population import (
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    AttributeSpec,
    SamplingConfig,
    GroundingInfo,
    NormalDistribution,
    CategoricalDistribution,
)
from extropy.core.models.scenario import (
    ScenarioSpec,
    ScenarioMeta,
    Event,
    EventType,
    SeedExposure,
    ExposureChannel,
    ExposureRule,
    InteractionConfig,
    InteractionType,
    SpreadConfig,
    OutcomeConfig,
    OutcomeDefinition,
    OutcomeType,
    SimulationConfig,
)


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def small_agents() -> list[dict]:
    """10 agents with age and role attributes."""
    agents = []
    for i in range(10):
        agents.append(
            {
                "_id": f"agent_{i:03d}",
                "age": 30 + i * 4,
                "role": "senior" if i >= 7 else "junior",
            }
        )
    return agents


@pytest.fixture
def small_network(small_agents) -> dict:
    """Simple network with known degree structure."""
    # Ring network: each agent connected to next, plus a few extra edges
    edges = []
    ids = [a["_id"] for a in small_agents]
    for i in range(len(ids)):
        edges.append({"source": ids[i], "target": ids[(i + 1) % len(ids)]})
    # Add a few extra edges
    edges.append({"source": ids[0], "target": ids[5]})
    edges.append({"source": ids[2], "target": ids[7]})
    return {"edges": edges}


@pytest.fixture
def small_pop_spec() -> PopulationSpec:
    """Minimal population spec with 2 attributes."""
    return PopulationSpec(
        meta=SpecMeta(
            description="Test population",
            size=10,
            geography="Test",
            created_at=datetime(2024, 1, 1),
            version="1.0",
        ),
        grounding=GroundingSummary(
            overall="low",
            sources_count=0,
            strong_count=0,
            medium_count=0,
            low_count=2,
            sources=[],
        ),
        attributes=[
            AttributeSpec(
                name="age",
                type="int",
                category="universal",
                description="Age",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal",
                        mean=45,
                        std=10,
                        min=25,
                        max=70,
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
            AttributeSpec(
                name="role",
                type="categorical",
                category="population_specific",
                description="Role",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["junior", "senior"],
                        weights=[0.7, 0.3],
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
        ],
        sampling_order=["age", "role"],
    )


@pytest.fixture
def small_scenario() -> ScenarioSpec:
    """Minimal scenario spec for estimation tests."""
    return ScenarioSpec(
        meta=ScenarioMeta(
            name="test_scenario",
            description="Test scenario for estimation",
            population_spec="pop.yaml",
            agents_file="agents.json",
            network_file="network.json",
        ),
        event=Event(
            type=EventType.ANNOUNCEMENT,
            content="A test announcement for the population to consider.",
            source="Test Corp",
            credibility=0.8,
            ambiguity=0.3,
            emotional_valence=0.2,
        ),
        seed_exposure=SeedExposure(
            channels=[
                ExposureChannel(
                    name="email",
                    description="Email broadcast",
                    reach="broadcast",
                    credibility_modifier=1.0,
                ),
            ],
            rules=[
                ExposureRule(
                    channel="email",
                    when="true",
                    probability=0.5,
                    timestep=0,
                ),
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Social media style observation",
        ),
        spread=SpreadConfig(
            share_probability=0.3,
        ),
        outcomes=OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="stance",
                    type=OutcomeType.CATEGORICAL,
                    description="Agent stance",
                    options=["support", "oppose", "neutral"],
                ),
            ],
        ),
        simulation=SimulationConfig(
            max_timesteps=50,
        ),
    )


# ─── Pricing Tests ────────────────────────────────────────────────────


class TestPricing:
    def test_known_model_pricing(self):
        p = get_pricing("gpt-5-mini")
        assert p is not None
        assert p.input_per_mtok == 0.30
        assert p.output_per_mtok == 1.50

    def test_unknown_model_returns_none(self):
        assert get_pricing("gpt-99-turbo") is None

    def test_claude_pricing_exists(self):
        p = get_pricing("claude-sonnet-4-5-20250929")
        assert p is not None
        assert p.input_per_mtok > 0
        assert p.output_per_mtok > 0

    def test_resolve_openai_default(self):
        assert resolve_default_model("openai", "reasoning") == "gpt-5"
        assert resolve_default_model("openai", "simple") == "gpt-5-mini"

    def test_resolve_claude_default(self):
        assert (
            resolve_default_model("claude", "reasoning") == "claude-sonnet-4-5-20250929"
        )
        assert resolve_default_model("claude", "simple") == "claude-haiku-4-5-20251001"

    def test_resolve_unknown_provider_falls_back(self):
        model = resolve_default_model("unknown_provider", "reasoning")
        # Should fall back to openai defaults
        assert model == "gpt-5"

    def test_model_pricing_frozen(self):
        p = ModelPricing(input_per_mtok=1.0, output_per_mtok=2.0)
        with pytest.raises(AttributeError):
            p.input_per_mtok = 5.0


# ─── Avg Degree Tests ─────────────────────────────────────────────────


class TestAvgDegree:
    def test_empty_network(self):
        assert _compute_avg_degree({"edges": []}) == 0.0
        assert _compute_avg_degree({}) == 0.0

    def test_simple_ring(self):
        # 4 nodes in a ring: each has degree 2
        edges = [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "d"},
            {"source": "d", "target": "a"},
        ]
        assert _compute_avg_degree({"edges": edges}) == 2.0

    def test_star_graph(self):
        # Center node connected to 3 leaves: center=3, each leaf=1
        edges = [
            {"source": "center", "target": "leaf1"},
            {"source": "center", "target": "leaf2"},
            {"source": "center", "target": "leaf3"},
        ]
        avg = _compute_avg_degree({"edges": edges})
        # center=3, leaf1=1, leaf2=1, leaf3=1 → avg = 6/4 = 1.5
        assert avg == 1.5


# ─── Rule Reach Tests ─────────────────────────────────────────────────


class TestRuleReach:
    def test_true_condition(self, small_agents):
        reach = _evaluate_rule_reach("true", 1.0, small_agents)
        assert reach == 10.0

    def test_true_with_probability(self, small_agents):
        reach = _evaluate_rule_reach("true", 0.5, small_agents)
        assert reach == 5.0

    def test_condition_filter(self, small_agents):
        # Only agents with age >= 58 → agents 7,8,9 (ages 58,62,66)
        reach = _evaluate_rule_reach("age >= 58", 1.0, small_agents)
        assert reach == 3.0

    def test_condition_with_probability(self, small_agents):
        reach = _evaluate_rule_reach("age >= 58", 0.5, small_agents)
        assert reach == 1.5

    def test_no_match(self, small_agents):
        reach = _evaluate_rule_reach("age > 100", 1.0, small_agents)
        assert reach == 0.0


# ─── Token Estimation Tests ──────────────────────────────────────────


class TestTokenEstimation:
    def test_basic_counts(self):
        tok = _estimate_token_counts(num_attributes=10, event_content_len=400)
        assert tok["pass1_input"] == 250 + (80 + 15 * 10) + 100 + 115
        assert tok["pass1_output"] == 200
        assert tok["pass2_input"] == 300
        assert tok["pass2_output"] == 70

    def test_more_attributes_more_tokens(self):
        small = _estimate_token_counts(num_attributes=5, event_content_len=100)
        large = _estimate_token_counts(num_attributes=30, event_content_len=100)
        assert large["pass1_input"] > small["pass1_input"]

    def test_longer_event_more_tokens(self):
        short = _estimate_token_counts(num_attributes=10, event_content_len=100)
        long = _estimate_token_counts(num_attributes=10, event_content_len=2000)
        assert long["pass1_input"] > short["pass1_input"]


# ─── Full Estimation Pipeline Tests ──────────────────────────────────


class TestEstimationPipeline:
    def test_basic_estimate(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
            provider="openai",
        )

        assert est.population_size == 10
        assert est.avg_degree > 0
        assert est.max_timesteps == 50
        assert est.effective_timesteps <= 50
        assert est.pass1_calls > 0
        assert est.pass2_calls == est.pass1_calls
        assert est.pass1_input_tokens > 0
        assert est.pass2_input_tokens > 0
        assert est.total_cost is not None
        assert est.total_cost > 0

    def test_pass1_equals_pass2_calls(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
        )
        assert est.pass1_calls == est.pass2_calls

    def test_per_timestep_populated(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
        )
        assert len(est.per_timestep) > 0
        assert est.per_timestep[0]["timestep"] == 0
        # First timestep should have some exposures from seed rule
        assert est.per_timestep[0]["new_exposures"] > 0

    def test_model_resolution_openai(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
            provider="openai",
        )
        assert est.pivotal_model == "gpt-5"
        assert est.routine_model == "gpt-5-mini"

    def test_model_resolution_claude(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
            provider="claude",
        )
        assert est.pivotal_model == "claude-sonnet-4-5-20250929"
        assert est.routine_model == "claude-haiku-4-5-20251001"

    def test_explicit_model_override(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
            provider="openai",
            pivotal_model="gpt-5-mini",
            routine_model="gpt-5-mini",
        )
        assert est.pivotal_model == "gpt-5-mini"
        assert est.routine_model == "gpt-5-mini"

    def test_unknown_model_pricing_none(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
            pivotal_model="unknown-model-x",
            routine_model="unknown-model-y",
        )
        assert est.pivotal_pricing is None
        assert est.routine_pricing is None
        assert est.pass1_cost is None
        assert est.pass2_cost is None
        assert est.total_cost is None
        # Token estimates should still be present
        assert est.pass1_input_tokens > 0

    def test_cost_increases_with_population(self, small_pop_spec, small_scenario):
        # 10 agents
        agents_10 = [
            {"_id": f"a{i}", "age": 30 + i, "role": "junior"} for i in range(10)
        ]
        edges_10 = [
            {"source": f"a{i}", "target": f"a{(i + 1) % 10}"} for i in range(10)
        ]

        # 50 agents
        agents_50 = [
            {"_id": f"a{i}", "age": 30 + (i % 40), "role": "junior"} for i in range(50)
        ]
        edges_50 = [
            {"source": f"a{i}", "target": f"a{(i + 1) % 50}"} for i in range(50)
        ]

        est_10 = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=agents_10,
            network={"edges": edges_10},
        )
        est_50 = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=agents_50,
            network={"edges": edges_50},
        )

        assert est_50.pass1_calls > est_10.pass1_calls
        assert est_50.total_cost > est_10.total_cost

    def test_zero_agents(self, small_scenario, small_pop_spec):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=[],
            network={"edges": []},
        )
        assert est.pass1_calls == 0
        assert est.pass2_calls == 0
        assert est.total_cost == 0.0

    def test_exposure_rate_increases(
        self, small_scenario, small_pop_spec, small_agents, small_network
    ):
        est = estimate_simulation_cost(
            scenario=small_scenario,
            population_spec=small_pop_spec,
            agents=small_agents,
            network=small_network,
        )
        # Exposure rate should be non-decreasing
        rates = [row["exposure_rate"] for row in est.per_timestep]
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1]
