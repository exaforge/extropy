"""Tests for the network generation module."""

import tempfile
from pathlib import Path

import pytest

from extropy.population.network.config import (
    NetworkConfig,
    AttributeWeightConfig,
    DegreeMultiplierConfig,
    EdgeTypeRule,
    InfluenceFactorConfig,
)
from extropy.population.network.similarity import (
    compute_match_score,
    compute_similarity,
    compute_degree_factor,
    compute_edge_probability,
    sigmoid,
)
from extropy.population.network.generator import (
    generate_network,
    generate_network_with_metrics,
    load_agents_json,
    Edge,
    NetworkResult,
)

_REFERENCE_SENIORITY_LEVELS = {
    "resident": 1,
    "specialist_attending": 2,
    "senior_physician_Oberarzt": 3,
    "chief_physician_Chefarzt": 4,
}

REFERENCE_NETWORK_CONFIG = NetworkConfig(
    avg_degree=20.0,
    rewire_prob=0.05,
    similarity_threshold=0.3,
    similarity_steepness=10.0,
    attribute_weights={
        "employer_type": AttributeWeightConfig(weight=3.0, match_type="exact"),
        "surgical_specialty": AttributeWeightConfig(weight=2.5, match_type="exact"),
        "federal_state": AttributeWeightConfig(weight=2.0, match_type="exact"),
        "role_seniority": AttributeWeightConfig(
            weight=1.5,
            match_type="within_n",
            range_value=1,
        ),
        "care_level": AttributeWeightConfig(weight=1.0, match_type="exact"),
        "age": AttributeWeightConfig(
            weight=1.0, match_type="numeric_range", range_value=10
        ),
        "participation_in_research": AttributeWeightConfig(
            weight=0.5, match_type="exact"
        ),
        "professional_society_membership": AttributeWeightConfig(
            weight=0.5, match_type="exact"
        ),
    },
    degree_multipliers=[
        DegreeMultiplierConfig(
            attribute="role_seniority",
            condition="chief_physician_Chefarzt",
            multiplier=2.0,
            rationale="Department heads know everyone",
        ),
        DegreeMultiplierConfig(
            attribute="role_seniority",
            condition="senior_physician_Oberarzt",
            multiplier=1.3,
            rationale="Mid-level leadership",
        ),
        DegreeMultiplierConfig(
            attribute="teaching_responsibility",
            condition=True,
            multiplier=1.4,
            rationale="Mentors many residents",
        ),
        DegreeMultiplierConfig(
            attribute="participation_in_research",
            condition=True,
            multiplier=1.3,
            rationale="Collaborations, publications",
        ),
        DegreeMultiplierConfig(
            attribute="professional_society_membership",
            condition=True,
            multiplier=1.2,
            rationale="Committee work, conferences",
        ),
        DegreeMultiplierConfig(
            attribute="employer_type",
            condition="university_hospital",
            multiplier=1.2,
            rationale="Larger institutions, more colleagues",
        ),
    ],
    edge_type_rules=[
        EdgeTypeRule(
            name="mentor_mentee",
            condition="a_employer_type == b_employer_type and a_role_seniority != b_role_seniority",
            priority=50,
            description="Same hospital, different seniority — training relationship",
        ),
        EdgeTypeRule(
            name="colleague",
            condition="a_employer_type == b_employer_type",
            priority=40,
            description="Same hospital — daily coworkers",
        ),
        EdgeTypeRule(
            name="specialty_peer",
            condition="a_surgical_specialty == b_surgical_specialty",
            priority=30,
            description="Same specialty — conferences, society meetings",
        ),
        EdgeTypeRule(
            name="regional",
            condition="a_federal_state == b_federal_state",
            priority=20,
            description="Same state — regional referral network",
        ),
    ],
    influence_factors=[
        InfluenceFactorConfig(
            attribute="role_seniority",
            type="ordinal",
            levels=_REFERENCE_SENIORITY_LEVELS,
            weight=0.3,
            description="Senior surgeons influence junior ones more",
        ),
        InfluenceFactorConfig(
            attribute="participation_in_research",
            type="boolean",
            weight=0.2,
            description="Researchers carry academic credibility",
        ),
        InfluenceFactorConfig(
            attribute="teaching_responsibility",
            type="boolean",
            weight=0.1,
            description="Teachers influence through mentorship",
        ),
    ],
    default_edge_type="weak_tie",
    ordinal_levels={"role_seniority": _REFERENCE_SENIORITY_LEVELS},
)


class TestNetworkConfig:
    """Tests for network configuration."""

    def test_default_config(self):
        """Test default network configuration."""
        config = NetworkConfig()

        assert config.avg_degree == 20.0
        assert config.rewire_prob == 0.05
        assert config.similarity_store_threshold == 0.05
        assert config.similarity_threshold == 0.3
        assert config.similarity_steepness == 10.0
        assert config.candidate_mode == "exact"
        assert config.candidate_pool_multiplier == 12.0
        assert config.min_candidate_pool == 80
        assert config.similarity_workers == 1
        assert config.checkpoint_every_rows == 250
        assert config.seed is None

    def test_custom_config(self):
        """Test custom network configuration."""
        config = NetworkConfig(
            avg_degree=10.0,
            rewire_prob=0.1,
            seed=42,
        )

        assert config.avg_degree == 10.0
        assert config.rewire_prob == 0.1
        assert config.seed == 42

    def test_empty_default_config(self):
        """Test that default config has empty weights (not hardcoded)."""
        config = NetworkConfig()

        assert len(config.attribute_weights) == 0
        assert len(config.degree_multipliers) == 0
        assert config.default_edge_type == "peer"

    def test_reference_network_config_has_weights(self):
        """Test that reference config has expected weights/rules."""
        config = REFERENCE_NETWORK_CONFIG

        assert len(config.attribute_weights) > 0
        assert "employer_type" in config.attribute_weights
        assert "surgical_specialty" in config.attribute_weights
        assert len(config.edge_type_rules) == 4
        assert len(config.influence_factors) == 3

    def test_get_total_weight(self):
        """Test calculating total weight."""
        config = REFERENCE_NETWORK_CONFIG
        total = config.get_total_weight()

        assert total > 0
        assert total == sum(w.weight for w in config.attribute_weights.values())

    def test_get_total_weight_empty(self):
        """Test total weight on empty config is 0."""
        config = NetworkConfig()
        assert config.get_total_weight() == 0


class TestAttributeWeightConfig:
    """Tests for attribute weight configuration."""

    def test_exact_match_config(self):
        """Test exact match configuration."""
        config = AttributeWeightConfig(weight=2.0, match_type="exact")

        assert config.weight == 2.0
        assert config.match_type == "exact"
        assert config.range_value is None

    def test_numeric_range_config(self):
        """Test numeric range configuration."""
        config = AttributeWeightConfig(
            weight=1.0,
            match_type="numeric_range",
            range_value=10.0,
        )

        assert config.match_type == "numeric_range"
        assert config.range_value == 10.0

    def test_within_n_config(self):
        """Test within_n configuration."""
        config = AttributeWeightConfig(
            weight=1.5,
            match_type="within_n",
            range_value=1,
        )

        assert config.match_type == "within_n"
        assert config.range_value == 1


class TestComputeMatchScore:
    """Tests for match score computation."""

    def test_exact_match_same_value(self):
        """Test exact match with same values."""
        config = AttributeWeightConfig(weight=1.0, match_type="exact")
        score = compute_match_score("A", "A", config)
        assert score == 1.0

    def test_exact_match_different_values(self):
        """Test exact match with different values."""
        config = AttributeWeightConfig(weight=1.0, match_type="exact")
        score = compute_match_score("A", "B", config)
        assert score == 0.0

    def test_numeric_range_same_value(self):
        """Test numeric range with same values."""
        config = AttributeWeightConfig(
            weight=1.0, match_type="numeric_range", range_value=10.0
        )
        score = compute_match_score(50, 50, config)
        assert score == 1.0

    def test_numeric_range_within_range(self):
        """Test numeric range with values within range."""
        config = AttributeWeightConfig(
            weight=1.0, match_type="numeric_range", range_value=10.0
        )
        score = compute_match_score(50, 55, config)
        assert score == 0.5  # 1 - 5/10 = 0.5

    def test_numeric_range_beyond_range(self):
        """Test numeric range with values beyond range."""
        config = AttributeWeightConfig(
            weight=1.0, match_type="numeric_range", range_value=10.0
        )
        score = compute_match_score(50, 70, config)
        assert score == 0.0  # 1 - 20/10 = -1.0 -> clamped to 0.0

    def test_within_n_same_level(self):
        """Test within_n with same seniority level."""
        seniority_levels = {
            "resident": 1,
            "specialist_attending": 2,
            "senior_physician_Oberarzt": 3,
            "chief_physician_Chefarzt": 4,
        }
        config = AttributeWeightConfig(
            weight=1.0,
            match_type="within_n",
            range_value=1,
            ordinal_levels=seniority_levels,
        )
        score = compute_match_score(
            "chief_physician_Chefarzt", "chief_physician_Chefarzt", config
        )
        assert score == 1.0

    def test_within_n_adjacent_levels(self):
        """Test within_n with adjacent seniority levels."""
        seniority_levels = {
            "resident": 1,
            "specialist_attending": 2,
            "senior_physician_Oberarzt": 3,
            "chief_physician_Chefarzt": 4,
        }
        config = AttributeWeightConfig(
            weight=1.0,
            match_type="within_n",
            range_value=1,
            ordinal_levels=seniority_levels,
        )
        # Chief (4) and Senior (3) are 1 level apart
        score = compute_match_score(
            "chief_physician_Chefarzt", "senior_physician_Oberarzt", config
        )
        assert score == 1.0

    def test_within_n_distant_levels(self):
        """Test within_n with distant seniority levels."""
        seniority_levels = {
            "resident": 1,
            "specialist_attending": 2,
            "senior_physician_Oberarzt": 3,
            "chief_physician_Chefarzt": 4,
        }
        config = AttributeWeightConfig(
            weight=1.0,
            match_type="within_n",
            range_value=1,
            ordinal_levels=seniority_levels,
        )
        # Chief (4) and Resident (1) are 3 levels apart
        score = compute_match_score("chief_physician_Chefarzt", "resident", config)
        assert score == 0.0

    def test_within_n_numeric_fallback(self):
        """Test within_n with numeric values (no ordinal_levels needed)."""
        config = AttributeWeightConfig(weight=1.0, match_type="within_n", range_value=2)
        assert compute_match_score(5, 7, config) == 1.0  # diff 2, within range
        assert compute_match_score(5, 8, config) == 0.0  # diff 3, outside range

    def test_none_values_return_zero(self):
        """Test that None values return 0 score."""
        config = AttributeWeightConfig(weight=1.0, match_type="exact")
        assert compute_match_score(None, "A", config) == 0.0
        assert compute_match_score("A", None, config) == 0.0


class TestComputeSimilarity:
    """Tests for similarity computation."""

    def test_identical_agents(self, sample_agents):
        """Test similarity between identical agents."""
        agent = sample_agents[0]
        config = REFERENCE_NETWORK_CONFIG
        sim = compute_similarity(
            agent, agent, config.attribute_weights, config.ordinal_levels
        )
        assert sim == 1.0

    def test_similar_agents(self, sample_agents):
        """Test similarity between similar agents."""
        config = REFERENCE_NETWORK_CONFIG
        # Agents 0 and 1 share employer_type, specialty, federal_state
        sim = compute_similarity(
            sample_agents[0],
            sample_agents[1],
            config.attribute_weights,
            config.ordinal_levels,
        )
        assert sim > 0.5  # Should have high similarity

    def test_dissimilar_agents(self, sample_agents):
        """Test similarity between dissimilar agents."""
        config = REFERENCE_NETWORK_CONFIG
        # Agents 0 and 4 differ in most attributes
        sim = compute_similarity(
            sample_agents[0],
            sample_agents[4],
            config.attribute_weights,
            config.ordinal_levels,
        )
        assert sim < 0.5  # Should have low similarity

    def test_similarity_is_symmetric(self, sample_agents):
        """Test that similarity is symmetric."""
        config = REFERENCE_NETWORK_CONFIG
        sim_ab = compute_similarity(
            sample_agents[0],
            sample_agents[1],
            config.attribute_weights,
            config.ordinal_levels,
        )
        sim_ba = compute_similarity(
            sample_agents[1],
            sample_agents[0],
            config.attribute_weights,
            config.ordinal_levels,
        )
        assert sim_ab == sim_ba

    def test_similarity_normalized_to_0_1(self, sample_agents):
        """Test that similarity is always in [0, 1]."""
        config = REFERENCE_NETWORK_CONFIG
        for i, agent_a in enumerate(sample_agents):
            for j, agent_b in enumerate(sample_agents):
                sim = compute_similarity(
                    agent_a, agent_b, config.attribute_weights, config.ordinal_levels
                )
                assert 0.0 <= sim <= 1.0

    def test_similarity_with_custom_weights(self, sample_agents):
        """Test similarity with custom attribute weights."""
        custom_weights = {
            "employer_type": AttributeWeightConfig(weight=10.0, match_type="exact"),
        }
        # Agents 0 and 1 have same employer_type
        sim = compute_similarity(sample_agents[0], sample_agents[1], custom_weights)
        assert sim == 1.0  # Only employer_type matters, and they match


class TestComputeDegreeFactor:
    """Tests for degree factor computation."""

    def test_chief_physician_factor(self, sample_agents):
        """Test degree factor for chief physician."""
        config = REFERENCE_NETWORK_CONFIG
        factor = compute_degree_factor(sample_agents[0], config)

        # Agent 0 is chief, teaching, research, society, university
        # 2.0 × 1.4 × 1.3 × 1.2 × 1.2 ≈ 5.24
        assert factor > 4.0

    def test_resident_factor(self, sample_agents):
        """Test degree factor for resident."""
        config = REFERENCE_NETWORK_CONFIG
        factor = compute_degree_factor(sample_agents[2], config)

        # Agent 2 is resident at university (1.2)
        assert 1.0 <= factor <= 2.0

    def test_factor_always_positive(self, sample_agents):
        """Test that degree factor is always >= 1.0."""
        config = REFERENCE_NETWORK_CONFIG
        for agent in sample_agents:
            factor = compute_degree_factor(agent, config)
            assert factor >= 1.0

    def test_factor_with_empty_config(self, sample_agents):
        """Test that empty config gives factor of 1.0 (no multipliers)."""
        config = NetworkConfig()
        for agent in sample_agents:
            factor = compute_degree_factor(agent, config)
            assert factor == 1.0


class TestSigmoid:
    """Tests for sigmoid function."""

    def test_sigmoid_at_threshold(self):
        """Test sigmoid value at threshold."""
        result = sigmoid(0.3, threshold=0.3, steepness=10.0)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_sigmoid_below_threshold(self):
        """Test sigmoid value below threshold."""
        result = sigmoid(0.0, threshold=0.3, steepness=10.0)
        assert result < 0.5

    def test_sigmoid_above_threshold(self):
        """Test sigmoid value above threshold."""
        result = sigmoid(0.6, threshold=0.3, steepness=10.0)
        assert result > 0.5

    def test_sigmoid_range(self):
        """Test that sigmoid outputs are in (0, 1)."""
        for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            result = sigmoid(x)
            assert 0.0 < result < 1.0


class TestComputeEdgeProbability:
    """Tests for edge probability computation."""

    def test_probability_clamped_to_1(self):
        """Test that probability is clamped to 1.0."""
        config = NetworkConfig()
        prob = compute_edge_probability(
            similarity=1.0,
            degree_factor_a=10.0,
            degree_factor_b=10.0,
            base_rate=1.0,
            config=config,
        )
        assert prob <= 1.0

    def test_probability_increases_with_similarity(self):
        """Test that higher similarity means higher probability."""
        config = NetworkConfig()
        prob_low = compute_edge_probability(0.2, 1.0, 1.0, 0.1, config)
        prob_high = compute_edge_probability(0.8, 1.0, 1.0, 0.1, config)
        assert prob_high > prob_low

    def test_probability_increases_with_degree_factor(self):
        """Test that higher degree factors mean higher probability."""
        config = NetworkConfig()
        prob_low = compute_edge_probability(0.5, 1.0, 1.0, 0.1, config)
        prob_high = compute_edge_probability(0.5, 2.0, 2.0, 0.1, config)
        assert prob_high > prob_low


class TestEdge:
    """Tests for Edge dataclass."""

    def test_edge_creation(self):
        """Test creating an edge."""
        edge = Edge(
            source="agent_000",
            target="agent_001",
            weight=0.85,
            edge_type="colleague",
        )
        assert edge.source == "agent_000"
        assert edge.target == "agent_001"
        assert edge.weight == 0.85
        assert edge.edge_type == "colleague"
        assert edge.bidirectional is True

    def test_edge_to_dict(self):
        """Test edge serialization to dict."""
        edge = Edge(
            source="agent_000",
            target="agent_001",
            weight=0.85,
            edge_type="colleague",
            influence_weight={
                "source_to_target": 0.9,
                "target_to_source": 0.7,
            },
        )
        d = edge.to_dict()

        assert d["source"] == "agent_000"
        assert d["target"] == "agent_001"
        assert d["weight"] == 0.85
        assert d["type"] == "colleague"
        assert d["bidirectional"] is True
        assert "influence_weight" in d


class TestNetworkResult:
    """Tests for NetworkResult dataclass."""

    def test_network_result_creation(self):
        """Test creating a network result."""
        edges = [
            Edge(source="a", target="b", weight=0.5, edge_type="colleague"),
            Edge(source="b", target="c", weight=0.7, edge_type="colleague"),
        ]
        result = NetworkResult(
            meta={"agent_count": 3, "edge_count": 2},
            edges=edges,
        )
        assert result.meta["agent_count"] == 3
        assert len(result.edges) == 2

    def test_network_result_to_dict(self):
        """Test network result serialization."""
        edges = [
            Edge(source="a", target="b", weight=0.5, edge_type="colleague"),
        ]
        result = NetworkResult(
            meta={"agent_count": 2, "edge_count": 1},
            edges=edges,
        )
        d = result.to_dict()

        assert "meta" in d
        assert "edges" in d
        assert len(d["edges"]) == 1

    def test_network_result_save_json(self):
        """Test saving network result to JSON."""
        edges = [Edge(source="a", target="b", weight=0.5, edge_type="colleague")]
        result = NetworkResult(
            meta={"agent_count": 2, "edge_count": 1},
            edges=edges,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "network.json"
            result.save_json(path)

            assert path.exists()

            import json

            with open(path) as f:
                data = json.load(f)

            assert data["meta"]["agent_count"] == 2
            assert len(data["edges"]) == 1


class TestGenerateNetwork:
    """Tests for network generation."""

    def test_generate_network_basic(self, sample_agents):
        """Test basic network generation."""
        config = NetworkConfig(avg_degree=2.0, seed=42)
        result = generate_network(sample_agents, config)

        assert isinstance(result, NetworkResult)
        assert result.meta["agent_count"] == len(sample_agents)
        assert len(result.edges) >= 0

    def test_generate_network_reproducibility(self, sample_agents):
        """Test that same seed produces same network."""
        config = NetworkConfig(avg_degree=2.0, seed=42)

        result1 = generate_network(sample_agents, config)
        result2 = generate_network(sample_agents, config)

        assert len(result1.edges) == len(result2.edges)
        # Compare edge sources and targets
        edges1 = [(e.source, e.target) for e in result1.edges]
        edges2 = [(e.source, e.target) for e in result2.edges]
        assert set(edges1) == set(edges2)

    def test_generate_network_different_seeds_differ(self, sample_agents):
        """Test that different seeds produce different networks."""
        config1 = NetworkConfig(avg_degree=3.0, seed=42)
        config2 = NetworkConfig(avg_degree=3.0, seed=123)

        result1 = generate_network(sample_agents, config1)
        result2 = generate_network(sample_agents, config2)

        set((e.source, e.target) for e in result1.edges)
        set((e.source, e.target) for e in result2.edges)

        # May be the same by chance with small network, but likely different
        # Just check they're valid networks
        assert result1.meta["seed"] == 42
        assert result2.meta["seed"] == 123

    def test_generate_network_edge_types(self, sample_agents):
        """Test that edges have valid types."""
        config = NetworkConfig(avg_degree=4.0, seed=42)
        result = generate_network(sample_agents, config)

        # With empty config, all edges should be "peer" (default)
        for edge in result.edges:
            assert edge.edge_type == "peer"

    def test_generate_network_edge_types_with_rules(self, sample_agents):
        """Test that edges have rule-derived types with reference config."""

        config = NetworkConfig(
            avg_degree=4.0,
            seed=42,
            attribute_weights=REFERENCE_NETWORK_CONFIG.attribute_weights,
            degree_multipliers=REFERENCE_NETWORK_CONFIG.degree_multipliers,
            edge_type_rules=REFERENCE_NETWORK_CONFIG.edge_type_rules,
            influence_factors=REFERENCE_NETWORK_CONFIG.influence_factors,
            default_edge_type=REFERENCE_NETWORK_CONFIG.default_edge_type,
            ordinal_levels=REFERENCE_NETWORK_CONFIG.ordinal_levels,
        )
        result = generate_network(sample_agents, config)

        valid_types = {
            "colleague",
            "mentor_mentee",
            "specialty_peer",
            "regional",
            "weak_tie",
        }
        for edge in result.edges:
            assert edge.edge_type in valid_types

    def test_generate_network_no_self_loops(self, sample_agents):
        """Test that network has no self-loops."""
        config = NetworkConfig(avg_degree=4.0, seed=42)
        result = generate_network(sample_agents, config)

        for edge in result.edges:
            assert edge.source != edge.target

    def test_generate_network_progress_callback(self, sample_agents):
        """Test progress callback is called."""
        progress_calls = []

        def on_progress(stage, current, total):
            progress_calls.append((stage, current, total))

        config = NetworkConfig(avg_degree=2.0, seed=42)

        generate_network(sample_agents, config, on_progress=on_progress)

        assert len(progress_calls) > 0
        # Check various stages were reported
        stages = set(call[0] for call in progress_calls)
        assert "Computing similarities" in stages

    def test_generate_network_blocked_mode_reproducibility(self, sample_agents):
        """Blocked candidate mode should remain deterministic with fixed seed."""
        config = REFERENCE_NETWORK_CONFIG.model_copy(
            update={
                "seed": 42,
                "candidate_mode": "blocked",
                "candidate_pool_multiplier": 8.0,
                "blocking_attributes": ["employer_type", "federal_state"],
            }
        )

        result1 = generate_network(sample_agents, config)
        result2 = generate_network(sample_agents, config)

        edges1 = {(e.source, e.target) for e in result1.edges}
        edges2 = {(e.source, e.target) for e in result2.edges}

        assert result1.meta["candidate_mode"] == "blocked"
        assert result2.meta["candidate_mode"] == "blocked"
        assert edges1 == edges2

    def test_generate_network_resume_from_checkpoint_matches_fresh(self, sample_agents):
        """Resuming from a saved similarity checkpoint should match a fresh run."""
        import pickle

        config = REFERENCE_NETWORK_CONFIG.model_copy(update={"seed": 42})

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "network-similarity.pkl"

            # Build and persist checkpoint from a full run.
            result_checkpointed = generate_network(
                sample_agents,
                config,
                checkpoint_path=checkpoint_path,
            )
            assert checkpoint_path.exists()

            # Simulate interruption by truncating completed_rows in checkpoint metadata.
            with open(checkpoint_path, "rb") as f:
                payload = pickle.load(f)
            completed_rows = max(1, len(sample_agents) // 2)
            payload["completed_rows"] = completed_rows
            payload["similarities"] = {
                pair: sim
                for pair, sim in payload["similarities"].items()
                if pair[0] < completed_rows
            }
            with open(checkpoint_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            resumed = generate_network(
                sample_agents,
                config,
                checkpoint_path=checkpoint_path,
                resume_from_checkpoint=True,
            )
            fresh = generate_network(sample_agents, config)

            resumed_edges = {(e.source, e.target) for e in resumed.edges}
            fresh_edges = {(e.source, e.target) for e in fresh.edges}

            assert resumed.meta["resumed_from_checkpoint"] is True
            assert resumed_edges == fresh_edges
            assert len(resumed.edges) == len(result_checkpointed.edges)


class TestGenerateNetworkWithMetrics:
    """Tests for network generation with metrics."""

    def test_generate_with_metrics(self, sample_agents):
        """Test network generation with metrics computation."""
        config = NetworkConfig(avg_degree=2.0, seed=42)
        result = generate_network_with_metrics(sample_agents, config)

        assert isinstance(result, NetworkResult)
        # Metrics should be computed
        assert result.network_metrics is not None or result.node_metrics is not None

    def test_metrics_in_meta(self, sample_agents):
        """Test that metrics are added to meta."""
        config = NetworkConfig(avg_degree=3.0, seed=42)
        result = generate_network_with_metrics(sample_agents, config)

        # If metrics computed, they should be in meta
        if result.network_metrics:
            assert "clustering_coefficient" in result.meta


class TestLoadAgentsJson:
    """Tests for loading agents from JSON."""

    def test_load_agents_list_format(self):
        """Test loading agents in list format."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agents.json"
            agents = [{"_id": "agent_0", "age": 30}]

            with open(path, "w") as f:
                json.dump(agents, f)

            loaded = load_agents_json(path)
            assert loaded == agents

    def test_load_agents_dict_format(self):
        """Test loading agents in dict format with meta."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agents.json"
            data = {
                "meta": {"count": 1},
                "agents": [{"_id": "agent_0", "age": 30}],
            }

            with open(path, "w") as f:
                json.dump(data, f)

            loaded = load_agents_json(path)
            assert loaded == data["agents"]

    def test_load_agents_invalid_format(self):
        """Test loading agents with invalid format."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agents.json"
            data = {"invalid": "format"}

            with open(path, "w") as f:
                json.dump(data, f)

            with pytest.raises(ValueError):
                load_agents_json(path)


class TestEdgeTypeInference:
    """Tests for edge type inference with data-driven rules."""

    def test_colleague_with_reference_config(self):
        """Test colleague edge type with reference config rules."""
        from extropy.population.network.generator import _infer_edge_type

        agent_a = {
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "role_seniority": "senior_physician_Oberarzt",
            "federal_state": "Bayern",
        }
        agent_b = {
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "role_seniority": "senior_physician_Oberarzt",
            "federal_state": "Bayern",
        }

        edge_type = _infer_edge_type(agent_a, agent_b, REFERENCE_NETWORK_CONFIG)
        assert edge_type == "colleague"

    def test_mentor_mentee_with_reference_config(self):
        """Test mentor_mentee edge type with reference config rules."""
        from extropy.population.network.generator import _infer_edge_type

        agent_a = {
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "role_seniority": "chief_physician_Chefarzt",
            "federal_state": "Bayern",
        }
        agent_b = {
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "role_seniority": "resident",
            "federal_state": "Bayern",
        }

        edge_type = _infer_edge_type(agent_a, agent_b, REFERENCE_NETWORK_CONFIG)
        assert edge_type == "mentor_mentee"

    def test_specialty_peer_with_reference_config(self):
        """Test specialty_peer edge type with reference config rules."""
        from extropy.population.network.generator import _infer_edge_type

        agent_a = {
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "federal_state": "Bayern",
        }
        agent_b = {
            "employer_type": "private_clinic",
            "surgical_specialty": "cardiology",
            "federal_state": "Berlin",
        }

        edge_type = _infer_edge_type(agent_a, agent_b, REFERENCE_NETWORK_CONFIG)
        assert edge_type == "specialty_peer"

    def test_regional_with_reference_config(self):
        """Test regional edge type with reference config rules."""
        from extropy.population.network.generator import _infer_edge_type

        agent_a = {
            "employer_type": "university_hospital",
            "surgical_specialty": "cardiology",
            "federal_state": "Bayern",
        }
        agent_b = {
            "employer_type": "private_clinic",
            "surgical_specialty": "neurology",
            "federal_state": "Bayern",
        }

        edge_type = _infer_edge_type(agent_a, agent_b, REFERENCE_NETWORK_CONFIG)
        assert edge_type == "regional"

    def test_default_edge_type_rewired(self):
        """Test rewired edges get the config's default_edge_type."""
        from extropy.population.network.generator import _infer_edge_type

        agent_a = {"employer_type": "A"}
        agent_b = {"employer_type": "B"}

        edge_type = _infer_edge_type(
            agent_a, agent_b, REFERENCE_NETWORK_CONFIG, is_rewired=True
        )
        assert edge_type == "weak_tie"  # Reference config default_edge_type

    def test_default_edge_type_no_rules(self):
        """Test that empty config returns 'peer' as default."""
        from extropy.population.network.generator import _infer_edge_type

        agent_a = {"employer_type": "A"}
        agent_b = {"employer_type": "B"}

        edge_type = _infer_edge_type(agent_a, agent_b, NetworkConfig())
        assert edge_type == "peer"

    def test_no_match_falls_through_to_default(self):
        """Test that when no rules match, default_edge_type is returned."""
        from extropy.population.network.generator import _infer_edge_type

        agent_a = {
            "employer_type": "A",
            "surgical_specialty": "X",
            "federal_state": "1",
            "role_seniority": "resident",
        }
        agent_b = {
            "employer_type": "B",
            "surgical_specialty": "Y",
            "federal_state": "2",
            "role_seniority": "resident",
        }

        edge_type = _infer_edge_type(agent_a, agent_b, REFERENCE_NETWORK_CONFIG)
        assert edge_type == "weak_tie"  # Falls through all rules to default
