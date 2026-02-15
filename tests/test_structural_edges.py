"""Tests for structural edge generation (Phase B2) and identity clustering (Phase B3)."""

import random

from extropy.core.models.network import Edge
from extropy.population.network.generator import _generate_structural_edges
from extropy.population.network.config import NetworkConfig


def _make_household_agents(n_households: int = 10) -> tuple[list[dict], list[str]]:
    """Create test agents with household structure."""
    agents = []
    agent_ids = []
    idx = 0

    for hh in range(n_households):
        hh_id = f"household_{hh:04d}"
        # Adult 1
        a1_id = f"agent_{idx:04d}"
        a1 = {
            "_id": a1_id,
            "household_id": hh_id,
            "household_role": "adult_primary",
            "partner_id": f"agent_{idx + 1:04d}" if hh % 2 == 0 else None,
            "age": 35 + hh,
            "gender": "male" if hh % 3 != 0 else "female",
            "state": "CA" if hh < 5 else "TX",
            "urban_rural": "urban" if hh < 7 else "rural",
            "occupation_sector": "tech" if hh < 6 else "healthcare",
            "religious_affiliation": "christian" if hh < 4 else "none",
            "dependents": (
                [
                    {
                        "age": 10,
                        "gender": "male",
                        "relationship": "son",
                        "school_status": "elementary",
                    }
                ]
                if hh % 3 == 0
                else []
            ),
        }
        agents.append(a1)
        agent_ids.append(a1_id)
        idx += 1

        # Adult 2 (partner) for even-numbered households
        if hh % 2 == 0:
            a2_id = f"agent_{idx:04d}"
            a2 = {
                "_id": a2_id,
                "household_id": hh_id,
                "household_role": "adult_secondary",
                "partner_id": a1_id,
                "age": 33 + hh,
                "gender": "female" if hh % 3 != 0 else "male",
                "state": a1["state"],
                "urban_rural": a1["urban_rural"],
                "occupation_sector": "education" if hh < 3 else "tech",
                "religious_affiliation": a1["religious_affiliation"],
                "dependents": a1["dependents"],
            }
            agents.append(a2)
            agent_ids.append(a2_id)
            idx += 1

    return agents, agent_ids


class TestStructuralEdgeGeneration:
    def test_partner_edges_created(self):
        agents, agent_ids = _make_household_agents(10)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)

        partner_edges = [e for e in edges if e.edge_type == "partner"]
        # Count agents that have a partner_id set
        agents_with_partners = [a for a in agents if a.get("partner_id")]
        expected_pairs = len(agents_with_partners) // 2
        assert len(partner_edges) == expected_pairs

    def test_partner_edge_weight(self):
        agents, agent_ids = _make_household_agents(4)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)
        partner_edges = [e for e in edges if e.edge_type == "partner"]
        for e in partner_edges:
            assert e.weight == 1.0
            assert e.structural is True
            assert e.context == "household"

    def test_household_edges_connect_members(self):
        agents, agent_ids = _make_household_agents(6)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)

        household_edges = [e for e in edges if e.edge_type == "household"]
        # Household edges connect adults in the same household (weight 0.9)
        for e in household_edges:
            assert e.weight == 0.9
            assert e.structural is True

    def test_coworker_edges_respect_sector_state(self):
        agents, agent_ids = _make_household_agents(10)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)

        coworker_edges = [e for e in edges if e.edge_type == "coworker"]
        id_to_agent = {a["_id"]: a for a in agents}
        for e in coworker_edges:
            a = id_to_agent[e.source]
            b = id_to_agent[e.target]
            assert a["occupation_sector"] == b["occupation_sector"]
            assert a["state"] == b["state"]
            assert e.weight == 0.6
            assert e.context == "workplace"

    def test_neighbor_edges_respect_age_constraint(self):
        agents, agent_ids = _make_household_agents(10)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)

        neighbor_edges = [e for e in edges if e.edge_type == "neighbor"]
        id_to_agent = {a["_id"]: a for a in agents}
        for e in neighbor_edges:
            a = id_to_agent[e.source]
            b = id_to_agent[e.target]
            assert abs(a["age"] - b["age"]) <= 15
            assert a["state"] == b["state"]
            assert a["urban_rural"] == b["urban_rural"]

    def test_all_edges_are_structural(self):
        agents, agent_ids = _make_household_agents(10)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)
        for e in edges:
            assert e.structural is True
            assert e.context is not None

    def test_school_parent_edges(self):
        agents, agent_ids = _make_household_agents(10)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)
        school_edges = [e for e in edges if e.edge_type == "school_parent"]
        # Agents with school-age dependents in same state+urban should connect
        id_to_agent = {a["_id"]: a for a in agents}
        for e in school_edges:
            a = id_to_agent[e.source]
            b = id_to_agent[e.target]
            assert a["state"] == b["state"]
            assert a["urban_rural"] == b["urban_rural"]
            # Both should have school-age dependents
            for agent in (a, b):
                has_school = any(
                    d.get("school_status")
                    in ("elementary", "middle_school", "high_school")
                    for d in agent.get("dependents", [])
                )
                assert has_school

    def test_no_self_edges(self):
        agents, agent_ids = _make_household_agents(10)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)
        for e in edges:
            assert e.source != e.target

    def test_no_duplicate_edges(self):
        agents, agent_ids = _make_household_agents(10)
        rng = random.Random(42)
        edges = _generate_structural_edges(agents, agent_ids, rng)
        pairs = set()
        for e in edges:
            pair = (min(e.source, e.target), max(e.source, e.target))
            assert pair not in pairs, f"Duplicate edge: {pair}"
            pairs.add(pair)


class TestEdgeModelEnhancements:
    def test_edge_structural_default(self):
        e = Edge(source="a", target="b", weight=0.5, edge_type="peer")
        assert e.structural is False
        assert e.context is None

    def test_edge_structural_to_dict(self):
        e = Edge(
            source="a",
            target="b",
            weight=0.5,
            edge_type="partner",
            structural=True,
            context="household",
        )
        d = e.to_dict()
        assert d["structural"] is True
        assert d["context"] == "household"

    def test_edge_non_structural_to_dict_omits_fields(self):
        e = Edge(source="a", target="b", weight=0.5, edge_type="peer")
        d = e.to_dict()
        assert "structural" not in d
        assert "context" not in d


class TestNetworkConfigEnhancements:
    def test_degree_distribution_target_default(self):
        config = NetworkConfig()
        assert config.degree_distribution_target is None
        assert config.power_law_exponent == 2.5

    def test_identity_clustering_defaults(self):
        config = NetworkConfig()
        assert config.identity_clustering_attributes == []
        assert config.identity_clustering_boost == 1.5

    def test_identity_clustering_config(self):
        config = NetworkConfig(
            identity_clustering_attributes=[
                "political_orientation",
                "religious_affiliation",
            ],
            identity_clustering_boost=2.0,
        )
        assert len(config.identity_clustering_attributes) == 2
        assert config.identity_clustering_boost == 2.0
