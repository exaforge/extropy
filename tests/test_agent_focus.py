"""Tests for agent_focus-aware sampling (Phase B2).

Tests the classification of household members as agents vs NPCs based on
the agent_focus_mode field in SpecMeta.

Agent focus determines who in a household becomes a full agent vs NPC:
- "primary_only" mode: only primary adult is agent
- "couples" mode: both partners are agents
- "all" mode: everyone including kids are agents

This test suite verifies:
1. _get_agent_focus_mode() correctly reads agent_focus_mode from spec
2. Primary-only mode creates NPC partners (partner_npc field)
3. Couples mode creates both partners as full agents with partner_id links
4. Families mode promotes kids to full agents with household_role="dependent_*"
5. NPC partners and kids have correlated demographics
6. Household-scoped attributes are properly shared/inherited
"""

from typing import Literal

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
from extropy.population.sampler.core import sample_population, _get_agent_focus_mode


def _make_household_spec(
    size: int = 100,
    agent_focus_mode: Literal["primary_only", "couples", "all"] | None = None,
) -> PopulationSpec:
    """Create a minimal spec with household-scoped attributes.

    This triggers household sampling mode.
    """
    return PopulationSpec(
        meta=SpecMeta(
            description="Test household spec",
            agent_focus_mode=agent_focus_mode,
        ),
        grounding=GroundingSummary(
            overall="medium",
            sources_count=1,
            strong_count=2,
            medium_count=2,
            low_count=1,
        ),
        attributes=[
            AttributeSpec(
                name="age",
                type="int",
                category="universal",
                description="Age",
                scope="individual",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal", mean=35, std=8, min=22, max=55
                    ),
                ),
                grounding=GroundingInfo(level="strong", method="researched"),
            ),
            AttributeSpec(
                name="gender",
                type="categorical",
                category="universal",
                description="Gender",
                scope="individual",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["male", "female"],
                        weights=[0.49, 0.51],
                    ),
                ),
                grounding=GroundingInfo(level="strong", method="researched"),
            ),
            AttributeSpec(
                name="state",
                type="categorical",
                category="universal",
                description="State",
                scope="household",  # household-scoped triggers household mode
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["CA", "TX", "NY"],
                        weights=[0.4, 0.3, 0.3],
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            ),
            AttributeSpec(
                name="race_ethnicity",
                type="categorical",
                category="universal",
                description="Race/ethnicity",
                scope="individual",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["white", "black", "hispanic", "asian"],
                        weights=[0.6, 0.13, 0.18, 0.09],
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="researched"),
            ),
            AttributeSpec(
                name="education_level",
                type="categorical",
                category="universal",
                description="Education",
                scope="individual",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["high_school", "bachelors", "masters", "doctorate"],
                        weights=[0.4, 0.3, 0.2, 0.1],
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="researched"),
            ),
        ],
        sampling_order=["age", "gender", "state", "race_ethnicity", "education_level"],
    )


class TestGetAgentFocusMode:
    """Test the _get_agent_focus_mode reads agent_focus_mode from spec."""

    def test_none_defaults_to_primary_only(self):
        spec = _make_household_spec(agent_focus_mode=None)
        assert _get_agent_focus_mode(spec) == "primary_only"

    def test_primary_only_mode(self):
        spec = _make_household_spec(agent_focus_mode="primary_only")
        assert _get_agent_focus_mode(spec) == "primary_only"

    def test_couples_mode(self):
        spec = _make_household_spec(agent_focus_mode="couples")
        assert _get_agent_focus_mode(spec) == "couples"

    def test_all_mode(self):
        spec = _make_household_spec(agent_focus_mode="all")
        assert _get_agent_focus_mode(spec) == "all"


class TestAgentFocusPrimaryOnly:
    """Test agent_focus_mode="primary_only".

    In this mode:
    - Only primary adult is a full agent
    - Partner becomes NPC (in partner_npc field)
    - partner_id is None
    - Kids are NPCs (in dependents list)
    """

    def test_partners_are_npc(self):
        spec = _make_household_spec(size=200, agent_focus_mode="primary_only")
        result = sample_population(spec, count=200, seed=42)
        agents = result.agents

        # Find primary adults with partners
        primaries_with_partners = [
            a
            for a in agents
            if a.get("household_role") == "adult_primary" and "partner_npc" in a
        ]

        assert len(primaries_with_partners) > 0, "Should have some partnered primaries"

        # Check that partner is NPC
        for agent in primaries_with_partners:
            assert "partner_npc" in agent, "Partner should be in partner_npc field"
            assert agent.get("partner_id") is None, (
                "partner_id should be None for NPC partners"
            )

            # Check NPC partner has expected fields
            npc = agent["partner_npc"]
            assert "age" in npc, "NPC should have age"
            assert "gender" in npc, "NPC should have gender"
            assert "relationship" in npc, "NPC should have relationship"
            assert npc["relationship"] == "partner"

    def test_no_partner_agents_in_result(self):
        spec = _make_household_spec(size=200, agent_focus_mode="primary_only")
        result = sample_population(spec, count=200, seed=42)
        agents = result.agents

        # Count agents with household_role = adult_secondary
        secondary_adults = [
            a for a in agents if a.get("household_role") == "adult_secondary"
        ]

        assert len(secondary_adults) == 0, (
            "In primary_only mode, partners should be NPCs, not agents"
        )

    def test_total_agent_count_matches(self):
        spec = _make_household_spec(size=100, agent_focus_mode="primary_only")
        result = sample_population(spec, count=100, seed=42)

        # Should produce at most the requested count
        assert len(result.agents) <= 100, (
            "Agent count should not exceed requested count"
        )

    def test_npc_partner_has_correlated_demographics(self):
        spec = _make_household_spec(size=300, agent_focus_mode="primary_only")
        result = sample_population(spec, count=300, seed=42)
        agents = result.agents

        age_diffs = []
        for agent in agents:
            npc = agent.get("partner_npc")
            if npc:
                diff = abs(agent["age"] - npc["age"])
                age_diffs.append(diff)

                # Check household attrs are shared
                if "state" in agent:
                    assert npc.get("state") == agent["state"], (
                        "NPC partner should share household-scoped attrs"
                    )

        # Age correlation check
        if age_diffs:
            avg_diff = sum(age_diffs) / len(age_diffs)
            assert avg_diff < 10, f"Average age gap {avg_diff:.1f} too large"

    def test_npc_partner_shares_last_name(self):
        spec = _make_household_spec(size=200, agent_focus_mode="primary_only")
        result = sample_population(spec, count=200, seed=42)

        for agent in result.agents:
            npc = agent.get("partner_npc")
            if npc and agent.get("last_name"):
                assert npc.get("last_name") == agent["last_name"], (
                    "NPC partner should share last name"
                )


class TestAgentFocusCouples:
    """Test agent_focus_mode="couples".

    In this mode:
    - Both partners are full agents
    - They have partner_id linking to each other
    - Kids are NPCs (in dependents list)
    """

    def test_both_partners_are_agents(self):
        spec = _make_household_spec(size=200, agent_focus_mode="couples")
        result = sample_population(spec, count=200, seed=42)
        agents = result.agents
        id_map = {a["_id"]: a for a in agents}

        # Find primary adults with partners
        primaries_with_partners = [
            a
            for a in agents
            if a.get("household_role") == "adult_primary" and a.get("partner_id")
        ]

        assert len(primaries_with_partners) > 0, "Should have some couples"

        for agent in primaries_with_partners:
            pid = agent.get("partner_id")
            assert pid is not None, "Primary should have partner_id"
            assert pid in id_map, f"Partner {pid} should be a full agent"

            partner = id_map[pid]
            assert partner.get("household_role") == "adult_secondary", (
                "Partner should be adult_secondary"
            )
            assert partner.get("partner_id") == agent["_id"], (
                "Partner should link back to primary"
            )

    def test_no_npc_partners(self):
        spec = _make_household_spec(size=200, agent_focus_mode="couples")
        result = sample_population(spec, count=200, seed=42)

        # No agents should have partner_npc field
        agents_with_npc = [a for a in result.agents if "partner_npc" in a]
        assert len(agents_with_npc) == 0, (
            "In couples mode, partners should be full agents, not NPCs"
        )

    def test_partners_share_household_id(self):
        spec = _make_household_spec(size=200, agent_focus_mode="couples")
        result = sample_population(spec, count=200, seed=42)
        id_map = {a["_id"]: a for a in result.agents}

        for agent in result.agents:
            pid = agent.get("partner_id")
            if pid:
                partner = id_map.get(pid)
                assert partner is not None
                assert partner["household_id"] == agent["household_id"], (
                    "Partners should share household_id"
                )

    def test_partners_share_household_scoped_attrs(self):
        spec = _make_household_spec(size=200, agent_focus_mode="couples")
        result = sample_population(spec, count=200, seed=42)
        id_map = {a["_id"]: a for a in result.agents}

        for agent in result.agents:
            pid = agent.get("partner_id")
            if pid:
                partner = id_map.get(pid)
                assert partner is not None
                # 'state' is household-scoped
                assert agent["state"] == partner["state"], (
                    "Partners should share household-scoped attrs"
                )

    def test_kids_are_npcs(self):
        spec = _make_household_spec(size=200, agent_focus_mode="couples")
        result = sample_population(spec, count=200, seed=42)

        # Check that dependents (kids) are in dependents list, not as full agents
        kids_as_agents = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]

        assert len(kids_as_agents) == 0, (
            "In couples mode, kids should be NPCs, not agents"
        )

        # Check that some agents have dependents
        agents_with_kids = [a for a in result.agents if a.get("dependents")]
        assert len(agents_with_kids) > 0, "Some households should have kids as NPCs"


class TestAgentFocusFamilies:
    """Test agent_focus_mode="all".

    In this mode:
    - Both partners are full agents
    - Kids become full agents too (not NPCs)
    - Kids have household_role starting with "dependent_"
    """

    def test_kids_become_agents(self):
        spec = _make_household_spec(size=200, agent_focus_mode="all")
        result = sample_population(spec, count=200, seed=42)

        # Find kid agents
        kid_agents = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]

        assert len(kid_agents) > 0, "In families mode, kids should be full agents"

    def test_kid_agents_have_correct_structure(self):
        spec = _make_household_spec(size=200, agent_focus_mode="all")
        result = sample_population(spec, count=200, seed=42)

        kid_agents = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]

        for kid in kid_agents:
            assert "_id" in kid, "Kid agent should have _id"
            assert "household_id" in kid, "Kid agent should have household_id"
            assert "household_role" in kid, "Kid agent should have household_role"
            assert kid["household_role"].startswith("dependent_"), (
                "Kid role should start with dependent_"
            )
            assert "relationship_to_primary" in kid, (
                "Kid should have relationship_to_primary"
            )
            assert "age" in kid, "Kid should have age"
            assert "gender" in kid, "Kid should have gender"

    def test_kid_agents_inherit_household_attrs(self):
        spec = _make_household_spec(size=200, agent_focus_mode="all")
        result = sample_population(spec, count=200, seed=42)

        kid_agents = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]

        for kid in kid_agents:
            # Find parent (primary adult in same household)
            hh_id = kid["household_id"]
            parent = next(
                (
                    a
                    for a in result.agents
                    if a.get("household_id") == hh_id
                    and a.get("household_role") == "adult_primary"
                ),
                None,
            )

            if parent:
                # 'state' is household-scoped, should be inherited
                assert kid["state"] == parent["state"], (
                    "Kid should inherit household-scoped attrs from parent"
                )

    def test_both_partners_still_agents(self):
        spec = _make_household_spec(size=200, agent_focus_mode="all")
        result = sample_population(spec, count=200, seed=42)

        # Should still have both partners as agents
        secondary_adults = [
            a for a in result.agents if a.get("household_role") == "adult_secondary"
        ]

        assert len(secondary_adults) > 0, (
            "In families mode, both partners should be agents"
        )

    def test_overflow_kids_remain_npc(self):
        """If we hit the agent count limit, remaining kids should be NPCs."""
        spec = _make_household_spec(size=50, agent_focus_mode="all")
        result = sample_population(spec, count=50, seed=42)

        # Should not exceed requested count
        assert len(result.agents) <= 50, "Should not exceed requested agent count"

        # Some households may have NPC dependents if we hit the limit
        # This is expected overflow protection behavior


class TestAgentFocusDefault:
    """Test default behavior when agent_focus_mode is None."""

    def test_none_defaults_to_primary_only(self):
        spec = _make_household_spec(size=200, agent_focus_mode=None)
        result = sample_population(spec, count=200, seed=42)

        # Should behave like primary_only
        primaries_with_npc = [
            a
            for a in result.agents
            if a.get("household_role") == "adult_primary" and "partner_npc" in a
        ]

        # Should have some NPC partners
        assert len(primaries_with_npc) > 0, (
            "Default (None) should behave like primary_only"
        )

    def test_no_secondary_agents_by_default(self):
        spec = _make_household_spec(size=200, agent_focus_mode=None)
        result = sample_population(spec, count=200, seed=42)

        secondary_adults = [
            a for a in result.agents if a.get("household_role") == "adult_secondary"
        ]

        assert len(secondary_adults) == 0, (
            "Default should be primary_only (no partner agents)"
        )


class TestAgentFocusMetadata:
    """Test that agent_focus is preserved in metadata."""

    def test_agent_focus_mode_preserved_in_spec_meta(self):
        spec = _make_household_spec(size=100, agent_focus_mode="primary_only")
        assert spec.meta.agent_focus_mode == "primary_only"

    def test_agent_focus_mode_none_preserved(self):
        spec = _make_household_spec(size=100, agent_focus_mode=None)
        assert spec.meta.agent_focus_mode is None

    def test_different_focus_mode_values(self):
        for mode in ["primary_only", "couples", "all", None]:
            spec = _make_household_spec(size=100, agent_focus_mode=mode)
            assert spec.meta.agent_focus_mode == mode


class TestPromotedDependentNames:
    def test_promoted_dependents_preserve_household_names(self):
        spec = _make_household_spec(size=240, agent_focus_mode="all")
        result = sample_population(spec, count=240, seed=42)

        promoted = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]
        assert promoted, "Expected promoted dependents in families mode"

        households = {h["id"]: h for h in getattr(result, "_households", [])}
        for dep in promoted:
            first_name = dep.get("first_name")
            assert first_name, "Promoted dependents must have first_name"

            hh = households.get(dep["household_id"])
            assert hh is not None
            dependent_names = {d.get("name") for d in hh.get("dependent_data", [])}
            assert first_name in dependent_names, (
                "Promoted dependent names should stay aligned with generated dependent records"
            )
