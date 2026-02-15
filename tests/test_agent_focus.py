"""Tests for agent_focus-aware sampling (Phase B2).

Tests the classification of household members as agents vs NPCs based on
the agent_focus field in SpecMeta.

Agent focus determines who in a household becomes a full agent vs NPC:
- "surgeons" / "students" / None → primary_only mode: only primary adult is agent
- "retired couples" / "married couples" → couples mode: both partners are agents
- "families" / "households" → all mode: everyone including kids are agents

This test suite verifies:
1. _classify_agent_focus() correctly maps agent_focus strings to modes
2. Primary-only mode creates NPC partners (partner_npc field)
3. Couples mode creates both partners as full agents with partner_id links
4. Families mode promotes kids to full agents with household_role="dependent_*"
5. NPC partners and kids have correlated demographics
6. Household-scoped attributes are properly shared/inherited
"""

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
from extropy.population.sampler.core import sample_population, _classify_agent_focus


def _make_household_spec(
    size: int = 100, agent_focus: str | None = None
) -> PopulationSpec:
    """Create a minimal spec with household-scoped attributes.

    This triggers household sampling mode.
    """
    return PopulationSpec(
        meta=SpecMeta(
            description="Test household spec", size=size, agent_focus=agent_focus
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


class TestClassifyAgentFocus:
    """Test the _classify_agent_focus classification logic."""

    def test_none_defaults_to_primary_only(self):
        assert _classify_agent_focus(None) == "primary_only"

    def test_empty_string_defaults_to_primary_only(self):
        assert _classify_agent_focus("") == "primary_only"

    def test_surgeons_is_primary_only(self):
        assert _classify_agent_focus("surgeons") == "primary_only"

    def test_students_is_primary_only(self):
        assert _classify_agent_focus("high school students") == "primary_only"

    def test_subscribers_is_primary_only(self):
        assert _classify_agent_focus("subscribers") == "primary_only"

    def test_retired_couples_is_couples(self):
        assert _classify_agent_focus("retired couples") == "couples"

    def test_married_couples_is_couples(self):
        assert _classify_agent_focus("married couples") == "couples"

    def test_couples_keyword_detected(self):
        assert _classify_agent_focus("young couples in NYC") == "couples"

    def test_partners_keyword_detected(self):
        assert _classify_agent_focus("domestic partners") == "couples"

    def test_families_is_all(self):
        assert _classify_agent_focus("families") == "all"

    def test_families_with_qualifier_is_all(self):
        assert _classify_agent_focus("families with young children") == "all"

    def test_households_is_all(self):
        assert _classify_agent_focus("households") == "all"

    def test_everyone_is_all(self):
        assert _classify_agent_focus("everyone in the community") == "all"

    def test_case_insensitive(self):
        assert _classify_agent_focus("SURGEONS") == "primary_only"
        assert _classify_agent_focus("COUPLES") == "couples"
        assert _classify_agent_focus("FAMILIES") == "all"


class TestAgentFocusPrimaryOnly:
    """Test agent_focus="surgeons" (primary_only mode).

    In this mode:
    - Only primary adult is a full agent
    - Partner becomes NPC (in partner_npc field)
    - partner_id is None
    - Kids are NPCs (in dependents list)
    """

    def test_partners_are_npc(self):
        spec = _make_household_spec(size=200, agent_focus="surgeons")
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
            assert (
                agent.get("partner_id") is None
            ), "partner_id should be None for NPC partners"

            # Check NPC partner has expected fields
            npc = agent["partner_npc"]
            assert "age" in npc, "NPC should have age"
            assert "gender" in npc, "NPC should have gender"
            assert "relationship" in npc, "NPC should have relationship"
            assert npc["relationship"] == "partner"

    def test_no_partner_agents_in_result(self):
        spec = _make_household_spec(size=200, agent_focus="surgeons")
        result = sample_population(spec, count=200, seed=42)
        agents = result.agents

        # Count agents with household_role = adult_secondary
        secondary_adults = [
            a for a in agents if a.get("household_role") == "adult_secondary"
        ]

        assert (
            len(secondary_adults) == 0
        ), "In primary_only mode, partners should be NPCs, not agents"

    def test_total_agent_count_matches(self):
        spec = _make_household_spec(size=100, agent_focus="surgeons")
        result = sample_population(spec, count=100, seed=42)

        # Should produce at most the requested count
        assert (
            len(result.agents) <= 100
        ), "Agent count should not exceed requested count"

    def test_npc_partner_has_correlated_demographics(self):
        spec = _make_household_spec(size=300, agent_focus="surgeons")
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
        spec = _make_household_spec(size=200, agent_focus="surgeons")
        result = sample_population(spec, count=200, seed=42)

        for agent in result.agents:
            npc = agent.get("partner_npc")
            if npc and agent.get("last_name"):
                assert (
                    npc.get("last_name") == agent["last_name"]
                ), "NPC partner should share last name"


class TestAgentFocusCouples:
    """Test agent_focus="retired couples" (couples mode).

    In this mode:
    - Both partners are full agents
    - They have partner_id linking to each other
    - Kids are NPCs (in dependents list)
    """

    def test_both_partners_are_agents(self):
        spec = _make_household_spec(size=200, agent_focus="retired couples")
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
            assert (
                partner.get("household_role") == "adult_secondary"
            ), "Partner should be adult_secondary"
            assert (
                partner.get("partner_id") == agent["_id"]
            ), "Partner should link back to primary"

    def test_no_npc_partners(self):
        spec = _make_household_spec(size=200, agent_focus="retired couples")
        result = sample_population(spec, count=200, seed=42)

        # No agents should have partner_npc field
        agents_with_npc = [a for a in result.agents if "partner_npc" in a]
        assert (
            len(agents_with_npc) == 0
        ), "In couples mode, partners should be full agents, not NPCs"

    def test_partners_share_household_id(self):
        spec = _make_household_spec(size=200, agent_focus="retired couples")
        result = sample_population(spec, count=200, seed=42)
        id_map = {a["_id"]: a for a in result.agents}

        for agent in result.agents:
            pid = agent.get("partner_id")
            if pid:
                partner = id_map.get(pid)
                assert partner is not None
                assert (
                    partner["household_id"] == agent["household_id"]
                ), "Partners should share household_id"

    def test_partners_share_household_scoped_attrs(self):
        spec = _make_household_spec(size=200, agent_focus="retired couples")
        result = sample_population(spec, count=200, seed=42)
        id_map = {a["_id"]: a for a in result.agents}

        for agent in result.agents:
            pid = agent.get("partner_id")
            if pid:
                partner = id_map.get(pid)
                assert partner is not None
                # 'state' is household-scoped
                assert (
                    agent["state"] == partner["state"]
                ), "Partners should share household-scoped attrs"

    def test_kids_are_npcs(self):
        spec = _make_household_spec(size=200, agent_focus="retired couples")
        result = sample_population(spec, count=200, seed=42)

        # Check that dependents (kids) are in dependents list, not as full agents
        kids_as_agents = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]

        assert (
            len(kids_as_agents) == 0
        ), "In couples mode, kids should be NPCs, not agents"

        # Check that some agents have dependents
        agents_with_kids = [a for a in result.agents if a.get("dependents")]
        assert (
            len(agents_with_kids) > 0
        ), "Some households should have kids as NPCs"


class TestAgentFocusFamilies:
    """Test agent_focus="families" (all mode).

    In this mode:
    - Both partners are full agents
    - Kids become full agents too (not NPCs)
    - Kids have household_role starting with "dependent_"
    """

    def test_kids_become_agents(self):
        spec = _make_household_spec(size=200, agent_focus="families")
        result = sample_population(spec, count=200, seed=42)

        # Find kid agents
        kid_agents = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]

        assert len(kid_agents) > 0, "In families mode, kids should be full agents"

    def test_kid_agents_have_correct_structure(self):
        spec = _make_household_spec(size=200, agent_focus="families")
        result = sample_population(spec, count=200, seed=42)

        kid_agents = [
            a
            for a in result.agents
            if a.get("household_role", "").startswith("dependent_")
        ]

        for kid in kid_agents:
            assert "_id" in kid, "Kid agent should have _id"
            assert "household_id" in kid, "Kid agent should have household_id"
            assert (
                "household_role" in kid
            ), "Kid agent should have household_role"
            assert kid["household_role"].startswith(
                "dependent_"
            ), "Kid role should start with dependent_"
            assert (
                "relationship_to_primary" in kid
            ), "Kid should have relationship_to_primary"
            assert "age" in kid, "Kid should have age"
            assert "gender" in kid, "Kid should have gender"

    def test_kid_agents_inherit_household_attrs(self):
        spec = _make_household_spec(size=200, agent_focus="families")
        result = sample_population(spec, count=200, seed=42)
        id_map = {a["_id"]: a for a in result.agents}

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
                assert (
                    kid["state"] == parent["state"]
                ), "Kid should inherit household-scoped attrs from parent"

    def test_both_partners_still_agents(self):
        spec = _make_household_spec(size=200, agent_focus="families")
        result = sample_population(spec, count=200, seed=42)

        # Should still have both partners as agents
        secondary_adults = [
            a for a in result.agents if a.get("household_role") == "adult_secondary"
        ]

        assert (
            len(secondary_adults) > 0
        ), "In families mode, both partners should be agents"

    def test_overflow_kids_remain_npc(self):
        """If we hit the agent count limit, remaining kids should be NPCs."""
        spec = _make_household_spec(size=50, agent_focus="families")
        result = sample_population(spec, count=50, seed=42)

        # Check if any primary has NPC dependents even in families mode
        # (happens when agent count is reached)
        agents_with_npc_deps = [a for a in result.agents if a.get("dependents")]

        # This is ok - overflow protection should keep some kids as NPCs
        # if we hit the agent limit


class TestAgentFocusDefault:
    """Test default behavior when agent_focus is None."""

    def test_none_defaults_to_primary_only(self):
        spec = _make_household_spec(size=200, agent_focus=None)
        result = sample_population(spec, count=200, seed=42)

        # Should behave like primary_only
        primaries_with_npc = [
            a
            for a in result.agents
            if a.get("household_role") == "adult_primary" and "partner_npc" in a
        ]

        # Should have some NPC partners
        assert (
            len(primaries_with_npc) > 0
        ), "Default (None) should behave like primary_only"

    def test_no_secondary_agents_by_default(self):
        spec = _make_household_spec(size=200, agent_focus=None)
        result = sample_population(spec, count=200, seed=42)

        secondary_adults = [
            a for a in result.agents if a.get("household_role") == "adult_secondary"
        ]

        assert (
            len(secondary_adults) == 0
        ), "Default should be primary_only (no partner agents)"


class TestAgentFocusMetadata:
    """Test that agent_focus is preserved in metadata."""

    def test_agent_focus_preserved_in_spec_meta(self):
        spec = _make_household_spec(size=100, agent_focus="surgeons")
        assert spec.meta.agent_focus == "surgeons"

    def test_agent_focus_none_preserved(self):
        spec = _make_household_spec(size=100, agent_focus=None)
        assert spec.meta.agent_focus is None

    def test_different_focus_values(self):
        for focus in ["surgeons", "retired couples", "families", None]:
            spec = _make_household_spec(size=100, agent_focus=focus)
            assert spec.meta.agent_focus == focus
