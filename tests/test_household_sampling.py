"""Tests for household-based sampling (Phase B1)."""

from extropy.core.models.population import (
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    AttributeSpec,
    SamplingConfig,
    GroundingInfo,
    NormalDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    HouseholdType,
    HouseholdConfig,
    Dependent,
    STANDARD_PERSONALITY_ATTRIBUTES,
)
from extropy.population.sampler.core import sample_population
from extropy.population.sampler.households import (
    sample_household_type,
    household_needs_partner,
    household_needs_kids,
    correlate_partner_attribute,
    generate_dependents,
    estimate_household_count,
)

import random

# US defaults — identical behavior to old hardcoded constants
_DEFAULT_CONFIG = HouseholdConfig()


def _make_household_spec(
    size: int = 100, agent_focus: str | None = "couples"
) -> PopulationSpec:
    """Create a minimal spec with household-scoped attributes."""
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
                scope="partner_correlated",  # Uses gaussian offset for partners
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal", mean=40, std=12, min=18, max=85
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
                scope="household",  # shared within household
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
                scope="partner_correlated",  # Uses per-group rates
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
                scope="partner_correlated",  # Correlated between partners
                correlation_rate=0.6,  # ~60% same education
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


def _make_individual_spec(size: int = 50) -> PopulationSpec:
    """Create a spec with NO household-scoped attributes (legacy mode)."""
    return PopulationSpec(
        meta=SpecMeta(description="Test individual spec", size=size),
        grounding=GroundingSummary(
            overall="medium",
            sources_count=1,
            strong_count=1,
            medium_count=1,
            low_count=0,
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
                        type="normal", mean=40, std=12, min=18, max=85
                    ),
                ),
                grounding=GroundingInfo(level="strong", method="researched"),
            ),
            AttributeSpec(
                name="gender",
                type="categorical",
                category="universal",
                description="Gender",
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
        ],
        sampling_order=["age", "gender"],
    )


def _make_household_consistency_spec(size: int = 200) -> PopulationSpec:
    """Spec with household-related attributes for reconciliation tests."""
    return PopulationSpec(
        meta=SpecMeta(description="Household consistency spec", size=size),
        grounding=GroundingSummary(
            overall="medium",
            sources_count=1,
            strong_count=2,
            medium_count=2,
            low_count=2,
        ),
        attributes=[
            AttributeSpec(
                name="age",
                type="int",
                category="universal",
                description="Age",
                scope="partner_correlated",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal", mean=38, std=12, min=18, max=85
                    ),
                ),
                grounding=GroundingInfo(level="strong", method="researched"),
            ),
            AttributeSpec(
                name="gender",
                type="categorical",
                category="universal",
                description="Gender",
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
                name="marital_status",
                type="categorical",
                category="universal",
                description="Marital status",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["Single", "Married", "Divorced", "Widowed"],
                        weights=[0.62, 0.25, 0.09, 0.04],
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            ),
            AttributeSpec(
                name="household_size",
                type="int",
                category="universal",
                description="Number of people in household",
                scope="household",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=NormalDistribution(
                        type="normal", mean=1.3, std=0.6, min=1, max=6
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            ),
            AttributeSpec(
                name="has_children",
                type="boolean",
                category="universal",
                description="Whether the agent has children under 18",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=BooleanDistribution(
                        type="boolean",
                        probability_true=0.35,
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            ),
        ],
        sampling_order=[
            "age",
            "gender",
            "marital_status",
            "household_size",
            "has_children",
        ],
    )


class TestHouseholdModels:
    def test_household_type_enum(self):
        assert HouseholdType.SINGLE.value == "single"
        assert HouseholdType.COUPLE_WITH_KIDS.value == "couple_with_kids"

    def test_dependent_model(self):
        dep = Dependent(
            name="Child",
            age=10,
            gender="male",
            relationship="son",
            school_status="elementary",
        )
        assert dep.age == 10
        assert dep.school_status == "elementary"

    def test_standard_personality_attributes(self):
        assert "conformity" in STANDARD_PERSONALITY_ATTRIBUTES
        assert "extraversion" in STANDARD_PERSONALITY_ATTRIBUTES
        assert len(STANDARD_PERSONALITY_ATTRIBUTES) == 6


class TestHouseholdSamplingHelpers:
    def test_sample_household_type_returns_valid(self):
        rng = random.Random(42)
        for age in [22, 35, 55, 70]:
            htype = sample_household_type(age, rng, _DEFAULT_CONFIG)
            assert isinstance(htype, HouseholdType)

    def test_household_needs_partner(self):
        assert not household_needs_partner(HouseholdType.SINGLE)
        assert household_needs_partner(HouseholdType.COUPLE)
        assert not household_needs_partner(HouseholdType.SINGLE_PARENT)
        assert household_needs_partner(HouseholdType.COUPLE_WITH_KIDS)
        assert household_needs_partner(HouseholdType.MULTI_GENERATIONAL)

    def test_household_needs_kids(self):
        assert not household_needs_kids(HouseholdType.SINGLE)
        assert not household_needs_kids(HouseholdType.COUPLE)
        assert household_needs_kids(HouseholdType.SINGLE_PARENT)
        assert household_needs_kids(HouseholdType.COUPLE_WITH_KIDS)
        assert household_needs_kids(HouseholdType.MULTI_GENERATIONAL)

    def test_correlate_age(self):
        """Age correlation uses gaussian offset from HouseholdConfig."""
        rng = random.Random(42)
        partner_age = correlate_partner_attribute(
            "age", "int", 35, None, rng, _DEFAULT_CONFIG
        )
        assert isinstance(partner_age, int)
        assert partner_age >= 18

    def test_correlate_race_same_rate(self):
        """Race/ethnicity uses per-group rates from HouseholdConfig."""
        rng = random.Random(42)
        same_count = 0
        trials = 500
        for _ in range(trials):
            result = correlate_partner_attribute(
                "race_ethnicity",
                "categorical",
                "white",
                None,  # Uses per-group rates from config
                rng,
                _DEFAULT_CONFIG,
                available_options=["white", "black", "hispanic"],
            )
            if result == "white":
                same_count += 1
        # Expect ~90% same-race for white
        rate = same_count / trials
        assert 0.80 < rate < 0.97, f"Same-race rate {rate:.2f} outside expected range"

    def test_correlate_education_assortative(self):
        """Education uses explicit correlation_rate."""
        rng = random.Random(42)
        same_count = 0
        trials = 500
        for _ in range(trials):
            result = correlate_partner_attribute(
                "education_level",
                "categorical",
                "bachelors",
                0.6,  # Explicit correlation rate
                rng,
                _DEFAULT_CONFIG,
                available_options=["high_school", "bachelors", "masters", "doctorate"],
            )
            if result == "bachelors":
                same_count += 1
        rate = same_count / trials
        # Expect ~60% same education
        assert 0.50 < rate < 0.75, f"Assortative rate {rate:.2f} outside expected range"

    def test_correlate_with_default_rate(self):
        """Unknown attributes use default_same_group_rate when no explicit rate."""
        rng = random.Random(42)
        same_count = 0
        trials = 500
        for _ in range(trials):
            result = correlate_partner_attribute(
                "some_attribute",
                "categorical",
                "value_a",
                None,  # No explicit rate, uses default
                rng,
                _DEFAULT_CONFIG,
                available_options=["value_a", "value_b", "value_c"],
            )
            if result == "value_a":
                same_count += 1
        rate = same_count / trials
        # Expect ~85% (default_same_group_rate)
        assert 0.75 < rate < 0.95, f"Default rate {rate:.2f} outside expected range"

    def test_correlate_uses_semantic_type_for_age_policy(self):
        """Semantic metadata should trigger gaussian age policy without name matching."""
        rng = random.Random(42)
        values = [
            correlate_partner_attribute(
                "years_old",
                "int",
                35,
                None,
                rng,
                _DEFAULT_CONFIG,
                semantic_type="age",
            )
            for _ in range(200)
        ]
        assert all(v >= _DEFAULT_CONFIG.min_adult_age for v in values)
        assert len(set(values)) > 1

    def test_correlate_uses_identity_type_for_group_rate_policy(self):
        """Identity metadata should trigger same-group rate without name matching."""
        rng = random.Random(42)
        same_count = 0
        trials = 500
        for _ in range(trials):
            result = correlate_partner_attribute(
                "ethnic_group",
                "categorical",
                "white",
                None,
                rng,
                _DEFAULT_CONFIG,
                available_options=["white", "black", "hispanic"],
                identity_type="race_ethnicity",
            )
            if result == "white":
                same_count += 1
        rate = same_count / trials
        assert 0.80 < rate < 0.97, f"Same-group rate {rate:.2f} outside expected range"

    def test_correlate_respects_explicit_policy_override(self):
        """Explicit partner policy should override inferred/default behavior."""
        rng = random.Random(42)
        values = [
            correlate_partner_attribute(
                "custom_numeric",
                "int",
                40,
                None,
                rng,
                _DEFAULT_CONFIG,
                partner_correlation_policy="gaussian_offset",
            )
            for _ in range(100)
        ]
        assert all(v >= _DEFAULT_CONFIG.min_adult_age for v in values)
        assert len(set(values)) > 1

    def test_generate_dependents_no_kids(self):
        rng = random.Random(42)
        deps = generate_dependents(
            HouseholdType.COUPLE, 2, 2, 40, rng, _DEFAULT_CONFIG, name_config=None
        )
        assert len(deps) == 0

    def test_generate_dependents_with_kids(self):
        rng = random.Random(42)
        deps = generate_dependents(
            HouseholdType.COUPLE_WITH_KIDS,
            4,
            2,
            40,
            rng,
            _DEFAULT_CONFIG,
            name_config=None,
        )
        assert len(deps) == 2
        for d in deps:
            assert isinstance(d, Dependent)
            assert d.age >= 0
            assert d.relationship in ("son", "daughter")

    def test_generate_dependents_multi_generational(self):
        rng = random.Random(42)
        deps = generate_dependents(
            HouseholdType.MULTI_GENERATIONAL,
            4,
            2,
            45,
            rng,
            _DEFAULT_CONFIG,
            name_config=None,
        )
        assert len(deps) == 2
        relationships = [d.relationship for d in deps]
        assert any(r in ("father", "mother") for r in relationships)

    def test_estimate_household_count(self):
        assert estimate_household_count(100, _DEFAULT_CONFIG) == 40
        assert estimate_household_count(1, _DEFAULT_CONFIG) == 1


class TestHouseholdPopulationSampling:
    def test_household_ids_assigned(self):
        spec = _make_household_spec(size=50)
        result = sample_population(spec, count=50, seed=42)
        agents = result.agents
        agents_with_hh = [a for a in agents if a.get("household_id")]
        assert len(agents_with_hh) == len(agents), "All agents should have household_id"

    def test_partner_agents_share_household(self):
        spec = _make_household_spec(size=100)
        result = sample_population(spec, count=100, seed=42)
        agents = result.agents
        id_map = {a["_id"]: a for a in agents}
        for agent in agents:
            pid = agent.get("partner_id")
            if pid:
                partner = id_map.get(pid)
                assert partner is not None, f"Partner {pid} not found"
                assert partner["household_id"] == agent["household_id"]

    def test_household_scoped_attrs_shared(self):
        spec = _make_household_spec(size=100)
        result = sample_population(spec, count=100, seed=42)
        agents = result.agents
        id_map = {a["_id"]: a for a in agents}
        for agent in agents:
            pid = agent.get("partner_id")
            if pid:
                partner = id_map.get(pid)
                assert partner is not None
                # 'state' is household-scoped, should be shared
                assert agent["state"] == partner["state"], (
                    f"Household-scoped attr 'state' should match: "
                    f"{agent['state']} != {partner['state']}"
                )

    def test_dependent_count_matches(self):
        spec = _make_household_spec(size=50)
        result = sample_population(spec, count=50, seed=42)
        for agent in result.agents:
            deps = agent.get("dependents", [])
            assert isinstance(deps, list)
            # Each dependent should have required fields
            for d in deps:
                assert "age" in d
                assert "gender" in d
                assert "relationship" in d

    def test_single_households_no_partner(self):
        spec = _make_household_spec(size=200)
        result = sample_population(spec, count=200, seed=42)
        single_agents = [
            a
            for a in result.agents
            if a.get("household_role") == "adult_primary"
            and a.get("partner_id") is None
        ]
        # There should be some single-person households
        assert len(single_agents) > 0

    def test_total_agent_count_matches_requested(self):
        spec = _make_household_spec(size=100)
        result = sample_population(spec, count=100, seed=42)
        assert len(result.agents) == 100

    def test_meta_has_household_info(self):
        spec = _make_household_spec(size=50)
        result = sample_population(spec, count=50, seed=42)
        assert result.meta.get("household_mode") is True
        assert "household_count" in result.meta
        assert "household_type_distribution" in result.meta

    def test_backward_compat_individual_spec(self):
        """Specs without household attributes should sample as before."""
        spec = _make_individual_spec(size=50)
        result = sample_population(spec, count=50, seed=42)
        assert len(result.agents) == 50
        # No household fields
        assert result.agents[0].get("household_id") is None
        assert result.meta.get("household_mode") is None

    def test_households_attached_to_result(self):
        spec = _make_household_spec(size=50)
        result = sample_population(spec, count=50, seed=42)
        households = getattr(result, "_households", [])
        assert len(households) > 0
        for hh in households:
            assert "id" in hh
            assert "household_type" in hh
            assert "adult_ids" in hh
            assert len(hh["adult_ids"]) >= 1


class TestCorrelatedDemographics:
    def test_partner_age_correlation(self):
        """Partners should have correlated ages (within a few years)."""
        spec = _make_household_spec(size=200)
        result = sample_population(spec, count=200, seed=42)
        id_map = {a["_id"]: a for a in result.agents}

        age_diffs = []
        for agent in result.agents:
            pid = agent.get("partner_id")
            if pid and pid in id_map:
                diff = abs(agent["age"] - id_map[pid]["age"])
                age_diffs.append(diff)

        if age_diffs:
            avg_diff = sum(age_diffs) / len(age_diffs)
            # Average age gap should be small (< 10 years)
            assert avg_diff < 10, f"Average age gap {avg_diff:.1f} too large"

    def test_country_correlation(self):
        """Country should be correlated with high same-country rate (~95%)."""
        rng = random.Random(42)
        config = HouseholdConfig(same_country_rate=0.95)
        countries = ["USA", "India", "UK", "Japan", "Brazil"]

        same_country = 0
        total = 1000
        for _ in range(total):
            primary_country = rng.choice(countries)
            partner_country = correlate_partner_attribute(
                "country", "categorical", primary_country, None, rng, config, countries
            )
            if partner_country == primary_country:
                same_country += 1

        rate = same_country / total
        # Should be close to 0.95 (within statistical margin)
        assert 0.90 < rate < 0.99, f"Same-country rate {rate:.2%} out of expected range"


class TestHouseholdReconciliation:
    def test_partnered_agents_are_not_left_single(self):
        spec = _make_household_consistency_spec(size=500)
        result = sample_population(spec, count=500, seed=42)

        partnered = [
            a
            for a in result.agents
            if a.get("partner_id") is not None or a.get("partner_npc") is not None
        ]
        assert partnered, "Expected at least some partnered agents"
        assert all(a.get("marital_status") == "Married" for a in partnered)

    def test_household_size_matches_actual_membership(self):
        spec = _make_household_consistency_spec(size=300)
        result = sample_population(spec, count=300, seed=7, agent_focus_mode="couples")

        by_household: dict[str, list[dict]] = {}
        for agent in result.agents:
            by_household.setdefault(agent["household_id"], []).append(agent)

        for members in by_household.values():
            primary = next(
                (m for m in members if m.get("household_role") == "adult_primary"),
                members[0],
            )
            dependents = primary.get("dependents", [])
            expected_size = len(members) + (len(dependents) if isinstance(dependents, list) else 0)
            for member in members:
                assert member.get("household_size") == expected_size

    def test_has_children_matches_generated_dependents(self):
        spec = _make_household_consistency_spec(size=300)
        result = sample_population(spec, count=300, seed=21)

        by_household: dict[str, list[dict]] = {}
        for agent in result.agents:
            by_household.setdefault(agent["household_id"], []).append(agent)

        for members in by_household.values():
            primary = next(
                (m for m in members if m.get("household_role") == "adult_primary"),
                members[0],
            )
            dependents = primary.get("dependents", [])
            child_count = 0
            if isinstance(dependents, list):
                for dep in dependents:
                    relationship = str(dep.get("relationship", "")).lower()
                    if any(token in relationship for token in ("son", "daughter", "child", "kid")):
                        child_count += 1
            expected = child_count > 0
            for member in members:
                assert member.get("has_children") is expected
