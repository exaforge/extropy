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


def _make_household_spec(size: int = 100) -> PopulationSpec:
    """Create a minimal spec with household-scoped attributes."""
    return PopulationSpec(
        meta=SpecMeta(description="Test household spec", size=size),
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
            "age", "int", 35, rng, _DEFAULT_CONFIG
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
        """Education uses household assortative-mating defaults."""
        rng = random.Random(42)
        same_count = 0
        trials = 500
        for _ in range(trials):
            result = correlate_partner_attribute(
                "education_level",
                "categorical",
                "bachelors",
                rng,
                _DEFAULT_CONFIG,
                available_options=["high_school", "bachelors", "masters", "doctorate"],
            )
            if result == "bachelors":
                same_count += 1
        rate = same_count / trials
        # Expect moderate assortative matching from default config table
        assert 0.50 < rate < 0.70, f"Assortative rate {rate:.2f} outside expected range"

    def test_correlate_with_default_rate(self):
        """Unknown attributes use default_same_group_rate."""
        rng = random.Random(42)
        same_count = 0
        trials = 500
        for _ in range(trials):
            result = correlate_partner_attribute(
                "some_attribute",
                "categorical",
                "value_a",
                rng,
                _DEFAULT_CONFIG,
                available_options=["value_a", "value_b", "value_c"],
            )
            if result == "value_a":
                same_count += 1
        rate = same_count / trials
        # Expect ~85% (default_same_group_rate)
        assert 0.75 < rate < 0.95, f"Default rate {rate:.2f} outside expected range"

    def test_generate_dependents_no_kids(self):
        rng = random.Random(42)
        deps = generate_dependents(HouseholdType.COUPLE, 2, 2, 40, rng, _DEFAULT_CONFIG)
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
        )
        assert len(deps) == 2
        relationships = [d.relationship for d in deps]
        assert any(r in ("father", "mother") for r in relationships)

    def test_generate_dependents_caps_elderly_age(self):
        rng = random.Random(42)
        config = HouseholdConfig(
            elderly_min_offset=25,
            elderly_max_offset=35,
            max_elderly_dependent_age=100,
        )
        deps = generate_dependents(
            HouseholdType.MULTI_GENERATIONAL,
            4,
            2,
            80,
            rng,
            config,
        )
        elder_ages = [d.age for d in deps if d.relationship in ("father", "mother")]
        assert elder_ages
        assert max(elder_ages) <= 100

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

    def test_household_members_share_surname(self):
        spec = _make_household_spec(size=120)
        result = sample_population(spec, count=120, seed=7)
        by_household: dict[str, list[dict]] = {}
        for agent in result.agents:
            hid = agent.get("household_id")
            if isinstance(hid, str):
                by_household.setdefault(hid, []).append(agent)

        for members in by_household.values():
            primary = next(
                (m for m in members if m.get("household_role") == "adult_primary"),
                members[0],
            )
            primary_last = primary.get("last_name")
            if not primary_last:
                continue

            for member in members:
                assert member.get("last_name") == primary_last

            partner_npc = primary.get("partner_npc")
            if isinstance(partner_npc, dict) and partner_npc.get("last_name"):
                assert partner_npc.get("last_name") == primary_last

            for dep in primary.get("dependents", []) or []:
                if not isinstance(dep, dict):
                    continue
                dep_name = dep.get("name")
                if isinstance(dep_name, str) and dep_name.strip():
                    assert dep_name.split()[-1] == primary_last

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
                "country", "categorical", primary_country, rng, config, countries
            )
            if partner_country == primary_country:
                same_country += 1

        rate = same_country / total
        # Should be close to 0.95 (within statistical margin)
        assert 0.90 < rate < 0.99, f"Same-country rate {rate:.2%} out of expected range"


class TestHouseholdCoherenceNormalization:
    def test_non_partnered_marital_categories_preserved(self):
        spec = _make_household_spec(size=500)
        spec.attributes.append(
            AttributeSpec(
                name="marital_status",
                type="categorical",
                category="universal",
                description="Marital status",
                scope="individual",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=[
                            "Married/partnered",
                            "Single",
                            "Divorced/separated",
                            "Widowed",
                        ],
                        weights=[0.25, 0.35, 0.25, 0.15],
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            )
        )
        spec.sampling_order.append("marital_status")

        result = sample_population(spec, count=500, seed=42, agent_focus_mode="couples")
        adults = [
            a
            for a in result.agents
            if str(a.get("household_role", "")).startswith("adult_")
        ]
        non_partnered = [
            a for a in adults if not (a.get("partner_id") or a.get("partner_npc"))
        ]
        assert non_partnered
        assert all(
            a.get("marital_status") != "Married/partnered" for a in non_partnered
        )
        preserved = sum(
            1
            for a in non_partnered
            if a.get("marital_status") in {"Divorced/separated", "Widowed"}
        )
        assert preserved > 0

    def test_has_children_matches_realized_dependents_for_adults(self):
        spec = _make_household_spec(size=400)
        spec.attributes.append(
            AttributeSpec(
                name="has_children",
                type="boolean",
                category="universal",
                description="Children in household",
                scope="individual",
                identity_type="parental_status",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=BooleanDistribution(
                        type="boolean", probability_true=0.5
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            )
        )
        spec.attributes.append(
            AttributeSpec(
                name="marital_status",
                type="categorical",
                category="universal",
                description="Marital status",
                scope="individual",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=[
                            "Married/partnered",
                            "Single",
                            "Divorced/separated",
                            "Widowed",
                        ],
                        weights=[0.25, 0.45, 0.2, 0.1],
                    ),
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            )
        )
        spec.sampling_order.extend(["has_children", "marital_status"])

        result = sample_population(spec, count=400, seed=42, agent_focus_mode="couples")
        adults = [
            a
            for a in result.agents
            if str(a.get("household_role", "")).startswith("adult_")
        ]
        assert adults
        for agent in adults:
            dependents = agent.get("dependents", []) or []
            has_child_dep = any(
                isinstance(dep, dict)
                and (
                    str(dep.get("relationship", "")).lower()
                    in {"son", "daughter", "child", "stepchild"}
                    or (
                        isinstance(dep.get("age"), (int, float))
                        and 0 <= int(dep["age"]) <= 17
                    )
                )
                for dep in dependents
            )
            assert bool(agent.get("has_children")) == has_child_dep

    def test_young_adult_retired_values_are_normalized(self):
        spec = _make_individual_spec(size=300)
        spec.attributes.append(
            AttributeSpec(
                name="employment_sector",
                type="categorical",
                category="universal",
                description="Employment sector",
                semantic_type="employment",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=["Retired", "Private sector", "Unemployed"],
                        weights=[0.8, 0.1, 0.1],
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            )
        )
        spec.sampling_order.append("employment_sector")
        for attr in spec.attributes:
            if attr.name == "age":
                attr.sampling.distribution = NormalDistribution(
                    type="normal",
                    mean=24,
                    std=2,
                    min=18,
                    max=29,
                )

        result = sample_population(spec, count=300, seed=42)
        assert all(a["age"] < 30 for a in result.agents)
        assert all(a.get("employment_sector") != "Retired" for a in result.agents)

    def test_education_normalization_for_young_adults(self):
        spec = _make_individual_spec(size=400)
        spec.attributes.append(
            AttributeSpec(
                name="education_level",
                type="categorical",
                category="universal",
                description="Education level",
                semantic_type="education",
                sampling=SamplingConfig(
                    strategy="independent",
                    distribution=CategoricalDistribution(
                        type="categorical",
                        options=[
                            "HS diploma/GED",
                            "Some college",
                            "Bachelor's degree",
                            "Graduate Degree",
                        ],
                        weights=[0.05, 0.05, 0.2, 0.7],
                    ),
                ),
                grounding=GroundingInfo(level="low", method="estimated"),
            )
        )
        spec.sampling_order.append("education_level")
        for attr in spec.attributes:
            if attr.name == "age":
                attr.sampling.distribution = NormalDistribution(
                    type="normal",
                    mean=19.5,
                    std=1.2,
                    min=18,
                    max=21,
                )

        result = sample_population(spec, count=400, seed=42)
        assert all(a["age"] < 22 for a in result.agents)
        assert all(a.get("education_level") != "Graduate Degree" for a in result.agents)
        assert all(
            not (
                a["age"] in (18, 19)
                and a.get("education_level") in {"Bachelor's degree", "Graduate Degree"}
            )
            for a in result.agents
        )
