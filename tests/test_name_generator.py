"""Tests for name generation module.

Functions under test in extropy/population/names/generator.py.
"""

from extropy.population.names import generate_name
from extropy.population.names.generator import age_to_birth_decade
from extropy.core.models.population import NameConfig, NameEntry


class TestGenerateName:
    """Test generate_name() for various demographics."""

    def test_returns_tuple(self):
        first, last = generate_name(seed=42)
        assert isinstance(first, str)
        assert isinstance(last, str)
        assert len(first) > 0
        assert len(last) > 0

    def test_male_name(self):
        first, last = generate_name(gender="male", seed=0)
        assert isinstance(first, str)

    def test_female_name(self):
        first, last = generate_name(gender="female", seed=0)
        assert isinstance(first, str)

    def test_hispanic_surname(self):
        """Hispanic ethnicity produces a Hispanic surname."""
        # Run several seeds — at least one should be recognizably Hispanic
        surnames = set()
        for s in range(20):
            _, last = generate_name(ethnicity="hispanic", seed=s)
            surnames.add(last)
        hispanic_names = {
            "Garcia",
            "Rodriguez",
            "Martinez",
            "Hernandez",
            "Lopez",
            "Gonzalez",
            "Perez",
            "Sanchez",
            "Ramirez",
            "Torres",
        }
        assert surnames & hispanic_names, f"Expected Hispanic surnames, got {surnames}"

    def test_asian_surname(self):
        """Asian ethnicity produces an Asian surname."""
        surnames = set()
        for s in range(20):
            _, last = generate_name(ethnicity="asian", seed=s)
            surnames.add(last)
        asian_names = {"Kim", "Lee", "Park", "Nguyen", "Chen", "Wang", "Patel", "Singh"}
        assert surnames & asian_names, f"Expected Asian surnames, got {surnames}"

    def test_black_surname(self):
        """Black ethnicity produces surnames from the black distribution."""
        surnames = set()
        for s in range(20):
            _, last = generate_name(ethnicity="black", seed=s)
            surnames.add(last)
        assert len(surnames) > 1  # Should have variety

    def test_decade_affects_first_name(self):
        """Different decades should produce different name distributions."""
        names_1950 = set()
        names_2010 = set()
        for s in range(30):
            first, _ = generate_name(gender="female", birth_decade=1950, seed=s)
            names_1950.add(first)
            first, _ = generate_name(gender="female", birth_decade=2010, seed=s)
            names_2010.add(first)
        # 1950s and 2010s should have some non-overlapping names
        assert names_1950 != names_2010

    def test_seeded_reproducibility(self):
        """Same seed produces same name."""
        a1, b1 = generate_name(gender="male", ethnicity="white", seed=123)
        a2, b2 = generate_name(gender="male", ethnicity="white", seed=123)
        assert a1 == a2
        assert b1 == b2

    def test_different_seeds_different_names(self):
        """Different seeds usually produce different names."""
        results = set()
        for s in range(10):
            results.add(generate_name(seed=s))
        assert len(results) > 1

    def test_age_derives_birth_decade(self):
        """Passing age= should work like explicit birth_decade."""
        first, last = generate_name(gender="female", age=30, seed=42)
        assert isinstance(first, str)

    def test_unknown_ethnicity_fallback(self):
        """Unknown ethnicity gracefully falls back to white surnames."""
        first, last = generate_name(ethnicity="martian", seed=0)
        assert isinstance(first, str)
        assert isinstance(last, str)

    def test_unknown_gender_fallback(self):
        """Unknown gender gracefully falls back to female names."""
        first, last = generate_name(gender="nonbinary", seed=0)
        assert isinstance(first, str)

    def test_none_params_fallback(self):
        """All-None params still produces valid names."""
        first, last = generate_name(
            gender=None, ethnicity=None, birth_decade=None, seed=0
        )
        assert isinstance(first, str)
        assert isinstance(last, str)

    def test_ethnicity_aliases(self):
        """Various ethnicity alias strings are recognized."""
        aliases = [
            "african american",
            "African_American",
            "latino",
            "Hispanic/Latino",
            "asian american",
            "caucasian",
            "AAPI",
        ]
        for alias in aliases:
            first, last = generate_name(ethnicity=alias, seed=0)
            assert isinstance(first, str)


class TestNameConfig:
    """Test generate_name() with NameConfig-based generation."""

    def _make_config(self) -> NameConfig:
        return NameConfig(
            male_first_names=[
                NameEntry(name="Haruto", weight=5.0),
                NameEntry(name="Ren", weight=3.0),
                NameEntry(name="Sota", weight=2.0),
            ],
            female_first_names=[
                NameEntry(name="Yui", weight=5.0),
                NameEntry(name="Hana", weight=3.0),
                NameEntry(name="Aoi", weight=2.0),
            ],
            last_names=[
                NameEntry(name="Tanaka", weight=5.0),
                NameEntry(name="Suzuki", weight=4.0),
                NameEntry(name="Sato", weight=3.0),
            ],
        )

    def test_config_male_draws_from_config(self):
        config = self._make_config()
        names = set()
        for s in range(20):
            first, last = generate_name(gender="male", seed=s, name_config=config)
            names.add(first)
        expected = {"Haruto", "Ren", "Sota"}
        assert names.issubset(expected), f"Got unexpected names: {names - expected}"
        assert len(names) > 1

    def test_config_female_draws_from_config(self):
        config = self._make_config()
        names = set()
        for s in range(20):
            first, last = generate_name(gender="female", seed=s, name_config=config)
            names.add(first)
        expected = {"Yui", "Hana", "Aoi"}
        assert names.issubset(expected), f"Got unexpected names: {names - expected}"
        assert len(names) > 1

    def test_config_last_names_from_config(self):
        config = self._make_config()
        surnames = set()
        for s in range(20):
            _, last = generate_name(seed=s, name_config=config)
            surnames.add(last)
        expected = {"Tanaka", "Suzuki", "Sato"}
        assert surnames.issubset(expected), (
            f"Got unexpected surnames: {surnames - expected}"
        )
        assert len(surnames) > 1

    def test_config_seeded_reproducibility(self):
        config = self._make_config()
        a1, b1 = generate_name(gender="male", seed=42, name_config=config)
        a2, b2 = generate_name(gender="male", seed=42, name_config=config)
        assert a1 == a2
        assert b1 == b2

    def test_empty_config_falls_back_to_csv(self):
        """NameConfig with empty lists falls back to CSV names."""
        empty_config = NameConfig()
        first, last = generate_name(gender="male", seed=42, name_config=empty_config)
        # Should still produce valid names from CSV fallback
        assert isinstance(first, str) and len(first) > 0
        assert isinstance(last, str) and len(last) > 0

    def test_none_config_uses_csv(self):
        """name_config=None is identical to no-config behavior."""
        first1, last1 = generate_name(gender="female", ethnicity="white", seed=99)
        first2, last2 = generate_name(
            gender="female", ethnicity="white", seed=99, name_config=None
        )
        assert first1 == first2
        assert last1 == last2


class TestAgeToBirthDecade:
    """Test age_to_birth_decade helper."""

    def test_age_30_reference_2025(self):
        assert age_to_birth_decade(30, 2025) == 1990

    def test_age_65_reference_2025(self):
        assert age_to_birth_decade(65, 2025) == 1960

    def test_age_5_reference_2025(self):
        assert age_to_birth_decade(5, 2025) == 2020
