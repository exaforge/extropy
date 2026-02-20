"""Tests for name generation module.

Functions under test in extropy/population/names/generator.py.
"""

from extropy.population.names import generate_name
from extropy.population.names.generator import age_to_birth_decade


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
        """Hispanic routing should differ from default white routing."""
        hispanic_surnames = set()
        white_surnames = set()
        for s in range(20):
            _, last = generate_name(ethnicity="hispanic", seed=s)
            hispanic_surnames.add(last)
            _, white_last = generate_name(ethnicity="white", seed=s)
            white_surnames.add(white_last)
        assert len(hispanic_surnames) > 1
        assert hispanic_surnames != white_surnames

    def test_asian_surname(self):
        """Asian routing should differ from default white routing."""
        asian_surnames = set()
        white_surnames = set()
        for s in range(20):
            _, last = generate_name(ethnicity="asian", seed=s)
            asian_surnames.add(last)
            _, white_last = generate_name(ethnicity="white", seed=s)
            white_surnames.add(white_last)
        assert len(asian_surnames) > 1
        assert asian_surnames != white_surnames

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

    def test_country_aliases_resolve_consistently(self):
        """Country aliases/codes should resolve to the same locale pool."""
        by_name = generate_name(gender="male", country="India", seed=17)
        by_alpha2 = generate_name(gender="male", country="IN", seed=17)
        by_alpha3 = generate_name(gender="male", country="IND", seed=17)
        assert by_name == by_alpha2 == by_alpha3

    def test_region_aliases_resolve_consistently(self):
        """Region spelling variants should normalize to a common region scope."""
        full = generate_name(gender="female", country="Southeast Asia", seed=31)
        short = generate_name(gender="female", country="SE Asia", seed=31)
        spaced = generate_name(gender="female", country="South East Asia", seed=31)
        assert full == short == spaced

    def test_us_city_hint_resolves_to_us_scope(self):
        """Subnational US hints (city + state) should route to US locale scope."""
        us_name = generate_name(gender="male", country="US", seed=9)
        city_name = generate_name(gender="male", country="Austin, TX", seed=9)
        assert city_name == us_name


class TestAgeToBirthDecade:
    """Test age_to_birth_decade helper."""

    def test_age_30_reference_2025(self):
        assert age_to_birth_decade(30, 2025) == 1990

    def test_age_65_reference_2025(self):
        assert age_to_birth_decade(65, 2025) == 1960

    def test_age_5_reference_2025(self):
        assert age_to_birth_decade(5, 2025) == 2020
