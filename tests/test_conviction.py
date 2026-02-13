"""Tests for the conviction system.

Tests conviction level mappings, float conversions, and map consistency.
Functions under test: conviction_to_float, float_to_conviction, CONVICTION_MAP.
"""

import pytest

from extropy.core.models import (
    ConvictionLevel,
    CONVICTION_MAP,
    CONVICTION_REVERSE_MAP,
    conviction_to_float,
    float_to_conviction,
)


class TestConvictionToFloat:
    """Test conviction level string → float conversion."""

    @pytest.mark.parametrize(
        "level,expected",
        [
            (ConvictionLevel.VERY_UNCERTAIN, 0.1),
            (ConvictionLevel.LEANING, 0.3),
            (ConvictionLevel.MODERATE, 0.5),
            (ConvictionLevel.FIRM, 0.7),
            (ConvictionLevel.ABSOLUTE, 0.9),
        ],
    )
    def test_all_levels(self, level, expected):
        assert conviction_to_float(level) == expected

    def test_none_returns_none(self):
        assert conviction_to_float(None) is None

    def test_invalid_string_returns_none(self):
        assert conviction_to_float("invalid") is None

    def test_string_values_work(self):
        """ConvictionLevel values are strings — plain strings should work too."""
        assert conviction_to_float("very_uncertain") == 0.1
        assert conviction_to_float("firm") == 0.7


class TestFloatToConviction:
    """Test float → nearest conviction level string conversion."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.1, "very_uncertain"),
            (0.3, "leaning"),
            (0.5, "moderate"),
            (0.7, "firm"),
            (0.9, "absolute"),
        ],
    )
    def test_exact_values(self, value, expected):
        assert float_to_conviction(value) == expected

    def test_none_returns_none(self):
        assert float_to_conviction(None) is None

    def test_zero_maps_to_very_uncertain(self):
        """0.0 is closest to 0.1 (very_uncertain)."""
        assert float_to_conviction(0.0) == "very_uncertain"

    def test_one_maps_to_absolute(self):
        """1.0 is closest to 0.9 (absolute)."""
        assert float_to_conviction(1.0) == "absolute"

    def test_midpoint_between_levels(self):
        """0.2 is equidistant between 0.1 and 0.3 — verify it picks one."""
        result = float_to_conviction(0.2)
        assert result in ("very_uncertain", "leaning")

    def test_just_above_boundary(self):
        """0.41 should map to moderate (0.5) not leaning (0.3)."""
        assert float_to_conviction(0.41) == "moderate"

    def test_just_below_boundary(self):
        """0.59 should map to moderate (0.5) not firm (0.7)."""
        assert float_to_conviction(0.59) == "moderate"


class TestConvictionMaps:
    """Test CONVICTION_MAP and CONVICTION_REVERSE_MAP consistency."""

    def test_map_has_five_levels(self):
        assert len(CONVICTION_MAP) == 5

    def test_reverse_map_has_five_levels(self):
        assert len(CONVICTION_REVERSE_MAP) == 5

    def test_roundtrip_forward_reverse(self):
        """Every key in CONVICTION_MAP appears as value in CONVICTION_REVERSE_MAP."""
        for level, value in CONVICTION_MAP.items():
            assert CONVICTION_REVERSE_MAP[value] == level

    def test_roundtrip_reverse_forward(self):
        """Every key in CONVICTION_REVERSE_MAP appears as value in CONVICTION_MAP."""
        for value, level in CONVICTION_REVERSE_MAP.items():
            assert CONVICTION_MAP[level] == value

    def test_values_are_ordered(self):
        """Float values increase with conviction level severity."""
        assert (
            CONVICTION_MAP[ConvictionLevel.VERY_UNCERTAIN]
            < CONVICTION_MAP[ConvictionLevel.LEANING]
            < CONVICTION_MAP[ConvictionLevel.MODERATE]
            < CONVICTION_MAP[ConvictionLevel.FIRM]
            < CONVICTION_MAP[ConvictionLevel.ABSOLUTE]
        )

    def test_values_in_zero_one_range(self):
        """All conviction floats are in (0, 1)."""
        for value in CONVICTION_MAP.values():
            assert 0 < value < 1

    def test_all_enum_members_in_map(self):
        """Every ConvictionLevel enum member has a mapping."""
        for level in ConvictionLevel:
            assert level in CONVICTION_MAP
