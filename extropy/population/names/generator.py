"""Demographically-plausible name generation for synthetic agents.

Uses bundled SSA first-name frequencies (by decade + gender) and Census
surname frequencies (by ethnicity) to produce names that are statistically
representative of US demographics. Seeded RNG ensures reproducibility.

When a NameConfig is provided (from LLM research), culturally-appropriate
names are drawn from those tables instead.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.models.population import NameConfig

_DATA_DIR = Path(__file__).parent / "data"

# Lazy-loaded lookup tables
_first_names: dict[tuple[str, str], list[tuple[str, float]]] | None = None
_last_names: dict[str, list[tuple[str, float]]] | None = None

# Ethnicity aliases — maps common attribute values to CSV ethnicity keys
_ETHNICITY_MAP: dict[str, str] = {
    "white": "white",
    "caucasian": "white",
    "european": "white",
    "black": "black",
    "african american": "black",
    "african_american": "black",
    "african-american": "black",
    "hispanic": "hispanic",
    "latino": "hispanic",
    "latina": "hispanic",
    "latinx": "hispanic",
    "hispanic/latino": "hispanic",
    "hispanic_latino": "hispanic",
    "asian": "asian",
    "asian american": "asian",
    "asian_american": "asian",
    "asian-american": "asian",
    "pacific islander": "asian",
    "aapi": "asian",
}

# Decade snapping thresholds
_VALID_DECADES = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]


def _load_first_names() -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Load first names CSV into {(gender, decade): [(name, weight), ...]}."""
    table: dict[tuple[str, str], list[tuple[str, float]]] = {}
    with open(_DATA_DIR / "first_names.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["gender"].strip().lower(), row["decade"].strip())
            table.setdefault(key, []).append(
                (row["name"].strip(), float(row["weight"]))
            )
    return table


def _load_last_names() -> dict[str, list[tuple[str, float]]]:
    """Load last names CSV into {ethnicity: [(name, weight), ...]}."""
    table: dict[str, list[tuple[str, float]]] = {}
    with open(_DATA_DIR / "last_names.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = row["ethnicity"].strip().lower()
            table.setdefault(key, []).append(
                (row["name"].strip(), float(row["weight"]))
            )
    return table


def _ensure_loaded() -> tuple[
    dict[tuple[str, str], list[tuple[str, float]]],
    dict[str, list[tuple[str, float]]],
]:
    global _first_names, _last_names
    if _first_names is None:
        _first_names = _load_first_names()
    if _last_names is None:
        _last_names = _load_last_names()
    return _first_names, _last_names


def _weighted_choice(options: list[tuple[str, float]], rng: random.Random) -> str:
    """Pick from [(name, weight), ...] using weighted random selection."""
    names = [n for n, _ in options]
    weights = [w for _, w in options]
    return rng.choices(names, weights=weights, k=1)[0]


def _snap_decade(birth_decade: int) -> str:
    """Snap a birth decade to the nearest available decade in the data."""
    closest = min(_VALID_DECADES, key=lambda d: abs(d - birth_decade))
    return str(closest)


def _normalize_gender(gender: str | None) -> str:
    """Map gender attribute to 'male' or 'female' for name lookup."""
    if not gender:
        return "female"  # default fallback
    g = gender.strip().lower()
    if g in ("male", "m", "man", "boy"):
        return "male"
    return "female"


def _normalize_ethnicity(ethnicity: str | None) -> str:
    """Map ethnicity attribute value to a CSV ethnicity key."""
    if not ethnicity:
        return "white"  # default fallback
    key = ethnicity.strip().lower()
    return _ETHNICITY_MAP.get(key, "white")


def age_to_birth_decade(age: int | float, reference_year: int = 2025) -> int:
    """Derive approximate birth decade from age."""
    birth_year = reference_year - int(age)
    return (birth_year // 10) * 10


def generate_name(
    gender: str | None = None,
    ethnicity: str | None = None,
    birth_decade: int | None = None,
    *,
    age: int | float | None = None,
    country: str = "US",
    seed: int | None = None,
    name_config: NameConfig | None = None,
) -> tuple[str, str]:
    """Generate a demographically-plausible (first_name, last_name) pair.

    Args:
        gender: Agent's gender attribute (e.g. "male", "female", "M", "F")
        ethnicity: Agent's ethnicity/race attribute
        birth_decade: Decade of birth (e.g. 1980). Derived from age if not given.
        age: Agent's age (used to derive birth_decade if not provided)
        country: Country code — only "US" supported for now
        seed: RNG seed for reproducibility
        name_config: LLM-researched name tables (None = use bundled CSVs)

    Returns:
        Tuple of (first_name, last_name)
    """
    rng = random.Random(seed)

    # Use NameConfig if available and populated
    if name_config is not None:
        result = _generate_from_config(name_config, gender, rng)
        if result is not None:
            return result

    # Fallback: bundled CSV path
    first_names, last_names = _ensure_loaded()

    # Resolve birth decade
    if birth_decade is None and age is not None:
        birth_decade = age_to_birth_decade(age)
    if birth_decade is None:
        birth_decade = 1980  # fallback

    norm_gender = _normalize_gender(gender)
    norm_ethnicity = _normalize_ethnicity(ethnicity)
    decade_str = _snap_decade(birth_decade)

    # Pick first name
    first_key = (norm_gender, decade_str)
    first_options = first_names.get(first_key)
    if not first_options:
        # Fallback: try any decade for this gender
        for d in reversed(_VALID_DECADES):
            fallback_key = (norm_gender, str(d))
            if fallback_key in first_names:
                first_options = first_names[fallback_key]
                break
    if not first_options:
        # Ultimate fallback
        first_options = [("Alex", 1.0)]

    first_name = _weighted_choice(first_options, rng)

    # Pick last name
    last_options = last_names.get(norm_ethnicity)
    if not last_options:
        # Fallback to white surnames
        last_options = last_names.get("white", [("Smith", 1.0)])

    last_name = _weighted_choice(last_options, rng)

    return first_name, last_name


def _generate_from_config(
    config: NameConfig, gender: str | None, rng: random.Random
) -> tuple[str, str] | None:
    """Try generating a name from NameConfig tables. Returns None if empty."""
    norm_gender = _normalize_gender(gender)

    if norm_gender == "male":
        first_entries = config.male_first_names
    else:
        first_entries = config.female_first_names

    if not first_entries or not config.last_names:
        return None

    first_options = [(e.name, e.weight) for e in first_entries]
    last_options = [(e.name, e.weight) for e in config.last_names]

    first_name = _weighted_choice(first_options, rng)
    last_name = _weighted_choice(last_options, rng)
    return first_name, last_name
