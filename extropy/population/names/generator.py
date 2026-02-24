"""Deterministic Faker-first name generation for synthetic agents.

Primary path uses Faker with locale routing from country/region and identity hints.
Bundled CSV registries remain as a safety fallback when Faker locale generation fails.
"""

from __future__ import annotations

import csv
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"

# Lazy-loaded lookup tables
_first_names: dict[tuple[str, str], list[tuple[str, float]]] | None = None
_last_names: dict[str, list[tuple[str, float]]] | None = None
_faker_cache: dict[str, Any] = {}

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

_ETHNICITY_LOCALE_MAP: dict[str, list[str]] = {
    "hispanic": ["es_MX", "es_ES", "pt_BR"],
    "asian": ["en_IN", "zh_CN", "ja_JP", "ko_KR", "vi_VN"],
    "black": ["en_US", "en_GB", "fr_FR"],
    "white": ["en_US", "en_GB", "de_DE", "fr_FR", "it_IT"],
}

_US_STATE_ABBREVIATIONS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

# Region aliases are kept data-driven here so free-text variants
# ("SE Asia", "South East Asia", etc.) resolve consistently.
_REGION_CODE_ALIASES: dict[str, tuple[list[str], list[str]]] = {
    "north_america": (
        ["north america", "n america", "na", "amer"],
        ["US", "CA", "MX"],
    ),
    "latin_america": (
        ["latin america", "latam", "south america", "central america", "caribbean"],
        [
            "MX",
            "BR",
            "AR",
            "CO",
            "CL",
            "PE",
            "VE",
            "UY",
            "PY",
            "BO",
            "EC",
            "GT",
            "HN",
            "SV",
            "NI",
            "CR",
            "PA",
            "DO",
            "CU",
            "PR",
        ],
    ),
    "europe": (
        [
            "europe",
            "eu",
            "emea",
            "western europe",
            "eastern europe",
            "nordic",
            "scandinavian",
            "balkan",
            "mediterranean",
        ],
        [
            "GB",
            "IE",
            "FR",
            "DE",
            "IT",
            "ES",
            "PT",
            "NL",
            "BE",
            "CH",
            "AT",
            "DK",
            "SE",
            "NO",
            "FI",
            "PL",
            "CZ",
            "HU",
            "RO",
            "GR",
            "RU",
            "UA",
            "TR",
        ],
    ),
    "east_asia": (
        ["east asia"],
        ["CN", "JP", "KR", "TW", "HK", "MO"],
    ),
    "south_asia": (
        ["south asia"],
        ["IN", "PK", "BD", "LK", "NP", "BT", "MV"],
    ),
    "southeast_asia": (
        ["southeast asia", "south east asia", "se asia", "sea"],
        ["ID", "MY", "SG", "TH", "VN", "PH", "MM", "KH", "LA", "BN", "TL"],
    ),
    "central_asia": (
        ["central asia"],
        ["KZ", "UZ", "KG", "TJ", "TM"],
    ),
    "middle_east": (
        ["middle east", "mena", "gulf"],
        [
            "AE",
            "SA",
            "QA",
            "KW",
            "BH",
            "OM",
            "JO",
            "LB",
            "IL",
            "PS",
            "IQ",
            "IR",
            "TR",
            "SY",
            "YE",
            "EG",
        ],
    ),
    "africa": (
        ["africa", "sub saharan africa", "sub-saharan africa", "north africa"],
        ["NG", "ZA", "KE", "GH", "ET", "TZ", "UG", "SN", "CM", "DZ", "MA", "TN", "EG"],
    ),
    "oceania": (
        ["oceania", "pacific"],
        ["AU", "NZ", "PG", "FJ"],
    ),
    "asia": (
        ["asia", "apac"],
        ["IN", "CN", "JP", "KR", "ID", "TH", "VN", "MY", "PH", "SG", "PK", "BD"],
    ),
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


def _normalize_geo_text(text: str | None) -> str:
    if not text:
        return ""
    token = text.strip().lower()
    token = token.replace("&", " and ")
    token = re.sub(r"[\s\-_\/\.]+", " ", token)
    token = re.sub(r"[^a-z0-9 ]", "", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


@lru_cache(maxsize=1)
def _country_lookup() -> dict[str, str]:
    """Build normalized country token -> ISO alpha2 map."""
    lookup: dict[str, str] = {}
    try:
        from iso3166 import countries  # Lightweight ISO registry.
    except Exception:
        return lookup

    for country in countries:
        alpha2 = country.alpha2.upper()
        candidates = {
            country.alpha2,
            country.alpha3,
            country.name,
            getattr(country, "apolitical_name", None),
            getattr(country, "official_name", None),
            getattr(country, "common_name", None),
        }
        for candidate in candidates:
            if not candidate:
                continue
            norm = _normalize_geo_text(str(candidate))
            if norm:
                lookup[norm] = alpha2

    # Common shorthands that are noisy in free-text geography.
    lookup.update(
        {
            "usa": "US",
            "u s": "US",
            "u s a": "US",
            "united states": "US",
            "uk": "GB",
            "u k": "GB",
            "uae": "AE",
            "south korea": "KR",
            "north korea": "KP",
            "russia": "RU",
            "ivory coast": "CI",
        }
    )
    return lookup


@lru_cache(maxsize=1)
def _available_locales() -> tuple[set[str], dict[str, list[str]], list[str]]:
    """Return supported Faker locales and reverse country->locale index."""
    try:
        from faker.config import AVAILABLE_LOCALES
    except Exception:
        return set(), {}, []

    available = set(AVAILABLE_LOCALES)
    by_country: dict[str, list[str]] = {}
    language_only: list[str] = []
    for locale in AVAILABLE_LOCALES:
        if "_" in locale:
            suffix = locale.rsplit("_", 1)[1].upper()
            if len(suffix) == 2 and suffix.isalpha():
                by_country.setdefault(suffix, []).append(locale)
                continue
        language_only.append(locale)
    return available, by_country, language_only


def _try_country_code_from_hint(geo_hint: str | None) -> str | None:
    if not geo_hint:
        return None
    lookup = _country_lookup()
    normalized = _normalize_geo_text(geo_hint)
    if not normalized:
        return None

    # Exact country token/code match (handles alpha2/alpha3 like IN/IND).
    if normalized in lookup:
        return lookup[normalized]

    parts = [p.strip() for p in re.split(r"[,;|]", geo_hint) if p.strip()]
    for part in reversed(parts):
        norm = _normalize_geo_text(part)
        if norm in lookup:
            return lookup[norm]

    tokens = normalized.split()
    for span in range(min(4, len(tokens)), 0, -1):
        for i in range(len(tokens) - span + 1):
            phrase = " ".join(tokens[i : i + span])
            # Avoid ambiguous short tokens inside region strings
            # (e.g. "SE Asia" should not resolve to country code SE).
            if len(phrase) <= 2:
                continue
            if phrase in lookup:
                return lookup[phrase]

    # US subnational fallback (e.g., "Austin, TX")
    if re.search(r",\s*([A-Za-z]{2})\b", geo_hint):
        abbr = re.search(r",\s*([A-Za-z]{2})\b", geo_hint)
        if abbr and abbr.group(1).upper() in _US_STATE_ABBREVIATIONS:
            return "US"

    return None


def _region_country_codes(geo_hint: str | None) -> list[str]:
    normalized = _normalize_geo_text(geo_hint)
    if not normalized:
        return []
    best_codes: list[str] = []
    best_len = -1
    for aliases, codes in _REGION_CODE_ALIASES.values():
        for alias in aliases:
            alias_norm = _normalize_geo_text(alias)
            if not alias_norm:
                continue
            if re.search(
                rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", normalized
            ):
                if len(alias_norm) > best_len:
                    best_len = len(alias_norm)
                    best_codes = codes
    if best_codes:
        return best_codes
    return []


def _build_locale_candidates(country: str | None, ethnicity_key: str) -> list[str]:
    available, by_country, _language_only = _available_locales()
    locales: list[str] = []

    raw_hint = (country or "").strip()
    norm_hint = _normalize_geo_text(raw_hint)

    # If caller already passes a locale token, honor directly.
    if raw_hint and re.fullmatch(r"[a-z]{2}[_-][A-Z]{2}", raw_hint):
        direct = raw_hint.replace("-", "_")
        if direct in available:
            locales.append(direct)

    # Country-first resolution via ISO lookup.
    country_code = _try_country_code_from_hint(raw_hint)
    if country_code:
        locales.extend(by_country.get(country_code, []))
    else:
        # Region-scope fallback (SE Asia, LATAM, EMEA, etc.)
        for code in _region_country_codes(norm_hint):
            locales.extend(by_country.get(code, []))

    # Ethnicity hint next.
    for locale in _ETHNICITY_LOCALE_MAP.get(ethnicity_key, []):
        if locale in available:
            locales.append(locale)

    # Global fallback locale always last.
    locales.append("en_US")

    seen: set[str] = set()
    deduped: list[str] = []
    for locale in locales:
        if locale not in seen:
            seen.add(locale)
            deduped.append(locale)
    return deduped


def _get_faker(locale: str) -> Any | None:
    """Get or create a Faker instance for locale; return None if unavailable."""
    if locale in _faker_cache:
        return _faker_cache[locale]

    try:
        from faker import Faker  # Imported lazily to keep module import cheap.
    except Exception:
        return None

    try:
        fake = Faker(locale)
    except Exception:
        return None

    _faker_cache[locale] = fake
    return fake


def _faker_generate_name(
    norm_gender: str,
    ethnicity_key: str,
    country: str | None,
    birth_decade: int,
    seed: int,
) -> tuple[str, str] | None:
    """Generate name via Faker locale routing. Returns None if all locales fail."""
    mixed_seed = (seed * 1315423911 + birth_decade * 2654435761) & 0x7FFFFFFF
    rng = random.Random(mixed_seed)
    locales = _build_locale_candidates(country, ethnicity_key)
    rng.shuffle(locales)

    for locale in locales:
        fake = _get_faker(locale)
        if fake is None:
            continue

        # Ensure deterministic draws per call.
        fake.seed_instance(mixed_seed)

        try:
            if norm_gender == "male":
                first_name = fake.first_name_male()
            else:
                first_name = fake.first_name_female()
        except Exception:
            try:
                first_name = fake.first_name()
            except Exception:
                continue

        try:
            last_name = fake.last_name()
        except Exception:
            continue

        if first_name and last_name:
            return str(first_name), str(last_name)

    return None


def generate_name(
    gender: str | None = None,
    ethnicity: str | None = None,
    birth_decade: int | None = None,
    *,
    age: int | float | None = None,
    country: str = "US",
    seed: int | None = None,
) -> tuple[str, str]:
    """Generate a demographically-plausible (first_name, last_name) pair.

    Args:
        gender: Agent's gender attribute (e.g. "male", "female", "M", "F")
        ethnicity: Agent's ethnicity/race attribute
        birth_decade: Decade of birth (e.g. 1980). Derived from age if not given.
        age: Agent's age (used to derive birth_decade if not provided)
        country: Country/region hint used for Faker locale routing
        seed: RNG seed for reproducibility

    Returns:
        Tuple of (first_name, last_name)
    """
    rng = random.Random(seed)

    # Resolve birth decade
    if birth_decade is None and age is not None:
        birth_decade = age_to_birth_decade(age)
    if birth_decade is None:
        birth_decade = 1980  # fallback

    norm_gender = _normalize_gender(gender)
    norm_ethnicity = _normalize_ethnicity(ethnicity)

    # Primary path: Faker (locale-routed by country/region + ethnicity hints)
    faker_result = _faker_generate_name(
        norm_gender=norm_gender,
        ethnicity_key=norm_ethnicity,
        country=country,
        birth_decade=birth_decade,
        seed=rng.randint(0, 2**31 - 1),
    )
    if faker_result is not None:
        return faker_result

    # Safety fallback: bundled CSVs
    first_names, last_names = _ensure_loaded()
    decade_str = _snap_decade(birth_decade)

    # Pick first name
    first_key = (norm_gender, decade_str)
    first_options = first_names.get(first_key)
    if not first_options:
        # Fallback tier 2: any decade for this demographic bucket
        for d in reversed(_VALID_DECADES):
            fallback_key = (norm_gender, str(d))
            if fallback_key in first_names:
                first_options = first_names[fallback_key]
                break
    if not first_options:
        # Fallback tier 3: global default
        first_options = [("Alex", 1.0)]

    first_name = _weighted_choice(first_options, rng)

    # Pick last name
    last_options = last_names.get(norm_ethnicity)
    if not last_options:
        pooled = [item for rows in last_names.values() for item in rows]
        last_options = pooled or [("Smith", 1.0)]

    last_name = _weighted_choice(last_options, rng)

    return first_name, last_name
