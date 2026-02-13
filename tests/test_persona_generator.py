"""Tests for persona config categorical phrasing generation."""

from extropy.core.models import PopulationSpec
from extropy.population.persona import generator


def _minimal_categorical_spec() -> PopulationSpec:
    return PopulationSpec.model_validate(
        {
            "meta": {"description": "Test population", "size": 10},
            "grounding": {
                "overall": "low",
                "sources_count": 0,
                "strong_count": 0,
                "medium_count": 0,
                "low_count": 1,
                "sources": [],
            },
            "attributes": [
                {
                    "name": "preferred_content_genre",
                    "type": "categorical",
                    "category": "population_specific",
                    "description": "Primary content genre preference",
                    "sampling": {
                        "strategy": "independent",
                        "distribution": {
                            "type": "categorical",
                            "options": ["Reality_TV", "Drama"],
                            "weights": [0.5, 0.5],
                        },
                        "formula": None,
                        "depends_on": [],
                        "modifiers": [],
                    },
                    "grounding": {"level": "low", "method": "estimated"},
                    "constraints": [],
                }
            ],
            "sampling_order": ["preferred_content_genre"],
        }
    )


def test_generate_categorical_phrasings_retries_on_invalid_coverage(monkeypatch):
    spec = _minimal_categorical_spec()
    calls: list[int] = []

    responses = iter(
        [
            {
                "phrasings": [
                    {
                        "attribute": "preferred_content_genre",
                        "option_phrases": [
                            {"option": "Drama", "phrase": "I mostly watch drama."},
                            {
                                "option": "Crime_Mystery",
                                "phrase": "I mostly watch crime mysteries.",
                            },
                        ],
                        "null_options": [],
                        "null_phrase": "",
                    }
                ]
            },
            {
                "phrasings": [
                    {
                        "attribute": "preferred_content_genre",
                        "option_phrases": [
                            {"option": "Drama", "phrase": "I mostly watch drama."},
                            {
                                "option": "Reality_TV",
                                "phrase": "I watch a lot of reality TV.",
                            },
                        ],
                        "null_options": [],
                        "null_phrase": "",
                    }
                ]
            },
        ]
    )

    def _fake_reasoning_call(**_kwargs):
        calls.append(1)
        return next(responses)

    monkeypatch.setattr(generator, "reasoning_call", _fake_reasoning_call)

    result = generator.generate_categorical_phrasings(spec)

    assert len(calls) == 2
    assert len(result) == 1
    assert set(result[0].phrases.keys()) == {"Drama", "Reality_TV"}


def test_generate_categorical_phrasings_accepts_normalized_option_tokens(monkeypatch):
    spec = _minimal_categorical_spec()

    response = {
        "phrasings": [
            {
                "attribute": "preferred_content_genre",
                "option_phrases": [
                    {"option": "Drama", "phrase": "I mostly watch drama."},
                    {"option": "Reality TV", "phrase": "I watch a lot of reality TV."},
                ],
                "null_options": [],
                "null_phrase": "",
            }
        ]
    }

    def _fake_reasoning_call(**_kwargs):
        return response

    monkeypatch.setattr(generator, "reasoning_call", _fake_reasoning_call)

    result = generator.generate_categorical_phrasings(spec)

    assert len(result) == 1
    assert set(result[0].phrases.keys()) == {"Drama", "Reality_TV"}
