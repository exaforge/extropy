"""Tests for scenario validation behavior."""

from pathlib import Path

from extropy.core.models.scenario import (
    Event,
    EventType,
    ExposureChannel,
    ExposureRule,
    InteractionConfig,
    InteractionType,
    OutcomeConfig,
    OutcomeDefinition,
    OutcomeType,
    ScenarioMeta,
    ScenarioSimConfig,
    ScenarioSpec,
    SeedExposure,
    SpreadModifier,
    SpreadConfig,
)
from extropy.scenario.validator import load_and_validate_scenario, validate_scenario
from extropy.storage import open_study_db


def _make_scenario_spec(
    population_path: str,
    study_db_path: str,
) -> ScenarioSpec:
    return ScenarioSpec(
        meta=ScenarioMeta(
            name="test_scenario",
            description="Validation test scenario",
            population_spec=population_path,
            study_db=study_db_path,
            population_id="default",
            network_id="default",
        ),
        event=Event(
            type=EventType.ANNOUNCEMENT,
            content="Test announcement",
            source="Test source",
            credibility=0.9,
            ambiguity=0.1,
            emotional_valence=0.0,
        ),
        seed_exposure=SeedExposure(
            channels=[
                ExposureChannel(
                    name="official_notice",
                    description="Official notice",
                    reach="broadcast",
                )
            ],
            rules=[
                ExposureRule(
                    channel="official_notice",
                    when="true",
                    probability=0.95,
                    timestep=0,
                )
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Passive observation test",
        ),
        spread=SpreadConfig(share_probability=0.3),
        outcomes=OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="response",
                    type=OutcomeType.CATEGORICAL,
                    description="Agent response",
                    options=["yes", "no"],
                )
            ]
        ),
        simulation=ScenarioSimConfig(max_timesteps=10),
    )


def test_validate_scenario_surfaces_errors(tmp_path: Path):
    """Validation should preserve and return discovered errors."""
    population_path = tmp_path / "population.yaml"
    study_db = tmp_path / "study.db"

    population_path.write_text("placeholder: true\n")
    with open_study_db(study_db) as db:
        db.save_sample_result(
            population_id="default",
            agents=[],
            meta={"source": "test"},
        )

    spec = _make_scenario_spec(
        str(population_path),
        str(study_db),
    )
    spec.seed_exposure.rules[0].channel = "missing_channel"

    result = validate_scenario(spec)

    assert not result.valid
    assert any(
        issue.location == "seed_exposure.rules[0].channel" for issue in result.errors
    )


def test_load_and_validate_scenario_resolves_relative_paths(
    tmp_path: Path, minimal_population_spec
):
    """Relative file references should resolve against scenario file location."""
    population_path = tmp_path / "population.yaml"
    study_db = tmp_path / "study.db"
    scenario_path = tmp_path / "scenario.yaml"

    minimal_population_spec.to_yaml(population_path)
    with open_study_db(study_db) as db:
        db.save_sample_result(
            population_id="default",
            agents=[{"_id": "agent_0", "age": 35, "gender": "male"}],
            meta={"source": "test"},
        )
        db.save_network_result(
            population_id="default",
            network_id="default",
            config={},
            result_meta={"node_count": 1},
            edges=[
                {
                    "source": "agent_0",
                    "target": "agent_0",
                    "weight": 1.0,
                    "type": "self",
                    "influence_weight": {
                        "source_to_target": 1.0,
                        "target_to_source": 1.0,
                    },
                }
            ],
            seed=None,
            candidate_mode="test",
        )

    spec = _make_scenario_spec("population.yaml", "study.db")
    spec.to_yaml(scenario_path)

    _, result = load_and_validate_scenario(scenario_path)

    file_errors = [
        issue for issue in result.errors if issue.category == "file_reference"
    ]
    assert not file_errors
    assert result.valid


def test_validate_scenario_allows_edge_weight_in_spread_modifier(tmp_path: Path):
    """edge_weight should be treated as a valid spread modifier reference."""
    population_path = tmp_path / "population.yaml"
    study_db = tmp_path / "study.db"

    population_path.write_text("placeholder: true\n")
    with open_study_db(study_db) as db:
        db.save_sample_result(
            population_id="default", agents=[], meta={"source": "test"}
        )

    spec = _make_scenario_spec(
        str(population_path),
        str(study_db),
    )
    spec.spread.share_modifiers = [
        SpreadModifier(when="edge_weight > 0.7", multiply=1.1, add=0.0)
    ]

    result = validate_scenario(spec)

    edge_weight_errors = [
        issue
        for issue in result.errors
        if issue.location == "spread.share_modifiers[0].when"
    ]
    assert not edge_weight_errors


def test_validate_scenario_allows_extended_attribute_reference(
    minimal_population_spec,
):
    """Extended attributes should be valid in scenario when-expressions."""
    spec = _make_scenario_spec("population.yaml", "study.db")
    spec.seed_exposure.rules[0].when = "extended_signal > 0"
    spec.extended_attributes = [
        minimal_population_spec.attributes[0].model_copy(
            update={"name": "extended_signal"}
        )
    ]

    result = validate_scenario(spec, population_spec=minimal_population_spec)

    ref_errors = [
        issue
        for issue in result.errors
        if issue.location == "seed_exposure.rules[0].when"
        and issue.category == "attribute_reference"
    ]
    assert not ref_errors


def test_validate_scenario_still_rejects_unknown_attribute_reference(
    minimal_population_spec,
):
    """Unknown attributes must still fail even with extended attrs present."""
    spec = _make_scenario_spec("population.yaml", "study.db")
    spec.seed_exposure.rules[0].when = "missing_signal > 0"
    spec.extended_attributes = [
        minimal_population_spec.attributes[0].model_copy(
            update={"name": "extended_signal"}
        )
    ]

    result = validate_scenario(spec, population_spec=minimal_population_spec)

    assert any(
        issue.location == "seed_exposure.rules[0].when"
        and issue.category == "attribute_reference"
        for issue in result.errors
    )
