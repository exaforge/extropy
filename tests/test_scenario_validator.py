"""Tests for scenario validation behavior."""

import json
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
    ScenarioSpec,
    SeedExposure,
    SimulationConfig,
    SpreadModifier,
    SpreadConfig,
)
from extropy.scenario.validator import load_and_validate_scenario, validate_scenario


def _make_scenario_spec(
    population_path: str,
    agents_path: str,
    network_path: str,
) -> ScenarioSpec:
    return ScenarioSpec(
        meta=ScenarioMeta(
            name="test_scenario",
            description="Validation test scenario",
            population_spec=population_path,
            agents_file=agents_path,
            network_file=network_path,
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
        simulation=SimulationConfig(max_timesteps=10),
    )


def test_validate_scenario_surfaces_errors(tmp_path: Path):
    """Validation should preserve and return discovered errors."""
    population_path = tmp_path / "population.yaml"
    agents_path = tmp_path / "agents.json"
    network_path = tmp_path / "network.json"

    population_path.write_text("placeholder: true\n")
    agents_path.write_text("[]\n")
    network_path.write_text('{"meta": {}, "edges": []}\n')

    spec = _make_scenario_spec(
        str(population_path),
        str(agents_path),
        str(network_path),
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
    agents_path = tmp_path / "agents.json"
    network_path = tmp_path / "network.json"
    scenario_path = tmp_path / "scenario.yaml"

    minimal_population_spec.to_yaml(population_path)
    agents_path.write_text('[{"_id": "agent_0", "age": 35, "gender": "male"}]\n')
    network_path.write_text(json.dumps({"meta": {"node_count": 1}, "edges": []}))

    spec = _make_scenario_spec("population.yaml", "agents.json", "network.json")
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
    agents_path = tmp_path / "agents.json"
    network_path = tmp_path / "network.json"

    population_path.write_text("placeholder: true\n")
    agents_path.write_text("[]\n")
    network_path.write_text('{"meta": {}, "edges": []}\n')

    spec = _make_scenario_spec(
        str(population_path),
        str(agents_path),
        str(network_path),
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
