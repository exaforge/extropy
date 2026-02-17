"""Tests for the scenario compiler with mocked LLM calls.

Tests the 5-step compilation pipeline and auto-configuration logic.
"""

from unittest.mock import patch

import pytest

from extropy.core.models.scenario import (
    EventType,
    TimestepUnit,
)
from extropy.scenario.compiler import (
    _generate_scenario_name,
    _determine_simulation_config,
    create_scenario,
)
from extropy.storage import open_study_db


class TestGenerateScenarioName:
    """Test scenario name generation from descriptions."""

    def test_basic_name(self):
        name = _generate_scenario_name("Netflix announces price increase")
        assert name == "netflix_announces_price_increase"

    def test_truncates_to_4_words(self):
        name = _generate_scenario_name("A very long description that goes on and on")
        assert name == "a_very_long_description"

    def test_removes_special_chars(self):
        name = _generate_scenario_name("Price +$3/mo for users!")
        assert "_" in name or name.isalnum()

    def test_empty_description(self):
        name = _generate_scenario_name("")
        assert name == "scenario"


class TestDetermineSimulationConfig:
    """Test default simulation configuration."""

    def test_default_config(self):
        config = _determine_simulation_config()
        assert config.max_timesteps == 100

    def test_has_stop_conditions(self):
        config = _determine_simulation_config()
        assert len(config.stop_conditions) > 0

    def test_timestep_unit_is_hour(self):
        config = _determine_simulation_config()
        assert config.timestep_unit == TimestepUnit.HOUR


class TestCreateScenario:
    """Test the full create_scenario pipeline with mocked LLM calls."""

    @pytest.fixture
    def mock_files(self, minimal_population_spec, tmp_path):
        """Create mock input files for the compiler."""
        # Save population spec
        pop_path = tmp_path / "population.yaml"
        minimal_population_spec.to_yaml(pop_path)

        agents = [
            {"_id": f"agent_{i:03d}", "age": 30 + i, "gender": "male"}
            for i in range(10)
        ]
        edges = [
            {
                "source": f"agent_{i:03d}",
                "target": f"agent_{(i + 1) % 10:03d}",
                "weight": 1.0,
                "type": "colleague",
                "influence_weight": {"source_to_target": 1.0, "target_to_source": 1.0},
            }
            for i in range(10)
        ]

        study_db = tmp_path / "study.db"
        with open_study_db(study_db) as db:
            db.save_sample_result(
                population_id="default",
                agents=agents,
                meta={"source": "test_fixture"},
            )
            db.save_network_result(
                population_id="default",
                network_id="default",
                config={},
                result_meta={"node_count": 10},
                edges=edges,
                seed=42,
                candidate_mode="test",
            )

        return pop_path, study_db

    @patch("extropy.scenario.compiler.parse_scenario")
    @patch("extropy.scenario.compiler.generate_seed_exposure")
    @patch("extropy.scenario.compiler.determine_interaction_model")
    @patch("extropy.scenario.compiler.define_outcomes")
    @patch("extropy.scenario.compiler.generate_timeline")
    def test_creates_valid_scenario(
        self,
        mock_timeline,
        mock_outcomes,
        mock_interaction,
        mock_exposure,
        mock_parse,
        mock_files,
    ):
        """Test that create_scenario produces a valid ScenarioSpec."""
        from extropy.core.models.scenario import (
            Event,
            SeedExposure,
            ExposureChannel,
            ExposureRule,
            InteractionConfig,
            InteractionType,
            SpreadConfig,
            OutcomeConfig,
            OutcomeDefinition,
            OutcomeType,
        )

        pop_path, study_db = mock_files

        # Configure mocks
        mock_parse.return_value = Event(
            type=EventType.PRODUCT_LAUNCH,
            content="New product launching.",
            source="Test Corp",
            credibility=0.9,
            ambiguity=0.2,
            emotional_valence=0.3,
        )

        mock_exposure.return_value = SeedExposure(
            channels=[
                ExposureChannel(
                    name="broadcast",
                    description="Mass broadcast",
                    reach="broadcast",
                    credibility_modifier=1.0,
                ),
            ],
            rules=[
                ExposureRule(
                    channel="broadcast", timestep=0, when="true", probability=0.8
                ),
            ],
        )

        mock_interaction.return_value = (
            InteractionConfig(
                primary_model=InteractionType.PASSIVE_OBSERVATION,
                description="Agents observe each other",
            ),
            SpreadConfig(share_probability=0.3),
        )

        mock_outcomes.return_value = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="adoption",
                    description="Whether the agent adopts the product",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["adopt", "reject"],
                ),
            ],
        )

        mock_timeline.return_value = ([], None)  # No timeline events, no background

        spec, validation_result = create_scenario(
            description="Test product launch scenario",
            population_spec_path=pop_path,
            study_db_path=study_db,
            population_id="default",
            network_id="default",
        )

        assert spec.meta.name is not None
        assert spec.event.type == EventType.PRODUCT_LAUNCH
        assert len(spec.seed_exposure.rules) == 1
        assert spec.simulation.max_timesteps == 100  # default config

    @patch("extropy.scenario.compiler.parse_scenario")
    @patch("extropy.scenario.compiler.generate_seed_exposure")
    @patch("extropy.scenario.compiler.determine_interaction_model")
    @patch("extropy.scenario.compiler.define_outcomes")
    @patch("extropy.scenario.compiler.generate_timeline")
    def test_progress_callback_called(
        self,
        mock_timeline,
        mock_outcomes,
        mock_interaction,
        mock_exposure,
        mock_parse,
        mock_files,
    ):
        """Test that progress callback is invoked for each step."""
        from extropy.core.models.scenario import (
            Event,
            SeedExposure,
            ExposureChannel,
            ExposureRule,
            InteractionConfig,
            InteractionType,
            SpreadConfig,
            OutcomeConfig,
            OutcomeDefinition,
            OutcomeType,
        )

        pop_path, study_db = mock_files

        mock_parse.return_value = Event(
            type=EventType.PRODUCT_LAUNCH,
            content="Test event content",
            source="Test Corp",
            credibility=0.9,
            ambiguity=0.2,
            emotional_valence=0.0,
        )
        mock_exposure.return_value = SeedExposure(
            channels=[
                ExposureChannel(
                    name="b",
                    description="Broadcast",
                    reach="broadcast",
                    credibility_modifier=1.0,
                )
            ],
            rules=[ExposureRule(channel="b", timestep=0, when="true", probability=1.0)],
        )
        mock_interaction.return_value = (
            InteractionConfig(
                primary_model=InteractionType.PASSIVE_OBSERVATION,
                description="Agents observe each other",
            ),
            SpreadConfig(share_probability=0.3),
        )
        mock_outcomes.return_value = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="x",
                    description="Boolean outcome",
                    type=OutcomeType.BOOLEAN,
                    required=True,
                ),
            ],
        )

        mock_timeline.return_value = ([], None)  # No timeline events, no background

        progress_calls = []

        def on_progress(step, status):
            progress_calls.append((step, status))

        create_scenario(
            description="Test",
            population_spec_path=pop_path,
            study_db_path=study_db,
            population_id="default",
            network_id="default",
            on_progress=on_progress,
        )

        # Should get 6 progress calls (steps 1/6 through 6/6)
        # Note: step 1/6 is called twice (once for loading, once for parsing)
        assert len(progress_calls) >= 6
        steps = [call[0] for call in progress_calls]
        assert "6/6" in steps
