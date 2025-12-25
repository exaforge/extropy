"""Tests for the scenario module components."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from entropy.core.models.scenario import (
    EventType,
    Event,
    ExposureChannel,
    ExposureRule,
    SeedExposure,
    InteractionType,
    InteractionConfig,
    SpreadModifier,
    SpreadConfig,
    OutcomeType,
    OutcomeDefinition,
    OutcomeConfig,
    TimestepUnit,
    SimulationConfig,
    ScenarioMeta,
    ScenarioSpec,
    ValidationError,
    ValidationWarning,
    ValidationResult,
)


class TestEventTypes:
    """Tests for event type enum."""

    def test_all_event_types(self):
        """Test all event types are defined."""
        assert EventType.ANNOUNCEMENT.value == "announcement"
        assert EventType.NEWS.value == "news"
        assert EventType.RUMOR.value == "rumor"
        assert EventType.POLICY_CHANGE.value == "policy_change"
        assert EventType.PRODUCT_LAUNCH.value == "product_launch"
        assert EventType.EMERGENCY.value == "emergency"
        assert EventType.OBSERVATION.value == "observation"


class TestEvent:
    """Tests for Event model."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            type=EventType.ANNOUNCEMENT,
            content="Hospital announces new AI diagnostic tool",
            source="Hospital Administration",
            credibility=0.9,
            ambiguity=0.2,
            emotional_valence=-0.1,
        )
        assert event.type == EventType.ANNOUNCEMENT
        assert event.credibility == 0.9
        assert event.emotional_valence == -0.1

    def test_event_credibility_bounds(self):
        """Test credibility is bounded 0-1."""
        with pytest.raises(ValueError):
            Event(
                type=EventType.NEWS,
                content="Test",
                source="Test",
                credibility=1.5,  # Invalid
                ambiguity=0.0,
                emotional_valence=0.0,
            )

    def test_event_emotional_valence_bounds(self):
        """Test emotional valence is bounded -1 to 1."""
        # Valid boundaries
        event = Event(
            type=EventType.NEWS,
            content="Test",
            source="Test",
            credibility=0.5,
            ambiguity=0.5,
            emotional_valence=-1.0,
        )
        assert event.emotional_valence == -1.0

        event = Event(
            type=EventType.NEWS,
            content="Test",
            source="Test",
            credibility=0.5,
            ambiguity=0.5,
            emotional_valence=1.0,
        )
        assert event.emotional_valence == 1.0


class TestExposureChannel:
    """Tests for ExposureChannel model."""

    def test_channel_creation(self):
        """Test creating an exposure channel."""
        channel = ExposureChannel(
            name="email_notification",
            description="Official email from administration",
            reach="broadcast",
            credibility_modifier=1.0,
        )
        assert channel.name == "email_notification"
        assert channel.reach == "broadcast"

    def test_channel_reach_options(self):
        """Test different reach options."""
        for reach in ["broadcast", "targeted", "organic"]:
            channel = ExposureChannel(
                name="test",
                description="Test",
                reach=reach,
            )
            assert channel.reach == reach


class TestExposureRule:
    """Tests for ExposureRule model."""

    def test_rule_creation(self):
        """Test creating an exposure rule."""
        rule = ExposureRule(
            channel="email_notification",
            when="true",
            probability=0.95,
            timestep=0,
        )
        assert rule.channel == "email_notification"
        assert rule.probability == 0.95
        assert rule.timestep == 0

    def test_rule_with_condition(self):
        """Test rule with complex condition."""
        rule = ExposureRule(
            channel="staff_meeting",
            when="employer_type == 'university_hospital'",
            probability=0.8,
            timestep=1,
        )
        assert "employer_type" in rule.when

    def test_rule_probability_bounds(self):
        """Test probability is bounded 0-1."""
        with pytest.raises(ValueError):
            ExposureRule(
                channel="test",
                when="true",
                probability=1.5,
                timestep=0,
            )


class TestSeedExposure:
    """Tests for SeedExposure model."""

    def test_seed_exposure_creation(self):
        """Test creating seed exposure config."""
        channels = [
            ExposureChannel(name="email", description="Email", reach="broadcast"),
            ExposureChannel(name="meeting", description="Meeting", reach="targeted"),
        ]
        rules = [
            ExposureRule(channel="email", when="true", probability=0.9, timestep=0),
            ExposureRule(channel="meeting", when="role == 'manager'", probability=0.8, timestep=1),
        ]
        exposure = SeedExposure(channels=channels, rules=rules)

        assert len(exposure.channels) == 2
        assert len(exposure.rules) == 2


class TestInteractionConfig:
    """Tests for InteractionConfig model."""

    def test_interaction_config_creation(self):
        """Test creating interaction config."""
        config = InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            secondary_model=InteractionType.DIRECT_CONVERSATION,
            description="Social media style with some direct discussions",
        )
        assert config.primary_model == InteractionType.PASSIVE_OBSERVATION
        assert config.secondary_model == InteractionType.DIRECT_CONVERSATION

    def test_all_interaction_types(self):
        """Test all interaction types."""
        assert InteractionType.PASSIVE_OBSERVATION.value == "passive_observation"
        assert InteractionType.DIRECT_CONVERSATION.value == "direct_conversation"
        assert InteractionType.BROADCAST_RESPONSE.value == "broadcast_response"
        assert InteractionType.DELIBERATIVE.value == "deliberative"


class TestSpreadConfig:
    """Tests for SpreadConfig model."""

    def test_spread_config_basic(self):
        """Test basic spread config."""
        config = SpreadConfig(
            share_probability=0.35,
            decay_per_hop=0.1,
        )
        assert config.share_probability == 0.35
        assert config.decay_per_hop == 0.1
        assert config.share_modifiers == []

    def test_spread_config_with_modifiers(self):
        """Test spread config with modifiers."""
        modifiers = [
            SpreadModifier(when="sentiment < -0.5", multiply=1.8),
            SpreadModifier(when="age < 40", multiply=1.3),
        ]
        config = SpreadConfig(
            share_probability=0.35,
            share_modifiers=modifiers,
            max_hops=5,
        )
        assert len(config.share_modifiers) == 2
        assert config.max_hops == 5


class TestOutcomeDefinition:
    """Tests for OutcomeDefinition model."""

    def test_categorical_outcome(self):
        """Test categorical outcome definition."""
        outcome = OutcomeDefinition(
            name="adoption_intent",
            type=OutcomeType.CATEGORICAL,
            description="Intention to adopt the tool",
            options=["will_adopt", "considering", "resistant", "strongly_opposed"],
            required=True,
        )
        assert outcome.type == OutcomeType.CATEGORICAL
        assert len(outcome.options) == 4

    def test_float_outcome(self):
        """Test float outcome definition."""
        outcome = OutcomeDefinition(
            name="sentiment",
            type=OutcomeType.FLOAT,
            description="Overall sentiment",
            range=(-1.0, 1.0),
        )
        assert outcome.type == OutcomeType.FLOAT
        assert outcome.range == (-1.0, 1.0)

    def test_boolean_outcome(self):
        """Test boolean outcome definition."""
        outcome = OutcomeDefinition(
            name="will_advocate",
            type=OutcomeType.BOOLEAN,
            description="Will actively promote or oppose",
        )
        assert outcome.type == OutcomeType.BOOLEAN

    def test_open_ended_outcome(self):
        """Test open-ended outcome definition."""
        outcome = OutcomeDefinition(
            name="concerns",
            type=OutcomeType.OPEN_ENDED,
            description="Main concerns or questions",
            required=False,
        )
        assert outcome.type == OutcomeType.OPEN_ENDED
        assert outcome.required is False


class TestOutcomeConfig:
    """Tests for OutcomeConfig model."""

    def test_outcome_config_creation(self):
        """Test creating outcome config."""
        outcomes = [
            OutcomeDefinition(
                name="adoption_intent",
                type=OutcomeType.CATEGORICAL,
                description="Adoption intent",
                options=["adopt", "consider", "resist"],
            ),
            OutcomeDefinition(
                name="sentiment",
                type=OutcomeType.FLOAT,
                description="Sentiment",
                range=(-1.0, 1.0),
            ),
        ]
        config = OutcomeConfig(
            suggested_outcomes=outcomes,
            capture_full_reasoning=True,
        )
        assert len(config.suggested_outcomes) == 2
        assert config.capture_full_reasoning is True


class TestSimulationConfig:
    """Tests for SimulationConfig model."""

    def test_simulation_config_creation(self):
        """Test creating simulation config."""
        config = SimulationConfig(
            max_timesteps=100,
            timestep_unit=TimestepUnit.HOUR,
            seed=42,
        )
        assert config.max_timesteps == 100
        assert config.timestep_unit == TimestepUnit.HOUR
        assert config.seed == 42

    def test_all_timestep_units(self):
        """Test all timestep units."""
        assert TimestepUnit.MINUTE.value == "minute"
        assert TimestepUnit.HOUR.value == "hour"
        assert TimestepUnit.DAY.value == "day"
        assert TimestepUnit.WEEK.value == "week"
        assert TimestepUnit.MONTH.value == "month"

    def test_simulation_config_with_stop_conditions(self):
        """Test simulation config with stop conditions."""
        config = SimulationConfig(
            max_timesteps=500,
            stop_conditions=["exposure_rate > 0.95", "no_state_changes_for > 10"],
        )
        assert len(config.stop_conditions) == 2


class TestScenarioMeta:
    """Tests for ScenarioMeta model."""

    def test_scenario_meta_creation(self):
        """Test creating scenario metadata."""
        meta = ScenarioMeta(
            name="ai_tool_announcement",
            description="Hospital announces new AI diagnostic tool",
            population_spec="surgeons.yaml",
            agents_file="agents.json",
            network_file="network.json",
        )
        assert meta.name == "ai_tool_announcement"
        assert meta.population_spec == "surgeons.yaml"
        assert meta.created_at is not None


class TestScenarioSpec:
    """Tests for ScenarioSpec model."""

    @pytest.fixture
    def sample_scenario_spec(self):
        """Create a sample scenario spec for testing."""
        return ScenarioSpec(
            meta=ScenarioMeta(
                name="test_scenario",
                description="Test scenario",
                population_spec="pop.yaml",
                agents_file="agents.json",
                network_file="network.json",
            ),
            event=Event(
                type=EventType.ANNOUNCEMENT,
                content="Test announcement content",
                source="Test Source",
                credibility=0.9,
                ambiguity=0.1,
                emotional_valence=0.0,
            ),
            seed_exposure=SeedExposure(
                channels=[
                    ExposureChannel(name="email", description="Email", reach="broadcast"),
                ],
                rules=[
                    ExposureRule(channel="email", when="true", probability=0.9, timestep=0),
                ],
            ),
            interaction=InteractionConfig(
                primary_model=InteractionType.PASSIVE_OBSERVATION,
                description="Social media style observation",
            ),
            spread=SpreadConfig(share_probability=0.3),
            outcomes=OutcomeConfig(
                suggested_outcomes=[
                    OutcomeDefinition(
                        name="sentiment",
                        type=OutcomeType.FLOAT,
                        description="Sentiment",
                        range=(-1.0, 1.0),
                    ),
                ],
            ),
            simulation=SimulationConfig(max_timesteps=50),
        )

    def test_scenario_spec_creation(self, sample_scenario_spec):
        """Test creating a scenario spec."""
        assert sample_scenario_spec.meta.name == "test_scenario"
        assert sample_scenario_spec.event.type == EventType.ANNOUNCEMENT
        assert len(sample_scenario_spec.seed_exposure.channels) == 1

    def test_scenario_spec_summary(self, sample_scenario_spec):
        """Test scenario spec summary."""
        summary = sample_scenario_spec.summary()

        assert "test_scenario" in summary
        assert "announcement" in summary
        assert "50" in summary

    def test_scenario_spec_yaml_roundtrip(self, sample_scenario_spec):
        """Test saving and loading scenario spec to/from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenario.yaml"

            # Save to YAML
            sample_scenario_spec.to_yaml(path)
            assert path.exists()

            # Load from YAML
            loaded = ScenarioSpec.from_yaml(path)
            assert loaded.meta.name == sample_scenario_spec.meta.name
            assert loaded.event.type == sample_scenario_spec.event.type
            assert loaded.spread.share_probability == sample_scenario_spec.spread.share_probability


class TestValidation:
    """Tests for validation models."""

    def test_validation_error(self):
        """Test validation error model."""
        error = ValidationError(
            category="attribute_reference",
            location="seed_exposure.rules[0]",
            message="Attribute 'nonexistent' not found in population spec",
            suggestion="Check attribute name or add to population spec",
        )
        assert error.category == "attribute_reference"
        assert "nonexistent" in error.message

    def test_validation_warning(self):
        """Test validation warning model."""
        warning = ValidationWarning(
            category="potential_issue",
            location="spread",
            message="High share_probability may cause rapid saturation",
        )
        assert warning.category == "potential_issue"

    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(valid=True, errors=[], warnings=[])
        assert result.valid is True

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        errors = [
            ValidationError(
                category="test",
                location="test",
                message="Test error",
            ),
        ]
        result = ValidationResult(valid=False, errors=errors)
        assert result.valid is False
        assert len(result.errors) == 1


class TestComplexScenarios:
    """Tests for complex scenario configurations."""

    def test_multi_channel_exposure(self):
        """Test scenario with multiple exposure channels."""
        channels = [
            ExposureChannel(name="email", description="Email notification", reach="broadcast"),
            ExposureChannel(name="meeting", description="Staff meeting", reach="targeted", credibility_modifier=0.95),
            ExposureChannel(name="word_of_mouth", description="Informal discussion", reach="organic", credibility_modifier=0.7),
        ]
        rules = [
            ExposureRule(channel="email", when="true", probability=0.98, timestep=0),
            ExposureRule(channel="meeting", when="employer_type == 'university_hospital'", probability=0.8, timestep=1),
            ExposureRule(channel="word_of_mouth", when="age < 40", probability=0.3, timestep=2),
        ]
        exposure = SeedExposure(channels=channels, rules=rules)

        assert len(exposure.channels) == 3
        assert len(exposure.rules) == 3

    def test_complex_spread_modifiers(self):
        """Test spread config with complex modifiers."""
        modifiers = [
            SpreadModifier(when="sentiment < -0.5", multiply=1.8, add=0.0),
            SpreadModifier(when="age < 40", multiply=1.3, add=0.0),
            SpreadModifier(when="role_seniority == 'chief_physician_Chefarzt'", multiply=0.8, add=0.0),
            SpreadModifier(when="edge_type == 'colleague'", multiply=1.5, add=0.05),
        ]
        config = SpreadConfig(
            share_probability=0.35,
            share_modifiers=modifiers,
            decay_per_hop=0.1,
            max_hops=3,
        )

        assert len(config.share_modifiers) == 4
        # Last modifier references edge attribute
        assert "edge_type" in config.share_modifiers[3].when

    def test_comprehensive_outcomes(self):
        """Test comprehensive outcome configuration."""
        outcomes = [
            OutcomeDefinition(
                name="adoption_intent",
                type=OutcomeType.CATEGORICAL,
                description="Intent to adopt the tool",
                options=["will_adopt", "considering", "resistant", "strongly_opposed"],
                required=True,
            ),
            OutcomeDefinition(
                name="sentiment",
                type=OutcomeType.FLOAT,
                description="Overall emotional response",
                range=(-1.0, 1.0),
                required=True,
            ),
            OutcomeDefinition(
                name="will_advocate",
                type=OutcomeType.BOOLEAN,
                description="Will actively promote or oppose",
                required=False,
            ),
            OutcomeDefinition(
                name="main_concerns",
                type=OutcomeType.OPEN_ENDED,
                description="Primary concerns or questions",
                required=False,
            ),
        ]
        config = OutcomeConfig(
            suggested_outcomes=outcomes,
            capture_full_reasoning=True,
            extraction_instructions="Focus on concrete evidence for positions",
        )

        assert len(config.suggested_outcomes) == 4
        assert config.capture_full_reasoning is True

        # Check types
        categorical = next(o for o in config.suggested_outcomes if o.name == "adoption_intent")
        assert categorical.type == OutcomeType.CATEGORICAL

        float_outcome = next(o for o in config.suggested_outcomes if o.name == "sentiment")
        assert float_outcome.type == OutcomeType.FLOAT

    def test_full_scenario_with_all_features(self):
        """Test a complete scenario with all features."""
        spec = ScenarioSpec(
            meta=ScenarioMeta(
                name="ai_tool_full_scenario",
                description="Hospital announces mandatory AI diagnostic tool",
                population_spec="german_surgeons.yaml",
                agents_file="agents_500.json",
                network_file="network_500.json",
            ),
            event=Event(
                type=EventType.ANNOUNCEMENT,
                content="Starting next quarter, all surgical departments will implement the DiagAI system for pre-operative assessments. Training sessions will be scheduled.",
                source="Hospital Administration",
                credibility=0.95,
                ambiguity=0.2,
                emotional_valence=-0.1,
            ),
            seed_exposure=SeedExposure(
                channels=[
                    ExposureChannel(name="email", description="Official email", reach="broadcast"),
                    ExposureChannel(name="meeting", description="Department meeting", reach="targeted", credibility_modifier=0.95),
                ],
                rules=[
                    ExposureRule(channel="email", when="true", probability=0.98, timestep=0),
                    ExposureRule(channel="meeting", when="role_seniority in ['chief_physician_Chefarzt', 'senior_physician_Oberarzt']", probability=0.9, timestep=1),
                ],
            ),
            interaction=InteractionConfig(
                primary_model=InteractionType.PASSIVE_OBSERVATION,
                secondary_model=InteractionType.DIRECT_CONVERSATION,
                description="Surgeons observe peer reactions and discuss informally",
            ),
            spread=SpreadConfig(
                share_probability=0.35,
                share_modifiers=[
                    SpreadModifier(when="sentiment < -0.5", multiply=1.8),
                ],
                decay_per_hop=0.1,
            ),
            outcomes=OutcomeConfig(
                suggested_outcomes=[
                    OutcomeDefinition(name="adoption_intent", type=OutcomeType.CATEGORICAL, description="Adoption intent", options=["will_adopt", "considering", "resistant"]),
                    OutcomeDefinition(name="sentiment", type=OutcomeType.FLOAT, description="Sentiment", range=(-1.0, 1.0)),
                ],
            ),
            simulation=SimulationConfig(
                max_timesteps=100,
                timestep_unit=TimestepUnit.HOUR,
                stop_conditions=["exposure_rate > 0.95"],
                seed=42,
            ),
        )

        assert spec.meta.name == "ai_tool_full_scenario"
        assert spec.event.type == EventType.ANNOUNCEMENT
        assert len(spec.seed_exposure.channels) == 2
        assert spec.interaction.secondary_model == InteractionType.DIRECT_CONVERSATION
        assert spec.simulation.seed == 42
