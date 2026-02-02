"""Tests for reasoning prompt construction and schema generation.

Tests prompt building, schema generation, sentiment-to-tone mapping,
and primary position outcome extraction. No LLM calls.
Functions under test in entropy/simulation/reasoning.py.
"""

from datetime import datetime

import pytest

from entropy.core.models import (
    ExposureRecord,
    MemoryEntry,
    PeerOpinion,
    ReasoningContext,
)
from entropy.core.models.scenario import (
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
    SpreadConfig,
)
from entropy.simulation.reasoning import (
    _get_primary_position_outcome,
    _sentiment_to_tone,
    build_pass1_prompt,
    build_pass1_schema,
    build_pass2_prompt,
    build_pass2_schema,
)


def _make_scenario(**overrides):
    """Create a minimal scenario for prompt tests."""
    defaults = dict(
        meta=ScenarioMeta(
            name="test",
            description="Test",
            population_spec="test.yaml",
            agents_file="test.json",
            network_file="test_network.json",
            created_at=datetime(2024, 1, 1),
        ),
        event=Event(
            type=EventType.PRODUCT_LAUNCH,
            content="A revolutionary product is launching next month.",
            source="TechCorp",
            credibility=0.9,
            ambiguity=0.2,
            emotional_valence=0.3,
        ),
        seed_exposure=SeedExposure(
            channels=[
                ExposureChannel(
                    name="broadcast",
                    description="Broadcast",
                    reach="broadcast",
                    credibility_modifier=1.0,
                )
            ],
            rules=[
                ExposureRule(
                    channel="broadcast", timestep=0, when="true", probability=1.0
                )
            ],
        ),
        interaction=InteractionConfig(
            primary_model=InteractionType.PASSIVE_OBSERVATION,
            description="Observe",
        ),
        spread=SpreadConfig(share_probability=0.3),
        outcomes=OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="adoption",
                    description="Product adoption decision",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["adopt", "reject", "wait"],
                ),
            ],
        ),
        simulation=SimulationConfig(max_timesteps=5),
    )
    defaults.update(overrides)
    return ScenarioSpec(**defaults)


def _make_context(**overrides):
    """Create a minimal ReasoningContext."""
    defaults = dict(
        agent_id="a0",
        persona="You are a 35-year-old software engineer in San Francisco.",
        event_content="A revolutionary product is launching.",
        exposure_history=[
            ExposureRecord(
                timestep=0,
                channel="broadcast",
                content="A revolutionary product is launching.",
                credibility=0.9,
            ),
        ],
        peer_opinions=[],
        current_state=None,
        memory_trace=[],
    )
    defaults.update(overrides)
    return ReasoningContext(**defaults)


# ============================================================================
# Pass 1 Prompt
# ============================================================================


class TestBuildPass1Prompt:
    """Test build_pass1_prompt(context, scenario)."""

    def test_persona_included(self):
        context = _make_context(persona="You are a 50-year-old surgeon in Berlin.")
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        assert "50-year-old surgeon in Berlin" in prompt

    def test_event_content_included(self):
        scenario = _make_scenario()
        context = _make_context()
        prompt = build_pass1_prompt(context, scenario)
        assert "revolutionary product is launching" in prompt

    def test_event_source_included(self):
        scenario = _make_scenario()
        context = _make_context()
        prompt = build_pass1_prompt(context, scenario)
        assert "TechCorp" in prompt

    def test_seed_exposure_shows_channel(self):
        """Non-network exposure shows channel name."""
        context = _make_context(
            exposure_history=[
                ExposureRecord(
                    timestep=0,
                    channel="email",
                    content="test",
                    credibility=0.9,
                    source_agent_id=None,
                ),
            ]
        )
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        assert "via email" in prompt

    def test_network_exposure_shows_peer(self):
        """Network exposure says 'someone in your network'."""
        context = _make_context(
            exposure_history=[
                ExposureRecord(
                    timestep=1,
                    channel="network",
                    content="test",
                    credibility=0.85,
                    source_agent_id="a5",
                ),
            ]
        )
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        assert "network" in prompt.lower()

    def test_no_memory_section_on_first_reasoning(self):
        """Empty memory_trace → no 'Previous Thinking' section."""
        context = _make_context(memory_trace=[])
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        assert "Previous Thinking" not in prompt

    def test_memory_trace_included(self):
        """Memory entries produce 'Your Previous Thinking' section."""
        context = _make_context(
            memory_trace=[
                MemoryEntry(
                    timestep=0,
                    sentiment=0.3,
                    conviction=0.5,
                    summary="I think this could be useful.",
                ),
                MemoryEntry(
                    timestep=2,
                    sentiment=0.6,
                    conviction=0.7,
                    summary="After hearing more, I'm convinced.",
                ),
            ]
        )
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        assert "Previous Thinking" in prompt
        assert "I think this could be useful" in prompt
        assert "After hearing more" in prompt

    def test_memory_trace_shows_conviction_label(self):
        """Memory trace includes conviction label from float_to_conviction."""
        context = _make_context(
            memory_trace=[
                MemoryEntry(
                    timestep=0, sentiment=0.3, conviction=0.7, summary="Some thought"
                ),
            ]
        )
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        # conviction 0.7 → "firm"
        assert "firm" in prompt

    def test_peer_opinions_included(self):
        """Peer opinions produce 'What People Around You Are Saying' section."""
        context = _make_context(
            peer_opinions=[
                PeerOpinion(
                    agent_id="a5",
                    relationship="colleague",
                    public_statement="This product will change everything.",
                    sentiment=0.8,
                ),
            ]
        )
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        assert "People Around You Are Saying" in prompt
        assert "This product will change everything" in prompt

    def test_peer_sentiment_fallback(self):
        """Peer without public_statement shows sentiment tone instead."""
        context = _make_context(
            peer_opinions=[
                PeerOpinion(
                    agent_id="a5",
                    relationship="mentor",
                    public_statement=None,
                    sentiment=-0.7,
                ),
            ]
        )
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)
        assert "mentor" in prompt
        # -0.7 → "strongly opposed"
        assert "strongly opposed" in prompt

    def test_peer_position_not_in_prompt(self):
        """Peer's position label should NOT appear in the prompt (semantic influence only)."""
        context = _make_context(
            peer_opinions=[
                PeerOpinion(
                    agent_id="a5",
                    relationship="colleague",
                    position="reject",
                    public_statement="I don't think this is worth it.",
                    sentiment=-0.5,
                ),
            ]
        )
        scenario = _make_scenario()
        prompt = build_pass1_prompt(context, scenario)

        # The position label "reject" should not appear as a structured field
        # (the word "position" may appear in instruction text, but not as
        # "position: reject" or similar structured label for the peer)
        assert "position: reject" not in prompt.lower()
        assert "position=reject" not in prompt.lower()

    def test_re_reasoning_instructions_differ(self):
        """With memory trace, prompt uses re-reasoning instructions."""
        context_first = _make_context(memory_trace=[])
        context_re = _make_context(
            memory_trace=[
                MemoryEntry(timestep=0, sentiment=0.3, conviction=0.5, summary="test"),
            ]
        )
        scenario = _make_scenario()

        prompt_first = build_pass1_prompt(context_first, scenario)
        prompt_re = build_pass1_prompt(context_re, scenario)

        # First reasoning has no "Previous Thinking" section
        assert "Previous Thinking" not in prompt_first
        # Re-reasoning has "Previous Thinking" section and asks about change
        assert "Previous Thinking" in prompt_re
        assert "changed" in prompt_re.lower() or "stand now" in prompt_re.lower()


# ============================================================================
# Pass 1 Schema
# ============================================================================


class TestBuildPass1Schema:
    """Test build_pass1_schema()."""

    def test_has_required_fields(self):
        schema = build_pass1_schema()
        required = schema["required"]
        for field in [
            "reasoning",
            "public_statement",
            "reasoning_summary",
            "sentiment",
            "conviction",
            "will_share",
        ]:
            assert field in required

    def test_sentiment_range(self):
        schema = build_pass1_schema()
        sentiment = schema["properties"]["sentiment"]
        assert sentiment["minimum"] == -1.0
        assert sentiment["maximum"] == 1.0

    def test_conviction_is_integer_0_100(self):
        """Conviction field is an integer from 0 to 100."""
        schema = build_pass1_schema()
        conviction = schema["properties"]["conviction"]
        assert conviction["type"] == "integer"
        assert conviction["minimum"] == 0
        assert conviction["maximum"] == 100

    def test_will_share_is_boolean(self):
        schema = build_pass1_schema()
        assert schema["properties"]["will_share"]["type"] == "boolean"

    def test_no_additional_properties(self):
        schema = build_pass1_schema()
        assert schema["additionalProperties"] is False


# ============================================================================
# Pass 2 Prompt
# ============================================================================


class TestBuildPass2Prompt:
    """Test build_pass2_prompt(reasoning_text, scenario)."""

    def test_reasoning_text_included(self):
        scenario = _make_scenario()
        prompt = build_pass2_prompt("I really love this product.", scenario)
        assert "I really love this product" in prompt

    def test_classification_instruction(self):
        scenario = _make_scenario()
        prompt = build_pass2_prompt("Some reasoning text.", scenario)
        assert "classif" in prompt.lower()

    def test_extraction_instructions_included(self):
        """If scenario has extraction_instructions, they appear in prompt."""
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="adoption",
                    description="Decision",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["adopt", "reject"],
                ),
            ],
            extraction_instructions="Focus on explicit intent, not hedging.",
        )
        scenario = _make_scenario(outcomes=outcomes)
        prompt = build_pass2_prompt("Some reasoning.", scenario)
        assert "Focus on explicit intent" in prompt


# ============================================================================
# Pass 2 Schema
# ============================================================================


class TestBuildPass2Schema:
    """Test build_pass2_schema(outcomes)."""

    def test_categorical_outcome(self):
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="decision",
                    description="User decision",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["buy", "skip", "wait"],
                ),
            ]
        )
        schema = build_pass2_schema(outcomes)
        assert schema is not None
        assert "decision" in schema["properties"]
        assert schema["properties"]["decision"]["enum"] == ["buy", "skip", "wait"]
        assert "decision" in schema["required"]

    def test_boolean_outcome(self):
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="will_try",
                    description="Will try product",
                    type=OutcomeType.BOOLEAN,
                    required=True,
                ),
            ]
        )
        schema = build_pass2_schema(outcomes)
        assert schema["properties"]["will_try"]["type"] == "boolean"

    def test_float_outcome(self):
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="satisfaction",
                    description="Satisfaction score",
                    type=OutcomeType.FLOAT,
                    required=False,
                    range=[0.0, 10.0],
                ),
            ]
        )
        schema = build_pass2_schema(outcomes)
        prop = schema["properties"]["satisfaction"]
        assert prop["type"] == "number"
        assert prop["minimum"] == 0.0
        assert prop["maximum"] == 10.0
        # Not required
        assert "satisfaction" not in schema["required"]

    def test_open_ended_outcome(self):
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="feedback",
                    description="Free-text feedback",
                    type=OutcomeType.OPEN_ENDED,
                    required=False,
                ),
            ]
        )
        schema = build_pass2_schema(outcomes)
        assert schema["properties"]["feedback"]["type"] == "string"

    def test_multiple_outcomes(self):
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="decision",
                    description="Decision",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["a", "b"],
                ),
                OutcomeDefinition(
                    name="confident",
                    description="Confident",
                    type=OutcomeType.BOOLEAN,
                    required=True,
                ),
                OutcomeDefinition(
                    name="score",
                    description="Score",
                    type=OutcomeType.FLOAT,
                    required=False,
                    range=[1.0, 5.0],
                ),
            ]
        )
        schema = build_pass2_schema(outcomes)
        assert len(schema["properties"]) == 3
        assert "decision" in schema["required"]
        assert "confident" in schema["required"]
        assert "score" not in schema["required"]

    def test_no_outcomes_returns_none(self):
        outcomes = OutcomeConfig(suggested_outcomes=[])
        schema = build_pass2_schema(outcomes)
        assert schema is None


# ============================================================================
# Helper Functions
# ============================================================================


class TestSentimentToTone:
    """Test _sentiment_to_tone(sentiment)."""

    @pytest.mark.parametrize(
        "sentiment,expected",
        [
            (-0.9, "strongly opposed"),
            (-0.7, "strongly opposed"),
            (-0.4, "skeptical"),
            (-0.1, "neutral"),
            (0.0, "neutral"),
            (0.1, "neutral"),
            (0.3, "positive"),
            (0.5, "positive"),
            (0.7, "enthusiastic"),
            (0.9, "enthusiastic"),
        ],
    )
    def test_tone_mapping(self, sentiment, expected):
        assert _sentiment_to_tone(sentiment) == expected

    def test_boundary_negative(self):
        """At exactly -0.6, should be 'skeptical'."""
        assert _sentiment_to_tone(-0.6) == "skeptical"

    def test_boundary_positive(self):
        """At exactly 0.6, should be 'enthusiastic'."""
        assert _sentiment_to_tone(0.6) == "enthusiastic"


class TestGetPrimaryPositionOutcome:
    """Test _get_primary_position_outcome(scenario)."""

    def test_required_categorical_found(self):
        scenario = _make_scenario()
        assert _get_primary_position_outcome(scenario) == "adoption"

    def test_no_categorical_returns_none(self):
        """Scenario with only boolean outcomes → None."""
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="interested",
                    description="Interested",
                    type=OutcomeType.BOOLEAN,
                    required=True,
                ),
            ]
        )
        scenario = _make_scenario(outcomes=outcomes)
        assert _get_primary_position_outcome(scenario) is None

    def test_unrequired_categorical_used_as_fallback(self):
        """If no required categorical, first categorical is used."""
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="preference",
                    description="Preference",
                    type=OutcomeType.CATEGORICAL,
                    required=False,
                    options=["a", "b"],
                ),
            ]
        )
        scenario = _make_scenario(outcomes=outcomes)
        assert _get_primary_position_outcome(scenario) == "preference"

    def test_first_required_categorical_preferred(self):
        """Multiple categoricals — first required one is chosen."""
        outcomes = OutcomeConfig(
            suggested_outcomes=[
                OutcomeDefinition(
                    name="secondary",
                    description="Secondary",
                    type=OutcomeType.CATEGORICAL,
                    required=False,
                    options=["x", "y"],
                ),
                OutcomeDefinition(
                    name="primary",
                    description="Primary",
                    type=OutcomeType.CATEGORICAL,
                    required=True,
                    options=["a", "b"],
                ),
            ]
        )
        scenario = _make_scenario(outcomes=outcomes)
        assert _get_primary_position_outcome(scenario) == "primary"
