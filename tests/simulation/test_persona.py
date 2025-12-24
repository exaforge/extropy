"""Tests for persona generation."""

import pytest

from entropy.simulation.persona import (
    generate_persona,
    _format_age,
    _format_gender,
    _format_role,
    _generate_personality_summary,
)


class TestFormatters:
    """Tests for formatting helper functions."""

    def test_format_age(self):
        """Test age formatting."""
        assert _format_age(35) == "35-year-old"
        assert _format_age(35.5) == "35-year-old"

    def test_format_gender(self):
        """Test gender formatting."""
        assert _format_gender("male") == "man"
        assert _format_gender("Male") == "man"
        assert _format_gender("M") == "man"
        assert _format_gender("female") == "woman"
        assert _format_gender("Female") == "woman"
        assert _format_gender("F") == "woman"
        assert _format_gender(None) == "person"
        assert _format_gender("non-binary") == "person"

    def test_format_role(self):
        """Test role formatting."""
        agent = {"role_seniority": "senior surgeon"}
        assert "senior surgeon" in _format_role(agent)

        agent = {"role": "developer", "specialty": "backend"}
        result = _format_role(agent)
        assert "developer" in result

        agent = {}
        assert _format_role(agent) == ""


class TestGeneratePersona:
    """Tests for persona generation."""

    def test_basic_persona(self):
        """Test generating a basic persona."""
        agent = {
            "_id": "agent_001",
            "age": 35,
            "gender": "male",
        }

        persona = generate_persona(agent)

        assert "35-year-old" in persona
        assert "man" in persona

    def test_persona_with_profession(self):
        """Test persona with professional attributes."""
        agent = {
            "_id": "agent_001",
            "age": 42,
            "gender": "female",
            "role_seniority": "senior surgeon",
            "surgical_specialty": "cardiology",
            "employer_type": "university hospital",
            "years_experience": 15,
        }

        persona = generate_persona(agent)

        assert "42-year-old" in persona
        assert "woman" in persona
        assert "senior surgeon" in persona.lower() or "surgeon" in persona.lower()

    def test_persona_with_location(self):
        """Test persona with location."""
        agent = {
            "_id": "agent_001",
            "age": 30,
            "gender": "male",
            "federal_state": "Bavaria",
        }

        persona = generate_persona(agent)

        assert "Bavaria" in persona

    def test_persona_with_research(self):
        """Test persona with research involvement."""
        agent = {
            "_id": "agent_001",
            "age": 45,
            "gender": "female",
            "participation_in_research": True,
            "teaching_responsibility": True,
        }

        persona = generate_persona(agent)

        assert "research" in persona.lower()
        assert "teaching" in persona.lower()

    def test_persona_empty_agent(self):
        """Test persona with minimal agent data."""
        agent = {"_id": "agent_001"}

        persona = generate_persona(agent)

        # Should still generate something
        assert len(persona) > 0

    def test_persona_with_population_spec(self):
        """Test persona with population spec for personality attributes."""
        from entropy.models import PopulationSpec, AttributeSpec, SamplingConfig, GroundingInfo

        # Create a minimal population spec with personality attribute
        agent = {
            "_id": "agent_001",
            "age": 35,
            "gender": "male",
            "openness_to_experience": 0.8,
            "risk_tolerance": 0.7,
        }

        # Create attribute specs for personality
        personality_attrs = [
            AttributeSpec(
                name="openness_to_experience",
                type="float",
                category="personality",
                description="Openness to new experiences",
                sampling=SamplingConfig(strategy="independent"),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
            AttributeSpec(
                name="risk_tolerance",
                type="float",
                category="personality",
                description="Risk tolerance",
                sampling=SamplingConfig(strategy="independent"),
                grounding=GroundingInfo(level="low", method="estimated"),
            ),
        ]

        # Create minimal population spec
        from entropy.models.spec import SpecMeta, GroundingSummary
        from datetime import datetime

        pop_spec = PopulationSpec(
            meta=SpecMeta(
                description="Test population",
                size=100,
            ),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=2,
            ),
            attributes=personality_attrs,
            sampling_order=["openness_to_experience", "risk_tolerance"],
        )

        persona = generate_persona(agent, pop_spec)

        # Should include personality description
        assert len(persona) > 0
        # Personality traits should influence the persona
        # High openness = "curious and creative" or similar
