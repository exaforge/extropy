"""Tests for Entropy core models (Pydantic models)."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from entropy.core.models.population import (
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    GroundingInfo,
    AttributeSpec,
    SamplingConfig,
    NormalDistribution,
    LognormalDistribution,
    UniformDistribution,
    BetaDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    Modifier,
    Constraint,
    DiscoveredAttribute,
    HydratedAttribute,
    SufficiencyResult,
)


class TestDistributions:
    """Tests for distribution model classes."""

    def test_normal_distribution_creation(self):
        """Test creating a normal distribution with all parameters."""
        dist = NormalDistribution(
            mean=50.0,
            std=10.0,
            min=20.0,
            max=80.0,
        )
        assert dist.type == "normal"
        assert dist.mean == 50.0
        assert dist.std == 10.0
        assert dist.min == 20.0
        assert dist.max == 80.0

    def test_normal_distribution_with_formula(self):
        """Test normal distribution with mean_formula."""
        dist = NormalDistribution(
            mean_formula="age - 28",
            std=5.0,
        )
        assert dist.mean_formula == "age - 28"
        assert dist.mean is None

    def test_lognormal_distribution_creation(self):
        """Test creating a lognormal distribution."""
        dist = LognormalDistribution(
            mean=100000.0,
            std=30000.0,
            min=40000.0,
            max=500000.0,
        )
        assert dist.type == "lognormal"
        assert dist.mean == 100000.0

    def test_uniform_distribution_creation(self):
        """Test creating a uniform distribution."""
        dist = UniformDistribution(
            min=0.0,
            max=100.0,
        )
        assert dist.type == "uniform"
        assert dist.min == 0.0
        assert dist.max == 100.0

    def test_beta_distribution_creation(self):
        """Test creating a beta distribution."""
        dist = BetaDistribution(
            alpha=2.0,
            beta=5.0,
        )
        assert dist.type == "beta"
        assert dist.alpha == 2.0
        assert dist.beta == 5.0

    def test_beta_distribution_with_scaling(self):
        """Test beta distribution with min/max scaling."""
        dist = BetaDistribution(
            alpha=2.0,
            beta=5.0,
            min=0.0,
            max=10.0,
        )
        assert dist.min == 0.0
        assert dist.max == 10.0

    def test_categorical_distribution_creation(self):
        """Test creating a categorical distribution."""
        dist = CategoricalDistribution(
            options=["A", "B", "C"],
            weights=[0.5, 0.3, 0.2],
        )
        assert dist.type == "categorical"
        assert dist.options == ["A", "B", "C"]
        assert dist.weights == [0.5, 0.3, 0.2]
        assert sum(dist.weights) == pytest.approx(1.0)

    def test_boolean_distribution_creation(self):
        """Test creating a boolean distribution."""
        dist = BooleanDistribution(
            probability_true=0.7,
        )
        assert dist.type == "boolean"
        assert dist.probability_true == 0.7

    def test_boolean_distribution_bounds(self):
        """Test that boolean probability is bounded 0-1."""
        # Valid boundary values
        dist = BooleanDistribution(probability_true=0.0)
        assert dist.probability_true == 0.0

        dist = BooleanDistribution(probability_true=1.0)
        assert dist.probability_true == 1.0

        # Invalid values should raise
        with pytest.raises(ValueError):
            BooleanDistribution(probability_true=-0.1)

        with pytest.raises(ValueError):
            BooleanDistribution(probability_true=1.1)


class TestModifiers:
    """Tests for modifier model classes."""

    def test_numeric_modifier(self):
        """Test modifier with multiply and add (for numeric distributions)."""
        mod = Modifier(
            when="age > 50",
            multiply=1.2,
            add=5000.0,
        )
        assert mod.when == "age > 50"
        assert mod.multiply == 1.2
        assert mod.add == 5000.0

    def test_categorical_modifier(self):
        """Test modifier with weight overrides (for categorical distributions)."""
        mod = Modifier(
            when="gender == 'female'",
            weight_overrides={"A": 0.6, "B": 0.3, "C": 0.1},
        )
        assert mod.when == "gender == 'female'"
        assert mod.weight_overrides == {"A": 0.6, "B": 0.3, "C": 0.1}

    def test_boolean_modifier(self):
        """Test modifier with probability override (for boolean distributions)."""
        mod = Modifier(
            when="education == 'phd'",
            probability_override=0.9,
        )
        assert mod.when == "education == 'phd'"
        assert mod.probability_override == 0.9

    def test_modifier_defaults(self):
        """Test modifier with only 'when' field."""
        mod = Modifier(when="age > 50")
        assert mod.multiply is None
        assert mod.add is None
        assert mod.weight_overrides is None
        assert mod.probability_override is None


class TestConstraints:
    """Tests for constraint model classes."""

    def test_hard_min_constraint(self):
        """Test hard minimum constraint."""
        constraint = Constraint(
            type="hard_min",
            value=0.0,
            reason="Cannot be negative",
        )
        assert constraint.type == "hard_min"
        assert constraint.value == 0.0

    def test_hard_max_constraint(self):
        """Test hard maximum constraint."""
        constraint = Constraint(
            type="hard_max",
            value=100.0,
            reason="Maximum allowed value",
        )
        assert constraint.type == "hard_max"
        assert constraint.value == 100.0

    def test_expression_constraint(self):
        """Test expression constraint."""
        constraint = Constraint(
            type="expression",
            expression="value <= age - 24",
            reason="Experience cannot exceed years since minimum working age",
        )
        assert constraint.type == "expression"
        assert constraint.expression == "value <= age - 24"


class TestSamplingConfig:
    """Tests for sampling configuration."""

    def test_independent_sampling(self, simple_normal_distribution):
        """Test independent sampling configuration."""
        config = SamplingConfig(
            strategy="independent",
            distribution=simple_normal_distribution,
        )
        assert config.strategy == "independent"
        assert config.distribution is not None
        assert config.formula is None
        assert config.depends_on == []

    def test_derived_sampling(self):
        """Test derived sampling configuration."""
        config = SamplingConfig(
            strategy="derived",
            formula="max(0, age - 26)",
            depends_on=["age"],
        )
        assert config.strategy == "derived"
        assert config.formula == "max(0, age - 26)"
        assert config.depends_on == ["age"]

    def test_conditional_sampling(self, simple_normal_distribution):
        """Test conditional sampling configuration."""
        config = SamplingConfig(
            strategy="conditional",
            distribution=simple_normal_distribution,
            depends_on=["role"],
            modifiers=[
                Modifier(when="role == 'senior'", multiply=1.5),
            ],
        )
        assert config.strategy == "conditional"
        assert len(config.modifiers) == 1


class TestAttributeSpec:
    """Tests for attribute specification."""

    def test_attribute_spec_creation(self, simple_attribute_spec):
        """Test creating an attribute spec."""
        assert simple_attribute_spec.name == "age"
        assert simple_attribute_spec.type == "int"
        assert simple_attribute_spec.category == "universal"

    def test_attribute_spec_with_constraints(self, simple_normal_distribution):
        """Test attribute spec with constraints."""
        attr = AttributeSpec(
            name="experience",
            type="int",
            category="population_specific",
            description="Years of experience",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=simple_normal_distribution,
            ),
            grounding=GroundingInfo(level="medium", method="estimated"),
            constraints=[
                Constraint(type="hard_min", value=0.0),
                Constraint(type="hard_max", value=50.0),
            ],
        )
        assert len(attr.constraints) == 2


class TestGroundingInfo:
    """Tests for grounding information."""

    def test_grounding_info_creation(self):
        """Test creating grounding info."""
        info = GroundingInfo(
            level="strong",
            method="researched",
            source="Census Bureau 2023",
            note="Based on 5-year average",
        )
        assert info.level == "strong"
        assert info.method == "researched"
        assert info.source == "Census Bureau 2023"

    def test_grounding_info_minimal(self):
        """Test grounding info with minimal fields."""
        info = GroundingInfo(
            level="low",
            method="estimated",
        )
        assert info.source is None
        assert info.note is None


class TestSpecMeta:
    """Tests for spec metadata."""

    def test_spec_meta_creation(self):
        """Test creating spec metadata."""
        meta = SpecMeta(
            description="500 German surgeons",
            size=500,
            geography="Germany",
        )
        assert meta.description == "500 German surgeons"
        assert meta.size == 500
        assert meta.geography == "Germany"
        assert meta.version == "1.0"

    def test_spec_meta_default_created_at(self):
        """Test that created_at defaults to now."""
        meta = SpecMeta(description="Test", size=100)
        assert meta.created_at is not None
        assert isinstance(meta.created_at, datetime)


class TestGroundingSummary:
    """Tests for grounding summary."""

    def test_grounding_summary_creation(self):
        """Test creating grounding summary."""
        summary = GroundingSummary(
            overall="medium",
            sources_count=5,
            strong_count=10,
            medium_count=15,
            low_count=5,
            sources=["Source 1", "Source 2"],
        )
        assert summary.overall == "medium"
        assert summary.sources_count == 5
        assert len(summary.sources) == 2


class TestPopulationSpec:
    """Tests for the complete population spec."""

    def test_population_spec_creation(self, minimal_population_spec):
        """Test creating a population spec."""
        assert minimal_population_spec.meta.description == "Test population"
        assert len(minimal_population_spec.attributes) == 2
        assert minimal_population_spec.sampling_order == ["age", "gender"]

    def test_get_attribute(self, minimal_population_spec):
        """Test getting an attribute by name."""
        age_attr = minimal_population_spec.get_attribute("age")
        assert age_attr is not None
        assert age_attr.name == "age"

        missing = minimal_population_spec.get_attribute("nonexistent")
        assert missing is None

    def test_population_spec_summary(self, minimal_population_spec):
        """Test getting spec summary."""
        summary = minimal_population_spec.summary()
        assert "Test population" in summary
        assert "Size: 100" in summary
        assert "age" in summary

    def test_population_spec_yaml_roundtrip(self, minimal_population_spec):
        """Test saving and loading spec to/from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_spec.yaml"

            # Save to YAML
            minimal_population_spec.to_yaml(path)
            assert path.exists()

            # Load from YAML
            loaded = PopulationSpec.from_yaml(path)
            assert loaded.meta.description == minimal_population_spec.meta.description
            assert len(loaded.attributes) == len(minimal_population_spec.attributes)
            assert loaded.sampling_order == minimal_population_spec.sampling_order

    def test_population_spec_merge(self, minimal_population_spec):
        """Test merging two population specs."""
        # Create an overlay spec with new attributes
        overlay = PopulationSpec(
            meta=SpecMeta(
                description="Overlay attributes",
                size=100,
            ),
            grounding=GroundingSummary(
                overall="low",
                sources_count=1,
                strong_count=0,
                medium_count=0,
                low_count=1,
                sources=["Overlay source"],
            ),
            attributes=[
                AttributeSpec(
                    name="experience",
                    type="int",
                    category="population_specific",
                    description="Years of experience",
                    sampling=SamplingConfig(
                        strategy="derived",
                        formula="max(0, age - 25)",
                        depends_on=["age"],
                    ),
                    grounding=GroundingInfo(level="low", method="computed"),
                ),
            ],
            sampling_order=["experience"],
        )

        merged = minimal_population_spec.merge(overlay)

        # Check merged has all attributes
        assert len(merged.attributes) == 3
        assert merged.get_attribute("age") is not None
        assert merged.get_attribute("gender") is not None
        assert merged.get_attribute("experience") is not None

        # Check sampling order includes new attribute
        assert "experience" in merged.sampling_order
        # experience depends on age, so age should come before experience
        age_idx = merged.sampling_order.index("age")
        exp_idx = merged.sampling_order.index("experience")
        assert age_idx < exp_idx

    def test_compute_sampling_order(self, complex_population_spec):
        """Test that sampling order respects dependencies."""
        order = complex_population_spec.sampling_order

        # years_experience depends on age
        assert order.index("age") < order.index("years_experience")

        # salary depends on role
        assert order.index("role") < order.index("salary")


class TestIntermediateTypes:
    """Tests for intermediate types used during spec building."""

    def test_discovered_attribute(self):
        """Test DiscoveredAttribute model."""
        attr = DiscoveredAttribute(
            name="age",
            type="int",
            category="universal",
            description="Age of the person",
            strategy="independent",
        )
        assert attr.name == "age"
        assert attr.strategy == "independent"
        assert attr.depends_on == []

    def test_discovered_attribute_with_dependencies(self):
        """Test DiscoveredAttribute with dependencies."""
        attr = DiscoveredAttribute(
            name="experience",
            type="int",
            category="population_specific",
            description="Years of experience",
            strategy="derived",
            depends_on=["age"],
        )
        assert attr.strategy == "derived"
        assert attr.depends_on == ["age"]

    def test_hydrated_attribute(self):
        """Test HydratedAttribute model."""
        attr = HydratedAttribute(
            name="age",
            type="int",
            category="universal",
            description="Age",
            strategy="independent",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=NormalDistribution(mean=45.0, std=10.0),
            ),
            grounding=GroundingInfo(level="medium", method="estimated"),
        )
        assert attr.name == "age"
        assert attr.sampling.distribution is not None

    def test_sufficiency_result(self):
        """Test SufficiencyResult model."""
        result = SufficiencyResult(
            sufficient=True,
            size=500,
            geography="Germany",
        )
        assert result.sufficient is True
        assert result.size == 500
        assert result.clarifications_needed == []

    def test_sufficiency_result_insufficient(self):
        """Test SufficiencyResult when insufficient."""
        result = SufficiencyResult(
            sufficient=False,
            size=1000,  # default
            clarifications_needed=[
                "Please specify the target population",
                "What region should be covered?",
            ],
        )
        assert result.sufficient is False
        assert len(result.clarifications_needed) == 2
