"""Tests for the population sampler module."""

import tempfile
from pathlib import Path

import pytest

from entropy.core.models.population import (
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    AttributeSpec,
    SamplingConfig,
    GroundingInfo,
    NormalDistribution,
    LognormalDistribution,
    CategoricalDistribution,
    BetaDistribution,
    Modifier,
    Constraint,
)
from entropy.population.sampler.core import (
    sample_population,
    save_json,
    save_sqlite,
    SamplingResult,
    SamplingError,
)
from entropy.population.sampler.distributions import (
    sample_distribution,
    coerce_to_type,
)
from entropy.utils.eval_safe import (
    eval_safe,
    eval_formula,
    eval_condition,
    FormulaError,
)


class TestEvalSafe:
    """Tests for safe expression evaluation."""

    def test_simple_arithmetic(self):
        """Test basic arithmetic operations."""
        result = eval_safe("1 + 2", {})
        assert result == 3

        result = eval_safe("10 * 5", {})
        assert result == 50

        result = eval_safe("100 / 4", {})
        assert result == 25.0

    def test_with_variables(self):
        """Test evaluation with context variables."""
        result = eval_safe("age - 26", {"age": 45})
        assert result == 19

        result = eval_safe("x * y", {"x": 3, "y": 7})
        assert result == 21

    def test_builtins_allowed(self):
        """Test that safe builtins are available."""
        result = eval_safe("max(1, 5, 3)", {})
        assert result == 5

        result = eval_safe("min(10, 2, 8)", {})
        assert result == 2

        result = eval_safe("abs(-5)", {})
        assert result == 5

        result = eval_safe("round(3.7)", {})
        assert result == 4

    def test_conditional_expression(self):
        """Test conditional expressions."""
        result = eval_safe("'senior' if age > 50 else 'junior'", {"age": 55})
        assert result == "senior"

        result = eval_safe("'senior' if age > 50 else 'junior'", {"age": 30})
        assert result == "junior"

    def test_comparison_operators(self):
        """Test comparison operators."""
        result = eval_safe("age > 30", {"age": 45})
        assert result is True

        result = eval_safe("role == 'senior'", {"role": "senior"})
        assert result is True

        result = eval_safe("age >= 18 and age <= 65", {"age": 35})
        assert result is True

    def test_string_operations(self):
        """Test string operations in expressions."""
        result = eval_safe("role == 'chief'", {"role": "chief"})
        assert result is True

        result = eval_safe("'18-24' if age < 25 else '25-34'", {"age": 22})
        assert result == "18-24"

    def test_invalid_expression_raises(self):
        """Test that invalid expressions raise FormulaError."""
        with pytest.raises(FormulaError):
            eval_safe("invalid syntax (", {})

    def test_missing_variable_raises(self):
        """Test that missing variables raise FormulaError."""
        with pytest.raises(FormulaError):
            eval_safe("age + 10", {})


class TestEvalFormula:
    """Tests for formula evaluation."""

    def test_derived_attribute_formula(self):
        """Test formula for derived attributes."""
        result = eval_formula("max(0, age - 26)", {"age": 35})
        assert result == 9

        result = eval_formula("max(0, age - 26)", {"age": 20})
        assert result == 0

    def test_categorical_binning(self):
        """Test categorical binning formula."""
        formula = "'18-24' if age < 25 else '25-34' if age < 35 else '35-44' if age < 45 else '45+'"

        assert eval_formula(formula, {"age": 22}) == "18-24"
        assert eval_formula(formula, {"age": 30}) == "25-34"
        assert eval_formula(formula, {"age": 40}) == "35-44"
        assert eval_formula(formula, {"age": 50}) == "45+"

    def test_boolean_flag_formula(self):
        """Test boolean flag formula."""
        result = eval_formula("years_experience >= 15", {"years_experience": 20})
        assert result is True

        result = eval_formula("years_experience >= 15", {"years_experience": 10})
        assert result is False


class TestEvalCondition:
    """Tests for condition evaluation."""

    def test_simple_condition(self):
        """Test simple condition evaluation."""
        result = eval_condition("age > 30", {"age": 45})
        assert result is True

        result = eval_condition("age > 30", {"age": 25})
        assert result is False

    def test_equality_condition(self):
        """Test equality condition."""
        result = eval_condition("role == 'senior'", {"role": "senior"})
        assert result is True

        result = eval_condition("role == 'senior'", {"role": "junior"})
        assert result is False

    def test_compound_condition(self):
        """Test compound conditions."""
        agent = {"age": 45, "role": "senior", "years_experience": 20}

        result = eval_condition("age > 40 and role == 'senior'", agent)
        assert result is True

        result = eval_condition("age > 50 or years_experience > 15", agent)
        assert result is True

    def test_condition_failure_returns_false(self):
        """Test that condition failures return False (not raise)."""
        # Missing variable returns False, not raises
        result = eval_condition("nonexistent > 10", {"age": 45})
        assert result is False

        # Invalid syntax returns False
        result = eval_condition("invalid (( syntax", {"age": 45})
        assert result is False


class TestDistributionSampling:
    """Tests for distribution sampling."""

    def test_normal_distribution_sampling(self, rng, simple_normal_distribution):
        """Test sampling from normal distribution."""
        values = [
            sample_distribution(simple_normal_distribution, rng) for _ in range(1000)
        ]

        # Check values are within bounds
        assert all(
            simple_normal_distribution.min <= v <= simple_normal_distribution.max
            for v in values
        )

        # Check mean is approximately correct (within 2 std of expected)
        mean_val = sum(values) / len(values)
        assert abs(
            mean_val - simple_normal_distribution.mean
        ) < 2 * simple_normal_distribution.std / (len(values) ** 0.5)

    def test_normal_with_mean_formula(self, rng):
        """Test normal distribution with mean_formula."""
        dist = NormalDistribution(
            mean_formula="age - 28",
            std=3.0,
            min=0.0,
            max=50.0,
        )
        agent = {"age": 45}

        values = [sample_distribution(dist, rng, agent) for _ in range(100)]
        mean_val = sum(values) / len(values)

        # Expected mean is 45 - 28 = 17
        assert abs(mean_val - 17) < 2

    def test_lognormal_distribution_sampling(self, rng):
        """Test sampling from lognormal distribution."""
        dist = LognormalDistribution(
            mean=100000.0,
            std=30000.0,
            min=40000.0,
            max=300000.0,
        )
        values = [sample_distribution(dist, rng) for _ in range(100)]

        # All values should be positive
        assert all(v > 0 for v in values)
        # Should be within bounds
        assert all(dist.min <= v <= dist.max for v in values)

    def test_uniform_distribution_sampling(self, rng, simple_uniform_distribution):
        """Test sampling from uniform distribution."""
        values = [
            sample_distribution(simple_uniform_distribution, rng) for _ in range(1000)
        ]

        # Check values are within bounds
        assert all(
            simple_uniform_distribution.min <= v <= simple_uniform_distribution.max
            for v in values
        )

        # Check mean is approximately at midpoint
        mean_val = sum(values) / len(values)
        expected_mean = (
            simple_uniform_distribution.min + simple_uniform_distribution.max
        ) / 2
        assert abs(mean_val - expected_mean) < 5

    def test_beta_distribution_sampling(self, rng, simple_beta_distribution):
        """Test sampling from beta distribution."""
        values = [
            sample_distribution(simple_beta_distribution, rng) for _ in range(100)
        ]

        # Beta distribution outputs 0-1 by default
        assert all(0 <= v <= 1 for v in values)

    def test_beta_distribution_scaled(self, rng):
        """Test beta distribution with scaling."""
        dist = BetaDistribution(
            alpha=2.0,
            beta=5.0,
            min=0.0,
            max=100.0,
        )
        values = [sample_distribution(dist, rng) for _ in range(100)]

        # Scaled to 0-100
        assert all(0 <= v <= 100 for v in values)

    def test_categorical_distribution_sampling(
        self, rng, simple_categorical_distribution
    ):
        """Test sampling from categorical distribution."""
        values = [
            sample_distribution(simple_categorical_distribution, rng)
            for _ in range(1000)
        ]

        # All values should be valid options
        assert all(v in simple_categorical_distribution.options for v in values)

        # Check approximate distribution
        counts = {
            opt: values.count(opt) for opt in simple_categorical_distribution.options
        }
        # "A" should have roughly 50%
        assert 400 < counts["A"] < 600

    def test_boolean_distribution_sampling(self, rng, simple_boolean_distribution):
        """Test sampling from boolean distribution."""
        values = [
            sample_distribution(simple_boolean_distribution, rng) for _ in range(1000)
        ]

        # All values should be boolean
        assert all(isinstance(v, bool) for v in values)

        # Check approximate probability
        true_count = sum(values)
        assert 600 < true_count < 800  # Expecting ~70%


class TestCoerceToType:
    """Tests for type coercion."""

    def test_coerce_to_int(self):
        """Test coercing values to int."""
        assert coerce_to_type(42.7, "int") == 43
        assert coerce_to_type(42.3, "int") == 42
        assert coerce_to_type("5", "int") == 5
        assert coerce_to_type("6+", "int") == 6

    def test_coerce_to_float(self):
        """Test coercing values to float."""
        assert coerce_to_type(42, "float") == 42.0
        assert coerce_to_type("3.14", "float") == 3.14

    def test_coerce_to_boolean(self):
        """Test coercing values to boolean."""
        assert coerce_to_type(True, "boolean") is True
        assert coerce_to_type(False, "boolean") is False
        assert coerce_to_type("true", "boolean") is True
        assert coerce_to_type("yes", "boolean") is True
        assert coerce_to_type("1", "boolean") is True
        assert coerce_to_type("false", "boolean") is False

    def test_coerce_to_categorical(self):
        """Test coercing values to categorical (string)."""
        assert coerce_to_type("test", "categorical") == "test"
        assert coerce_to_type(123, "categorical") == "123"


class TestSamplePopulation:
    """Tests for the main sample_population function."""

    def test_sample_minimal_spec(self, minimal_population_spec):
        """Test sampling from a minimal population spec."""
        result = sample_population(minimal_population_spec, count=10, seed=42)

        assert isinstance(result, SamplingResult)
        assert len(result.agents) == 10
        assert result.meta["seed"] == 42

    def test_sample_agents_have_correct_fields(self, minimal_population_spec):
        """Test that sampled agents have all required fields."""
        result = sample_population(minimal_population_spec, count=5, seed=42)

        for agent in result.agents:
            assert "_id" in agent
            assert "age" in agent
            assert "gender" in agent
            assert agent["_id"].startswith("agent_")

    def test_sample_agents_respect_types(self, minimal_population_spec):
        """Test that sampled values have correct types."""
        result = sample_population(minimal_population_spec, count=100, seed=42)

        for agent in result.agents:
            assert isinstance(agent["age"], int)
            assert isinstance(agent["gender"], str)
            assert agent["gender"] in ["male", "female", "other"]

    def test_sample_reproducibility(self, minimal_population_spec):
        """Test that same seed produces same results."""
        result1 = sample_population(minimal_population_spec, count=10, seed=42)
        result2 = sample_population(minimal_population_spec, count=10, seed=42)

        assert result1.agents == result2.agents

    def test_sample_different_seeds_differ(self, minimal_population_spec):
        """Test that different seeds produce different results."""
        result1 = sample_population(minimal_population_spec, count=10, seed=42)
        result2 = sample_population(minimal_population_spec, count=10, seed=123)

        assert result1.agents != result2.agents

    def test_sample_complex_spec(self, complex_population_spec):
        """Test sampling from a complex spec with derived and conditional attributes."""
        result = sample_population(complex_population_spec, count=100, seed=42)

        assert len(result.agents) == 100

        for agent in result.agents:
            # Check all attributes present
            assert "age" in agent
            assert "role" in agent
            assert "years_experience" in agent
            assert "salary" in agent

            # Check derived attribute is computed correctly
            expected_exp = max(0, agent["age"] - 26)
            assert agent["years_experience"] == expected_exp

    def test_sample_statistics(self, complex_population_spec):
        """Test that sampling statistics are collected."""
        result = sample_population(complex_population_spec, count=100, seed=42)

        stats = result.stats

        # Check numeric stats
        assert "age" in stats.attribute_means
        assert "age" in stats.attribute_stds
        assert stats.attribute_means["age"] > 0

        # Check categorical stats
        assert "role" in stats.categorical_counts
        assert sum(stats.categorical_counts["role"].values()) == 100

    def test_sample_modifier_triggers(self, complex_population_spec):
        """Test that modifier trigger counts are tracked."""
        result = sample_population(complex_population_spec, count=100, seed=42)

        # salary has modifiers, so should have trigger counts
        assert "salary" in result.stats.modifier_triggers

    def test_sample_uses_spec_size_by_default(self, minimal_population_spec):
        """Test that sample_population uses spec size when count not provided."""
        result = sample_population(minimal_population_spec, seed=42)
        assert len(result.agents) == minimal_population_spec.meta.size

    def test_sample_progress_callback(self, minimal_population_spec):
        """Test progress callback is called."""
        progress_calls = []

        def on_progress(current, total):
            progress_calls.append((current, total))

        sample_population(
            minimal_population_spec, count=10, seed=42, on_progress=on_progress
        )

        assert len(progress_calls) == 10
        assert progress_calls[-1] == (10, 10)


class TestHardConstraints:
    """Tests for hard constraint application."""

    def test_hard_min_constraint(self):
        """Test that hard_min constraints are applied."""
        spec = PopulationSpec(
            meta=SpecMeta(description="Test", size=100),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=1,
                sources=[],
            ),
            attributes=[
                AttributeSpec(
                    name="experience",
                    type="int",
                    category="universal",
                    description="Experience",
                    sampling=SamplingConfig(
                        strategy="independent",
                        distribution=NormalDistribution(
                            mean=5.0, std=5.0
                        ),  # Can go negative
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[
                        Constraint(type="hard_min", value=0.0),
                    ],
                ),
            ],
            sampling_order=["experience"],
        )

        result = sample_population(spec, count=100, seed=42)

        # All values should be >= 0
        for agent in result.agents:
            assert agent["experience"] >= 0

    def test_hard_max_constraint(self):
        """Test that hard_max constraints are applied."""
        spec = PopulationSpec(
            meta=SpecMeta(description="Test", size=100),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=1,
                sources=[],
            ),
            attributes=[
                AttributeSpec(
                    name="score",
                    type="float",
                    category="universal",
                    description="Score",
                    sampling=SamplingConfig(
                        strategy="independent",
                        distribution=NormalDistribution(mean=90.0, std=20.0),
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[
                        Constraint(type="hard_max", value=100.0),
                    ],
                ),
            ],
            sampling_order=["score"],
        )

        result = sample_population(spec, count=100, seed=42)

        # All values should be <= 100
        for agent in result.agents:
            assert agent["score"] <= 100


class TestSaveResults:
    """Tests for saving sampling results."""

    def test_save_json(self, minimal_population_spec):
        """Test saving results to JSON."""
        result = sample_population(minimal_population_spec, count=10, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agents.json"
            save_json(result, path)

            assert path.exists()

            import json

            with open(path) as f:
                data = json.load(f)

            assert "meta" in data
            assert "agents" in data
            assert len(data["agents"]) == 10

    def test_save_sqlite(self, minimal_population_spec):
        """Test saving results to SQLite."""
        result = sample_population(minimal_population_spec, count=10, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agents.db"
            save_sqlite(result, path)

            assert path.exists()

            import sqlite3

            conn = sqlite3.connect(path)
            cursor = conn.cursor()

            # Check agents table
            cursor.execute("SELECT COUNT(*) FROM agents")
            count = cursor.fetchone()[0]
            assert count == 10

            # Check meta table
            cursor.execute("SELECT COUNT(*) FROM meta")
            meta_count = cursor.fetchone()[0]
            assert meta_count > 0

            conn.close()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_derived_attribute_error(self):
        """Test error handling for invalid derived formula."""
        spec = PopulationSpec(
            meta=SpecMeta(description="Test", size=10),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=1,
                sources=[],
            ),
            attributes=[
                AttributeSpec(
                    name="broken",
                    type="int",
                    category="universal",
                    description="Broken attribute",
                    sampling=SamplingConfig(
                        strategy="derived",
                        formula="nonexistent_var * 2",  # References missing var
                        depends_on=["nonexistent_var"],
                    ),
                    grounding=GroundingInfo(level="low", method="computed"),
                ),
            ],
            sampling_order=["broken"],
        )

        with pytest.raises(SamplingError):
            sample_population(spec, count=1, seed=42)

    def test_single_agent_sampling(self, minimal_population_spec):
        """Test sampling a single agent."""
        result = sample_population(minimal_population_spec, count=1, seed=42)
        assert len(result.agents) == 1
        assert result.agents[0]["_id"] == "agent_0"

    def test_large_population_sampling(self, minimal_population_spec):
        """Test sampling a large population."""
        # Modify spec for larger population
        minimal_population_spec.meta.size = 1000
        result = sample_population(minimal_population_spec, count=1000, seed=42)

        assert len(result.agents) == 1000
        # Check ID padding
        assert result.agents[0]["_id"] == "agent_000"
        assert result.agents[999]["_id"] == "agent_999"


class TestSpecExpressionConstraints:
    """Tests for spec_expression constraint type (spec-level validation, not agent-level)."""

    def test_spec_expression_not_evaluated_against_agents(self):
        """Constraints with type='spec_expression' should not appear in violation reports."""
        spec = PopulationSpec(
            meta=SpecMeta(description="Test", size=100),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=1,
                sources=[],
            ),
            attributes=[
                AttributeSpec(
                    name="category",
                    type="categorical",
                    category="universal",
                    description="A category",
                    sampling=SamplingConfig(
                        strategy="independent",
                        distribution=CategoricalDistribution(
                            options=["a", "b", "c"],
                            weights=[0.5, 0.3, 0.2],
                        ),
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[
                        # This is a spec-level constraint - validates YAML definition, not agents
                        Constraint(
                            type="spec_expression",
                            expression="sum(weights)==1",
                            reason="Weights must sum to 1",
                        ),
                    ],
                ),
            ],
            sampling_order=["category"],
        )

        result = sample_population(spec, count=100, seed=42)

        # spec_expression constraints should NOT appear in violation reports
        # (they validate the spec definition, not individual agents)
        assert "sum(weights)" not in str(result.stats.constraint_violations)
        assert len(result.stats.constraint_violations) == 0

    def test_agent_expression_still_checked(self):
        """Constraints with type='expression' should still be validated against agents."""
        spec = PopulationSpec(
            meta=SpecMeta(description="Test", size=100),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=2,
                sources=[],
            ),
            attributes=[
                AttributeSpec(
                    name="value_a",
                    type="int",
                    category="universal",
                    description="Value A",
                    sampling=SamplingConfig(
                        strategy="independent",
                        distribution=NormalDistribution(mean=10.0, std=3.0),
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[],
                ),
                AttributeSpec(
                    name="value_b",
                    type="int",
                    category="universal",
                    description="Value B (should be <= value_a)",
                    sampling=SamplingConfig(
                        strategy="independent",
                        distribution=NormalDistribution(
                            mean=15.0, std=3.0
                        ),  # Often exceeds value_a
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[
                        # This is an agent-level constraint - should be checked against each agent
                        Constraint(
                            type="expression",
                            expression="value_b <= value_a",
                            reason="B should not exceed A",
                        ),
                    ],
                ),
            ],
            sampling_order=["value_a", "value_b"],
        )

        result = sample_population(spec, count=100, seed=42)

        # expression constraints SHOULD appear in violation reports if violated
        # Since value_b (mean=15) often exceeds value_a (mean=10), we expect violations
        assert len(result.stats.constraint_violations) > 0
        assert any(
            "value_b <= value_a" in k for k in result.stats.constraint_violations
        )


class TestDynamicBounds:
    """Tests for formula-based min/max bounds (min_formula, max_formula)."""

    def test_max_formula_enforced(self, rng):
        """max_formula should clamp values to dynamic upper bound."""
        dist = NormalDistribution(
            mean_formula="household_size - 2",
            std=0.9,
            min=0,
            max_formula="household_size - 1",
        )
        agent = {"household_size": 3}

        # Sample many times - all should be clamped to max = 3-1 = 2
        values = [sample_distribution(dist, rng, agent) for _ in range(100)]
        assert all(v <= 2 for v in values), f"Found value > 2: {max(values)}"
        assert all(v >= 0 for v in values), f"Found value < 0: {min(values)}"

    def test_min_formula_enforced(self, rng):
        """min_formula should clamp values to dynamic lower bound."""
        dist = NormalDistribution(
            mean=5.0,
            std=3.0,
            min_formula="baseline",  # Dynamic min
            max=20,
        )
        agent = {"baseline": 3}

        # Sample many times - all should be >= baseline (3)
        values = [sample_distribution(dist, rng, agent) for _ in range(100)]
        assert all(v >= 3 for v in values), f"Found value < 3: {min(values)}"

    def test_both_formula_bounds(self, rng):
        """Both min_formula and max_formula should work together."""
        dist = NormalDistribution(
            mean=50.0,
            std=20.0,
            min_formula="lower",
            max_formula="upper",
        )
        agent = {"lower": 30, "upper": 70}

        values = [sample_distribution(dist, rng, agent) for _ in range(100)]
        assert all(30 <= v <= 70 for v in values), (
            f"Values out of range: min={min(values)}, max={max(values)}"
        )

    def test_formula_takes_precedence_over_static(self, rng):
        """Formula bounds should take precedence over static bounds."""
        dist = NormalDistribution(
            mean=50.0,
            std=10.0,
            min=0,  # Static min
            max=100,  # Static max
            min_formula="dynamic_min",  # Formula min - should take precedence
            max_formula="dynamic_max",  # Formula max - should take precedence
        )
        agent = {"dynamic_min": 40, "dynamic_max": 60}

        values = [sample_distribution(dist, rng, agent) for _ in range(100)]
        # Formula bounds (40, 60) should be used, not static (0, 100)
        assert all(40 <= v <= 60 for v in values), (
            f"Values out of range: min={min(values)}, max={max(values)}"
        )

    def test_lognormal_with_max_formula(self, rng):
        """max_formula should work with lognormal distribution."""
        from entropy.core.models.population import LognormalDistribution

        dist = LognormalDistribution(
            mean=100.0,
            std=30.0,
            min=50,
            max_formula="limit",
        )
        agent = {"limit": 150}

        values = [sample_distribution(dist, rng, agent) for _ in range(100)]
        assert all(v <= 150 for v in values), f"Found value > 150: {max(values)}"
        assert all(v >= 50 for v in values), f"Found value < 50: {min(values)}"

    def test_beta_with_formula_bounds(self, rng):
        """Formula bounds should work with beta distribution scaling."""
        from entropy.core.models.population import BetaDistribution

        dist = BetaDistribution(
            alpha=2.0,
            beta=5.0,
            min_formula="scale_min",
            max_formula="scale_max",
        )
        agent = {"scale_min": 10, "scale_max": 20}

        values = [sample_distribution(dist, rng, agent) for _ in range(100)]
        # Beta should be scaled to [10, 20]
        assert all(10 <= v <= 20 for v in values), (
            f"Values out of range: min={min(values)}, max={max(values)}"
        )

    def test_integration_children_count_pattern(self):
        """Integration test: children_count should never exceed household_size - 1."""
        spec = PopulationSpec(
            meta=SpecMeta(description="Test household", size=500),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=2,
                sources=[],
            ),
            attributes=[
                AttributeSpec(
                    name="household_size",
                    type="int",
                    category="universal",
                    description="Household size",
                    sampling=SamplingConfig(
                        strategy="independent",
                        distribution=NormalDistribution(
                            mean=3.0, std=1.0, min=1, max=8
                        ),
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[],
                ),
                AttributeSpec(
                    name="children_count",
                    type="int",
                    category="universal",
                    description="Number of children",
                    sampling=SamplingConfig(
                        strategy="conditional",
                        distribution=NormalDistribution(
                            mean_formula="max(0, household_size - 2)",
                            std=0.9,
                            min=0,
                            max_formula="max(0, household_size - 1)",
                        ),
                        depends_on=["household_size"],
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[
                        Constraint(
                            type="expression",
                            expression="children_count <= max(0, household_size - 1)",
                            reason="Children cannot exceed household size minus one adult",
                        ),
                    ],
                ),
            ],
            sampling_order=["household_size", "children_count"],
        )

        result = sample_population(spec, count=500, seed=42)

        # With max_formula, there should be ZERO violations
        violations = result.stats.constraint_violations
        children_violations = {
            k: v for k, v in violations.items() if "children_count" in k
        }
        assert len(children_violations) == 0, f"Found violations: {children_violations}"

        # Also verify by checking actual values
        for agent in result.agents:
            assert agent["children_count"] <= max(0, agent["household_size"] - 1), (
                f"Agent {agent['_id']}: children={agent['children_count']}, household={agent['household_size']}"
            )

    def test_modifier_respects_formula_bounds(self):
        """Modifiers should respect max_formula bounds after applying multiply/add."""

        spec = PopulationSpec(
            meta=SpecMeta(description="Test modifier bounds", size=500),
            grounding=GroundingSummary(
                overall="low",
                sources_count=0,
                strong_count=0,
                medium_count=0,
                low_count=2,
                sources=[],
            ),
            attributes=[
                AttributeSpec(
                    name="household_size",
                    type="int",
                    category="universal",
                    description="Household size",
                    sampling=SamplingConfig(
                        strategy="independent",
                        distribution=NormalDistribution(
                            mean=4.0, std=1.0, min=2, max=6
                        ),
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[],
                ),
                AttributeSpec(
                    name="children_count",
                    type="int",
                    category="universal",
                    description="Number of children",
                    sampling=SamplingConfig(
                        strategy="conditional",
                        distribution=NormalDistribution(
                            mean_formula="max(0, household_size - 2)",
                            std=0.9,
                            min=0,
                            max_formula="max(0, household_size - 1)",  # Dynamic max bound
                        ),
                        depends_on=["household_size"],
                        modifiers=[
                            # This modifier adds values - but result should still be clamped
                            Modifier(when="household_size >= 4", multiply=1.2, add=0.5),
                        ],
                    ),
                    grounding=GroundingInfo(level="low", method="estimated"),
                    constraints=[],
                ),
            ],
            sampling_order=["household_size", "children_count"],
        )

        result = sample_population(spec, count=500, seed=42)

        # Even with modifier multiply/add, values should still respect max_formula
        for agent in result.agents:
            max_allowed = max(0, agent["household_size"] - 1)
            assert agent["children_count"] <= max_allowed, (
                f"Agent {agent['_id']}: children={agent['children_count']}, household={agent['household_size']}, max_allowed={max_allowed}"
            )


class TestConstraintValidation:
    """Tests for constraint validation logic in hydrator_utils and quick_validate."""

    def test_is_spec_level_constraint_sum_weights(self):
        """Test detection of sum(weights) as spec-level constraint."""
        from entropy.population.validator import is_spec_level_constraint

        assert is_spec_level_constraint("sum(weights)==1") is True
        assert is_spec_level_constraint("sum(weights) == 1.0") is True

    def test_is_spec_level_constraint_weights_index(self):
        """Test detection of weights[i] as spec-level constraint."""
        from entropy.population.validator import is_spec_level_constraint

        assert is_spec_level_constraint("weights[0]+weights[1]==1") is True
        assert is_spec_level_constraint("weights[0] >= 0") is True

    def test_is_spec_level_constraint_options(self):
        """Test detection of options as spec-level constraint."""
        from entropy.population.validator import is_spec_level_constraint

        assert is_spec_level_constraint("len(options) > 0") is True
        assert is_spec_level_constraint("options[0] != ''") is True

    def test_is_spec_level_constraint_agent_expression(self):
        """Test that agent-level expressions are not flagged as spec-level."""
        from entropy.population.validator import is_spec_level_constraint

        assert is_spec_level_constraint("children_count <= household_size - 1") is False
        assert is_spec_level_constraint("years_experience <= age - 23") is False
        assert is_spec_level_constraint("x > 0") is False

    def test_extract_bound_upper_bound(self):
        """Test extraction of upper bound from constraint."""
        from entropy.population.validator import extract_bound_from_constraint

        bound_type, bound_expr, is_strict = extract_bound_from_constraint(
            "children_count <= household_size - 1", "children_count"
        )
        assert bound_type == "max"
        assert bound_expr == "household_size - 1"
        assert is_strict is False

    def test_extract_bound_strict_upper_bound(self):
        """Test extraction of strict upper bound."""
        from entropy.population.validator import extract_bound_from_constraint

        bound_type, bound_expr, is_strict = extract_bound_from_constraint("x < y", "x")
        assert bound_type == "max"
        assert bound_expr == "y"
        assert is_strict is True

    def test_extract_bound_lower_bound(self):
        """Test extraction of lower bound from constraint."""
        from entropy.population.validator import extract_bound_from_constraint

        bound_type, bound_expr, is_strict = extract_bound_from_constraint(
            "x >= min_value + 5", "x"
        )
        assert bound_type == "min"
        assert bound_expr == "min_value + 5"
        assert is_strict is False

    def test_extract_bound_reversed_form(self):
        """Test extraction when attr is on the right side."""
        from entropy.population.validator import extract_bound_from_constraint

        bound_type, bound_expr, is_strict = extract_bound_from_constraint(
            "household_size - 1 >= children_count", "children_count"
        )
        assert bound_type == "max"
        assert bound_expr == "household_size - 1"

    def test_extract_bound_complex_expression(self):
        """Test extraction with complex bound expression."""
        from entropy.population.validator import extract_bound_from_constraint

        bound_type, bound_expr, is_strict = extract_bound_from_constraint(
            "children_count <= max(0, household_size - 1)", "children_count"
        )
        assert bound_type == "max"
        assert bound_expr == "max(0, household_size - 1)"

    def test_extract_bound_no_match(self):
        """Test extraction returns None for non-bound constraints."""
        from entropy.population.validator import extract_bound_from_constraint

        bound_type, bound_expr, is_strict = extract_bound_from_constraint(
            "a + b == c", "a"
        )
        assert bound_type is None
        assert bound_expr is None


class TestQuickValidation:
    """Tests for LLM response validation functions (validator/llm_response.py)."""

    def test_quick_validate_catches_spec_level_constraint(self):
        """validate_independent_response should catch spec-level constraints with wrong type."""
        from entropy.population.validator import (
            validate_independent_response,
        )

        data = {
            "attributes": [
                {
                    "name": "gender",
                    "distribution": {
                        "type": "categorical",
                        "options": ["male", "female"],
                        "weights": [0.5, 0.5],
                    },
                    "constraints": [
                        {
                            "type": "expression",  # WRONG
                            "expression": "sum(weights)==1",
                        },
                    ],
                },
            ],
        }

        result = validate_independent_response(data, ["gender"])
        assert not result.valid
        assert len(result.errors) > 0
        assert any("spec_expression" in e.suggestion for e in result.errors)

    def test_quick_validate_conditional_catches_missing_max_formula(self):
        """validate_conditional_base_response should catch missing max_formula."""
        from entropy.population.validator import (
            validate_conditional_base_response,
        )

        data = {
            "attributes": [
                {
                    "name": "children_count",
                    "distribution": {
                        "type": "normal",
                        "mean": 0,
                        "std": 0.9,
                        "mean_formula": "household_size - 2",
                        "std_formula": None,
                        "min": 0,
                        "max": None,
                        "min_formula": None,
                        "max_formula": None,  # Missing!
                    },
                    "constraints": [
                        {
                            "type": "expression",
                            "expression": "children_count <= household_size - 1",
                        },
                    ],
                },
            ],
        }

        result = validate_conditional_base_response(data, ["children_count"])
        assert not result.valid
        assert len(result.errors) > 0
        assert any("max_formula" in e.suggestion for e in result.errors)

    def test_quick_validate_conditional_passes_with_max_formula(self):
        """validate_conditional_base_response should pass when max_formula is present."""
        from entropy.population.validator import (
            validate_conditional_base_response,
        )

        data = {
            "attributes": [
                {
                    "name": "children_count",
                    "distribution": {
                        "type": "normal",
                        "mean": 0,
                        "std": 0.9,
                        "mean_formula": "household_size - 2",
                        "std_formula": None,
                        "min": 0,
                        "max": None,
                        "min_formula": None,
                        "max_formula": "household_size - 1",  # Present!
                    },
                    "constraints": [
                        {
                            "type": "expression",
                            "expression": "children_count <= household_size - 1",
                        },
                    ],
                },
            ],
        }

        result = validate_conditional_base_response(data, ["children_count"])
        # Should not have errors about max_formula
        assert not any("max_formula" in str(e.suggestion) for e in result.errors)

    def test_quick_validate_error_message_is_prescriptive(self):
        """Error messages should tell LLM exactly what to fix."""
        from entropy.population.validator import (
            validate_conditional_base_response,
        )

        data = {
            "attributes": [
                {
                    "name": "children_count",
                    "distribution": {
                        "type": "normal",
                        "mean": 0,
                        "std": 0.9,
                        "mean_formula": "household_size - 2",
                        "std_formula": None,
                        "min": 0,
                        "max": None,
                        "min_formula": None,
                        "max_formula": None,
                    },
                    "constraints": [
                        {
                            "type": "expression",
                            "expression": "children_count <= max(0, household_size - 1)",
                        },
                    ],
                },
            ],
        }

        result = validate_conditional_base_response(data, ["children_count"])

        # The suggestion should include the exact formula to add
        formatted = result.format_for_retry()
        assert "max_formula" in formatted
        assert "max(0, household_size - 1)" in formatted

    def test_quick_validate_accepts_static_max_for_constraint(self):
        """Static max should satisfy max bound requirement (not just max_formula)."""
        from entropy.population.validator import (
            validate_conditional_base_response,
        )

        data = {
            "attributes": [
                {
                    "name": "score",
                    "distribution": {
                        "type": "normal",
                        "mean": 50,
                        "std": 20,
                        "mean_formula": None,
                        "std_formula": None,
                        "min": 0,
                        "max": 100,  # Static max enforces the constraint
                        "min_formula": None,
                        "max_formula": None,
                    },
                    "constraints": [
                        {
                            "type": "expression",
                            "expression": "score <= 100",
                        },
                    ],
                },
            ],
        }

        result = validate_conditional_base_response(data, ["score"])
        # Should NOT error - static max=100 satisfies the constraint
        assert not any("max_formula" in str(e.suggestion) for e in result.errors)
