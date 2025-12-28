"""Tests for the population validator module."""

from entropy.core.models.population import (
    PopulationSpec,
    SpecMeta,
    GroundingSummary,
    AttributeSpec,
    SamplingConfig,
    GroundingInfo,
    NormalDistribution,
    CategoricalDistribution,
    BooleanDistribution,
    BetaDistribution,
    Modifier,
)
from entropy.population.validator import (
    validate_spec,
    ValidationResult,
    ValidationIssue,
    Severity,
)
from entropy.utils.expressions import extract_names_from_expression


def make_spec(
    attributes: list[AttributeSpec], sampling_order: list[str] | None = None
) -> PopulationSpec:
    """Helper to create a PopulationSpec for testing."""
    if sampling_order is None:
        sampling_order = [a.name for a in attributes]

    return PopulationSpec(
        meta=SpecMeta(description="Test", size=100),
        grounding=GroundingSummary(
            overall="low",
            sources_count=0,
            strong_count=0,
            medium_count=0,
            low_count=len(attributes),
            sources=[],
        ),
        attributes=attributes,
        sampling_order=sampling_order,
    )


def make_attr(
    name: str,
    attr_type: str = "int",
    strategy: str = "independent",
    distribution=None,
    formula: str | None = None,
    depends_on: list[str] | None = None,
    modifiers: list[Modifier] | None = None,
) -> AttributeSpec:
    """Helper to create an AttributeSpec for testing."""
    if distribution is None and strategy != "derived":
        if attr_type == "int" or attr_type == "float":
            distribution = NormalDistribution(mean=50.0, std=10.0)
        elif attr_type == "categorical":
            distribution = CategoricalDistribution(
                options=["A", "B"], weights=[0.5, 0.5]
            )
        elif attr_type == "boolean":
            distribution = BooleanDistribution(probability_true=0.5)

    return AttributeSpec(
        name=name,
        type=attr_type,
        category="universal",
        description=f"Test attribute {name}",
        sampling=SamplingConfig(
            strategy=strategy,
            distribution=distribution,
            formula=formula,
            depends_on=depends_on or [],
            modifiers=modifiers or [],
        ),
        grounding=GroundingInfo(level="low", method="estimated"),
    )


class TestValidateSpec:
    """Tests for the main validate_spec function."""

    def test_valid_spec(self, minimal_population_spec):
        """Test that a valid spec passes validation."""
        result = validate_spec(minimal_population_spec)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validation_result_properties(self, minimal_population_spec):
        """Test ValidationResult properties."""
        result = validate_spec(minimal_population_spec)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, "valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "info")

    def test_all_issues_property(self):
        """Test all_issues property combines all issue types."""
        spec = make_spec(
            [
                make_attr("age", "int"),
                make_attr("age", "int"),  # Duplicate - will cause ERROR
            ]
        )
        result = validate_spec(spec)

        all_issues = result.all_issues
        assert len(all_issues) == len(result.errors) + len(result.warnings) + len(
            result.info
        )


class TestExtractNamesFromExpression:
    """Tests for extracting variable names from expressions."""

    def test_simple_variable(self):
        """Test extracting a single variable."""
        names = extract_names_from_expression("age")
        assert names == {"age"}

    def test_arithmetic_expression(self):
        """Test extracting variables from arithmetic."""
        names = extract_names_from_expression("age - 26")
        assert names == {"age"}

    def test_comparison_expression(self):
        """Test extracting variables from comparisons."""
        names = extract_names_from_expression("role == 'senior'")
        assert names == {"role"}

    def test_compound_expression(self):
        """Test extracting variables from compound expressions."""
        names = extract_names_from_expression("age > 30 and role == 'senior'")
        assert names == {"age", "role"}

    def test_function_call(self):
        """Test that function names are not extracted."""
        names = extract_names_from_expression("max(0, age - 26)")
        assert names == {"age"}

    def test_string_literals_ignored(self):
        """Test that strings inside literals are not extracted."""
        names = extract_names_from_expression("role == 'University_hospital'")
        assert names == {"role"}
        assert "University_hospital" not in names

    def test_boolean_literals_ignored(self):
        """Test that True/False are not extracted as variables."""
        names = extract_names_from_expression("is_active == True")
        assert names == {"is_active"}
        assert "True" not in names


class TestTypeModifierCompatibility:
    """Tests for Category 1: Type/Modifier Compatibility."""

    def test_numeric_with_weight_overrides_error(self):
        """Test that numeric distribution with weight_overrides is an error."""
        attr = make_attr(
            "salary",
            "float",
            strategy="conditional",
            depends_on=["role"],
            modifiers=[
                Modifier(
                    when="role == 'senior'", weight_overrides={"A": 0.5, "B": 0.5}
                ),
            ],
        )
        spec = make_spec([make_attr("role", "categorical"), attr], ["role", "salary"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("weight_overrides" in str(e) for e in result.errors)

    def test_categorical_with_multiply_error(self):
        """Test that categorical distribution with multiply is an error."""
        attr = make_attr(
            "category",
            "categorical",
            strategy="conditional",
            depends_on=["age"],
            modifiers=[
                Modifier(when="age > 50", multiply=1.5),
            ],
        )
        spec = make_spec([make_attr("age", "int"), attr], ["age", "category"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("multiply" in str(e) for e in result.errors)

    def test_boolean_with_add_error(self):
        """Test that boolean distribution with add is an error."""
        attr = make_attr(
            "is_active",
            "boolean",
            strategy="conditional",
            depends_on=["age"],
            modifiers=[
                Modifier(when="age > 50", add=0.1),
            ],
        )
        spec = make_spec([make_attr("age", "int"), attr], ["age", "is_active"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("add" in str(e) for e in result.errors)


class TestRangeViolations:
    """Tests for Category 2: Range Violations."""

    def test_beta_large_add_error(self):
        """Test that large add on beta distribution is an error."""
        attr = AttributeSpec(
            name="score",
            type="float",
            category="universal",
            description="Score",
            sampling=SamplingConfig(
                strategy="conditional",
                distribution=BetaDistribution(alpha=2.0, beta=5.0),
                depends_on=["age"],
                modifiers=[
                    Modifier(when="age > 50", add=0.6),  # Too large for beta
                ],
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([make_attr("age", "int"), attr], ["age", "score"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("beta" in str(e).lower() and "add" in str(e) for e in result.errors)

    def test_boolean_probability_out_of_bounds(self):
        """Test that probability_override out of [0,1] is an error."""
        attr = make_attr(
            "is_active",
            "boolean",
            strategy="conditional",
            depends_on=["age"],
            modifiers=[
                Modifier(when="age > 50", probability_override=1.5),  # Out of bounds
            ],
        )
        spec = make_spec([make_attr("age", "int"), attr], ["age", "is_active"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("probability" in str(e).lower() for e in result.errors)


class TestWeightValidity:
    """Tests for Category 3: Weight Validity."""

    def test_categorical_weights_dont_sum_to_one(self):
        """Test that weights not summing to 1.0 is an error."""
        attr = AttributeSpec(
            name="category",
            type="categorical",
            category="universal",
            description="Category",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=CategoricalDistribution(
                    options=["A", "B", "C"],
                    weights=[0.3, 0.3, 0.3],  # Sums to 0.9
                ),
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("weights sum" in str(e) for e in result.errors)

    def test_categorical_options_weights_mismatch(self):
        """Test that options/weights length mismatch is an error."""
        attr = AttributeSpec(
            name="category",
            type="categorical",
            category="universal",
            description="Category",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=CategoricalDistribution(
                    options=["A", "B", "C"],
                    weights=[0.5, 0.5],  # Wrong length
                ),
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("mismatch" in str(e).lower() for e in result.errors)

    def test_weight_override_unknown_option(self):
        """Test that weight_override with unknown option is an error."""
        attr = AttributeSpec(
            name="category",
            type="categorical",
            category="universal",
            description="Category",
            sampling=SamplingConfig(
                strategy="conditional",
                distribution=CategoricalDistribution(
                    options=["A", "B", "C"],
                    weights=[0.4, 0.3, 0.3],
                ),
                depends_on=["age"],
                modifiers=[
                    Modifier(
                        when="age > 50",
                        weight_overrides={
                            "A": 0.5,
                            "B": 0.3,
                            "D": 0.2,
                        },  # "D" doesn't exist
                    ),
                ],
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([make_attr("age", "int"), attr], ["age", "category"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("'D'" in str(e) for e in result.errors)


class TestDistributionParameters:
    """Tests for Category 4: Distribution Parameters."""

    def test_negative_std_error(self):
        """Test that negative std is an error."""
        attr = AttributeSpec(
            name="value",
            type="float",
            category="universal",
            description="Value",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=NormalDistribution(mean=50.0, std=-10.0),
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("std" in str(e) and "negative" in str(e) for e in result.errors)

    def test_zero_std_error(self):
        """Test that zero std is an error (should be derived)."""
        attr = AttributeSpec(
            name="value",
            type="float",
            category="universal",
            description="Value",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=NormalDistribution(mean=50.0, std=0.0),
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("std" in str(e) and "0" in str(e) for e in result.errors)

    def test_min_greater_than_max_error(self):
        """Test that min >= max is an error."""
        attr = AttributeSpec(
            name="value",
            type="float",
            category="universal",
            description="Value",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=NormalDistribution(
                    mean=50.0, std=10.0, min=100.0, max=50.0
                ),
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("min" in str(e) and "max" in str(e) for e in result.errors)


class TestDependencyValidation:
    """Tests for Category 5: Dependency Validation."""

    def test_dependency_references_nonexistent(self):
        """Test that referencing a nonexistent attribute is an error."""
        attr = make_attr(
            "derived_value",
            "int",
            strategy="derived",
            formula="nonexistent + 10",
            depends_on=["nonexistent"],
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("nonexistent" in str(e) for e in result.errors)

    def test_sampling_order_missing_attribute(self):
        """Test that missing attribute in sampling_order is an error."""
        attrs = [make_attr("a"), make_attr("b")]
        spec = make_spec(attrs, sampling_order=["a"])  # Missing "b"
        result = validate_spec(spec)

        assert not result.valid
        assert any("sampling_order" in str(e) for e in result.errors)

    def test_sampling_order_wrong_order(self):
        """Test that wrong dependency order is an error."""
        attrs = [
            make_attr("a"),
            make_attr("b", strategy="derived", formula="a + 1", depends_on=["a"]),
        ]
        spec = make_spec(
            attrs, sampling_order=["b", "a"]
        )  # b depends on a but comes first
        result = validate_spec(spec)

        assert not result.valid
        assert any("before" in str(e) for e in result.errors)


class TestConditionSyntax:
    """Tests for Category 6: Condition Syntax & References."""

    def test_invalid_condition_syntax(self):
        """Test that invalid Python in condition is an error."""
        attr = make_attr(
            "value",
            "int",
            strategy="conditional",
            depends_on=["age"],
            modifiers=[
                Modifier(when="age > (( invalid syntax"),
            ],
        )
        spec = make_spec([make_attr("age"), attr], ["age", "value"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("syntax" in str(e).lower() for e in result.errors)

    def test_condition_references_not_in_depends_on(self):
        """Test that condition referencing attr not in depends_on is an error."""
        attr = make_attr(
            "value",
            "int",
            strategy="conditional",
            depends_on=["age"],
            modifiers=[
                Modifier(
                    when="role == 'senior'", multiply=1.5
                ),  # role not in depends_on
            ],
        )
        spec = make_spec(
            [make_attr("age"), make_attr("role", "categorical"), attr],
            ["age", "role", "value"],
        )
        result = validate_spec(spec)

        assert not result.valid
        assert any("role" in str(e) and "depends_on" in str(e) for e in result.errors)


class TestFormulaValidation:
    """Tests for Category 7: Formula Validation."""

    def test_invalid_formula_syntax(self):
        """Test that invalid formula syntax is an error."""
        attr = make_attr(
            "derived",
            "int",
            strategy="derived",
            formula="age + ((missing_paren",  # Actual syntax error
            depends_on=["age"],
        )
        spec = make_spec([make_attr("age"), attr], ["age", "derived"])
        result = validate_spec(spec)

        assert not result.valid
        assert any(
            "formula" in str(e).lower() and "syntax" in str(e).lower()
            for e in result.errors
        )

    def test_formula_references_not_in_depends_on(self):
        """Test that formula referencing attr not in depends_on is an error."""
        attr = make_attr(
            "derived",
            "int",
            strategy="derived",
            formula="age + bonus",
            depends_on=["age"],  # bonus not in depends_on
        )
        spec = make_spec(
            [make_attr("age"), make_attr("bonus"), attr], ["age", "bonus", "derived"]
        )
        result = validate_spec(spec)

        assert not result.valid
        assert any("bonus" in str(e) and "depends_on" in str(e) for e in result.errors)


class TestDuplicateDetection:
    """Tests for Category 8: Duplicate Detection."""

    def test_duplicate_attribute_names(self):
        """Test that duplicate attribute names is an error."""
        attrs = [make_attr("age"), make_attr("age")]
        spec = make_spec(attrs, sampling_order=["age"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("duplicate" in str(e).lower() for e in result.errors)


class TestStrategyConsistency:
    """Tests for Category 9: Strategy Consistency."""

    def test_independent_without_distribution(self):
        """Test that independent strategy without distribution is an error."""
        attr = AttributeSpec(
            name="value",
            type="int",
            category="universal",
            description="Value",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=None,
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("distribution" in str(e) for e in result.errors)

    def test_independent_with_modifiers(self):
        """Test that independent strategy with modifiers is an error."""
        attr = make_attr(
            "value",
            "int",
            strategy="independent",
            modifiers=[Modifier(when="True", multiply=1.5)],
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("modifiers" in str(e) for e in result.errors)

    def test_independent_with_depends_on(self):
        """Test that independent strategy with depends_on is an error."""
        attr = AttributeSpec(
            name="value",
            type="int",
            category="universal",
            description="Value",
            sampling=SamplingConfig(
                strategy="independent",
                distribution=NormalDistribution(mean=50.0, std=10.0),
                depends_on=["age"],
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([make_attr("age"), attr], ["age", "value"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("depends_on" in str(e) for e in result.errors)

    def test_derived_without_formula(self):
        """Test that derived strategy without formula is an error."""
        attr = AttributeSpec(
            name="derived",
            type="int",
            category="universal",
            description="Derived",
            sampling=SamplingConfig(
                strategy="derived",
                formula=None,
                depends_on=["age"],
            ),
            grounding=GroundingInfo(level="low", method="computed"),
        )
        spec = make_spec([make_attr("age"), attr], ["age", "derived"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("formula" in str(e) for e in result.errors)

    def test_derived_without_depends_on(self):
        """Test that derived strategy without depends_on is an error."""
        attr = AttributeSpec(
            name="derived",
            type="int",
            category="universal",
            description="Derived",
            sampling=SamplingConfig(
                strategy="derived",
                formula="42",  # Constant formula
                depends_on=[],
            ),
            grounding=GroundingInfo(level="low", method="computed"),
        )
        spec = make_spec([attr])
        result = validate_spec(spec)

        assert not result.valid
        assert any("depends_on" in str(e) for e in result.errors)

    def test_conditional_without_distribution(self):
        """Test that conditional strategy without distribution is an error."""
        attr = AttributeSpec(
            name="value",
            type="int",
            category="universal",
            description="Value",
            sampling=SamplingConfig(
                strategy="conditional",
                distribution=None,
                depends_on=["age"],
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([make_attr("age"), attr], ["age", "value"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("distribution" in str(e) for e in result.errors)

    def test_conditional_with_formula(self):
        """Test that conditional strategy with formula is an error."""
        attr = AttributeSpec(
            name="value",
            type="int",
            category="universal",
            description="Value",
            sampling=SamplingConfig(
                strategy="conditional",
                distribution=NormalDistribution(mean=50.0, std=10.0),
                formula="age * 2",  # Should not have formula
                depends_on=["age"],
            ),
            grounding=GroundingInfo(level="low", method="estimated"),
        )
        spec = make_spec([make_attr("age"), attr], ["age", "value"])
        result = validate_spec(spec)

        assert not result.valid
        assert any("formula" in str(e) for e in result.errors)


class TestValidationIssue:
    """Tests for ValidationIssue class."""

    def test_issue_str_representation(self):
        """Test string representation of issues."""
        issue = ValidationIssue(
            severity=Severity.ERROR,
            category="TEST",
            location="age",
            message="Test error message",
        )
        str_repr = str(issue)
        assert "age" in str_repr
        assert "Test error message" in str_repr

    def test_issue_with_modifier_index(self):
        """Test issue with modifier index."""
        issue = ValidationIssue(
            severity=Severity.ERROR,
            category="TEST",
            location="salary",
            message="Test error",
            modifier_index=2,
        )
        str_repr = str(issue)
        assert "salary[2]" in str_repr


class TestComplexSpec:
    """Tests for validation of complex specs."""

    def test_complex_spec_valid(self, complex_population_spec):
        """Test that a well-formed complex spec passes validation."""
        result = validate_spec(complex_population_spec)
        assert result.valid is True

    def test_spec_with_mean_formula(self):
        """Test validation of spec with mean_formula."""
        attrs = [
            make_attr("age"),
            AttributeSpec(
                name="experience",
                type="int",
                category="population_specific",
                description="Years of experience",
                sampling=SamplingConfig(
                    strategy="conditional",
                    distribution=NormalDistribution(
                        mean_formula="age - 28",
                        std=3.0,
                        min=0.0,
                        max=50.0,
                    ),
                    depends_on=["age"],
                ),
                grounding=GroundingInfo(level="medium", method="estimated"),
            ),
        ]
        spec = make_spec(attrs, ["age", "experience"])
        result = validate_spec(spec)

        assert result.valid is True
