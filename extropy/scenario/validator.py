"""Step 6: Scenario Validation.

Validates a ScenarioSpec against the population spec, agents, and network
to ensure all references are valid and configurations are consistent.
"""

import json
import re
from pathlib import Path

from ..core.models import (
    OutcomeType,
    PopulationSpec,
    ScenarioSpec,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from ..utils.expressions import (
    extract_names_from_expression,
    validate_expression_syntax,
)


# Helper functions to create ValidationIssue with appropriate severity
def ValidationError(
    category: str,
    location: str,
    message: str,
    suggestion: str | None = None,
) -> ValidationIssue:
    """Create an ERROR-level validation issue."""
    return ValidationIssue(
        severity=Severity.ERROR,
        category=category,
        location=location,
        message=message,
        suggestion=suggestion,
    )


def ValidationWarning(
    category: str,
    location: str,
    message: str,
    suggestion: str | None = None,
) -> ValidationIssue:
    """Create a WARNING-level validation issue."""
    return ValidationIssue(
        severity=Severity.WARNING,
        category=category,
        location=location,
        message=message,
        suggestion=suggestion,
    )


def validate_scenario(
    spec: ScenarioSpec,
    population_spec: PopulationSpec | None = None,
    agent_count: int | None = None,
    network: dict | None = None,
    spec_file: Path | str | None = None,
) -> ValidationResult:
    """
    Validate a scenario spec for correctness.

    Checks:
    - All 'when' clauses reference valid attributes
    - Probabilities are in valid ranges
    - Timesteps are valid
    - Outcome definitions are consistent
    - Channel references are valid
    - File references exist (if paths provided)

    Args:
        spec: The scenario spec to validate
        population_spec: Optional population spec for attribute validation
        agent_count: Optional count of agents for consistency checks
        network: Optional network dict for edge type validation
        spec_file: Optional scenario file path used to resolve relative file references

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    # Build set of known attributes from population spec
    known_attributes: set[str] = set()
    if population_spec:
        known_attributes = {attr.name for attr in population_spec.attributes}

    # Build set of known edge types from network
    # Check both 'edge_type' and 'type' fields (different network formats)
    known_edge_types: set[str] = set()
    if network and "edges" in network:
        for edge in network["edges"]:
            if "edge_type" in edge:
                known_edge_types.add(edge["edge_type"])
            elif "type" in edge:
                known_edge_types.add(edge["type"])

    # Build set of defined channels
    defined_channels = {ch.name for ch in spec.seed_exposure.channels}

    # =========================================================================
    # Validate Event
    # =========================================================================

    if not spec.event.content.strip():
        errors.append(
            ValidationError(
                category="event",
                location="event.content",
                message="Event content cannot be empty",
                suggestion="Add a description of the event/information",
            )
        )

    if not spec.event.source.strip():
        errors.append(
            ValidationError(
                category="event",
                location="event.source",
                message="Event source cannot be empty",
                suggestion="Specify the source of the event (e.g., 'Netflix', 'government')",
            )
        )

    # =========================================================================
    # Validate Exposure Channels
    # =========================================================================

    channel_names = set()
    for i, channel in enumerate(spec.seed_exposure.channels):
        if channel.name in channel_names:
            errors.append(
                ValidationError(
                    category="exposure_channel",
                    location=f"seed_exposure.channels[{i}]",
                    message=f"Duplicate channel name: '{channel.name}'",
                    suggestion="Use unique names for each channel",
                )
            )
        channel_names.add(channel.name)

        if not re.match(r"^[a-z][a-z0-9_]*$", channel.name):
            warnings.append(
                ValidationWarning(
                    category="exposure_channel",
                    location=f"seed_exposure.channels[{i}].name",
                    message=f"Channel name '{channel.name}' should be snake_case",
                )
            )

    # =========================================================================
    # Validate Exposure Rules
    # =========================================================================

    for i, rule in enumerate(spec.seed_exposure.rules):
        # Check channel reference
        if rule.channel not in defined_channels:
            errors.append(
                ValidationError(
                    category="exposure_rule",
                    location=f"seed_exposure.rules[{i}].channel",
                    message=f"Rule references undefined channel: '{rule.channel}'",
                    suggestion=f"Define the channel first or use one of: {', '.join(sorted(defined_channels))}",
                )
            )

        # Check expression syntax
        syntax_error = validate_expression_syntax(rule.when)
        if syntax_error:
            errors.append(
                ValidationError(
                    category="exposure_rule",
                    location=f"seed_exposure.rules[{i}].when",
                    message=f"Invalid expression syntax: {syntax_error}",
                    suggestion="Use valid Python expression syntax",
                )
            )
        else:
            # Check attribute references
            if population_spec:
                refs = extract_names_from_expression(rule.when)
                unknown_refs = refs - known_attributes
                if unknown_refs:
                    errors.append(
                        ValidationError(
                            category="attribute_reference",
                            location=f"seed_exposure.rules[{i}].when",
                            message=f"References unknown attribute(s): {', '.join(sorted(unknown_refs))}",
                            suggestion="Check attribute names in population spec",
                        )
                    )

        # Check probability bounds (already enforced by Pydantic, but double-check)
        if not 0 <= rule.probability <= 1:
            errors.append(
                ValidationError(
                    category="probability",
                    location=f"seed_exposure.rules[{i}].probability",
                    message=f"Probability {rule.probability} out of range [0, 1]",
                    suggestion="Use a value between 0 and 1",
                )
            )

        # Check timestep
        if rule.timestep < 0:
            errors.append(
                ValidationError(
                    category="timestep",
                    location=f"seed_exposure.rules[{i}].timestep",
                    message=f"Timestep cannot be negative: {rule.timestep}",
                    suggestion="Use a non-negative integer",
                )
            )

        if rule.timestep > spec.simulation.max_timesteps:
            warnings.append(
                ValidationWarning(
                    category="timestep",
                    location=f"seed_exposure.rules[{i}].timestep",
                    message=f"Timestep {rule.timestep} exceeds max_timesteps {spec.simulation.max_timesteps}",
                )
            )

    # Check that at least one exposure rule exists
    if not spec.seed_exposure.rules:
        errors.append(
            ValidationError(
                category="exposure_rule",
                location="seed_exposure.rules",
                message="No exposure rules defined",
                suggestion="Add at least one exposure rule to seed the event",
            )
        )

    # =========================================================================
    # Validate Spread Modifiers
    # =========================================================================

    for i, modifier in enumerate(spec.spread.share_modifiers):
        # Check expression syntax
        syntax_error = validate_expression_syntax(modifier.when)
        if syntax_error:
            errors.append(
                ValidationError(
                    category="spread_modifier",
                    location=f"spread.share_modifiers[{i}].when",
                    message=f"Invalid expression syntax: {syntax_error}",
                    suggestion="Use valid Python expression syntax",
                )
            )
        else:
            # Check attribute/edge type references
            refs = extract_names_from_expression(modifier.when)

            # Allow edge attributes injected during propagation
            refs_without_edge_fields = refs - {"edge_type", "edge_weight"}

            if population_spec:
                unknown_refs = refs_without_edge_fields - known_attributes
                if unknown_refs:
                    errors.append(
                        ValidationError(
                            category="attribute_reference",
                            location=f"spread.share_modifiers[{i}].when",
                            message=f"References unknown attribute(s): {', '.join(sorted(unknown_refs))}",
                            suggestion="Check attribute names in population spec",
                        )
                    )

            # Check edge type references
            if "edge_type" in refs:
                # Extract the edge type being compared
                edge_type_match = re.search(
                    r"edge_type\s*==\s*['\"]([^'\"]+)['\"]", modifier.when
                )
                if edge_type_match and network:
                    referenced_edge_type = edge_type_match.group(1)
                    if referenced_edge_type not in known_edge_types:
                        warnings.append(
                            ValidationWarning(
                                category="edge_type_reference",
                                location=f"spread.share_modifiers[{i}].when",
                                message=f"References edge_type '{referenced_edge_type}' not found in network",
                            )
                        )

        # Warn about potentially problematic multipliers
        if modifier.multiply < 0:
            warnings.append(
                ValidationWarning(
                    category="spread_modifier",
                    location=f"spread.share_modifiers[{i}].multiply",
                    message=f"Negative multiplier {modifier.multiply} may cause unexpected behavior",
                )
            )

        if modifier.multiply > 5:
            warnings.append(
                ValidationWarning(
                    category="spread_modifier",
                    location=f"spread.share_modifiers[{i}].multiply",
                    message=f"Large multiplier {modifier.multiply} may cause probability > 1",
                )
            )

    # =========================================================================
    # Validate Outcomes
    # =========================================================================

    outcome_names = set()
    for i, outcome in enumerate(spec.outcomes.suggested_outcomes):
        # Check for duplicate names
        if outcome.name in outcome_names:
            errors.append(
                ValidationError(
                    category="outcome",
                    location=f"outcomes.suggested_outcomes[{i}]",
                    message=f"Duplicate outcome name: '{outcome.name}'",
                    suggestion="Use unique names for each outcome",
                )
            )
        outcome_names.add(outcome.name)

        # Check name format
        if not re.match(r"^[a-z][a-z0-9_]*$", outcome.name):
            warnings.append(
                ValidationWarning(
                    category="outcome",
                    location=f"outcomes.suggested_outcomes[{i}].name",
                    message=f"Outcome name '{outcome.name}' should be snake_case",
                )
            )

        # Validate categorical outcomes have options
        if outcome.type == OutcomeType.CATEGORICAL:
            if not outcome.options or len(outcome.options) < 2:
                errors.append(
                    ValidationError(
                        category="outcome",
                        location=f"outcomes.suggested_outcomes[{i}].options",
                        message="Categorical outcomes must have at least 2 options",
                        suggestion="Add options list with at least 2 values",
                    )
                )

        # Validate float outcomes have valid range
        if outcome.type == OutcomeType.FLOAT:
            if outcome.range:
                min_val, max_val = outcome.range
                if min_val >= max_val:
                    errors.append(
                        ValidationError(
                            category="outcome",
                            location=f"outcomes.suggested_outcomes[{i}].range",
                            message=f"Invalid range: min ({min_val}) >= max ({max_val})",
                            suggestion="Ensure min < max",
                        )
                    )

    # Check for at least one outcome
    if not spec.outcomes.suggested_outcomes:
        warnings.append(
            ValidationWarning(
                category="outcome",
                location="outcomes.suggested_outcomes",
                message="No outcomes defined - simulation won't measure anything",
            )
        )

    # =========================================================================
    # Validate Simulation Config
    # =========================================================================

    if spec.simulation.max_timesteps < 1:
        errors.append(
            ValidationError(
                category="simulation",
                location="simulation.max_timesteps",
                message="max_timesteps must be at least 1",
                suggestion="Set max_timesteps to a positive integer",
            )
        )

    # Validate stop conditions if present
    if spec.simulation.stop_conditions:
        for i, condition in enumerate(spec.simulation.stop_conditions):
            syntax_error = validate_expression_syntax(condition)
            if syntax_error:
                errors.append(
                    ValidationError(
                        category="simulation",
                        location=f"simulation.stop_conditions[{i}]",
                        message=f"Invalid stop condition syntax: {syntax_error}",
                        suggestion="Use valid Python expression syntax",
                    )
                )

    # =========================================================================
    # Validate File References
    # =========================================================================

    # Check if referenced files exist
    if spec_file is not None:
        from ..utils import resolve_relative_to

        base_file = Path(spec_file)
        population_path = resolve_relative_to(spec.meta.population_spec, base_file)
        agents_path = resolve_relative_to(spec.meta.agents_file, base_file)
        network_path = resolve_relative_to(spec.meta.network_file, base_file)
    else:
        population_path = Path(spec.meta.population_spec)
        agents_path = Path(spec.meta.agents_file)
        network_path = Path(spec.meta.network_file)

    if not population_path.exists():
        errors.append(
            ValidationError(
                category="file_reference",
                location="meta.population_spec",
                message=f"Population spec not found: {spec.meta.population_spec}",
                suggestion="Check the file path",
            )
        )

    if not agents_path.exists():
        errors.append(
            ValidationError(
                category="file_reference",
                location="meta.agents_file",
                message=f"Agents file not found: {spec.meta.agents_file}",
                suggestion="Check the file path",
            )
        )

    if not network_path.exists():
        errors.append(
            ValidationError(
                category="file_reference",
                location="meta.network_file",
                message=f"Network file not found: {spec.meta.network_file}",
                suggestion="Check the file path",
            )
        )

    # =========================================================================
    # Validate Agent Count Consistency
    # =========================================================================

    if agent_count is not None and population_spec:
        if agent_count != population_spec.meta.size:
            warnings.append(
                ValidationWarning(
                    category="consistency",
                    location="agents",
                    message=f"Agent count ({agent_count}) differs from population spec size ({population_spec.meta.size})",
                )
            )

    return ValidationResult(issues=[*errors, *warnings])


def get_agent_count(path: Path) -> int | None:
    """
    Safely get agent count from file using standard JSON parsing.

    Prioritizes correctness over memory optimization by fully parsing the JSON.
    This ensures we handle all valid JSON formats correctly.
    """
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        # Case 1: Standard format {"meta": {"count": N}, "agents": [...]}
        if isinstance(data, dict):
            # Trust metadata count if present
            if "meta" in data and isinstance(data["meta"], dict):
                count = data["meta"].get("count")
                if isinstance(count, int):
                    return count

            # Fallback to counting agents list
            agents = data.get("agents")
            if isinstance(agents, list):
                return len(agents)

            # Legacy/Alternative format: data is the dict, maybe agents is missing?
            # If data is a dict but no agents key, it's not a valid agent file we recognize
            return None

        # Case 2: Simple list format [{"id": ...}, ...]
        if isinstance(data, list):
            return len(data)

    except Exception:
        return None

    return None


def load_and_validate_scenario(
    scenario_path: Path | str,
) -> tuple[ScenarioSpec, ValidationResult]:
    """
    Load a scenario spec and validate it against its referenced files.

    Args:
        scenario_path: Path to the scenario YAML file

    Returns:
        Tuple of (ScenarioSpec, ValidationResult)

    Raises:
        FileNotFoundError: If scenario file doesn't exist
        ValueError: If scenario YAML is invalid
    """
    scenario_path = Path(scenario_path)

    # Load scenario spec
    spec = ScenarioSpec.from_yaml(scenario_path)

    # Try to load referenced files for validation
    population_spec = None
    agent_count = None
    network = None

    from ..utils import resolve_relative_to

    pop_path = resolve_relative_to(spec.meta.population_spec, scenario_path)
    if pop_path.exists():
        try:
            population_spec = PopulationSpec.from_yaml(pop_path)
        except Exception:
            pass  # Will be caught as validation error

    agents_path = resolve_relative_to(spec.meta.agents_file, scenario_path)
    if agents_path.exists():
        agent_count = get_agent_count(agents_path)

    network_path = resolve_relative_to(spec.meta.network_file, scenario_path)
    if network_path.exists():
        try:
            with open(network_path) as f:
                network = json.load(f)
        except Exception:
            pass

    # Validate
    result = validate_scenario(
        spec,
        population_spec,
        agent_count,
        network,
        spec_file=scenario_path,
    )

    return spec, result
