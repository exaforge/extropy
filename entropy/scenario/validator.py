"""Step 6: Scenario Validation.

Validates a ScenarioSpec against the population spec, agents, and network
to ensure all references are valid and configurations are consistent.
"""

import ast
import json
import re
from pathlib import Path

from ..core.models import (
    PopulationSpec,
    ScenarioSpec,
    ValidationError,
    ValidationWarning,
    ValidationResult,
)


def _extract_attribute_references(expression: str) -> set[str]:
    """Extract attribute names referenced in a Python expression.

    Args:
        expression: Python expression like "age < 45 and income > 50000"

    Returns:
        Set of attribute names referenced (e.g., {'age', 'income'})
    """
    # Handle special case of 'true' or 'false'
    if expression.lower() in ('true', 'false'):
        return set()

    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        # Can't parse - return empty set, will be caught by syntax check
        return set()

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            # Skip built-in names and common operators
            if node.id not in ('true', 'false', 'True', 'False', 'None', 'and', 'or', 'not'):
                names.add(node.id)

    return names


def _validate_expression_syntax(expression: str) -> str | None:
    """Check if a Python expression is syntactically valid.

    Args:
        expression: Python expression to validate

    Returns:
        Error message if invalid, None if valid
    """
    # Handle special case of 'true' or 'false'
    if expression.lower() in ('true', 'false'):
        return None

    try:
        ast.parse(expression, mode='eval')
        return None
    except SyntaxError as e:
        return str(e)


def validate_scenario(
    spec: ScenarioSpec,
    population_spec: PopulationSpec | None = None,
    agents: list[dict] | None = None,
    network: dict | None = None,
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
        agents: Optional list of agent dicts for consistency checks
        network: Optional network dict for edge type validation

    Returns:
        ValidationResult with errors and warnings
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    # Build set of known attributes from population spec
    known_attributes: set[str] = set()
    if population_spec:
        known_attributes = {attr.name for attr in population_spec.attributes}

    # Build set of known edge types from network
    # Check both 'edge_type' and 'type' fields (different network formats)
    known_edge_types: set[str] = set()
    if network and 'edges' in network:
        for edge in network['edges']:
            if 'edge_type' in edge:
                known_edge_types.add(edge['edge_type'])
            elif 'type' in edge:
                known_edge_types.add(edge['type'])

    # Build set of defined channels
    defined_channels = {ch.name for ch in spec.seed_exposure.channels}

    # =========================================================================
    # Validate Event
    # =========================================================================

    if not spec.event.content.strip():
        errors.append(ValidationError(
            category="event",
            location="event.content",
            message="Event content cannot be empty",
            suggestion="Add a description of the event/information",
        ))

    if not spec.event.source.strip():
        errors.append(ValidationError(
            category="event",
            location="event.source",
            message="Event source cannot be empty",
            suggestion="Specify the source of the event (e.g., 'Netflix', 'government')",
        ))

    # =========================================================================
    # Validate Exposure Channels
    # =========================================================================

    channel_names = set()
    for i, channel in enumerate(spec.seed_exposure.channels):
        if channel.name in channel_names:
            errors.append(ValidationError(
                category="exposure_channel",
                location=f"seed_exposure.channels[{i}]",
                message=f"Duplicate channel name: '{channel.name}'",
                suggestion="Use unique names for each channel",
            ))
        channel_names.add(channel.name)

        if not re.match(r'^[a-z][a-z0-9_]*$', channel.name):
            warnings.append(ValidationWarning(
                category="exposure_channel",
                location=f"seed_exposure.channels[{i}].name",
                message=f"Channel name '{channel.name}' should be snake_case",
            ))

    # =========================================================================
    # Validate Exposure Rules
    # =========================================================================

    for i, rule in enumerate(spec.seed_exposure.rules):
        # Check channel reference
        if rule.channel not in defined_channels:
            errors.append(ValidationError(
                category="exposure_rule",
                location=f"seed_exposure.rules[{i}].channel",
                message=f"Rule references undefined channel: '{rule.channel}'",
                suggestion=f"Define the channel first or use one of: {', '.join(sorted(defined_channels))}",
            ))

        # Check expression syntax
        syntax_error = _validate_expression_syntax(rule.when)
        if syntax_error:
            errors.append(ValidationError(
                category="exposure_rule",
                location=f"seed_exposure.rules[{i}].when",
                message=f"Invalid expression syntax: {syntax_error}",
                suggestion="Use valid Python expression syntax",
            ))
        else:
            # Check attribute references
            if population_spec:
                refs = _extract_attribute_references(rule.when)
                unknown_refs = refs - known_attributes
                if unknown_refs:
                    errors.append(ValidationError(
                        category="attribute_reference",
                        location=f"seed_exposure.rules[{i}].when",
                        message=f"References unknown attribute(s): {', '.join(sorted(unknown_refs))}",
                        suggestion="Check attribute names in population spec",
                    ))

        # Check probability bounds (already enforced by Pydantic, but double-check)
        if not 0 <= rule.probability <= 1:
            errors.append(ValidationError(
                category="probability",
                location=f"seed_exposure.rules[{i}].probability",
                message=f"Probability {rule.probability} out of range [0, 1]",
                suggestion="Use a value between 0 and 1",
            ))

        # Check timestep
        if rule.timestep < 0:
            errors.append(ValidationError(
                category="timestep",
                location=f"seed_exposure.rules[{i}].timestep",
                message=f"Timestep cannot be negative: {rule.timestep}",
                suggestion="Use a non-negative integer",
            ))

        if rule.timestep > spec.simulation.max_timesteps:
            warnings.append(ValidationWarning(
                category="timestep",
                location=f"seed_exposure.rules[{i}].timestep",
                message=f"Timestep {rule.timestep} exceeds max_timesteps {spec.simulation.max_timesteps}",
            ))

    # Check that at least one exposure rule exists
    if not spec.seed_exposure.rules:
        errors.append(ValidationError(
            category="exposure_rule",
            location="seed_exposure.rules",
            message="No exposure rules defined",
            suggestion="Add at least one exposure rule to seed the event",
        ))

    # =========================================================================
    # Validate Spread Modifiers
    # =========================================================================

    for i, modifier in enumerate(spec.spread.share_modifiers):
        # Check expression syntax
        syntax_error = _validate_expression_syntax(modifier.when)
        if syntax_error:
            errors.append(ValidationError(
                category="spread_modifier",
                location=f"spread.share_modifiers[{i}].when",
                message=f"Invalid expression syntax: {syntax_error}",
                suggestion="Use valid Python expression syntax",
            ))
        else:
            # Check attribute/edge type references
            refs = _extract_attribute_references(modifier.when)

            # Allow 'edge_type' as a special reference
            refs_without_edge_type = refs - {'edge_type'}

            if population_spec:
                unknown_refs = refs_without_edge_type - known_attributes
                if unknown_refs:
                    errors.append(ValidationError(
                        category="attribute_reference",
                        location=f"spread.share_modifiers[{i}].when",
                        message=f"References unknown attribute(s): {', '.join(sorted(unknown_refs))}",
                        suggestion="Check attribute names in population spec",
                    ))

            # Check edge type references
            if 'edge_type' in refs:
                # Extract the edge type being compared
                edge_type_match = re.search(r"edge_type\s*==\s*['\"]([^'\"]+)['\"]", modifier.when)
                if edge_type_match and network:
                    referenced_edge_type = edge_type_match.group(1)
                    if referenced_edge_type not in known_edge_types:
                        warnings.append(ValidationWarning(
                            category="edge_type_reference",
                            location=f"spread.share_modifiers[{i}].when",
                            message=f"References edge_type '{referenced_edge_type}' not found in network",
                        ))

        # Warn about potentially problematic multipliers
        if modifier.multiply < 0:
            warnings.append(ValidationWarning(
                category="spread_modifier",
                location=f"spread.share_modifiers[{i}].multiply",
                message=f"Negative multiplier {modifier.multiply} may cause unexpected behavior",
            ))

        if modifier.multiply > 5:
            warnings.append(ValidationWarning(
                category="spread_modifier",
                location=f"spread.share_modifiers[{i}].multiply",
                message=f"Large multiplier {modifier.multiply} may cause probability > 1",
            ))

    # =========================================================================
    # Validate Outcomes
    # =========================================================================

    outcome_names = set()
    for i, outcome in enumerate(spec.outcomes.suggested_outcomes):
        # Check for duplicate names
        if outcome.name in outcome_names:
            errors.append(ValidationError(
                category="outcome",
                location=f"outcomes.suggested_outcomes[{i}]",
                message=f"Duplicate outcome name: '{outcome.name}'",
                suggestion="Use unique names for each outcome",
            ))
        outcome_names.add(outcome.name)

        # Check name format
        if not re.match(r'^[a-z][a-z0-9_]*$', outcome.name):
            warnings.append(ValidationWarning(
                category="outcome",
                location=f"outcomes.suggested_outcomes[{i}].name",
                message=f"Outcome name '{outcome.name}' should be snake_case",
            ))

        # Validate categorical outcomes have options
        if outcome.type.value == "categorical":
            if not outcome.options or len(outcome.options) < 2:
                errors.append(ValidationError(
                    category="outcome",
                    location=f"outcomes.suggested_outcomes[{i}].options",
                    message="Categorical outcomes must have at least 2 options",
                    suggestion="Add options list with at least 2 values",
                ))

        # Validate float outcomes have valid range
        if outcome.type.value == "float":
            if outcome.range:
                min_val, max_val = outcome.range
                if min_val >= max_val:
                    errors.append(ValidationError(
                        category="outcome",
                        location=f"outcomes.suggested_outcomes[{i}].range",
                        message=f"Invalid range: min ({min_val}) >= max ({max_val})",
                        suggestion="Ensure min < max",
                    ))

    # Check for at least one outcome
    if not spec.outcomes.suggested_outcomes:
        warnings.append(ValidationWarning(
            category="outcome",
            location="outcomes.suggested_outcomes",
            message="No outcomes defined - simulation won't measure anything",
        ))

    # =========================================================================
    # Validate Simulation Config
    # =========================================================================

    if spec.simulation.max_timesteps < 1:
        errors.append(ValidationError(
            category="simulation",
            location="simulation.max_timesteps",
            message="max_timesteps must be at least 1",
            suggestion="Set max_timesteps to a positive integer",
        ))

    # Validate stop conditions if present
    if spec.simulation.stop_conditions:
        for i, condition in enumerate(spec.simulation.stop_conditions):
            syntax_error = _validate_expression_syntax(condition)
            if syntax_error:
                errors.append(ValidationError(
                    category="simulation",
                    location=f"simulation.stop_conditions[{i}]",
                    message=f"Invalid stop condition syntax: {syntax_error}",
                    suggestion="Use valid Python expression syntax",
                ))

    # =========================================================================
    # Validate File References
    # =========================================================================

    # Check if referenced files exist
    population_path = Path(spec.meta.population_spec)
    if not population_path.exists():
        errors.append(ValidationError(
            category="file_reference",
            location="meta.population_spec",
            message=f"Population spec not found: {spec.meta.population_spec}",
            suggestion="Check the file path",
        ))

    agents_path = Path(spec.meta.agents_file)
    if not agents_path.exists():
        errors.append(ValidationError(
            category="file_reference",
            location="meta.agents_file",
            message=f"Agents file not found: {spec.meta.agents_file}",
            suggestion="Check the file path",
        ))

    network_path = Path(spec.meta.network_file)
    if not network_path.exists():
        errors.append(ValidationError(
            category="file_reference",
            location="meta.network_file",
            message=f"Network file not found: {spec.meta.network_file}",
            suggestion="Check the file path",
        ))

    # =========================================================================
    # Validate Agent Count Consistency
    # =========================================================================

    if agents and population_spec:
        if len(agents) != population_spec.meta.size:
            warnings.append(ValidationWarning(
                category="consistency",
                location="agents",
                message=f"Agent count ({len(agents)}) differs from population spec size ({population_spec.meta.size})",
            ))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


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
    agents = None
    network = None

    pop_path = Path(spec.meta.population_spec)
    if pop_path.exists():
        try:
            population_spec = PopulationSpec.from_yaml(pop_path)
        except Exception:
            pass  # Will be caught as validation error

    agents_path = Path(spec.meta.agents_file)
    if agents_path.exists():
        try:
            with open(agents_path) as f:
                data = json.load(f)
                agents = data.get('agents', data)  # Handle both formats
        except Exception:
            pass

    network_path = Path(spec.meta.network_file)
    if network_path.exists():
        try:
            with open(network_path) as f:
                network = json.load(f)
        except Exception:
            pass

    # Validate
    result = validate_scenario(spec, population_spec, agents, network)

    return spec, result
