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
from ..storage import open_study_db


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


def _try_resolve_base_population(
    scenario_path: Path, base_pop_ref: str
) -> PopulationSpec | None:
    """Try to resolve base_population reference from study folder structure."""
    # Parse "population.v2" -> ("population", 2)
    match = re.match(r"^(.+)\.v(\d+)$", base_pop_ref)
    if not match:
        return None

    pop_name, pop_version = match.group(1), int(match.group(2))

    # Navigate: {study_root}/scenario/{name}/scenario.v{N}.yaml -> {study_root}/
    scenario_dir = scenario_path.parent
    scenarios_dir = scenario_dir.parent
    study_root = scenarios_dir.parent

    if scenarios_dir.name != "scenario":
        return None

    pop_path = study_root / f"{pop_name}.v{pop_version}.yaml"
    if not pop_path.exists():
        return None

    try:
        return PopulationSpec.from_yaml(pop_path)
    except Exception:
        return None


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
    # Validate Timeline
    # =========================================================================

    if spec.timeline:
        seen_timesteps: set[int] = set()
        valid_intensity = {"normal", "high", "extreme"}
        for i, te in enumerate(spec.timeline):
            if te.timestep in seen_timesteps:
                errors.append(
                    ValidationError(
                        category="timeline",
                        location=f"timeline[{i}].timestep",
                        message=f"Duplicate timeline timestep: {te.timestep}",
                        suggestion="Use unique timesteps for timeline events",
                    )
                )
            seen_timesteps.add(te.timestep)

            if te.timestep >= spec.simulation.max_timesteps:
                warnings.append(
                    ValidationWarning(
                        category="timeline",
                        location=f"timeline[{i}].timestep",
                        message=(
                            f"Timeline timestep {te.timestep} is outside max_timesteps "
                            f"{spec.simulation.max_timesteps}"
                        ),
                        suggestion="Increase simulation.max_timesteps or move event earlier",
                    )
                )

            if te.re_reasoning_intensity is not None and (
                te.re_reasoning_intensity not in valid_intensity
            ):
                errors.append(
                    ValidationError(
                        category="timeline",
                        location=f"timeline[{i}].re_reasoning_intensity",
                        message=(
                            "Invalid re_reasoning_intensity. Must be one of: "
                            "normal, high, extreme"
                        ),
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

    if (
        spec.simulation.allow_early_convergence is True
        and spec.timeline
        and any(te.timestep > 0 for te in spec.timeline)
    ):
        warnings.append(
            ValidationWarning(
                category="simulation",
                location="simulation.allow_early_convergence",
                message=(
                    "allow_early_convergence=true with future timeline events may "
                    "truncate evolving scenarios before all events fire"
                ),
                suggestion=(
                    "Use null/auto or false unless early stop is intentionally desired"
                ),
            )
        )

    if spec.simulation.allow_early_convergence is False and not spec.timeline:
        warnings.append(
            ValidationWarning(
                category="simulation",
                location="simulation.allow_early_convergence",
                message=(
                    "allow_early_convergence=false without timeline events may run "
                    "unnecessarily long after opinions stabilize"
                ),
                suggestion=(
                    "Use null/auto unless full-length execution is intentionally desired"
                ),
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
    # Validate Agent Focus Mode
    # =========================================================================

    if spec.agent_focus_mode is not None:
        valid_modes = {"primary_only", "couples", "all"}
        if spec.agent_focus_mode not in valid_modes:
            errors.append(
                ValidationError(
                    category="agent_focus",
                    location="agent_focus_mode",
                    message=f"Invalid agent_focus_mode: '{spec.agent_focus_mode}'",
                    suggestion=f"Use one of: {', '.join(sorted(valid_modes))}",
                )
            )
    elif spec.household_config is not None:
        warnings.append(
            ValidationWarning(
                category="agent_focus",
                location="agent_focus_mode",
                message="household_config present but agent_focus_mode not set (defaults to primary_only)",
                suggestion="Set agent_focus_mode to 'primary_only', 'couples', or 'all'",
            )
        )

    # =========================================================================
    # Validate File References
    # =========================================================================

    # Determine which flow is being used
    has_base_population = bool(spec.meta.base_population)
    has_legacy_refs = bool(spec.meta.population_spec and spec.meta.study_db)

    if has_base_population:
        # New flow: validate base_population format
        if not re.match(r"^.+\.v\d+$", spec.meta.base_population):
            errors.append(
                ValidationError(
                    category="file_reference",
                    location="meta.base_population",
                    message=f"Invalid base_population format: {spec.meta.base_population}",
                    suggestion="Use format like 'population.v1' or 'population.v2'",
                )
            )
        # If we have a spec_file, try to resolve and validate the population exists
        elif spec_file is not None:
            resolved_pop = _try_resolve_base_population(
                Path(spec_file), spec.meta.base_population
            )
            if resolved_pop is None:
                warnings.append(
                    ValidationWarning(
                        category="file_reference",
                        location="meta.base_population",
                        message=f"Could not resolve base_population: {spec.meta.base_population}",
                        suggestion="Ensure the population file exists at the study root",
                    )
                )

    elif has_legacy_refs:
        # Legacy flow: validate population_spec + study_db paths
        if spec_file is not None:
            from ..utils import resolve_relative_to

            base_file = Path(spec_file)
            population_path = resolve_relative_to(spec.meta.population_spec, base_file)
            study_db_path = resolve_relative_to(spec.meta.study_db, base_file)
        else:
            population_path = Path(spec.meta.population_spec)
            study_db_path = Path(spec.meta.study_db)

        if not population_path.exists():
            errors.append(
                ValidationError(
                    category="file_reference",
                    location="meta.population_spec",
                    message=f"Population spec not found: {spec.meta.population_spec}",
                    suggestion="Check the file path",
                )
            )

        if not study_db_path.exists():
            errors.append(
                ValidationError(
                    category="file_reference",
                    location="meta.study_db",
                    message=f"Study DB not found: {spec.meta.study_db}",
                    suggestion="Check the file path",
                )
            )

        # Validate IDs inside study DB when available.
        if study_db_path.exists():
            try:
                with open_study_db(study_db_path) as db:
                    agent_count = db.get_agent_count_by_scenario(spec.meta.name)
                    if agent_count == 0:
                        agent_count = db.get_agent_count(spec.meta.population_id)
                    if agent_count == 0:
                        errors.append(
                            ValidationError(
                                category="file_reference",
                                location="meta.population_id",
                                message=f"No agents found for scenario '{spec.meta.name}' or population_id '{spec.meta.population_id}' in study DB",
                                suggestion="Run `extropy sample -s <scenario>` first",
                            )
                        )
                    edge_count = db.get_network_edge_count_by_scenario(spec.meta.name)
                    if edge_count == 0:
                        edge_count = db.get_network_edge_count(spec.meta.network_id)
                    if edge_count == 0:
                        errors.append(
                            ValidationError(
                                category="file_reference",
                                location="meta.network_id",
                                message=f"No network edges found for scenario '{spec.meta.name}' or network_id '{spec.meta.network_id}' in study DB",
                                suggestion="Run `extropy network -s <scenario>` first",
                            )
                        )
            except Exception:
                errors.append(
                    ValidationError(
                        category="file_reference",
                        location="meta.study_db",
                        message=f"Failed to read study DB: {spec.meta.study_db}",
                        suggestion="Check that the file is a valid SQLite study DB",
                    )
                )

    else:
        # Neither flow configured
        errors.append(
            ValidationError(
                category="file_reference",
                location="meta",
                message="Scenario must specify either base_population (new flow) or population_spec + study_db (legacy flow)",
                suggestion="Add meta.base_population or both meta.population_spec and meta.study_db",
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

    # Determine which flow to use
    if spec.meta.base_population:
        # New flow: resolve base_population from study folder structure
        population_spec = _try_resolve_base_population(
            scenario_path, spec.meta.base_population
        )
    elif spec.meta.population_spec:
        # Legacy flow: resolve population_spec path directly
        from ..utils import resolve_relative_to

        pop_path = resolve_relative_to(spec.meta.population_spec, scenario_path)
        if pop_path.exists():
            try:
                population_spec = PopulationSpec.from_yaml(pop_path)
            except Exception:
                pass  # Will be caught as validation error

    # Load study DB if using legacy flow
    if spec.meta.study_db:
        from ..utils import resolve_relative_to

        study_db_path = resolve_relative_to(spec.meta.study_db, scenario_path)
        if study_db_path.exists():
            try:
                with open_study_db(study_db_path) as db:
                    agent_count = db.get_agent_count(spec.meta.population_id)
                    network = db.get_network(spec.meta.network_id)
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
