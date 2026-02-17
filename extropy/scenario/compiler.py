"""Step 5: Scenario Compiler (Orchestrator).

Orchestrates the full scenario compilation pipeline:
1. Parse scenario description into Event
2. Generate seed exposure rules
3. Determine interaction model and spread config
4. Define outcomes
5. Assemble and validate ScenarioSpec
"""

import re
from datetime import datetime
from pathlib import Path

from ..core.models import (
    PopulationSpec,
    ScenarioMeta,
    ScenarioSpec,
    SimulationConfig,
    TimestepUnit,
    ValidationResult,
)
from .parser import parse_scenario
from .exposure import generate_seed_exposure
from .interaction import determine_interaction_model
from .outcomes import define_outcomes
from .timeline import generate_timeline
from ..utils.callbacks import StepProgressCallback
from .validator import validate_scenario
from ..storage import open_study_db


def _generate_scenario_name(description: str) -> str:
    """Generate a short snake_case name from a scenario description."""
    # Take first few words, lowercase, replace spaces with underscores
    words = description.lower().split()[:4]
    # Remove non-alphanumeric characters
    words = [re.sub(r"[^a-z0-9]", "", word) for word in words]
    # Filter empty strings
    words = [w for w in words if w]
    return "_".join(words) or "scenario"


def _determine_simulation_config() -> SimulationConfig:
    """Determine default simulation configuration.

    Uses a fixed default since population size is determined at sample time.
    """
    return SimulationConfig(
        max_timesteps=100,
        timestep_unit=TimestepUnit.HOUR,
        stop_conditions=["exposure_rate > 0.95 and no_state_changes_for > 10"],
        seed=None,
    )


def _load_network_summary(network_data: dict[str, object]) -> dict[str, object]:
    """Build network summary for exposure generation from network payload."""
    edge_types = set()
    node_count = 0

    meta = network_data.get("meta")
    if isinstance(meta, dict):
        raw_count = meta.get("node_count")
        if isinstance(raw_count, int):
            node_count = raw_count

    edges = network_data.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            edge_type = edge.get("edge_type") or edge.get("type")
            if isinstance(edge_type, str):
                edge_types.add(edge_type)

    return {
        "node_count": node_count,
        "edge_types": list(edge_types),
    }


def create_scenario(
    description: str,
    population_spec_path: str | Path,
    study_db_path: str | Path,
    population_id: str = "default",
    network_id: str = "default",
    output_path: str | Path | None = None,
    on_progress: StepProgressCallback | None = None,
    timeline_mode: str | None = None,
) -> tuple[ScenarioSpec, ValidationResult]:
    """
    Create a complete scenario spec from a description.

    Orchestrates the full pipeline:
    1. Load population spec and parse event
    2. Generate seed exposure rules
    3. Determine interaction model and spread config
    4. Define outcomes
    5. Generate timeline and background context
    6. Assemble ScenarioSpec and validate

    Args:
        description: Natural language scenario description
        population_spec_path: Path to population YAML file
        study_db_path: Path to canonical study DB
        population_id: Population ID in study DB
        network_id: Network ID in study DB
        output_path: Optional path to save scenario YAML
        on_progress: Optional callback(step, status) for progress updates
        timeline_mode: Timeline mode override. None = auto-detect, "static" = single event,
            "evolving" = multi-event timeline.

    Returns:
        Tuple of (ScenarioSpec, ValidationResult)

    Raises:
        FileNotFoundError: If required input files don't exist
        ValueError: If input files are invalid

    Example:
        >>> spec, result = create_scenario(
        ...     "Netflix announces $3 price increase",
        ...     "population.yaml",
        ...     "study.db",
        ...     "default",
        ...     "default",
        ...     "scenario.yaml"
        ... )
        >>> result.valid
        True
    """
    population_spec_path = Path(population_spec_path)
    study_db_path = Path(study_db_path)

    def progress(step: str, status: str):
        if on_progress:
            on_progress(step, status)

    # =========================================================================
    # Load inputs
    # =========================================================================

    progress("1/6", "Loading population spec...")

    if not population_spec_path.exists():
        raise FileNotFoundError(f"Population spec not found: {population_spec_path}")

    population_spec = PopulationSpec.from_yaml(population_spec_path)

    # Load network summary for exposure generation
    with open_study_db(study_db_path) as db:
        network = db.get_network(network_id)
        if not network.get("edges"):
            raise FileNotFoundError(
                f"Network '{network_id}' not found in study DB: {study_db_path}"
            )
        network_summary = _load_network_summary(network)

    # =========================================================================
    # Step 1: Parse scenario description
    # =========================================================================

    progress("1/6", "Parsing event definition...")

    event = parse_scenario(description, population_spec)

    # =========================================================================
    # Step 2: Generate seed exposure
    # =========================================================================

    progress("2/6", "Generating seed exposure rules...")

    seed_exposure = generate_seed_exposure(
        event,
        population_spec,
        network_summary,
    )

    # =========================================================================
    # Step 3: Determine interaction model
    # =========================================================================

    progress("3/6", "Determining interaction model...")

    interaction_config, spread_config = determine_interaction_model(
        event,
        population_spec,
        network_summary,
    )

    # =========================================================================
    # Step 4: Define outcomes
    # =========================================================================

    progress("4/6", "Defining outcomes...")

    outcome_config = define_outcomes(
        event,
        population_spec,
        description,
    )

    # =========================================================================
    # Step 5: Generate timeline + background context
    # =========================================================================

    progress("5/6", "Generating timeline...")

    # Generate simulation config
    simulation_config = _determine_simulation_config()

    timeline_events, background_context = generate_timeline(
        scenario_description=description,
        base_event=event,
        simulation_config=simulation_config,
        timeline_mode=timeline_mode,
    )

    # =========================================================================
    # Step 6: Assemble scenario spec
    # =========================================================================

    progress("6/6", "Assembling scenario spec...")

    # Generate scenario name
    scenario_name = _generate_scenario_name(description)

    # Create metadata
    meta = ScenarioMeta(
        name=scenario_name,
        description=description,
        population_spec=str(population_spec_path),
        study_db=str(study_db_path),
        population_id=population_id,
        network_id=network_id,
        created_at=datetime.now(),
    )

    # Assemble full spec
    spec = ScenarioSpec(
        meta=meta,
        event=event,
        timeline=timeline_events if timeline_events else None,
        seed_exposure=seed_exposure,
        interaction=interaction_config,
        spread=spread_config,
        outcomes=outcome_config,
        simulation=simulation_config,
        background_context=background_context,
    )

    # =========================================================================
    # Validate
    # =========================================================================

    with open_study_db(study_db_path) as db:
        agent_count = db.get_agent_count(population_id)
        network = db.get_network(network_id)

    validation_result = validate_scenario(spec, population_spec, agent_count, network)

    # =========================================================================
    # Save if requested
    # =========================================================================

    if output_path:
        spec.to_yaml(output_path)

    return spec, validation_result


def create_scenario_spec(
    description: str,
    population_spec: PopulationSpec,
    extended_attributes: list | None = None,
    on_progress: StepProgressCallback | None = None,
    timeline_mode: str | None = None,
) -> tuple[ScenarioSpec, ValidationResult]:
    """
    Create a scenario spec without requiring agents or network.

    This is used in the new CLI flow where scenario is created before
    sampling and network generation. Exposure rules use generic patterns
    that work with any network topology.

    Args:
        description: Natural language scenario description
        population_spec: Loaded population spec
        extended_attributes: Optional list of extended AttributeSpecs from scenario
        on_progress: Optional callback(step, status) for progress updates
        timeline_mode: Timeline mode override. None = auto-detect.

    Returns:
        Tuple of (ScenarioSpec, ValidationResult)
    """

    def progress(step: str, status: str):
        if on_progress:
            on_progress(step, status)

    # Step 1: Parse scenario description
    progress("1/5", "Parsing event definition...")
    event = parse_scenario(description, population_spec)

    # Step 2: Generate seed exposure (generic, without network specifics)
    progress("2/5", "Generating seed exposure rules...")

    # Create a minimal network summary - exposure will use generic rules
    # Node count isn't needed for generic rules; will be determined at sample time
    generic_network_summary = {
        "edge_types": ["colleague", "friend", "family"],  # Generic types
    }

    seed_exposure = generate_seed_exposure(
        event,
        population_spec,
        generic_network_summary,
    )

    # Step 3: Determine interaction model
    progress("3/5", "Determining interaction model...")

    interaction_config, spread_config = determine_interaction_model(
        event,
        population_spec,
        generic_network_summary,
    )

    # Step 4: Define outcomes
    progress("4/5", "Defining outcomes...")

    outcome_config = define_outcomes(
        event,
        population_spec,
        description,
    )

    # Step 5: Generate timeline + background context
    progress("5/5", "Generating timeline...")

    simulation_config = _determine_simulation_config()

    timeline_events, background_context = generate_timeline(
        scenario_description=description,
        base_event=event,
        simulation_config=simulation_config,
        timeline_mode=timeline_mode,
    )

    # Assemble scenario spec
    scenario_name = _generate_scenario_name(description)

    meta = ScenarioMeta(
        name=scenario_name,
        description=description,
        created_at=datetime.now(),
    )

    spec = ScenarioSpec(
        meta=meta,
        event=event,
        timeline=timeline_events if timeline_events else None,
        seed_exposure=seed_exposure,
        interaction=interaction_config,
        spread=spread_config,
        outcomes=outcome_config,
        simulation=simulation_config,
        background_context=background_context,
    )

    # Store extended attributes if provided
    if extended_attributes:
        spec.extended_attributes = extended_attributes

    # Light validation (no agents/network to validate against)
    from ..core.models.validation import ValidationResult as VResult

    validation_result = VResult(valid=True, errors=[], warnings=[])

    return spec, validation_result


def compile_scenario_from_files(
    description: str,
    population_spec_path: str | Path,
    study_db_path: str | Path,
    population_id: str = "default",
    network_id: str = "default",
) -> ScenarioSpec:
    """
    Convenience function to create a scenario spec.

    Same as create_scenario but returns only the spec (for simpler usage).

    Args:
        description: Natural language scenario description
        population_spec_path: Path to population YAML file
        study_db_path: Path to canonical study DB
        population_id: Population ID in study DB
        network_id: Network ID in study DB

    Returns:
        ScenarioSpec

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If validation fails with errors
    """
    spec, result = create_scenario(
        description,
        population_spec_path,
        study_db_path,
        population_id,
        network_id,
    )

    if not result.valid:
        errors = "; ".join(e.message for e in result.errors[:3])
        raise ValueError(f"Scenario validation failed: {errors}")

    return spec
