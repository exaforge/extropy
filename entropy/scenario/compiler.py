"""Step 5: Scenario Compiler (Orchestrator).

Orchestrates the full scenario compilation pipeline:
1. Parse scenario description into Event
2. Generate seed exposure rules
3. Determine interaction model and spread config
4. Define outcomes
5. Assemble and validate ScenarioSpec
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..models import PopulationSpec
from .models import (
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
from .validator import validate_scenario


def _generate_scenario_name(description: str) -> str:
    """Generate a short snake_case name from a scenario description."""
    # Take first few words, lowercase, replace spaces with underscores
    words = description.lower().split()[:4]
    # Remove non-alphanumeric characters
    words = [re.sub(r'[^a-z0-9]', '', word) for word in words]
    # Filter empty strings
    words = [w for w in words if w]
    return '_'.join(words) or 'scenario'


def _determine_simulation_config(population_size: int) -> SimulationConfig:
    """Determine default simulation configuration based on population size."""
    if population_size < 500:
        max_timesteps = 50
    elif population_size <= 5000:
        max_timesteps = 100
    else:
        max_timesteps = 168  # 1 week of hours

    return SimulationConfig(
        max_timesteps=max_timesteps,
        timestep_unit=TimestepUnit.HOUR,
        stop_conditions=["exposure_rate > 0.95 and no_state_changes_for > 10"],
        seed=None,
    )


def _load_network_summary(network_path: Path) -> dict | None:
    """Load network summary for exposure generation."""
    if not network_path.exists():
        return None

    try:
        with open(network_path) as f:
            network = json.load(f)

        # Extract summary information
        edge_types = set()
        node_count = 0

        if 'meta' in network:
            node_count = network['meta'].get('node_count', 0)

        if 'edges' in network:
            for edge in network['edges']:
                # Check both 'edge_type' and 'type' fields (different network formats)
                if 'edge_type' in edge:
                    edge_types.add(edge['edge_type'])
                elif 'type' in edge:
                    edge_types.add(edge['type'])

        return {
            'node_count': node_count,
            'edge_types': list(edge_types),
        }
    except Exception:
        return None


def create_scenario(
    description: str,
    population_spec_path: str | Path,
    agents_path: str | Path,
    network_path: str | Path,
    output_path: str | Path | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> tuple[ScenarioSpec, ValidationResult]:
    """
    Create a complete scenario spec from a description.

    Orchestrates the full pipeline:
    1. Load population spec
    2. Parse scenario description into Event
    3. Generate seed exposure rules
    4. Determine interaction model and spread config
    5. Define outcomes
    6. Generate simulation config
    7. Assemble ScenarioSpec
    8. Validate
    9. Optionally save to YAML

    Args:
        description: Natural language scenario description
        population_spec_path: Path to population YAML file
        agents_path: Path to agents JSON file
        network_path: Path to network JSON file
        output_path: Optional path to save scenario YAML
        on_progress: Optional callback(step, status) for progress updates

    Returns:
        Tuple of (ScenarioSpec, ValidationResult)

    Raises:
        FileNotFoundError: If required input files don't exist
        ValueError: If input files are invalid

    Example:
        >>> spec, result = create_scenario(
        ...     "Netflix announces $3 price increase",
        ...     "population.yaml",
        ...     "agents.json",
        ...     "network.json",
        ...     "scenario.yaml"
        ... )
        >>> result.valid
        True
    """
    population_spec_path = Path(population_spec_path)
    agents_path = Path(agents_path)
    network_path = Path(network_path)

    def progress(step: str, status: str):
        if on_progress:
            on_progress(step, status)

    # =========================================================================
    # Load inputs
    # =========================================================================

    progress("1/5", "Loading population spec...")

    if not population_spec_path.exists():
        raise FileNotFoundError(f"Population spec not found: {population_spec_path}")

    population_spec = PopulationSpec.from_yaml(population_spec_path)

    # Load network summary for exposure generation
    network_summary = _load_network_summary(network_path)

    # =========================================================================
    # Step 1: Parse scenario description
    # =========================================================================

    progress("1/5", "Parsing event definition...")

    event = parse_scenario(description, population_spec)

    # =========================================================================
    # Step 2: Generate seed exposure
    # =========================================================================

    progress("2/5", "Generating seed exposure rules...")

    seed_exposure = generate_seed_exposure(
        event,
        population_spec,
        network_summary,
    )

    # =========================================================================
    # Step 3: Determine interaction model
    # =========================================================================

    progress("3/5", "Determining interaction model...")

    interaction_config, spread_config = determine_interaction_model(
        event,
        population_spec,
    )

    # =========================================================================
    # Step 4: Define outcomes
    # =========================================================================

    progress("4/5", "Defining outcomes...")

    outcome_config = define_outcomes(
        event,
        population_spec,
        description,
    )

    # =========================================================================
    # Step 5: Assemble scenario spec
    # =========================================================================

    progress("5/5", "Assembling scenario spec...")

    # Generate simulation config based on population size
    simulation_config = _determine_simulation_config(population_spec.meta.size)

    # Generate scenario name
    scenario_name = _generate_scenario_name(description)

    # Create metadata
    meta = ScenarioMeta(
        name=scenario_name,
        description=description,
        population_spec=str(population_spec_path),
        agents_file=str(agents_path),
        network_file=str(network_path),
        created_at=datetime.now(),
    )

    # Assemble full spec
    spec = ScenarioSpec(
        meta=meta,
        event=event,
        seed_exposure=seed_exposure,
        interaction=interaction_config,
        spread=spread_config,
        outcomes=outcome_config,
        simulation=simulation_config,
    )

    # =========================================================================
    # Validate
    # =========================================================================

    # Load agents for validation
    agents = None
    if agents_path.exists():
        try:
            with open(agents_path) as f:
                data = json.load(f)
                agents = data.get('agents', data)
        except Exception:
            pass

    # Load network for validation
    network = None
    if network_path.exists():
        try:
            with open(network_path) as f:
                network = json.load(f)
        except Exception:
            pass

    validation_result = validate_scenario(spec, population_spec, agents, network)

    # =========================================================================
    # Save if requested
    # =========================================================================

    if output_path:
        spec.to_yaml(output_path)

    return spec, validation_result


def compile_scenario_from_files(
    description: str,
    population_spec_path: str | Path,
    agents_path: str | Path,
    network_path: str | Path,
) -> ScenarioSpec:
    """
    Convenience function to create a scenario spec.

    Same as create_scenario but returns only the spec (for simpler usage).

    Args:
        description: Natural language scenario description
        population_spec_path: Path to population YAML file
        agents_path: Path to agents JSON file
        network_path: Path to network JSON file

    Returns:
        ScenarioSpec

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If validation fails with errors
    """
    spec, result = create_scenario(
        description,
        population_spec_path,
        agents_path,
        network_path,
    )

    if not result.valid:
        errors = "; ".join(e.message for e in result.errors[:3])
        raise ValueError(f"Scenario validation failed: {errors}")

    return spec
