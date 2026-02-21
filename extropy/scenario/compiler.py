"""Scenario Compiler (Orchestrator).

Orchestrates the full scenario compilation pipeline:
1. Parse scenario description into Event
2. Generate seed exposure rules
3. Determine interaction model and spread config
4. Generate timeline, outcomes, and background context
5. Assemble and validate ScenarioSpec
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..core.models import (
    IdentityDimension,
    HouseholdConfig,
    PopulationSpec,
    ScenarioMeta,
    ScenarioSimConfig,
    ScenarioSpec,
    TimestepUnit,
    ValidationResult,
)
from ..core.llm import simple_call
from .parser import parse_scenario
from .exposure import generate_seed_exposure
from .interaction import determine_interaction_model
from .timeline import generate_timeline_and_outcomes
from .sampling_semantics import generate_sampling_semantic_roles
from ..utils.callbacks import StepProgressCallback
from ..utils import topological_sort
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


def _determine_simulation_config() -> ScenarioSimConfig:
    """Determine default simulation configuration.

    Uses a fixed default since population size is determined at sample time.
    """
    return ScenarioSimConfig(
        max_timesteps=100,
        timestep_unit=TimestepUnit.HOUR,
        stop_conditions=["exposure_rate > 0.95 and no_state_changes_for > 10"],
        seed=None,
    )


# Schema for identity dimension detection
_IDENTITY_DIMENSION_SCHEMA = {
    "type": "object",
    "properties": {
        "identity_dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": [
                            "political_orientation",
                            "religious_affiliation",
                            "race_ethnicity",
                            "gender_identity",
                            "sexual_orientation",
                            "parental_status",
                            "citizenship",
                            "socioeconomic_class",
                            "professional_identity",
                            "generational_identity",
                        ],
                        "description": "The identity dimension being activated",
                    },
                    "relevance": {
                        "type": "string",
                        "description": "Why this dimension is relevant (1-2 sentences)",
                    },
                },
                "required": ["dimension", "relevance"],
            },
            "description": "Identity dimensions that may feel threatened or activated by this scenario. Only include dimensions that are clearly relevant.",
        },
    },
    "required": ["identity_dimensions"],
}


def _detect_identity_dimensions(
    description: str,
    event_content: str,
    background_context: str | None = None,
) -> list[IdentityDimension]:
    """Detect which identity dimensions are activated by the scenario.

    Uses LLM to analyze the scenario and determine which aspects of
    people's identity might feel threatened or activated.
    """
    context_parts = [
        f"Scenario: {description}",
        f"Event: {event_content}",
    ]
    if background_context:
        context_parts.append(f"Background: {background_context}")

    prompt = f"""Analyze this scenario and identify which identity dimensions might feel threatened or activated.

{chr(10).join(context_parts)}

Identity dimensions to consider:
- political_orientation: Liberal/conservative, party affiliation, political ideology
- religious_affiliation: Faith traditions, religious identity, moral frameworks
- race_ethnicity: Racial/ethnic identity, minority/majority status
- gender_identity: Gender expression, gender roles
- sexual_orientation: LGBTQ+ identity
- parental_status: Parent/caregiver identity, family roles
- citizenship: Immigration status, national identity
- socioeconomic_class: Working class, professional class, wealth identity
- professional_identity: Career/occupational identity
- generational_identity: Boomer, millennial, gen-z identity

Only include dimensions that are CLEARLY relevant to this specific scenario.
A scenario about book bans might activate political_orientation and parental_status.
A scenario about a new iPhone might not activate any identity dimensions.

If no identity dimensions are clearly relevant, return an empty array."""

    try:
        response = simple_call(
            prompt=prompt,
            response_schema=_IDENTITY_DIMENSION_SCHEMA,
            schema_name="identity_dimensions",
        )

        if not response:
            return []

        dimensions = []
        for item in response.get("identity_dimensions", []):
            try:
                dim = IdentityDimension(
                    dimension=item["dimension"],
                    relevance=item["relevance"],
                )
                dimensions.append(dim)
            except (KeyError, ValueError):
                continue

        return dimensions

    except Exception:
        # If detection fails, return empty list - not critical
        return []


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
    4. Generate timeline, outcomes, and background context
    5. Assemble ScenarioSpec and validate

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

    progress("1/5", "Loading population spec...")

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
        network_summary,
    )

    # =========================================================================
    # Step 4: Generate timeline, outcomes, and background context
    # =========================================================================

    progress("4/5", "Generating timeline and outcomes...")

    # Generate simulation config
    simulation_config = _determine_simulation_config()

    timeline_events, background_context, simulation_config, outcome_config = (
        generate_timeline_and_outcomes(
            scenario_description=description,
            base_event=event,
            simulation_config=simulation_config,
            timeline_mode=timeline_mode,
        )
    )

    # =========================================================================
    # Step 5: Assemble scenario spec
    # =========================================================================

    progress("5/5", "Assembling scenario spec...")

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
    sampling_semantic_roles = generate_sampling_semantic_roles(population_spec)

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
        sampling_semantic_roles=sampling_semantic_roles,
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
    household_config: HouseholdConfig | None = None,
    agent_focus_mode: Literal["primary_only", "couples", "all"] | None = None,
    on_progress: StepProgressCallback | None = None,
    timeline_mode: str | None = None,
    timestep_unit_override: str | None = None,
    max_timesteps_override: int | None = None,
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
        household_config: Optional household configuration for scenario-owned household semantics
        agent_focus_mode: Household agent scope (primary_only/couples/all)
        on_progress: Optional callback(step, status) for progress updates
        timeline_mode: Timeline mode override. None = auto-detect.
        timestep_unit_override: CLI override for timestep unit (e.g. "month").
        max_timesteps_override: CLI override for max timesteps.

    Returns:
        Tuple of (ScenarioSpec, ValidationResult)
    """

    def progress(step: str, status: str):
        if on_progress:
            on_progress(step, status)

    ext_attrs = list(extended_attributes or [])
    merged_population = population_spec
    if ext_attrs:
        merged_attributes = list(population_spec.attributes) + ext_attrs
        merged_deps: dict[str, list[str]] = {
            attr.name: list(attr.sampling.depends_on or []) for attr in merged_attributes
        }
        merged_names = set(merged_deps.keys())
        merged_deps = {
            name: [dep for dep in deps if dep in merged_names]
            for name, deps in merged_deps.items()
        }
        merged_population = PopulationSpec(
            meta=population_spec.meta.model_copy(),
            grounding=population_spec.grounding,
            attributes=merged_attributes,
            sampling_order=topological_sort(merged_deps),
        )

    # Step 1: Parse scenario description
    progress("1/5", "Parsing event definition...")
    event = parse_scenario(description, merged_population)

    # Step 2: Generate seed exposure without assumed edge types.
    progress("2/5", "Generating seed exposure rules...")

    seed_exposure = generate_seed_exposure(
        event,
        merged_population,
        None,
    )

    # Step 3: Determine interaction model
    progress("3/5", "Determining interaction model...")

    interaction_config, spread_config = determine_interaction_model(
        event,
        merged_population,
        None,
    )

    # Step 4: Generate timeline, outcomes, and background context
    progress("4/5", "Generating timeline and outcomes...")

    simulation_config = _determine_simulation_config()

    timeline_events, background_context, simulation_config, outcome_config = (
        generate_timeline_and_outcomes(
            scenario_description=description,
            base_event=event,
            simulation_config=simulation_config,
            timeline_mode=timeline_mode,
            timestep_unit_override=timestep_unit_override,
            max_timesteps_override=max_timesteps_override,
        )
    )

    # Step 5: Detect identity dimensions + sampling semantics
    progress("5/5", "Detecting identity dimensions...")

    identity_dimensions = _detect_identity_dimensions(
        description=description,
        event_content=event.content,
        background_context=background_context,
    )
    sampling_semantic_roles = generate_sampling_semantic_roles(merged_population)

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
        identity_dimensions=identity_dimensions if identity_dimensions else None,
        sampling_semantic_roles=sampling_semantic_roles,
        extended_attributes=ext_attrs,
        household_config=household_config,
        agent_focus_mode=agent_focus_mode,
    )

    validation_result = validate_scenario(
        spec=spec,
        population_spec=merged_population,
        agent_count=None,
        network=None,
    )

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
