"""Validate command for population specs and scenario specs."""

import re
from pathlib import Path

import typer
import yaml

from ...core.models import PopulationSpec, ScenarioSpec
from ...population.validator import validate_spec
from ..app import app, console, get_json_mode, is_agent_mode
from ..utils import Output, ExitCode, format_validation_for_json


def _is_json_output() -> bool:
    """Check if JSON output is enabled (via --json flag or agent mode config)."""
    return get_json_mode() or is_agent_mode()


def _is_scenario_file(path: Path) -> bool:
    """Check if file is a scenario spec based on naming convention."""
    name = path.name
    # Legacy patterns
    if name.endswith(".scenario.yaml") or name.endswith(".scenario.yml"):
        return True
    if name in {"scenario.yaml", "scenario.yml"}:
        return True
    # Versioned pattern: scenario.v{N}.yaml or scenario.v{N}.yml
    if re.match(r"^scenario\.v\d+\.ya?ml$", name):
        return True
    return False


def _is_persona_file(path: Path) -> bool:
    """Check if file is a persona config based on naming convention."""
    name = path.name
    # Legacy patterns
    if name.endswith(".persona.yaml") or name.endswith(".persona.yml"):
        return True
    if name in {"persona.yaml", "persona.yml"}:
        return True
    # Versioned pattern: persona.v{N}.yaml or persona.v{N}.yml
    if re.match(r"^persona\.v\d+\.ya?ml$", name):
        return True
    return False


def _detect_spec_type(path: Path) -> str:
    """Detect whether a file is population/scenario/persona.

    Uses filename conventions first, then falls back to top-level YAML keys.
    """
    if _is_persona_file(path):
        return "persona"
    if _is_scenario_file(path):
        return "scenario"

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return "population"

    if isinstance(data, dict):
        keys = set(data.keys())
        if {"intro_template", "treatments", "groups", "phrasings"}.issubset(keys):
            return "persona"
        if {
            "event",
            "seed_exposure",
            "interaction",
            "spread",
            "outcomes",
            "simulation",
        }.issubset(keys):
            return "scenario"

    return "population"


def _extract_version(path: Path, prefix: str) -> int | None:
    """Extract version from versioned filename (e.g., persona.v2.yaml)."""
    match = re.match(rf"^{re.escape(prefix)}\.v(\d+)\.ya?ml$", path.name)
    if not match:
        return None
    return int(match.group(1))


def _find_scenario_for_persona(persona_path: Path) -> Path | None:
    """Find the most likely scenario YAML alongside a persona config."""
    scenario_dir = persona_path.parent
    persona_version = _extract_version(persona_path, "persona")

    # Prefer matching version if persona.vN and scenario.vN exist together.
    if persona_version is not None:
        for ext in ("yaml", "yml"):
            candidate = scenario_dir / f"scenario.v{persona_version}.{ext}"
            if candidate.exists():
                return candidate

    # Otherwise prefer highest versioned scenario.
    versioned: list[tuple[int, Path]] = []
    for candidate in scenario_dir.iterdir():
        if not candidate.is_file():
            continue
        match = re.match(r"^scenario\.v(\d+)\.ya?ml$", candidate.name)
        if match:
            versioned.append((int(match.group(1)), candidate))
    if versioned:
        versioned.sort(key=lambda x: x[0], reverse=True)
        return versioned[0][1]

    # Legacy names.
    for name in ("scenario.yaml", "scenario.yml"):
        candidate = scenario_dir / name
        if candidate.exists():
            return candidate

    # Legacy suffix forms.
    suffix_matches = sorted(
        [
            p
            for p in scenario_dir.iterdir()
            if p.is_file()
            and (p.name.endswith(".scenario.yaml") or p.name.endswith(".scenario.yml"))
        ]
    )
    return suffix_matches[0] if suffix_matches else None


def _resolve_population_from_scenario(
    scenario_spec: ScenarioSpec, scenario_path: Path
) -> tuple[PopulationSpec | None, str | None]:
    """Resolve and load the population spec referenced by a scenario."""
    try:
        pop_name, pop_version = scenario_spec.meta.get_population_ref()
    except ValueError as e:
        return None, str(e)

    pop_path: Path
    if scenario_spec.meta.base_population:
        if pop_version is None:
            return (
                None,
                f"Unsupported base_population reference: {scenario_spec.meta.base_population}",
            )
        # Expected layout: {study_root}/scenario/{scenario_name}/scenario.vN.yaml
        scenario_dir = scenario_path.parent
        scenarios_dir = scenario_dir.parent
        if scenarios_dir.name == "scenario":
            study_root = scenarios_dir.parent
        else:
            study_root = scenario_dir
        pop_path = study_root / f"{pop_name}.v{pop_version}.yaml"
    elif scenario_spec.meta.population_spec:
        from ...utils import resolve_relative_to

        pop_path = resolve_relative_to(
            scenario_spec.meta.population_spec, scenario_path
        )
    else:
        return None, "Scenario does not define base_population or population_spec"

    if not pop_path.exists():
        return None, f"Population spec not found: {pop_path}"

    try:
        return PopulationSpec.from_yaml(pop_path), None
    except Exception as e:
        return None, f"Failed to load population spec: {e}"


def _categorical_options_for_attribute(attr_spec) -> set[str]:
    """Return categorical options for an attribute when available."""
    dist = attr_spec.sampling.distribution
    if dist is None:
        return set()
    if getattr(dist, "type", None) != "categorical":
        return set()
    options = getattr(dist, "options", None)
    if not options:
        return set()
    return {str(opt) for opt in options}


def _validate_persona_config(spec_file: Path, out: Output) -> int:
    """Validate a persona rendering config."""
    from ...population.persona import PersonaConfig
    from ...population.persona.renderer import extract_intro_attributes

    # Load config
    if not _is_json_output():
        with console.status("[cyan]Loading persona config...[/cyan]"):
            try:
                config = PersonaConfig.from_yaml(spec_file)
            except Exception as e:
                out.error(
                    f"Failed to load persona config: {e}",
                    exit_code=ExitCode.VALIDATION_ERROR,
                )
                return out.finish()
    else:
        try:
            config = PersonaConfig.from_yaml(spec_file)
        except Exception as e:
            out.error(
                f"Failed to load persona config: {e}",
                exit_code=ExitCode.VALIDATION_ERROR,
            )
            return out.finish()

    out.success(
        "Loaded persona config",
        spec_file=str(spec_file),
        treatment_count=len(config.treatments),
        group_count=len(config.groups),
    )
    out.blank()

    errors: list[str] = []
    warnings: list[str] = []

    # Best-effort context resolution for cross-file validation.
    attribute_specs_by_name: dict[str, object] = {}
    scenario_path = _find_scenario_for_persona(spec_file)
    if scenario_path is None:
        warnings.append(
            "No sibling scenario file found; running structural-only persona validation"
        )
    else:
        try:
            scenario_spec = ScenarioSpec.from_yaml(scenario_path)
            population_spec, pop_error = _resolve_population_from_scenario(
                scenario_spec, scenario_path
            )
            if pop_error:
                warnings.append(pop_error)
            elif population_spec:
                merged_attributes = list(population_spec.attributes)
                if scenario_spec.extended_attributes:
                    merged_attributes.extend(scenario_spec.extended_attributes)
                attribute_specs_by_name = {a.name: a for a in merged_attributes}
                out.set_data("resolved_scenario", str(scenario_path))
                out.set_data("resolved_attribute_count", len(attribute_specs_by_name))
        except Exception as e:
            warnings.append(f"Failed to resolve scenario context: {e}")

    # Structural checks.
    group_names: list[str] = [g.name for g in config.groups]
    group_name_set = set(group_names)
    duplicate_group_names = sorted(
        {name for name in group_names if group_names.count(name) > 1}
    )
    for name in duplicate_group_names:
        errors.append(f"Duplicate group name: {name}")

    treatment_by_attr: dict[str, object] = {}
    for treatment in config.treatments:
        if treatment.attribute in treatment_by_attr:
            errors.append(f"Duplicate treatment for attribute: {treatment.attribute}")
        treatment_by_attr[treatment.attribute] = treatment
        if treatment.group not in group_name_set:
            errors.append(
                f"Treatment for {treatment.attribute} references unknown group: {treatment.group}"
            )

    grouped_attrs: dict[str, set[str]] = {}
    for group in config.groups:
        seen_in_group: set[str] = set()
        for attr in group.attributes:
            if attr in seen_in_group:
                errors.append(
                    f"Group {group.name} contains duplicate attribute: {attr}"
                )
            seen_in_group.add(attr)

            if attr not in treatment_by_attr:
                errors.append(
                    f"Group {group.name} references attribute without treatment: {attr}"
                )
            grouped_attrs.setdefault(attr, set()).add(group.name)

    for attr, owning_groups in grouped_attrs.items():
        if len(owning_groups) > 1:
            groups_str = ", ".join(sorted(owning_groups))
            errors.append(f"Attribute {attr} appears in multiple groups: {groups_str}")

    ungrouped = sorted(set(treatment_by_attr) - set(grouped_attrs))
    if ungrouped:
        errors.append(
            f"Attributes have treatments but are not present in any group: {', '.join(ungrouped)}"
        )

    phrasing_attr_to_kind: dict[str, str] = {}
    phrasing_kind_sets: dict[str, set[str]] = {
        "boolean": set(),
        "categorical": set(),
        "relative": set(),
        "concrete": set(),
    }

    def _register_phrasing(attr: str, kind: str) -> None:
        if attr in phrasing_kind_sets[kind]:
            errors.append(f"Duplicate {kind} phrasing for attribute: {attr}")
        phrasing_kind_sets[kind].add(attr)
        existing = phrasing_attr_to_kind.get(attr)
        if existing and existing != kind:
            errors.append(
                f"Attribute {attr} has multiple phrasing kinds: {existing}, {kind}"
            )
        phrasing_attr_to_kind[attr] = kind

    for phrasing in config.phrasings.boolean:
        _register_phrasing(phrasing.attribute, "boolean")
        if not phrasing.true_phrase.strip():
            errors.append(
                f"Boolean phrasing for {phrasing.attribute} has empty true_phrase"
            )
        if not phrasing.false_phrase.strip():
            errors.append(
                f"Boolean phrasing for {phrasing.attribute} has empty false_phrase"
            )

    for phrasing in config.phrasings.categorical:
        _register_phrasing(phrasing.attribute, "categorical")
        if not phrasing.phrases:
            errors.append(
                f"Categorical phrasing for {phrasing.attribute} has no option phrases"
            )

    for phrasing in config.phrasings.relative:
        _register_phrasing(phrasing.attribute, "relative")
        labels = phrasing.labels
        if not all(
            [
                labels.much_below.strip(),
                labels.below.strip(),
                labels.average.strip(),
                labels.above.strip(),
                labels.much_above.strip(),
            ]
        ):
            errors.append(
                f"Relative phrasing for {phrasing.attribute} has one or more empty labels"
            )

    concrete_template_pattern = re.compile(r"\{value(?::[^}]*)?\}")
    for phrasing in config.phrasings.concrete:
        _register_phrasing(phrasing.attribute, "concrete")
        if not concrete_template_pattern.search(phrasing.template):
            errors.append(
                f"Concrete phrasing for {phrasing.attribute} template must include {{value}}"
            )
        if phrasing.format_spec:
            if phrasing.format_spec not in {"time12", "time24"}:
                try:
                    format(1234.567, phrasing.format_spec)
                except Exception:
                    errors.append(
                        f"Concrete phrasing for {phrasing.attribute} has invalid format_spec: {phrasing.format_spec}"
                    )

    intro_attrs = extract_intro_attributes(config.intro_template)

    # Cross-file semantic checks when scenario + population can be resolved.
    if attribute_specs_by_name:
        known_attrs = set(attribute_specs_by_name)

        for source, attr_set in [
            ("treatments", set(treatment_by_attr)),
            ("groups", set(grouped_attrs)),
            ("phrasings", set(phrasing_attr_to_kind)),
            ("intro_template", set(intro_attrs)),
        ]:
            unknown = sorted(attr_set - known_attrs)
            if unknown:
                errors.append(
                    f"{source} references unknown attributes: {', '.join(unknown)}"
                )

        missing_treatments = sorted(known_attrs - set(treatment_by_attr))
        if missing_treatments:
            errors.append(
                f"Missing treatments for attributes: {', '.join(missing_treatments)}"
            )

        missing_phrasings = sorted(known_attrs - set(phrasing_attr_to_kind))
        if missing_phrasings:
            errors.append(
                f"Missing phrasing entries for attributes: {', '.join(missing_phrasings)}"
            )

        for attr_name, attr_spec in attribute_specs_by_name.items():
            attr_type = attr_spec.type
            treatment = treatment_by_attr.get(attr_name)
            phrasing_kind = phrasing_attr_to_kind.get(attr_name)

            if treatment is not None and treatment.treatment.value == "relative":
                if attr_type not in {"int", "float"}:
                    errors.append(
                        f"Attribute {attr_name} uses relative treatment but has non-numeric type: {attr_type}"
                    )
                if phrasing_kind != "relative":
                    errors.append(
                        f"Attribute {attr_name} uses relative treatment but has {phrasing_kind or 'no'} phrasing"
                    )
            elif attr_type in {"int", "float"} and phrasing_kind not in {
                None,
                "concrete",
                "relative",
            }:
                errors.append(
                    f"Numeric attribute {attr_name} has incompatible phrasing kind: {phrasing_kind}"
                )

            if attr_type == "boolean" and phrasing_kind not in {None, "boolean"}:
                errors.append(
                    f"Boolean attribute {attr_name} must use boolean phrasing (found {phrasing_kind})"
                )
            if attr_type == "categorical" and phrasing_kind not in {
                None,
                "categorical",
            }:
                errors.append(
                    f"Categorical attribute {attr_name} must use categorical phrasing (found {phrasing_kind})"
                )

        for phrasing in config.phrasings.categorical:
            attr_spec = attribute_specs_by_name.get(phrasing.attribute)
            if attr_spec is None:
                continue
            if attr_spec.type != "categorical":
                errors.append(
                    f"Attribute {phrasing.attribute} has categorical phrasing but type is {attr_spec.type}"
                )
                continue

            options = _categorical_options_for_attribute(attr_spec)
            if not options:
                warnings.append(
                    f"Could not verify option coverage for {phrasing.attribute} (no categorical options found in spec)"
                )
                continue

            covered = set(phrasing.phrases.keys()) | set(phrasing.null_options)
            missing = sorted(options - covered)
            extra = sorted(covered - options)
            if missing:
                errors.append(
                    f"Categorical phrasing for {phrasing.attribute} missing options: {', '.join(missing)}"
                )
            if extra:
                errors.append(
                    f"Categorical phrasing for {phrasing.attribute} includes unknown options: {', '.join(extra)}"
                )

    # Emit warnings.
    for warning in warnings:
        out.warning(warning)

    # Emit errors.
    if errors:
        out.error(
            f"Persona config has {len(errors)} error(s)",
            exit_code=ExitCode.VALIDATION_ERROR,
        )
        if _is_json_output():
            for err in errors:
                out.error(err, exit_code=ExitCode.VALIDATION_ERROR)
        else:
            for err in errors[:15]:
                out.text(f"  [red]✗[/red] {err}")
            if len(errors) > 15:
                out.text(f"  [dim]... and {len(errors) - 15} more[/dim]")
        return out.finish()

    if warnings:
        out.success(f"Persona config validated with {len(warnings)} warning(s)")
    else:
        out.success("Persona config validated")

    out.blank()
    out.divider()
    out.text("[green]Validation passed[/green]")
    out.divider()

    return out.finish()


def _validate_population_spec(spec_file: Path, strict: bool, out: Output) -> int:
    """Validate a population spec."""
    # Load spec
    if not _is_json_output():
        with console.status("[cyan]Loading spec...[/cyan]"):
            try:
                spec = PopulationSpec.from_yaml(spec_file)
            except Exception as e:
                out.error(
                    f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR
                )
                return out.finish()
    else:
        try:
            spec = PopulationSpec.from_yaml(spec_file)
        except Exception as e:
            out.error(f"Failed to load spec: {e}", exit_code=ExitCode.VALIDATION_ERROR)
            return out.finish()

    out.success(
        f"Loaded: [bold]{spec.meta.description}[/bold] ({len(spec.attributes)} attributes)",
        spec_file=str(spec_file),
        description=spec.meta.description,
        attribute_count=len(spec.attributes),
    )
    out.blank()

    # Validate spec
    if not _is_json_output():
        with console.status("[cyan]Validating spec...[/cyan]"):
            result = validate_spec(spec)
    else:
        result = validate_spec(spec)

    # Add validation result to JSON output
    out.set_data("validation", format_validation_for_json(result))

    # Handle errors
    if result.errors:
        out.error(
            f"Spec has {len(result.errors)} error(s)",
            exit_code=ExitCode.VALIDATION_ERROR,
        )

        if not _is_json_output():
            error_rows = []
            for err in result.errors[:15]:
                loc = err.location
                if err.modifier_index is not None:
                    loc = f"{err.location}[{err.modifier_index}]"
                error_rows.append([loc, err.category, err.message[:60]])

            if error_rows:
                out.table(
                    "Errors",
                    ["Location", "Category", "Message"],
                    error_rows,
                    styles=["red", "dim", None],
                )

            if len(result.errors) > 15:
                out.text(
                    f"  [dim]... and {len(result.errors) - 15} more error(s)[/dim]"
                )

            out.blank()
            out.text("[bold]Suggestions:[/bold]")
            for err in result.errors[:3]:
                if err.suggestion:
                    out.text(f"  [dim]→ {err.location}: {err.suggestion}[/dim]")

        return out.finish()

    # Handle warnings (with strict mode)
    if result.warnings:
        if strict:
            out.error(
                f"Spec has {len(result.warnings)} warning(s) (strict mode)",
                exit_code=ExitCode.VALIDATION_ERROR,
            )

            if not _is_json_output():
                warning_rows = []
                for warn in result.warnings[:10]:
                    loc = warn.location
                    if warn.modifier_index is not None:
                        loc = f"{warn.location}[{warn.modifier_index}]"
                    warning_rows.append([loc, warn.category, warn.message[:60]])

                out.table(
                    "Warnings",
                    ["Location", "Category", "Message"],
                    warning_rows,
                    styles=["yellow", "dim", None],
                )

            return out.finish()
        else:
            out.success(f"Spec validated with {len(result.warnings)} warning(s)")

            if not _is_json_output():
                for warn in result.warnings[:3]:
                    loc = warn.location
                    if warn.modifier_index is not None:
                        loc = f"{warn.location}[{warn.modifier_index}]"
                    out.warning(f"{loc}: {warn.message}")

                if len(result.warnings) > 3:
                    out.text(
                        f"  [dim]... and {len(result.warnings) - 3} more warning(s)[/dim]"
                    )
    else:
        out.success("Spec validated")

    out.blank()
    out.divider()
    out.text("[green]Validation passed[/green]")
    out.divider()

    return out.finish()


def _validate_scenario_spec(spec_file: Path, out: Output) -> int:
    """Validate a scenario spec."""
    from ...scenario import load_and_validate_scenario

    # Load and validate
    if not _is_json_output():
        with console.status("[cyan]Loading scenario spec...[/cyan]"):
            try:
                spec, result = load_and_validate_scenario(spec_file)
            except Exception as e:
                out.error(
                    f"Failed to load scenario: {e}", exit_code=ExitCode.VALIDATION_ERROR
                )
                return out.finish()
    else:
        try:
            spec, result = load_and_validate_scenario(spec_file)
        except Exception as e:
            out.error(
                f"Failed to load scenario: {e}", exit_code=ExitCode.VALIDATION_ERROR
            )
            return out.finish()

    out.success(
        f"Loaded scenario: [bold]{spec.meta.name}[/bold]",
        spec_file=str(spec_file),
        name=spec.meta.name,
    )
    out.blank()

    # Show file references (human mode only)
    if not _is_json_output():
        out.text("[bold]Scenario Details:[/bold]")

        # New flow: base_population
        if spec.meta.base_population:
            out.text(f"  [cyan]•[/cyan] base_population: {spec.meta.base_population}")
            if spec.extended_attributes:
                out.text(
                    f"  [cyan]•[/cyan] extended_attributes: {len(spec.extended_attributes)}"
                )

        # Legacy flow: population_spec + study_db
        if spec.meta.population_spec:
            from ...utils import resolve_relative_to

            pop_path = resolve_relative_to(spec.meta.population_spec, spec_file)
            if pop_path.exists():
                out.text(f"  [green]✓[/green] Population: {spec.meta.population_spec}")
            else:
                out.text(
                    f"  [red]✗[/red] Population: {spec.meta.population_spec} (not found)"
                )

        if spec.meta.study_db:
            from ...utils import resolve_relative_to

            study_db_path = resolve_relative_to(spec.meta.study_db, spec_file)
            if study_db_path.exists():
                out.text(f"  [green]✓[/green] Study DB: {spec.meta.study_db}")
            else:
                out.text(f"  [red]✗[/red] Study DB: {spec.meta.study_db} (not found)")
            out.text(f"  [cyan]•[/cyan] population_id: {spec.meta.population_id}")
            out.text(f"  [cyan]•[/cyan] network_id: {spec.meta.network_id}")

        out.blank()

    # Handle errors
    if result.errors:
        out.error(
            f"Scenario has {len(result.errors)} error(s)",
            exit_code=ExitCode.VALIDATION_ERROR,
        )

        if not _is_json_output():
            for err in result.errors[:10]:
                out.text(
                    f"  [red]✗[/red] [{err.category}] {err.location}: {err.message}"
                )
                if err.suggestion:
                    out.text(f"    [dim]→ {err.suggestion}[/dim]")

            if len(result.errors) > 10:
                out.text(f"  [dim]... and {len(result.errors) - 10} more[/dim]")

        return out.finish()

    # Handle warnings
    if result.warnings:
        out.success(f"Scenario validated with {len(result.warnings)} warning(s)")

        if not _is_json_output():
            for warn in result.warnings[:5]:
                out.warning(f"[{warn.category}] {warn.location}: {warn.message}")

            if len(result.warnings) > 5:
                out.text(f"  [dim]... and {len(result.warnings) - 5} more[/dim]")
    else:
        out.success("Scenario validated")

    out.blank()
    out.divider()
    out.text("[green]Validation passed[/green]")
    out.divider()

    return out.finish()


@app.command("validate")
def validate_command(
    spec_file: Path = typer.Argument(
        ..., help="Spec file to validate (.yaml, scenario.vN.yaml, or persona.vN.yaml)"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Treat warnings as errors (population specs only)"
    ),
):
    """
    Validate a population, scenario, or persona spec.

    Auto-detects file type based on naming:
    - *.scenario.yaml → scenario spec validation
    - *.persona.yaml → persona config validation
    - *.yaml → population spec validation

    EXIT CODES:
        0 = Success (valid spec)
        1 = Validation error (invalid spec)
        3 = File not found

    EXAMPLES:
        extropy validate surgeons.yaml              # Population spec
        extropy validate surgeons.scenario.yaml     # Scenario spec
        extropy validate scenario/ai-adoption/persona.v1.yaml  # Persona config
        extropy validate surgeons.yaml --strict     # Treat warnings as errors
    """
    out = Output(console=console, json_mode=_is_json_output())
    out.blank()

    # Check file exists
    if not spec_file.exists():
        out.error(
            f"File not found: {spec_file}",
            exit_code=ExitCode.FILE_NOT_FOUND,
            suggestion=f"Check the file path: {spec_file.absolute()}",
        )
        raise typer.Exit(out.finish())

    # Route to appropriate validator
    spec_type = _detect_spec_type(spec_file)
    if spec_type == "persona":
        exit_code = _validate_persona_config(spec_file, out)
    elif spec_type == "scenario":
        exit_code = _validate_scenario_spec(spec_file, out)
    else:
        exit_code = _validate_population_spec(spec_file, strict, out)

    raise typer.Exit(exit_code)
