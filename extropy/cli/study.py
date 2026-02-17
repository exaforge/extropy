"""Study folder detection and management utilities.

This module provides utilities for:
- Detecting study folders from cwd or --study flag
- Auto-versioning YAML files
- Study folder structure management
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class StudyContext:
    """Context for a study folder.

    Provides access to study paths and versioning utilities.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.db_path = self.root / "study.db"
        self.scenario_dir = self.root / "scenario"

    @property
    def exists(self) -> bool:
        """Check if this is a valid study folder."""
        return self.db_path.exists()

    def ensure_exists(self) -> None:
        """Create study folder structure if it doesn't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.scenario_dir.mkdir(exist_ok=True)

    def get_population_versions(self, name: str = "population") -> list[int]:
        """Get all version numbers for a population spec.

        Args:
            name: Base name of the population spec (default: "population")

        Returns:
            Sorted list of version numbers found
        """
        pattern = re.compile(rf"^{re.escape(name)}\.v(\d+)\.yaml$")
        versions = []
        for path in self.root.glob(f"{name}.v*.yaml"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))
        return sorted(versions)

    def get_latest_population_version(self, name: str = "population") -> int | None:
        """Get the latest version number for a population spec.

        Returns:
            Latest version number, or None if no versions exist
        """
        versions = self.get_population_versions(name)
        return versions[-1] if versions else None

    def get_next_population_version(self, name: str = "population") -> int:
        """Get the next version number for a population spec.

        Returns:
            Next version number (1 if none exist)
        """
        latest = self.get_latest_population_version(name)
        return (latest or 0) + 1

    def get_population_path(
        self, name: str = "population", version: int | None = None
    ) -> Path:
        """Get path to a population spec.

        Args:
            name: Base name of the population spec
            version: Specific version, or None for latest

        Returns:
            Path to the population spec
        """
        if version is None:
            version = self.get_latest_population_version(name)
            if version is None:
                raise FileNotFoundError(f"No population spec found: {name}")
        return self.root / f"{name}.v{version}.yaml"

    def get_scenario_dir(self, scenario_name: str) -> Path:
        """Get path to a scenario directory."""
        return self.scenario_dir / scenario_name

    def get_scenario_versions(self, scenario_name: str) -> list[int]:
        """Get all version numbers for a scenario spec.

        Returns:
            Sorted list of version numbers found
        """
        scenario_dir = self.get_scenario_dir(scenario_name)
        if not scenario_dir.exists():
            return []

        pattern = re.compile(r"^scenario\.v(\d+)\.yaml$")
        versions = []
        for path in scenario_dir.glob("scenario.v*.yaml"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))
        return sorted(versions)

    def get_latest_scenario_version(self, scenario_name: str) -> int | None:
        """Get the latest version number for a scenario spec.

        Returns:
            Latest version number, or None if no versions exist
        """
        versions = self.get_scenario_versions(scenario_name)
        return versions[-1] if versions else None

    def get_next_scenario_version(self, scenario_name: str) -> int:
        """Get the next version number for a scenario spec.

        Returns:
            Next version number (1 if none exist)
        """
        latest = self.get_latest_scenario_version(scenario_name)
        return (latest or 0) + 1

    def get_scenario_path(
        self, scenario_name: str, version: int | None = None
    ) -> Path:
        """Get path to a scenario spec.

        Args:
            scenario_name: Name of the scenario
            version: Specific version, or None for latest

        Returns:
            Path to the scenario spec
        """
        scenario_dir = self.get_scenario_dir(scenario_name)
        if version is None:
            version = self.get_latest_scenario_version(scenario_name)
            if version is None:
                raise FileNotFoundError(f"No scenario spec found: {scenario_name}")
        return scenario_dir / f"scenario.v{version}.yaml"

    def get_persona_versions(self, scenario_name: str) -> list[int]:
        """Get all version numbers for a persona config.

        Returns:
            Sorted list of version numbers found
        """
        scenario_dir = self.get_scenario_dir(scenario_name)
        if not scenario_dir.exists():
            return []

        pattern = re.compile(r"^persona\.v(\d+)\.yaml$")
        versions = []
        for path in scenario_dir.glob("persona.v*.yaml"):
            match = pattern.match(path.name)
            if match:
                versions.append(int(match.group(1)))
        return sorted(versions)

    def get_latest_persona_version(self, scenario_name: str) -> int | None:
        """Get the latest version number for a persona config.

        Returns:
            Latest version number, or None if no versions exist
        """
        versions = self.get_persona_versions(scenario_name)
        return versions[-1] if versions else None

    def get_next_persona_version(self, scenario_name: str) -> int:
        """Get the next version number for a persona config.

        Returns:
            Next version number (1 if none exist)
        """
        latest = self.get_latest_persona_version(scenario_name)
        return (latest or 0) + 1

    def get_persona_path(
        self, scenario_name: str, version: int | None = None
    ) -> Path:
        """Get path to a persona config.

        Args:
            scenario_name: Name of the scenario
            version: Specific version, or None for latest

        Returns:
            Path to the persona config
        """
        scenario_dir = self.get_scenario_dir(scenario_name)
        if version is None:
            version = self.get_latest_persona_version(scenario_name)
            if version is None:
                raise FileNotFoundError(
                    f"No persona config found for scenario: {scenario_name}"
                )
        return scenario_dir / f"persona.v{version}.yaml"

    def list_scenarios(self) -> list[str]:
        """List all scenario names in this study.

        Returns:
            List of scenario names (directory names under scenario/)
        """
        if not self.scenario_dir.exists():
            return []
        return sorted(
            d.name for d in self.scenario_dir.iterdir() if d.is_dir()
        )


def detect_study_folder(start: Path | None = None) -> Path | None:
    """Detect study folder from current directory or parent directories.

    Looks for a directory containing study.db.

    Args:
        start: Starting directory (default: cwd)

    Returns:
        Path to study folder, or None if not found
    """
    if start is None:
        start = Path.cwd()

    current = start.resolve()

    # Check current directory
    if (current / "study.db").exists():
        return current

    # Check parent directories (up to 5 levels)
    for _ in range(5):
        parent = current.parent
        if parent == current:
            break  # Reached root
        current = parent
        if (current / "study.db").exists():
            return current

    return None


def get_study_context(study_path: Path | str | None = None) -> StudyContext:
    """Get study context from explicit path or auto-detection.

    Args:
        study_path: Explicit study path, or None to auto-detect

    Returns:
        StudyContext for the study folder

    Raises:
        FileNotFoundError: If no study folder found
    """
    if study_path is not None:
        return StudyContext(Path(study_path))

    detected = detect_study_folder()
    if detected is None:
        raise FileNotFoundError(
            "Not in a study folder. Use --study to specify a study folder, "
            "or use -o to create a new one."
        )
    return StudyContext(detected)


def create_study_folder(path: Path | str, name: str | None = None) -> StudyContext:
    """Create a new study folder.

    Args:
        path: Path to create study folder at
        name: Optional study name for metadata

    Returns:
        StudyContext for the new study folder
    """
    root = Path(path).resolve()
    ctx = StudyContext(root)
    ctx.ensure_exists()

    # Create empty study.db with schema
    from ..storage import open_study_db

    with open_study_db(ctx.db_path):
        pass  # Just create with schema

    return ctx


def generate_study_name() -> str:
    """Generate a default study folder name with timestamp.

    Returns:
        Name like "study-20260216-143052"
    """
    return f"study-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def parse_version_ref(ref: str) -> tuple[str, int | None]:
    """Parse a version reference like "name@v2" or "name".

    Args:
        ref: Reference string

    Returns:
        Tuple of (name, version) where version is None for latest
    """
    if "@v" in ref:
        parts = ref.rsplit("@v", 1)
        name = parts[0]
        try:
            version = int(parts[1])
            return name, version
        except ValueError:
            pass
    if "@latest" in ref:
        name = ref.rsplit("@latest", 1)[0]
        return name, None
    return ref, None


def parse_population_ref(ref: str) -> tuple[str, int | None]:
    """Parse a population reference like "@pop:v2" or "@pop:latest".

    Args:
        ref: Reference string starting with @pop:

    Returns:
        Tuple of (name, version) where version is None for latest

    Raises:
        ValueError: If ref doesn't start with @pop:
    """
    if not ref.startswith("@pop:"):
        raise ValueError(f"Invalid population reference: {ref}")

    version_part = ref[5:]  # Remove "@pop:"

    if version_part == "latest":
        return "population", None
    elif version_part.startswith("v"):
        try:
            version = int(version_part[1:])
            return "population", version
        except ValueError:
            raise ValueError(f"Invalid version in population reference: {ref}")
    else:
        raise ValueError(f"Invalid population reference: {ref}")
