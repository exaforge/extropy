"""Path resolution utilities for Entropy.

This module provides consistent path handling across all CLI commands,
ensuring relative paths in YAML files are resolved correctly regardless
of the current working directory.
"""

from pathlib import Path


def resolve_relative_to(path: str | Path, base_file: Path) -> Path:
    """
    Resolve a path relative to a base file's directory.

    If path is absolute, returns it unchanged.
    If path is relative, resolves it against base_file's parent directory.

    Args:
        path: Path string or Path object to resolve
        base_file: The file that contains the path reference (e.g., scenario.yaml)

    Returns:
        Resolved absolute Path

    Example:
        >>> resolve_relative_to("population.yaml", Path("/project/study/scenario.yaml"))
        PosixPath('/project/study/population.yaml')

        >>> resolve_relative_to("/abs/path/pop.yaml", Path("/project/scenario.yaml"))
        PosixPath('/abs/path/pop.yaml')
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_file.parent / path).resolve()


def make_relative_to(path: str | Path, base_file: Path) -> str:
    """
    Convert a path to be relative to a base file's directory.

    Used when storing file references in YAML files (e.g., scenario.yaml).
    If the path cannot be made relative (different drives on Windows, etc.),
    returns the absolute path as a string.

    Args:
        path: Path to convert (can be relative to cwd or absolute)
        base_file: The file that will contain the reference (e.g., scenario.yaml)

    Returns:
        Relative path string if possible, otherwise absolute path string

    Example:
        >>> make_relative_to("/project/study/population.yaml", Path("/project/study/scenario.yaml"))
        'population.yaml'

        >>> make_relative_to("studies/netflix/pop.yaml", Path("studies/netflix/scenario.yaml"))
        'pop.yaml'
    """
    path = Path(path).resolve()
    base_dir = base_file.parent.resolve()

    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        # Path is not under base_dir, return absolute
        return str(path)
