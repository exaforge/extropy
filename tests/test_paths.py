"""Tests for path resolution utilities."""

from pathlib import Path
import tempfile
import os

from entropy.utils import resolve_relative_to, make_relative_to


class TestResolveRelativeTo:
    """Tests for resolve_relative_to function."""

    def test_relative_path_resolved(self):
        """Relative paths should be resolved against base file's directory."""
        result = resolve_relative_to(
            "population.yaml", Path("/project/study/scenario.yaml")
        )
        assert result == Path("/project/study/population.yaml")

    def test_absolute_path_unchanged(self):
        """Absolute paths should be returned unchanged."""
        result = resolve_relative_to(
            "/abs/path/pop.yaml", Path("/project/scenario.yaml")
        )
        assert result == Path("/abs/path/pop.yaml")

    def test_nested_relative_path(self):
        """Nested relative paths should resolve correctly."""
        result = resolve_relative_to(
            "data/agents.json", Path("/project/study/scenario.yaml")
        )
        assert result == Path("/project/study/data/agents.json")

    def test_parent_relative_path(self):
        """Parent-relative paths (../) should resolve correctly."""
        result = resolve_relative_to(
            "../common/pop.yaml", Path("/project/study/scenario.yaml")
        )
        assert result == Path("/project/common/pop.yaml")

    def test_path_object_input(self):
        """Should accept Path objects as input."""
        result = resolve_relative_to(
            Path("population.yaml"), Path("/project/study/scenario.yaml")
        )
        assert result == Path("/project/study/population.yaml")


class TestMakeRelativeTo:
    """Tests for make_relative_to function."""

    def test_sibling_file_relative(self):
        """Files in same directory should return just filename."""
        result = make_relative_to(
            "/project/study/population.yaml", Path("/project/study/scenario.yaml")
        )
        assert result == "population.yaml"

    def test_nested_file_relative(self):
        """Files in subdirectory should return relative path."""
        result = make_relative_to(
            "/project/study/data/agents.json", Path("/project/study/scenario.yaml")
        )
        assert result == "data/agents.json"

    def test_unrelated_path_absolute(self):
        """Unrelated paths should return absolute path."""
        result = make_relative_to(
            "/other/path/pop.yaml", Path("/project/study/scenario.yaml")
        )
        assert result == "/other/path/pop.yaml"

    def test_relative_input_resolved(self):
        """Relative input paths should be resolved against cwd first."""
        # Use a temp directory to test this
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure: tmpdir/study/scenario.yaml and tmpdir/study/pop.yaml
            study_dir = Path(tmpdir) / "study"
            study_dir.mkdir()
            scenario = study_dir / "scenario.yaml"
            pop = study_dir / "population.yaml"
            scenario.touch()
            pop.touch()

            # Save original cwd
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # When cwd is tmpdir, "study/population.yaml" should become "population.yaml"
                # relative to "study/scenario.yaml"
                result = make_relative_to("study/population.yaml", Path("study/scenario.yaml"))
                assert result == "population.yaml"
            finally:
                os.chdir(orig_cwd)


class TestRoundTrip:
    """Test that make_relative_to and resolve_relative_to are inverses."""

    def test_roundtrip_same_directory(self):
        """Roundtrip for files in same directory."""
        base = Path("/project/study/scenario.yaml")
        original = "/project/study/population.yaml"

        relative = make_relative_to(original, base)
        resolved = resolve_relative_to(relative, base)

        assert str(resolved) == original

    def test_roundtrip_nested(self):
        """Roundtrip for nested files."""
        base = Path("/project/study/scenario.yaml")
        original = "/project/study/data/agents.json"

        relative = make_relative_to(original, base)
        resolved = resolve_relative_to(relative, base)

        assert str(resolved) == original
