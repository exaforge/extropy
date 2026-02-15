"""CLI smoke tests using typer's CliRunner."""
from pathlib import Path

from typer.testing import CliRunner

from extropy.cli.app import app
from extropy.cli.commands.validate import _is_scenario_file
from extropy.population.network.config import NetworkConfig
from extropy.storage import open_study_db

runner = CliRunner()


class TestConfigCommand:
    """Tests for the config command."""

    def test_config_show(self):
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "Models" in result.output
        assert "Simulation" in result.output

    def test_config_set_invalid_key(self):
        result = runner.invoke(app, ["config", "set", "invalid.key", "value"])
        assert result.exit_code == 1
        assert "Unknown key" in result.output

    def test_config_set_invalid_int_value(self):
        result = runner.invoke(app, ["config", "set", "simulation.rate_tier", "abc"])
        assert result.exit_code == 1
        assert "Invalid integer" in result.output

    def test_config_set_missing_args(self):
        result = runner.invoke(app, ["config", "set"])
        assert result.exit_code == 1

    def test_config_unknown_action(self):
        result = runner.invoke(app, ["config", "unknown_action"])
        assert result.exit_code == 1
        assert "Unknown action" in result.output


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_nonexistent_file(self):
        result = runner.invoke(app, ["validate", "/nonexistent/path.yaml"])
        assert result.exit_code != 0

    def test_scenario_filename_detection(self):
        assert _is_scenario_file(Path("scenario.yaml"))
        assert _is_scenario_file(Path("foo.scenario.yaml"))
        assert not _is_scenario_file(Path("population.yaml"))


class TestVersionFlag:
    """Test the --version flag."""

    def test_version_output(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "extropy" in result.output


class TestNetworkCommand:
    """Smoke tests for the network command options."""

    def test_network_command_supports_fast_mode_and_checkpoint(self, tmp_path):
        study_db = tmp_path / "study.db"
        config_path = tmp_path / "network-config.yaml"
        output_path = tmp_path / "network.json"
        checkpoint_path = tmp_path / "network-checkpoint.pkl"

        agents = [
            {"_id": "a0", "role": "x", "team": "alpha"},
            {"_id": "a1", "role": "x", "team": "alpha"},
            {"_id": "a2", "role": "y", "team": "beta"},
            {"_id": "a3", "role": "y", "team": "beta"},
        ]
        with open_study_db(study_db) as db:
            db.save_sample_result(population_id="default", agents=agents, meta={"source": "test"})

        NetworkConfig(seed=42, avg_degree=2.0).to_yaml(config_path)

        result = runner.invoke(
            app,
            [
                "network",
                "--study-db",
                str(study_db),
                "-o",
                str(output_path),
                "-c",
                str(config_path),
                "--no-metrics",
                "--candidate-mode",
                "blocked",
                "--candidate-pool-multiplier",
                "4.0",
                "--block-attr",
                "role",
                "--similarity-workers",
                "1",
                "--similarity-chunk-size",
                "8",
                "--checkpoint",
                str(checkpoint_path),
                "--checkpoint-every",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()
        assert checkpoint_path.exists()

    def test_network_resume_requires_checkpoint(self):
        result = runner.invoke(
            app,
            [
                "network",
                "--study-db",
                "study.db",
                "-o",
                "network.json",
                "--resume-checkpoint",
            ],
        )
        assert result.exit_code == 1
        assert "Study DB not found" in result.output
