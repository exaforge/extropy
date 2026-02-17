"""CLI smoke tests using typer's CliRunner."""

import json
import sqlite3
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

    def _setup_study_with_scenario(self, tmp_path):
        """Helper to create a study folder with a scenario and sampled agents."""

        # Create study folder structure
        study_dir = tmp_path
        study_db = study_dir / "study.db"
        scenario_dir = study_dir / "scenario" / "test"
        scenario_dir.mkdir(parents=True)

        # Create minimal scenario file
        scenario_yaml = scenario_dir / "scenario.v1.yaml"
        scenario_yaml.write_text("""
meta:
  name: test
  description: Test scenario
  base_population: population.v1
  created_at: 2024-01-01T00:00:00

event:
  type: announcement
  content: Test announcement
  source: test_source
  credibility: 0.8
  ambiguity: 0.2
  emotional_valence: 0.0

seed_exposure:
  channels: []
  rules: []

interaction:
  primary_model: passive_observation
  description: Test interaction

spread:
  share_probability: 0.3

outcomes:
  suggested_outcomes: []
  capture_full_reasoning: true

simulation:
  max_timesteps: 10
  timestep_unit: day
""")

        # Create minimal population file
        pop_yaml = study_dir / "population.v1.yaml"
        pop_yaml.write_text("""
meta:
  description: Test population
  geography: USA
  agent_focus: test agents

grounding:
  summary: Test grounding

attributes:
  - name: role
    type: categorical
    distribution:
      type: categorical
      probabilities:
        x: 0.5
        y: 0.5
  - name: team
    type: categorical
    distribution:
      type: categorical
      probabilities:
        alpha: 0.5
        beta: 0.5

sampling_order:
  - role
  - team
""")

        # Create persona config
        persona_yaml = scenario_dir / "persona.v1.yaml"
        persona_yaml.write_text("""
intro_template: "You are {role} on team {team}."
treatments: []
groups: []
phrasings:
  boolean: []
  categorical: []
  relative: []
  concrete: []
""")

        # Create agents in DB with scenario_id
        agents = [
            {"_id": "a0", "role": "x", "team": "alpha"},
            {"_id": "a1", "role": "x", "team": "alpha"},
            {"_id": "a2", "role": "y", "team": "beta"},
            {"_id": "a3", "role": "y", "team": "beta"},
        ]
        with open_study_db(study_db) as db:
            db.save_sample_result(
                population_id="test",
                agents=agents,
                meta={"source": "test"},
                scenario_id="test",
            )

        return study_dir

    def test_network_command_supports_fast_mode_and_checkpoint(self, tmp_path):
        import os

        study_dir = self._setup_study_with_scenario(tmp_path)
        study_db = study_dir / "study.db"
        config_path = tmp_path / "network-config.yaml"
        output_path = tmp_path / "network.json"

        NetworkConfig(seed=42, avg_degree=2.0).to_yaml(config_path)

        old_cwd = os.getcwd()
        try:
            os.chdir(study_dir)
            result = runner.invoke(
                app,
                [
                    "network",
                    "-s",
                    "test",
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
                    str(study_db),
                    "--checkpoint-every",
                    "1",
                ],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, f"Output: {result.output}"
        assert output_path.exists()
        with open_study_db(study_db) as db:
            rows = db.run_select("SELECT COUNT(*) AS cnt FROM network_similarity_jobs")
            assert rows and int(rows[0]["cnt"]) >= 1

    def test_network_requires_study_folder(self, tmp_path):
        """Network command requires being in a study folder."""
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["network", "-s", "test"])
            # Should fail with "Not in a study folder" error
            assert result.exit_code != 0
            assert "study folder" in result.output.lower()
        finally:
            os.chdir(old_cwd)

    def test_network_requires_agents(self, tmp_path):
        """Network command requires sampled agents for the scenario."""
        import os

        # Create study folder with scenario but no agents
        study_dir = tmp_path
        study_db = study_dir / "study.db"
        scenario_dir = study_dir / "scenario" / "test"
        scenario_dir.mkdir(parents=True)

        # Create minimal scenario file
        scenario_yaml = scenario_dir / "scenario.v1.yaml"
        scenario_yaml.write_text("""
meta:
  name: test
  description: Test scenario
  base_population: population.v1
  created_at: 2024-01-01T00:00:00
event:
  type: announcement
  content: Test
  source: test
  credibility: 0.8
  ambiguity: 0.2
  emotional_valence: 0.0
seed_exposure:
  channels: []
  rules: []
interaction:
  primary_model: passive_observation
  description: Test
spread:
  share_probability: 0.3
outcomes:
  suggested_outcomes: []
  capture_full_reasoning: true
simulation:
  max_timesteps: 10
  timestep_unit: day
""")

        # Just create an empty DB
        with open_study_db(study_db):
            pass

        old_cwd = os.getcwd()
        try:
            os.chdir(study_dir)
            result = runner.invoke(app, ["network", "-s", "test"])
            # Should fail because no agents exist
            assert result.exit_code != 0
            assert (
                "no agents" in result.output.lower()
                or "sample" in result.output.lower()
            )
        finally:
            os.chdir(old_cwd)

    def test_network_checkpoint_must_match_study_db(self, tmp_path):
        import os

        study_dir = self._setup_study_with_scenario(tmp_path)
        other_db = tmp_path / "other.db"

        old_cwd = os.getcwd()
        try:
            os.chdir(study_dir)
            result = runner.invoke(
                app,
                [
                    "network",
                    "-s",
                    "test",
                    "--checkpoint",
                    str(other_db),
                ],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 1
        assert (
            "--checkpoint must point to the same canonical file as study.db"
            in result.output
        )


def _seed_run_scoped_state(study_db: Path) -> None:
    agents = [
        {"_id": "a0", "team": "alpha"},
        {"_id": "a1", "team": "beta"},
    ]
    with open_study_db(study_db) as db:
        db.save_sample_result(
            population_id="default", agents=agents, meta={"source": "test"}
        )
        db.create_simulation_run(
            run_id="run_old",
            scenario_name="s",
            population_id="default",
            network_id="default",
            config={},
            seed=1,
            status="completed",
        )
        db.create_simulation_run(
            run_id="run_new",
            scenario_name="s",
            population_id="default",
            network_id="default",
            config={},
            seed=2,
            status="running",
        )

    conn = sqlite3.connect(str(study_db))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO agent_states (run_id, agent_id, aware, position, private_position, updated_at)
        VALUES ('run_old', 'a0', 1, 'old_pos', 'old_pos', 0)
        """
    )
    cur.execute(
        """
        INSERT INTO agent_states (run_id, agent_id, aware, position, private_position, updated_at)
        VALUES ('run_new', 'a0', 1, 'new_pos', 'new_pos', 0)
        """
    )
    cur.execute(
        """
        INSERT INTO timestep_summaries (
            run_id, timestep, new_exposures, agents_reasoned, shares_occurred,
            state_changes, exposure_rate, position_distribution_json
        )
        VALUES ('run_new', 0, 1, 1, 0, 1, 0.5, '{}')
        """
    )
    conn.commit()
    conn.close()


class TestRunScopedCliReads:
    def test_results_defaults_to_latest_run(self, tmp_path):
        import os

        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["results"])
            assert result.exit_code == 0
            assert "run_id=run_new" in result.output
            assert "new_pos" in result.output
            assert "old_pos" not in result.output
        finally:
            os.chdir(old_cwd)


class TestInspectNetworkStatus:
    def test_inspect_network_status(self, tmp_path):
        study_db = tmp_path / "study.db"
        with open_study_db(study_db) as db:
            db.upsert_network_generation_status(
                network_run_id="run_123",
                phase="Calibrating network",
                current=2,
                total=8,
                message="Calibration restart 1/4",
            )
            cal_id = db.create_network_calibration_run(
                network_run_id="run_123",
                restart_index=0,
                seed=42,
            )
            db.append_network_calibration_iteration(
                calibration_run_id=cal_id,
                iteration=0,
                phase="calibration",
                intra_scale=1.0,
                inter_scale=1.1,
                edge_count=100,
                avg_degree=2.0,
                clustering=0.2,
                modularity=0.7,
                largest_component_ratio=0.9,
                score=1.23,
                accepted=False,
            )
        result = runner.invoke(
            app,
            [
                "inspect",
                "network-status",
                "--study-db",
                str(study_db),
                "--network-run-id",
                "run_123",
            ],
        )
        assert result.exit_code == 0
        assert "Network Run Status" in result.output
        assert "run_123" in result.output

    def test_export_states_defaults_to_latest_run(self, tmp_path):
        study_db = tmp_path / "study.db"
        out = tmp_path / "states.jsonl"
        _seed_run_scoped_state(study_db)

        result = runner.invoke(
            app,
            ["export", "states", "--study-db", str(study_db), "--to", str(out)],
        )
        assert result.exit_code == 0
        rows = [
            json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run_new"
        assert rows[0]["private_position"] == "new_pos"

    def test_chat_ask_reads_state_for_requested_run(self, tmp_path):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        result = runner.invoke(
            app,
            [
                "chat",
                "ask",
                "--study-db",
                str(study_db),
                "--run-id",
                "run_old",
                "--agent-id",
                "a0",
                "--prompt",
                "what is my stance",
                "--json",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout.strip())
        assert payload["session_id"]
        assert "old_pos" in payload["assistant_text"]
        assert "new_pos" not in payload["assistant_text"]


class TestPersonaCommand:
    def test_persona_requires_study_folder(self, tmp_path):
        """Persona command requires being in a study folder."""
        import os

        # Run from a non-study folder
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["persona", "-s", "test"])
            # Should fail with "Not in a study folder" error
            assert result.exit_code != 0
            assert "study folder" in result.output.lower()
        finally:
            os.chdir(old_cwd)

    def test_persona_requires_scenario(self, tmp_path):
        """Persona command requires a scenario to exist."""
        import os

        # Create a study folder without scenarios
        study_db = tmp_path / "study.db"
        with open_study_db(study_db):
            pass  # Just create the db

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["persona"])
            # Should fail because no scenarios exist
            assert result.exit_code != 0
            assert "scenario" in result.output.lower()
        finally:
            os.chdir(old_cwd)
