"""CLI smoke tests using typer's CliRunner."""

import json
import os
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from extropy.cli.app import app
from extropy.config import ExtropyConfig, configure, reset_config
from extropy.cli.commands.validate import (
    _canonical_yaml_path_for_invalid,
    _is_persona_file,
    _is_scenario_file,
)
from extropy.population.network.config import NetworkConfig
from extropy.storage import open_study_db

runner = CliRunner()


@pytest.fixture(autouse=True)
def _force_human_cli_mode():
    """Keep CLI tests deterministic regardless local ~/.config/extropy/config.json."""
    reset_config()
    cfg = ExtropyConfig()
    cfg.cli.mode = "human"
    configure(cfg)
    yield
    reset_config()


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

    def test_invalid_filename_detection(self):
        assert _is_scenario_file(Path("scenario.v1.invalid.v1.yaml"))
        assert _is_persona_file(Path("persona.v1.invalid.v2.yaml"))
        assert not _is_persona_file(Path("population.v1.invalid.v1.yaml"))

    def test_invalid_path_canonicalization(self):
        p = Path("scenario.v4.invalid.v3.yaml")
        canonical = _canonical_yaml_path_for_invalid(p)
        assert canonical is not None
        assert canonical.name == "scenario.v4.yaml"

    def test_validate_promotes_invalid_population_file(self, tmp_path):
        invalid_pop = tmp_path / "population.v1.invalid.v1.yaml"
        invalid_pop.write_text("""
meta:
  description: Test population
  geography: USA

grounding:
  overall: low
  sources_count: 0
  strong_count: 0
  medium_count: 0
  low_count: 1

attributes:
  - name: age
    type: int
    category: universal
    description: Age in years
    sampling:
      strategy: independent
      distribution:
        type: uniform
        min: 18
        max: 65
    grounding:
      level: low
      method: estimated

sampling_order:
  - age
""")

        result = runner.invoke(app, ["validate", str(invalid_pop)])
        assert result.exit_code == 0, result.output
        assert not invalid_pop.exists()
        assert (tmp_path / "population.v1.yaml").exists()


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

grounding:
  overall: low
  sources_count: 0
  strong_count: 0
  medium_count: 0
  low_count: 2

attributes:
  - name: role
    type: categorical
    category: population_specific
    description: Role in organization
    sampling:
      strategy: independent
      distribution:
        type: categorical
        options: [x, y]
        weights: [0.5, 0.5]
    grounding:
      level: low
      method: estimated
  - name: team
    type: categorical
    category: population_specific
    description: Team membership
    sampling:
      strategy: independent
      distribution:
        type: categorical
        options: [alpha, beta]
        weights: [0.5, 0.5]
    grounding:
      level: low
      method: estimated

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
            population_id="s", agents=agents, meta={"source": "test"}, scenario_id="s"
        )
        db.create_simulation_run(
            run_id="run_old",
            scenario_name="s",
            population_id="s",
            network_id="s",
            config={},
            seed=1,
            status="completed",
        )
        db.create_simulation_run(
            run_id="run_new",
            scenario_name="s",
            population_id="s",
            network_id="s",
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

    def test_results_timeline_subcommand(self, tmp_path):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["results", "timeline"])
            assert result.exit_code == 0
            assert "Timeline" in result.output
            assert "t=" in result.output
        finally:
            os.chdir(old_cwd)

    def test_results_segment_subcommand(self, tmp_path):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["results", "segment", "team"])
            assert result.exit_code == 0
            assert "Segment by team" in result.output
        finally:
            os.chdir(old_cwd)

    def test_results_agent_subcommand(self, tmp_path):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["results", "agent", "a0"])
            assert result.exit_code == 0
            assert "Agent a0" in result.output
            assert "new_pos" in result.output
        finally:
            os.chdir(old_cwd)


class TestQueryCommand:
    def test_query_network_status(self, tmp_path):
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
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["query", "network-status", "run_123"])
            assert result.exit_code == 0
            assert "Network Run Status" in result.output
            assert "run_123" in result.output
        finally:
            os.chdir(old_cwd)

    def test_query_states_defaults_to_latest_run(self, tmp_path):
        study_db = tmp_path / "study.db"
        out = tmp_path / "states.jsonl"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(
                app,
                ["query", "states", "--to", str(out)],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        rows = [
            json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run_new"
        assert rows[0]["private_position"] == "new_pos"

    def test_query_agents(self, tmp_path):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            out = tmp_path / "agents.jsonl"
            result = runner.invoke(
                app,
                ["query", "agents", "--to", str(out)],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        rows = [
            json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 2

    def test_query_summary(self, tmp_path):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["query", "summary"])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        assert "Study Summary" in result.output

    def test_query_sql_positional(self, tmp_path):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(
                app,
                ["query", "sql", "SELECT count(*) AS cnt FROM agents"],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0
        assert "cnt" in result.output


class TestChatCommand:
    def test_chat_ask_reads_state_for_requested_run(self, tmp_path, monkeypatch):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)
        import extropy.cli.commands.chat as chat_cmd

        captured: dict[str, str] = {}

        def fake_simple_call(
            prompt: str,
            response_schema: dict,
            schema_name: str = "response",
            model: str | None = None,
            log: bool = True,
            max_tokens: int | None = None,
        ):
            del response_schema, schema_name, log, max_tokens
            captured["prompt"] = prompt
            captured["model"] = model or ""
            return {"assistant_text": "I am still at old_pos."}

        monkeypatch.setattr(chat_cmd, "simple_call", fake_simple_call)
        monkeypatch.setattr(chat_cmd, "is_agent_mode", lambda: True)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(
                app,
                [
                    "chat",
                    "ask",
                    "--run-id",
                    "run_old",
                    "--agent-id",
                    "a0",
                    "--prompt",
                    "what is my stance",
                ],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout.strip())
        assert payload["session_id"]
        assert payload["assistant_text"] == "I am still at old_pos."
        assert "old_pos" in captured["prompt"]
        assert "new_pos" not in captured["prompt"]

    def test_chat_ask_includes_session_history(self, tmp_path, monkeypatch):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)
        import extropy.cli.commands.chat as chat_cmd

        with open_study_db(study_db) as db:
            sid = db.create_chat_session(
                run_id="run_old",
                agent_id="a0",
                mode="machine",
                meta={"entrypoint": "test"},
            )
            db.append_chat_message(sid, "user", "first question")
            db.append_chat_message(sid, "assistant", "first answer")

        captured: dict[str, str] = {}

        def fake_simple_call(
            prompt: str,
            response_schema: dict,
            schema_name: str = "response",
            model: str | None = None,
            log: bool = True,
            max_tokens: int | None = None,
        ):
            del response_schema, schema_name, model, log, max_tokens
            captured["prompt"] = prompt
            return {"assistant_text": "second answer"}

        monkeypatch.setattr(chat_cmd, "simple_call", fake_simple_call)
        monkeypatch.setattr(chat_cmd, "is_agent_mode", lambda: True)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(
                app,
                [
                    "chat",
                    "ask",
                    "--run-id",
                    "run_old",
                    "--agent-id",
                    "a0",
                    "--session-id",
                    sid,
                    "--prompt",
                    "second question",
                ],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert "first question" in captured["prompt"]
        assert "first answer" in captured["prompt"]
        assert "second question" in captured["prompt"]

    def test_chat_ask_defaults_to_latest_run_and_first_agent(
        self, tmp_path, monkeypatch
    ):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)
        import extropy.cli.commands.chat as chat_cmd

        captured: dict[str, str] = {}

        def fake_simple_call(
            prompt: str,
            response_schema: dict,
            schema_name: str = "response",
            model: str | None = None,
            log: bool = True,
            max_tokens: int | None = None,
        ):
            del response_schema, schema_name, model, log, max_tokens
            captured["prompt"] = prompt
            return {"assistant_text": "latest run default works"}

        monkeypatch.setattr(chat_cmd, "simple_call", fake_simple_call)
        monkeypatch.setattr(chat_cmd, "is_agent_mode", lambda: True)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(
                app,
                [
                    "chat",
                    "ask",
                    "--prompt",
                    "default target?",
                ],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout.strip())
        assert payload["run_id"] == "run_new"
        assert payload["agent_id"] == "a0"
        assert "new_pos" in captured["prompt"]
        assert "old_pos" not in captured["prompt"]

    def test_chat_list_outputs_recent_runs_and_sample_agents(
        self, tmp_path, monkeypatch
    ):
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)
        import extropy.cli.commands.chat as chat_cmd

        monkeypatch.setattr(chat_cmd, "is_agent_mode", lambda: True)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(
                app,
                [
                    "chat",
                    "list",
                    "--limit-runs",
                    "2",
                    "--agents-per-run",
                    "2",
                ],
            )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout.strip())
        assert payload["runs"]
        assert payload["runs"][0]["run_id"] == "run_new"
        assert "a0" in payload["runs"][0]["sample_agents"]


class TestPersonaCommand:
    def test_persona_requires_study_folder(self, tmp_path):
        """Persona command requires being in a study folder."""
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
