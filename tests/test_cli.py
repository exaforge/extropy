"""CLI smoke tests using typer's CliRunner."""

import json
import sqlite3
from types import SimpleNamespace
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

        agents = [
            {"_id": "a0", "role": "x", "team": "alpha"},
            {"_id": "a1", "role": "x", "team": "alpha"},
            {"_id": "a2", "role": "y", "team": "beta"},
            {"_id": "a3", "role": "y", "team": "beta"},
        ]
        with open_study_db(study_db) as db:
            db.save_sample_result(
                population_id="default", agents=agents, meta={"source": "test"}
            )

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
                str(study_db),
                "--checkpoint-every",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()
        with open_study_db(study_db) as db:
            rows = db.run_select("SELECT COUNT(*) AS cnt FROM network_similarity_jobs")
            assert rows and int(rows[0]["cnt"]) >= 1

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

    def test_network_supports_quality_profile_flag(self):
        result = runner.invoke(
            app,
            [
                "network",
                "--study-db",
                "study.db",
                "--quality-profile",
                "strict",
            ],
        )
        assert result.exit_code == 1
        assert "Study DB not found" in result.output

    def test_network_checkpoint_must_match_study_db(self, tmp_path):
        study_db = tmp_path / "study.db"
        other_db = tmp_path / "other.db"
        with open_study_db(study_db) as db:
            db.save_sample_result(
                population_id="default", agents=[{"_id": "a0"}], meta={}
            )

        result = runner.invoke(
            app,
            [
                "network",
                "--study-db",
                str(study_db),
                "--checkpoint",
                str(other_db),
            ],
        )
        assert result.exit_code == 1
        assert (
            "--checkpoint must point to the same canonical file as --study-db"
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
        study_db = tmp_path / "study.db"
        _seed_run_scoped_state(study_db)

        result = runner.invoke(app, ["results", "--study-db", str(study_db)])
        assert result.exit_code == 0
        assert "run_id=run_new" in result.output
        assert "new_pos" in result.output
        assert "old_pos" not in result.output


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
    def test_persona_show_loads_agents_from_study_db(self, tmp_path, monkeypatch):
        import extropy.cli.commands.persona as persona_cmd
        import extropy.population.persona as persona_pkg

        class DummyPopulationSpec:
            @classmethod
            def from_yaml(cls, _path):
                return SimpleNamespace(
                    meta=SimpleNamespace(description="test population"),
                    attributes=[{"name": "age"}],
                )

        class DummyPersonaConfig:
            @classmethod
            def from_file(cls, _path):
                return object()

        monkeypatch.setattr(persona_cmd, "PopulationSpec", DummyPopulationSpec)
        monkeypatch.setattr(persona_pkg, "PersonaConfig", DummyPersonaConfig)
        monkeypatch.setattr(
            persona_pkg,
            "preview_persona",
            lambda _agent, _config, max_width=80: "I am a test persona.",
        )

        spec_file = tmp_path / "population.yaml"
        spec_file.write_text("meta: {}\n", encoding="utf-8")
        persona_file = spec_file.with_suffix(".persona.yaml")
        persona_file.write_text("dummy: true\n", encoding="utf-8")

        study_db = tmp_path / "study.db"
        with open_study_db(study_db) as db:
            db.save_sample_result(
                population_id="default",
                agents=[{"_id": "a0", "age": 30}, {"_id": "a1", "age": 41}],
                meta={"source": "test"},
            )

        result = runner.invoke(
            app,
            [
                "persona",
                str(spec_file),
                "--study-db",
                str(study_db),
                "--population-id",
                "default",
                "--show",
            ],
        )
        assert result.exit_code == 0
        assert "Loaded 2 agents from study DB population_id=default" in result.output
        assert "Persona for Agent a0" in result.output

    def test_persona_rejects_agents_and_study_db_together(self, tmp_path, monkeypatch):
        import extropy.cli.commands.persona as persona_cmd

        monkeypatch.setattr(
            persona_cmd.PopulationSpec,
            "from_yaml",
            classmethod(
                lambda cls, _path: SimpleNamespace(
                    meta=SimpleNamespace(description="test population"),
                    attributes=[{"name": "age"}],
                )
            ),
        )
        spec_file = tmp_path / "population.yaml"
        spec_file.write_text("meta: {}\n", encoding="utf-8")
        agents_file = tmp_path / "agents.json"
        agents_file.write_text("[]\n", encoding="utf-8")
        study_db = tmp_path / "study.db"
        with open_study_db(study_db) as db:
            db.save_sample_result(
                population_id="default", agents=[{"_id": "a0"}], meta={}
            )

        result = runner.invoke(
            app,
            [
                "persona",
                str(spec_file),
                "--agents",
                str(agents_file),
                "--study-db",
                str(study_db),
            ],
        )
        assert result.exit_code == 1
        assert "Use either --agents or --study-db, not both" in result.output
