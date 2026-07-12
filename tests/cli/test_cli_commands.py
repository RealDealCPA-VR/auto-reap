"""CLI surface tests: every command answers --help; core flows run offline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from reaplab.cli.main import app

runner = CliRunner()

ALL_COMMANDS = [
    "version", "demo", "init", "doctor", "generate", "audit",
    "sweep", "report", "promote", "prune", "convert", "eval", "status",
]


def test_root_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ALL_COMMANDS:
        assert cmd in result.output


def test_every_command_help_exits_zero():
    for cmd in ALL_COMMANDS:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "reap-lab" in result.output


def test_init_yes_mock_writes_loadable_yamls(tmp_path):
    result = runner.invoke(
        app,
        [
            "init", "--yes", "--out", str(tmp_path), "--name", "Support Bot!",
            "--describe", "customer support chatbot", "--provider", "mock",
        ],
    )
    assert result.exit_code == 0, result.output
    pack_path = tmp_path / "support-bot-pack.yaml"
    sweep_path = tmp_path / "support-bot-sweep.yaml"
    assert pack_path.exists() and sweep_path.exists()

    from reaplab.core.config import DomainPack, SweepSpec

    spec = SweepSpec.from_yaml(sweep_path)
    pack = DomainPack.from_yaml(spec.domain_pack)
    assert pack.name == "support-bot"
    assert spec.generator.kind == "mock"
    assert pack.domains


def test_init_rejects_unknown_provider(tmp_path):
    result = runner.invoke(
        app, ["init", "--yes", "--out", str(tmp_path), "--provider", "gpt-nine"]
    )
    assert result.exit_code != 0


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "reap-lab doctor" in result.output


def test_doctor_with_bad_spec_reports_fail(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_id: [not, a, string]\ndomain_pack: 42\n", encoding="utf-8")
    result = runner.invoke(app, ["doctor", str(bad)])
    assert result.exit_code == 0  # non-strict never hard-exits
    assert "FAIL" in result.output
    strict = runner.invoke(app, ["doctor", str(bad), "--strict"])
    assert strict.exit_code == 1


def _mini_spec(tmp_path, **overrides) -> str:
    """A tiny all-mock sweep spec on disk; returns its path as str."""
    pack = {
        "name": "mini",
        "description": "mini pack",
        "domains": [
            {
                "name": "classify",
                "description": "classify a request",
                "task_type": "exact",
                "weight": 2.0,
            },
            {
                "name": "chat",
                "description": "general chat",
                "task_type": "open_ended",
                "weight": 1.0,
            },
        ],
    }
    pack_path = tmp_path / "mini-pack.yaml"
    pack_path.write_text(yaml.safe_dump(pack), encoding="utf-8")
    spec = {
        "model_id": "mini/moe",
        "domain_pack": "mini-pack.yaml",
        "retention": [0.75],
        "quants": ["Q4_K_M"],
        "generator": {"kind": "mock"},
        "judge": {"provider": {"kind": "mock"}},
        "data": {"calibration_size": 30, "eval_size": 30},
        "prune": {"execution_profile": "mock"},
        "runtime": {"kind": "mock"},
        # sandboxed LM Studio dir: `promote` must never touch the real ~/.lmstudio
        "promote": {"lmstudio_dir": str(tmp_path / "lms-models")},
        "workspace": str(tmp_path / "ws"),
        "min_free_disk_gb": 0.5,
    }
    spec.update(overrides)
    spec_path = tmp_path / "mini-sweep.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return str(spec_path)


def _load(spec_path: str):
    from reaplab.core.config import SweepSpec

    return SweepSpec.from_yaml(Path(spec_path))


def _data_dir(spec_path: str) -> Path:
    from reaplab.core.paths import Workspace

    spec = _load(spec_path)
    return Workspace(spec.workspace).data_dir(spec.config_hash())


def test_generate_then_audit(tmp_path):
    spec_path = _mini_spec(tmp_path)
    gen = runner.invoke(app, ["generate", spec_path])
    assert gen.exit_code == 0, gen.output
    assert "calibration" in gen.output
    # datasets are per-sweep: they live under runs/<config_hash>/data (C1)
    data_dir = _data_dir(spec_path)
    assert (data_dir / "eval_v1.jsonl").exists()
    assert (data_dir / "calibration_v1.jsonl").exists()
    audit = runner.invoke(app, ["audit", spec_path])
    assert audit.exit_code == 0, audit.output


def test_audit_before_generate_instructs(tmp_path):
    spec_path = _mini_spec(tmp_path)
    result = runner.invoke(app, ["audit", spec_path])
    assert result.exit_code == 1
    assert "generate" in result.output


def _sentinel_eval_set(spec_path: str) -> None:
    """Replace the generated eval set with one recognizable item, so any command that
    silently REGENERATES datasets is caught red-handed ([15]/[28])."""
    item = {
        "id": "sentinel-1",
        "domain": "classify",
        "prompt": "Which category?",
        "task_type": "exact",
        "gold": "billing",
    }
    (_data_dir(spec_path) / "eval_v1.jsonl").write_text(
        json.dumps(item) + "\n", encoding="utf-8"
    )


def test_eval_reuses_the_audited_dataset(tmp_path):
    spec_path = _mini_spec(tmp_path)
    assert runner.invoke(app, ["generate", spec_path]).exit_code == 0
    _sentinel_eval_set(spec_path)

    gguf = tmp_path / "candidate-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    result = runner.invoke(app, ["eval", spec_path, "--gguf", str(gguf)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())  # rich wraps the table title
    assert "(1 items)" in flat, "eval regenerated the dataset instead of reusing it"
    assert (_data_dir(spec_path) / "eval_v1.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1


def test_prune_reuses_the_audited_dataset(tmp_path):
    spec_path = _mini_spec(tmp_path)
    assert runner.invoke(app, ["generate", spec_path]).exit_code == 0
    _sentinel_eval_set(spec_path)

    prune = runner.invoke(app, ["prune", spec_path, "--retention", "0.75"])
    assert prune.exit_code == 0, prune.output
    kept = (_data_dir(spec_path) / "eval_v1.jsonl").read_text(encoding="utf-8")
    assert "sentinel-1" in kept, "prune regenerated (and overwrote) the audited eval set"


def test_sweep_report_status_flow(tmp_path):
    spec_path = _mini_spec(tmp_path)
    sweep = runner.invoke(app, ["sweep", spec_path])
    assert sweep.exit_code == 0, sweep.output
    assert "Report:" in sweep.output

    report = runner.invoke(app, ["report", spec_path])
    assert report.exit_code == 0, report.output

    status = runner.invoke(app, ["status", spec_path])
    assert status.exit_code == 0, status.output


def test_report_runs_no_new_work(tmp_path, monkeypatch):
    """[34]/[m3]: `report` promises 'no new work' — it must not resume the sweep."""
    spec_path = _mini_spec(tmp_path)
    assert runner.invoke(app, ["sweep", spec_path]).exit_code == 0

    import reaplab.orchestrate as orch

    def boom(*args, **kwargs):
        raise AssertionError("report must not run the sweep")

    monkeypatch.setattr(orch, "run_sweep", boom)
    result = runner.invoke(app, ["report", spec_path])
    assert result.exit_code == 0, result.output
    assert "Report:" in result.output


def test_report_before_any_sweep_is_instructive(tmp_path):
    spec_path = _mini_spec(tmp_path)
    result = runner.invoke(app, ["report", spec_path])
    assert result.exit_code == 1
    assert "Nothing has been evaluated yet" in result.output
    assert "reap-lab sweep" in result.output


def test_promote_places_the_winner_and_accepts_an_artifact_override(tmp_path):
    spec_path = _mini_spec(tmp_path)
    assert runner.invoke(app, ["sweep", spec_path]).exit_code == 0

    result = runner.invoke(app, ["promote", spec_path])
    assert result.exit_code == 0, result.output
    promoted = list((tmp_path / "lms-models").rglob("*.gguf"))
    assert len(promoted) == 1
    assert "r0.75-q4_k_m" in promoted[0].name

    override = runner.invoke(app, ["promote", spec_path, "--artifact", "r0.75-q4_k_m"])
    assert override.exit_code == 0, override.output


def test_promote_unknown_artifact_exits_one(tmp_path):
    spec_path = _mini_spec(tmp_path)
    assert runner.invoke(app, ["sweep", spec_path]).exit_code == 0
    result = runner.invoke(app, ["promote", spec_path, "--artifact", "nope-q4_k_m"])
    assert result.exit_code == 1
    assert "nope-q4_k_m" in result.output


def test_promote_before_any_sweep_exits_one(tmp_path):
    spec_path = _mini_spec(tmp_path)
    result = runner.invoke(app, ["promote", spec_path])
    assert result.exit_code == 1
    assert "reap-lab sweep" in result.output


def test_prune_and_convert_commands(tmp_path):
    spec_path = _mini_spec(tmp_path)
    prune = runner.invoke(app, ["prune", spec_path, "--retention", "0.75"])
    assert prune.exit_code == 0, prune.output
    assert "r0.75-q4_k_m" in prune.output
    convert = runner.invoke(app, ["convert", spec_path])
    assert convert.exit_code == 0, convert.output
    assert "baseline-q4_k_m" in convert.output


def test_eval_standalone_gguf(tmp_path):
    spec_path = _mini_spec(tmp_path)
    gguf = tmp_path / "candidate-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    result = runner.invoke(app, ["eval", spec_path, "--gguf", str(gguf)])
    assert result.exit_code == 0, result.output
    assert "classify" in result.output


def test_invalid_spec_fails_cleanly(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("retention: [2.0]\n", encoding="utf-8")
    result = runner.invoke(app, ["sweep", str(bad)])
    assert result.exit_code == 1
    assert "Invalid sweep spec" in result.output
