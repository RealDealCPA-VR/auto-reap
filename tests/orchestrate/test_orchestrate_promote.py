"""Promotion tests: LM Studio layout, decision page, smoke hook, archival,
disk guard. The only subprocess used is `python -c ...` (always present)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reaplab.core.config import PromoteCfg, SweepSpec
from reaplab.core.records import ArtifactManifest
from reaplab.orchestrate.promote import PromotionResult, promote_winner, split_command
from reaplab.orchestrate.scoring import GateResult

WINNER_BYTES = b"GGUF-winner-payload" * 16


@pytest.fixture
def lms_dir(tmp_path: Path) -> Path:
    return tmp_path / "lms-models"


@pytest.fixture
def spec(tmp_path: Path, lms_dir: Path) -> SweepSpec:
    return SweepSpec(
        model_id="Qwen/Qwen3-30B-A3B",
        domain_pack=str(tmp_path / "pack.yaml"),  # unused by promotion
        workspace=str(tmp_path / "ws"),
        promote=PromoteCfg(lmstudio_dir=str(lms_dir), publisher="reap-lab"),
    )


@pytest.fixture
def winner_manifest(spec: SweepSpec, ws) -> ArtifactManifest:
    gguf = ws.artifacts / "r0.5-q4_k_m.gguf"
    gguf.write_bytes(WINNER_BYTES)
    return ArtifactManifest(
        artifact_id="r0.5-q4_k_m",
        kind="gguf",
        model_id=spec.model_id,
        retention=0.5,
        quant="Q4_K_M",
        path=str(gguf),
        config_hash="cafe0123beef",
        artifact_hash="deadbeefcafe",
    )


@pytest.fixture
def loser_gguf(ws) -> Path:
    loser = ws.artifacts / "r0.75-q4_k_m.gguf"
    loser.write_bytes(b"GGUF-loser" * 8)
    return loser


@pytest.fixture
def loser_manifest(spec: SweepSpec, loser_gguf: Path) -> ArtifactManifest:
    """The manifest run_sweep hands to promote_winner for an EVALUATED non-winner."""
    return ArtifactManifest(
        artifact_id="r0.75-q4_k_m",
        kind="gguf",
        model_id=spec.model_id,
        retention=0.75,
        quant="Q4_K_M",
        path=str(loser_gguf),
        config_hash="cafe0123beef",
    )


@pytest.fixture
def report_path(ws) -> Path:
    path = ws.reports / "sweep-cafe0123beef.md"
    path.write_text("# report\n", encoding="utf-8")
    return path


def gates() -> list[GateResult]:
    return [
        GateResult(name="quality_retention", value=0.98, limit=0.95, passed=True),
        GateResult(name="decode_tps", value=42.0, limit=None, passed=True, blocking=False),
    ]


def test_promote_uses_required_two_level_layout(spec, winner_manifest, report_path, ws, lms_dir):
    result = promote_winner(spec, winner_manifest, report_path, ws, gates=gates())
    assert result.ok, result.message
    assert result.stage == "done"
    expected = (
        lms_dir / "reap-lab" / "Qwen3-30B-A3B-r0.5-q4_k_m" / "Qwen3-30B-A3B-r0.5-q4_k_m.gguf"
    )
    assert result.dest_path == expected
    assert expected.read_bytes() == WINNER_BYTES
    # exactly two directory levels between the models root and the gguf
    assert expected.parent.parent.parent == lms_dir
    # copy, not move: the source artifact stays in the workspace
    assert Path(winner_manifest.path).exists()


def test_decision_page_written_with_gates_and_hash(spec, winner_manifest, report_path, ws):
    result = promote_winner(
        spec, winner_manifest, report_path, ws, gates=gates(), rationale="Best tradeoff."
    )
    assert result.decision_page is not None and result.decision_page.exists()
    page = result.decision_page.read_text(encoding="utf-8")
    assert "quality_retention" in page
    assert "deadbeefcafe" in page  # artifact hash
    assert "cafe0123beef" in page  # config hash
    assert str(report_path) in page
    assert "Best tradeoff." in page
    # default decision dir is workspace.reports
    assert result.decision_page.parent == ws.reports


def test_decision_page_honors_decision_dir(spec, winner_manifest, report_path, ws, tmp_path):
    spec.promote.decision_dir = str(tmp_path / "brain" / "decisions")
    result = promote_winner(spec, winner_manifest, report_path, ws)
    assert result.ok
    assert result.decision_page is not None
    assert result.decision_page.parent == tmp_path / "brain" / "decisions"


def test_archive_moves_losers_but_never_the_winner(
    spec, winner_manifest, report_path, ws, loser_gguf, loser_manifest
):
    spec.promote.archive_losers = True
    result = promote_winner(
        spec, winner_manifest, report_path, ws, losers=[loser_manifest, winner_manifest]
    )
    assert result.ok
    assert not loser_gguf.exists()  # moved out of artifacts
    assert (ws.archive / loser_gguf.name).exists()  # moved, not deleted
    assert Path(winner_manifest.path).exists()  # winner source untouched even if listed


def test_archive_disabled_leaves_losers(
    spec, winner_manifest, report_path, ws, loser_gguf, loser_manifest
):
    spec.promote.archive_losers = False
    result = promote_winner(spec, winner_manifest, report_path, ws, losers=[loser_manifest])
    assert result.ok
    assert loser_gguf.exists()


def test_archive_never_discovers_files_on_its_own(spec, winner_manifest, report_path, ws):
    """[7]: with no loser list, promotion moves NOTHING — a never-evaluated candidate
    (or a bf16 intermediate) must never be archived out from under a partial grid."""
    spec.promote.archive_losers = True
    unevaluated = ws.artifacts / "r0.625-q4_k_m.gguf"
    unevaluated.write_bytes(b"GGUF-never-evaluated")
    bf16 = ws.artifacts / "Qwen3-30B-A3B-bf16.gguf"
    bf16.write_bytes(b"GGUF-bf16")

    result = promote_winner(spec, winner_manifest, report_path, ws)
    assert result.ok
    assert unevaluated.exists()
    assert bf16.exists()
    assert not list(ws.archive.glob("*.gguf"))
    assert "Archived" not in result.message


def test_smoke_failure_blocks_and_skips_archival(
    spec, winner_manifest, report_path, ws, loser_gguf, loser_manifest
):
    spec.promote.archive_losers = True
    spec.promote.smoke_command = 'python -c "import sys; sys.exit(1)"'
    result = promote_winner(spec, winner_manifest, report_path, ws, losers=[loser_manifest])
    assert not result.ok
    assert result.stage == "smoke"
    assert loser_gguf.exists(), "losers must NOT be archived on smoke failure"
    # decision page still written so there is a record of the failed attempt
    assert result.decision_page is not None and result.decision_page.exists()
    assert "FAILED" in result.decision_page.read_text(encoding="utf-8")


def test_smoke_success_substitutes_path_placeholder(
    spec, winner_manifest, report_path, ws, loser_gguf, loser_manifest
):
    spec.promote.archive_losers = True
    spec.promote.smoke_command = (
        'python -c "import sys; sys.exit(0 if sys.argv[1].endswith(\'.gguf\') else 1)" {path}'
    )
    result = promote_winner(spec, winner_manifest, report_path, ws, losers=[loser_manifest])
    assert result.ok, result.message
    assert not loser_gguf.exists()  # archived after a passing smoke test


def test_smoke_substitutes_model_placeholder(spec, winner_manifest, report_path, ws):
    spec.promote.smoke_command = (
        "python -c \"import sys; sys.exit(0 if sys.argv[1].startswith('reap-lab/') else 1)\" {model}"
    )
    result = promote_winner(spec, winner_manifest, report_path, ws)
    assert result.ok, result.message


def test_smoke_command_not_found_is_instructive(spec, winner_manifest, report_path, ws):
    spec.promote.smoke_command = "definitely-not-a-real-tool-4242 {path}"
    result = promote_winner(spec, winner_manifest, report_path, ws)
    assert not result.ok
    assert result.stage == "smoke"
    assert "not found" in result.message


def test_disk_guard_blocks_before_copy(
    spec, winner_manifest, report_path, ws, lms_dir, monkeypatch
):
    monkeypatch.setattr("reaplab.orchestrate.promote._free_bytes", lambda p: 0)
    result = promote_winner(spec, winner_manifest, report_path, ws)
    assert not result.ok
    assert result.stage == "disk"
    assert "free" in result.message.lower()
    assert not lms_dir.exists()  # nothing was copied


def test_missing_source_gguf_is_instructive(spec, report_path, ws):
    manifest = ArtifactManifest(
        artifact_id="r0.5-q4_k_m",
        kind="gguf",
        model_id=spec.model_id,
        retention=0.5,
        quant="Q4_K_M",
        path=str(ws.artifacts / "ghost.gguf"),
        config_hash="cafe0123beef",
    )
    result = promote_winner(spec, manifest, report_path, ws)
    assert not result.ok
    assert result.stage == "resolve"
    assert "reap-lab sweep" in result.message


def test_split_command_keeps_windows_backslashes():
    """[4]/[39]: POSIX shlex eats backslashes, so an unquoted Windows path is mangled
    into a command that can never be found."""
    tokens = split_command(r"C:\tools\smoke.exe --model {model}", windows=True)
    assert tokens == [r"C:\tools\smoke.exe", "--model", "{model}"]


def test_split_command_strips_surrounding_quotes_on_windows():
    """posix=False RETAINS the quote characters inside the token; subprocess.run would
    then look for a file literally named '"C:\\Program Files\\smoke.exe"'."""
    tokens = split_command(r'"C:\Program Files\smoke.exe" {path}', windows=True)
    assert tokens == [r"C:\Program Files\smoke.exe", "{path}"]


def test_split_command_posix_semantics_unchanged():
    tokens = split_command("/usr/bin/smoke --name 'my model' {path}", windows=False)
    assert tokens == ["/usr/bin/smoke", "--name", "my model", "{path}"]


def test_smoke_substitution_happens_after_splitting(spec, winner_manifest, report_path, ws):
    """A destination path containing spaces must stay ONE argv token."""
    spec.promote.lmstudio_dir = str(Path(spec.promote.lmstudio_dir).parent / "lm studio models")
    spec.promote.smoke_command = (
        "python -c \"import sys; sys.exit(0 if len(sys.argv) == 2 "
        "and sys.argv[1].endswith('.gguf') else 1)\" {path}"
    )
    result = promote_winner(spec, winner_manifest, report_path, ws)
    assert result.ok, result.message
    assert " " in str(result.dest_path)


def test_result_model_shape(spec, winner_manifest, report_path, ws):
    result = promote_winner(spec, winner_manifest, report_path, ws)
    assert isinstance(result, PromotionResult)
    dumped = result.model_dump()
    assert set(dumped) == {"ok", "stage", "message", "dest_path", "decision_page"}
