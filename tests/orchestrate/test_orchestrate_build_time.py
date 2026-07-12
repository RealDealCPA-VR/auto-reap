"""PRD §5 advisory metric: does the sweep fit in an overnight window (<= 12 h)?"""

from __future__ import annotations

from reaplab.core.records import ArtifactManifest
from reaplab.orchestrate.sweep import _build_seconds


def _manifest(artifact_id: str, wall_clock_s: float | None) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        kind="gguf",
        model_id="m/moe",
        path=f"{artifact_id}.gguf",
        config_hash="abc123",
        wall_clock_s=wall_clock_s,
    )


def test_sums_build_time_across_artifacts():
    assert _build_seconds([_manifest("a", 3600.0), _manifest("b", 1800.0)]) == 5400.0


def test_resumed_stages_contribute_zero_not_none():
    """A resumed artifact records 0 s of build time — it must still count as measured,
    so a fully-resumed sweep reports 0.00 h rather than 'unknown'."""
    assert _build_seconds([_manifest("a", 0.0), _manifest("b", 0.0)]) == 0.0


def test_no_recorded_durations_reports_nothing_rather_than_zero():
    assert _build_seconds([_manifest("a", None)]) is None
    assert _build_seconds([]) is None


def test_report_header_states_the_overnight_verdict(tmp_path, pack):
    from reaplab.core.config import SweepSpec
    from reaplab.orchestrate.report import ArtifactRow, render_report

    spec = SweepSpec(
        model_id="m/moe",
        domain_pack=str(tmp_path / "pack.yaml"),
        workspace=str(tmp_path / "ws"),
    )
    rows = [ArtifactRow(artifact_id="baseline-q4_k_m", weighted=0.9, is_baseline=True)]

    def render(build_seconds: float | None) -> str:
        return render_report(spec, "hash1234abcd", pack, rows, None, build_seconds=build_seconds)

    within = render(6 * 3600)
    assert "**Artifact build time:** 6.00 h" in within
    assert "within the overnight window" in within

    assert "OVER the 12 h overnight window" in render(13 * 3600)

    # unmeasured: the line is simply absent, never a misleading 0
    assert "Artifact build time" not in render(None)
