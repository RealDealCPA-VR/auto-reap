"""reap_cmd: ratio math, exact command content, dataset-folder conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reaplab.core.jsonl import write_jsonl
from reaplab.core.records import CalibrationRecord
from reaplab.prune.errors import PruneError
from reaplab.prune.reap_cmd import (
    build_prune_command,
    calib_to_dataset_dir,
    compression_ratio,
    format_ratio,
    retention_tag,
)

from .helpers import MODEL_ID, make_spec


class TestRatioMath:
    @pytest.mark.parametrize(
        ("retention", "expected"),
        [(0.5, "0.5"), (0.625, "0.375"), (0.75, "0.25"), (0.9, "0.1"), (1.0, "0")],
    )
    def test_format_ratio_is_clean(self, retention: float, expected: str):
        assert format_ratio(retention) == expected

    def test_no_binary_float_junk_at_0625(self):
        # the classic failure: 1 - 0.625 -> 0.37500000000000004 without rounding
        assert compression_ratio(0.625) == 0.375
        assert "00000" not in format_ratio(0.625)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_out_of_range_retention_rejected(self, bad: float):
        with pytest.raises(PruneError, match="retention"):
            compression_ratio(bad)

    @pytest.mark.parametrize(
        ("retention", "tag"), [(0.5, "r0.5"), (0.625, "r0.625"), (0.75, "r0.75")]
    )
    def test_retention_tag_g_format(self, retention: float, tag: str):
        assert retention_tag(retention) == tag


class TestBuildPruneCommand:
    def test_exact_command_content(self, tmp_path: Path):
        spec = make_spec(tmp_path, seeds=[42])
        dataset = tmp_path / "dataset"
        cmd = build_prune_command(spec, 0.625, dataset)
        assert cmd == [
            "python",
            "src/reap/prune.py",
            "--model-name",
            MODEL_ID,
            "--dataset-name",
            str(dataset),
            "--compression-ratio",
            "0.375",
            "--prune-method",
            "reap",
            "--seed",
            "42",
            "--distance_measure",
            "cosine",
            "--record_pruning_metrics_only",
            "false",
        ]

    def test_shell_variable_dataset_passes_through_verbatim(self, tmp_path: Path):
        """Remote scripts pass $DATASET; it must not be pathlib-normalized."""
        spec = make_spec(tmp_path)
        cmd = build_prune_command(spec, 0.5, "$DATASET")
        assert "$DATASET" in cmd
        assert "\\" not in cmd[cmd.index("--dataset-name") + 1]

    def test_uses_first_seed(self, tmp_path: Path):
        spec = make_spec(tmp_path, seeds=[7, 99])
        cmd = build_prune_command(spec, 0.5, "d")
        assert cmd[cmd.index("--seed") + 1] == "7"


class TestCalibToDatasetDir:
    def test_creates_messages_column_jsonl(self, calibration_jsonl: Path, tmp_path: Path):
        out = calib_to_dataset_dir(calibration_jsonl, tmp_path / "ds")
        data = out / "data.jsonl"
        assert data.exists()
        lines = data.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            row = json.loads(line)
            assert set(row) == {"messages"}
            (msg,) = row["messages"]
            assert msg["role"] == "user"
            assert msg["content"].startswith("Categorize transaction")

    def test_prompt_content_round_trips(self, tmp_path: Path):
        rec = CalibrationRecord(id="c1", domain="d", prompt="unicode éà & \"quotes\"")
        src = tmp_path / "cal.jsonl"
        write_jsonl(src, [rec])
        out = calib_to_dataset_dir(src, tmp_path / "ds")
        row = json.loads((out / "data.jsonl").read_text(encoding="utf-8"))
        assert row["messages"][0]["content"] == rec.prompt

    def test_missing_file_is_instructive(self, tmp_path: Path):
        with pytest.raises(PruneError, match="reap-lab generate"):
            calib_to_dataset_dir(tmp_path / "nope.jsonl", tmp_path / "ds")

    def test_empty_file_is_instructive(self, tmp_path: Path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(PruneError, match="empty"):
            calib_to_dataset_dir(empty, tmp_path / "ds")

    def test_idempotent_overwrite(self, calibration_jsonl: Path, tmp_path: Path):
        out_dir = tmp_path / "ds"
        first = (calib_to_dataset_dir(calibration_jsonl, out_dir) / "data.jsonl").read_bytes()
        second = (calib_to_dataset_dir(calibration_jsonl, out_dir) / "data.jsonl").read_bytes()
        assert first == second
