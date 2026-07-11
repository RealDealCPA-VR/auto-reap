from __future__ import annotations

from reaplab.core.hashing import artifact_hash, canonical_hash, canonical_json, dir_hash, file_hash


def test_canonical_json_key_order_independent():
    assert canonical_json({"b": 1, "a": [2, {"z": 3, "y": 4}]}) == canonical_json(
        {"a": [2, {"y": 4, "z": 3}], "b": 1}
    )


def test_canonical_hash_stable_and_short():
    h1 = canonical_hash({"retention": [0.5, 0.75], "model": "qwen"})
    h2 = canonical_hash({"model": "qwen", "retention": [0.5, 0.75]})
    assert h1 == h2
    assert len(h1) == 12


def test_canonical_hash_changes_on_content():
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_file_and_dir_hash(tmp_path):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"hello" * 1000)
    assert file_hash(f1) == file_hash(f1)

    sub = tmp_path / "ckpt"
    sub.mkdir()
    (sub / "model.safetensors").write_bytes(b"weights")
    (sub / "config.json").write_text('{"num_experts": 64}')
    h = dir_hash(sub)
    assert h == dir_hash(sub)
    assert artifact_hash(sub) == h
    assert artifact_hash(f1) == file_hash(f1)

    (sub / "config.json").write_text('{"num_experts": 96}')
    assert dir_hash(sub) != h
