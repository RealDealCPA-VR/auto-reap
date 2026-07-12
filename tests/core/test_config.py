from __future__ import annotations

import pytest

from reaplab.core.config import DomainPack, ProviderCfg, SweepSpec
from reaplab.core.records import TaskType


def test_example_sweep_loads_and_resolves_paths(example_sweep_path):
    spec = SweepSpec.from_yaml(example_sweep_path)
    assert spec.model_id == "Qwen/Qwen3-30B-A3B"
    assert spec.retention == [0.75, 0.625, 0.50]
    # domain_pack path resolved relative to the spec file
    pack = DomainPack.from_yaml(spec.domain_pack)
    assert pack.name == "cpa-firm"


def test_config_hash_stable_and_ignores_workspace(example_sweep_path):
    a = SweepSpec.from_yaml(example_sweep_path)
    b = SweepSpec.from_yaml(example_sweep_path)
    assert a.config_hash() == b.config_hash()
    b.workspace = "elsewhere"
    assert a.config_hash() == b.config_hash()
    b.retention = [0.5]
    assert a.config_hash() != b.config_hash()


def test_retention_validation():
    with pytest.raises(ValueError):
        SweepSpec(model_id="m", domain_pack="p.yaml", retention=[1.5])
    with pytest.raises(ValueError):
        SweepSpec(model_id="m", domain_pack="p.yaml", retention=[0.0])


def test_cpa_pack_weights_and_tools(cpa_pack_path):
    pack = DomainPack.from_yaml(cpa_pack_path)
    weights = pack.normalized_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["qbo_categorization"] == max(weights.values())
    tool_domains = [d for d in pack.domains if d.task_type is TaskType.TOOL_CALL]
    assert tool_domains and all(d.tools for d in tool_domains)
    long_ctx = [d for d in pack.domains if d.long_context]
    assert long_ctx, "pack must exercise long-context items (PRD FR-1.4)"


def test_provider_cfg_defaults():
    cfg = ProviderCfg(kind="openai-compat")
    assert cfg.temperature == 0.0
    assert cfg.api_key_env is None  # local servers need no key


def test_config_hash_tracks_pack_content(tmp_path, example_sweep_path, cpa_pack_path):
    import shutil

    from reaplab.core.config import SweepSpec as SS

    cfg_dir = tmp_path / "configs"
    shutil.copytree(example_sweep_path.parent, cfg_dir)
    spec_path = cfg_dir / example_sweep_path.name
    h1 = SS.from_yaml(spec_path).config_hash()

    pack = cfg_dir / "domain-packs" / "cpa-firm.yaml"
    pack.write_text(
        pack.read_text(encoding="utf-8").replace("weight: 3.0", "weight: 9.0"), encoding="utf-8"
    )
    h2 = SS.from_yaml(spec_path).config_hash()
    assert h1 != h2, "editing pack content must mint a new hash (fresh datasets/artifacts)"

    # comment-only edits don't invalidate completed work
    pack.write_text("# tuning note\n" + pack.read_text(encoding="utf-8"), encoding="utf-8")
    assert SS.from_yaml(spec_path).config_hash() == h2


def test_config_hash_ignores_gates_and_runtime_plumbing(example_sweep_path):
    a = SweepSpec.from_yaml(example_sweep_path)
    b = SweepSpec.from_yaml(example_sweep_path)
    b.gates.max_vram_gb = 24
    b.runtime.port = 19999
    b.runtime.llama_server_path = "C:/tools/llama-server.exe"
    assert a.config_hash() == b.config_hash()
    b.runtime.base_url = "http://localhost:5555/v1"  # selects which server answers -> matters
    assert a.config_hash() != b.config_hash()


def test_unknown_yaml_keys_fail_loudly():
    with pytest.raises(ValueError, match="wieght|extra"):
        DomainPack.model_validate(
            {
                "name": "p",
                "domains": [{"name": "d", "description": "x", "wieght": 2.0}],
            }
        )


def test_duplicate_domain_names_rejected():
    dom = {"name": "d", "description": "x"}
    with pytest.raises(ValueError, match="duplicate domain name"):
        DomainPack.model_validate({"name": "p", "domains": [dom, dict(dom)]})
