from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments import AblationStudy, ComponentMask


SEEDS = [1, 42, 123, 456, 2024]


def _sig(obj: dict) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def test_imports_layer7() -> None:
    assert AblationStudy is not None
    assert ComponentMask is not None


def test_generate_8_variants_and_component_removal_effect() -> None:
    s = AblationStudy(output_dir=Path("results/ablation_framework_test"))
    base = {"batch_size": 16}
    vars8 = s.generate_variants(base, ["gla", "swa", "adaptive_threshold"])
    assert len(vars8) == 8

    cfg = {"disabled_components": []}
    v_base = s._deterministic_score(cfg, seed=42)
    v_removed = s.run_with_gla_removed(cfg, seed=42)
    assert v_base != v_removed


def test_component_mask_context_manager() -> None:
    cfg = {"disabled_components": ["swa"]}
    with ComponentMask(cfg, ["gla"]):
        assert "gla" in cfg["disabled_components"]
        assert "swa" in cfg["disabled_components"]
    assert cfg["disabled_components"] == ["swa"]


def test_aggregation_json_and_waterfall_output() -> None:
    out_dir = Path("results/ablation_framework_test")
    s = AblationStudy(output_dir=out_dir)
    result = s.run_component_ablation({"batch_size": 16}, ["gla", "swa", "adaptive_threshold"], SEEDS)

    assert "baseline" in result and "variants" in result and "ranked" in result
    assert len(result["variants"]) == 8
    assert all("delta_vs_baseline" in r for r in result["variants"])

    j = out_dir / "ablation_results.json"
    p = out_dir / "contribution_waterfall.png"
    assert j.exists()
    assert p.exists()


def test_reproducibility_signature() -> None:
    _set_seed(42)
    s1 = AblationStudy(output_dir=Path("results/ablation_framework_test_r1"))
    r1 = s1.run_component_ablation({"batch_size": 16}, ["gla", "swa", "adaptive_threshold"], SEEDS)

    _set_seed(42)
    s2 = AblationStudy(output_dir=Path("results/ablation_framework_test_r2"))
    r2 = s2.run_component_ablation({"batch_size": 16}, ["gla", "swa", "adaptive_threshold"], SEEDS)
    assert _sig(r1) == _sig(r2)


def test_smoke_batch16() -> None:
    s = AblationStudy(output_dir=Path("results/ablation_framework_smoke"))
    r = s.run_component_ablation({"batch_size": 16}, ["gla", "swa", "adaptive_threshold"], [42])
    assert r["baseline"]["avg_score"] >= 0.0


if __name__ == "__main__":
    test_imports_layer7()
    test_generate_8_variants_and_component_removal_effect()
    test_component_mask_context_manager()
    test_aggregation_json_and_waterfall_output()
    test_reproducibility_signature()
    test_smoke_batch16()
    print("ABLATION_FRAMEWORK_TESTS_OK")

