from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments import BaselineExperiment, CRSExperiment, FullExperimentSuite


def _sig(obj: dict) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def test_imports_experiment_protocol() -> None:
    assert BaselineExperiment is not None
    assert CRSExperiment is not None
    assert FullExperimentSuite is not None


def test_schedule_and_artifacts_and_resume() -> None:
    root = Path("results/experiment_protocol_test")
    if root.exists():
        shutil.rmtree(root)

    suite = FullExperimentSuite(results_root=root, max_retries=1)
    out = suite.run_48h_schedule()
    assert out.exists()
    state_p = root / "state.json"
    summary_p = root / "summary.json"
    assert state_p.exists() and summary_p.exists()

    state_before = json.loads(state_p.read_text(encoding="utf-8"))
    done_before = len(state_before.get("done", {}))
    assert done_before == 12  # 4 models x 3 datasets

    # 再跑一次，应走断点续跑（跳过已完成）
    out2 = suite.run_48h_schedule()
    assert out2.exists()
    state_after = json.loads(state_p.read_text(encoding="utf-8"))
    assert len(state_after.get("done", {})) == done_before


def test_reproducibility_same_seed_outputs() -> None:
    r1 = BaselineExperiment().run("T-Base", "lc25000", n_folds=5, seeds=[1, 42, 123, 456, 2024])
    r2 = BaselineExperiment().run("T-Base", "lc25000", n_folds=5, seeds=[1, 42, 123, 456, 2024])
    assert _sig(r1) == _sig(r2)


def test_smoke_constraints() -> None:
    # 仅编排层 smoke：不触发重训练，验证调度可运行
    suite = FullExperimentSuite(results_root=Path("results/experiment_protocol_smoke"), max_retries=0)
    out = suite.run_48h_schedule()
    assert (out / "summary.json").exists()


if __name__ == "__main__":
    test_imports_experiment_protocol()
    test_schedule_and_artifacts_and_resume()
    test_reproducibility_same_seed_outputs()
    test_smoke_constraints()
    print("EXPERIMENT_PROTOCOL_TESTS_OK")

