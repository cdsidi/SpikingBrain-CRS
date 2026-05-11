from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout_sec: int = 600) -> str:
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=True,
    )
    return (p.stdout or "").strip()


def _sig(p: Path) -> str:
    d = json.loads(p.read_text(encoding="utf-8"))
    # 使用稳定签名字段，避免输出目录路径差异导致误判
    if "signature" in d:
        return str(d["signature"])
    return hashlib.md5(json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def test_entrypoints_import_and_run() -> None:
    o1 = _run([
        sys.executable,
        "-m",
        "training.train_crs",
        "--epochs",
        "1",
        "--quick-smoke",
        "--batch-size",
        "4",
        "--seed",
        "42",
        "--output-dir",
        "results/execution_guide/t_single",
    ], timeout_sec=300)
    assert "TRAIN_CRS_OK" in o1

    o2 = _run([sys.executable, "-m", "experiments.ablation_study", "--base-config", "configs/s_crs.yaml", "--seed", "42", "--output-dir", "results/execution_guide/t_ablation"])
    assert "ABLATION_RUN_OK" in o2


def test_full_flow_and_artifacts() -> None:
    o = _run([
        sys.executable,
        "scripts/run_full_experiment.py",
        "--epochs",
        "1",
        "--quick-smoke",
        "--batch-size",
        "4",
        "--seed",
        "42",
        "--output-dir",
        "results/execution_guide/t_full",
    ], timeout_sec=600)
    assert "FULL_EXPERIMENT_OK" in o

    root = ROOT / "results" / "execution_guide" / "t_full"
    assert (root / "full_summary.json").exists()
    assert (root / "full_report.md").exists()
    assert (root / "full.log").exists()


def test_doc_consistency_and_reproducibility() -> None:
    doc = (ROOT / "docs" / "EXECUTION_GUIDE.md").read_text(encoding="utf-8")
    assert "python scripts/download_data.py" in doc
    assert "python -m training.train_crs" in doc
    assert "python -m experiments.ablation_study" in doc

    _run([
        sys.executable,
        "scripts/run_full_experiment.py",
        "--epochs",
        "1",
        "--quick-smoke",
        "--batch-size",
        "4",
        "--seed",
        "42",
        "--output-dir",
        "results/execution_guide/t_rep1",
    ], timeout_sec=600)
    _run([
        sys.executable,
        "scripts/run_full_experiment.py",
        "--epochs",
        "1",
        "--quick-smoke",
        "--batch-size",
        "4",
        "--seed",
        "42",
        "--output-dir",
        "results/execution_guide/t_rep2",
    ], timeout_sec=600)
    s1 = _sig(ROOT / "results" / "execution_guide" / "t_rep1" / "full_summary.json")
    s2 = _sig(ROOT / "results" / "execution_guide" / "t_rep2" / "full_summary.json")
    assert s1 == s2


if __name__ == "__main__":
    test_entrypoints_import_and_run()
    test_full_flow_and_artifacts()
    test_doc_consistency_and_reproducibility()
    print("EXECUTION_GUIDE_TESTS_OK")

