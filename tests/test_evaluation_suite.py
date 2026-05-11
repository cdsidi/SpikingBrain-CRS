from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation import HardwareMonitor, MedicalMetrics, ReportGenerator, SparsityAnalyzer, StatisticalTesting


def _sig(obj: dict) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def test_imports() -> None:
    assert MedicalMetrics is not None
    assert HardwareMonitor is not None
    assert StatisticalTesting is not None
    assert ReportGenerator is not None


def test_metrics_compute() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.2, 0.9, 0.4, 0.3, 0.8, 0.7])
    m = MedicalMetrics.compute_all(y_true, y_pred, y_prob)
    for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "sensitivity", "specificity"]:
        assert k in m
    assert 0.0 <= m["accuracy"] <= 1.0


def test_monitor_and_reports_and_stats() -> None:
    out = Path("results/evaluation_suite_test")
    if out.exists():
        shutil.rmtree(out)

    mon = HardwareMonitor(log_interval=1)
    mon.start_monitoring()
    mon.sample(steps=1, samples=8)
    mon.sample(steps=2, samples=16)
    rep = mon.get_report()
    assert "peak_gpu_mem_mb" in rep and "throughput_samples_per_sec" in rep

    rg = ReportGenerator(out)
    p_json = rg.save_json("metrics.json", {"monitor": rep})
    p_md = rg.save_markdown("report.md", "Evaluation Report", {"monitor": rep})
    p_log = rg.append_log("eval.log", "evaluation done")
    assert p_json.exists() and p_md.exists() and p_log.exists()

    t = StatisticalTesting.paired_ttest([0.8, 0.7, 0.9], [0.75, 0.68, 0.88])
    d = StatisticalTesting.cohens_d([0.8, 0.7, 0.9], [0.75, 0.68, 0.88])
    a = StatisticalTesting.anova_oneway([[0.8, 0.7], [0.75, 0.68], [0.9, 0.88]])
    assert "t_stat" in t and isinstance(d, float) and "f_stat" in a


def test_reproducibility() -> None:
    random.seed(42)
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=32)
    y_prob = np.random.rand(32)
    y_pred = (y_prob > 0.5).astype(int)
    m1 = MedicalMetrics.compute_all(y_true, y_pred, y_prob)

    random.seed(42)
    np.random.seed(42)
    y_true2 = np.random.randint(0, 2, size=32)
    y_prob2 = np.random.rand(32)
    y_pred2 = (y_prob2 > 0.5).astype(int)
    m2 = MedicalMetrics.compute_all(y_true2, y_pred2, y_prob2)
    assert _sig(m1) == _sig(m2)


def test_smoke_batch16_sparsity() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    x = torch.randn(16, 8)
    _ = model(x)
    s = SparsityAnalyzer().analyze_model(model)
    assert "global_sparsity" in s


if __name__ == "__main__":
    test_imports()
    test_metrics_compute()
    test_monitor_and_reports_and_stats()
    test_reproducibility()
    test_smoke_batch16_sparsity()
    print("EVALUATION_SUITE_TESTS_OK")

