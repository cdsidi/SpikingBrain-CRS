from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation import ReportGenerator
from experiments import FullExperimentSuite
from training import CRSMedicalTrainer, MemoryOptimizer, RTX4050Config


def _sig(obj: dict) -> str:
    clean = dict(obj)
    # 时间字段不参与确定性签名（允许真实耗时统计）
    clean.pop("avg_step_time_sec", None)
    clean.pop("eta_sec", None)
    s = json.dumps(clean, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyModel(nn.Module):
    def __init__(self, d_in: int = 16, n_cls: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, n_cls))

    def forward(self, x: torch.Tensor, stage: str = "comprehension", sample_ids: torch.Tensor | None = None):
        return {"logits": self.net(x.mean(dim=1))}


class TinyLoader:
    def __init__(self, batches: int = 2, bsz: int = 16, l: int = 12, d: int = 16, n_cls: int = 3):
        self.items = []
        for i in range(batches):
            self.items.append(
                {
                    "input": torch.randn(bsz, l, d),
                    "label": torch.randint(0, n_cls, (bsz,)),
                    "sample_id": torch.arange(i * bsz, (i + 1) * bsz),
                }
            )

    def __iter__(self):
        return iter(self.items)


def test_imports_layer6() -> None:
    assert MemoryOptimizer is not None
    assert RTX4050Config is not None


def test_feature_switches_effective() -> None:
    cfg = RTX4050Config().to_training_config({"seed": 42, "lr": 1e-3, "device": "cpu"})
    cfg = MemoryOptimizer.optimize_config(cfg)
    t = CRSMedicalTrainer(TinyModel(), cfg)
    r = t.get_hardware_report()
    assert r["effective_batch_size"] <= 16
    assert "optimizer" in r and "enable_amp" in r


def test_resource_metrics_and_outputs() -> None:
    _set_seed(42)
    cfg = RTX4050Config(batch_size=16, enable_amp=True, optimizer="adam8bit").to_training_config({"seed": 42, "lr": 1e-3, "device": "cpu"})
    t = CRSMedicalTrainer(TinyModel(), MemoryOptimizer.optimize_config(cfg))
    ts = time.time()
    m = t.train_epoch(TinyLoader(batches=2, bsz=16), epoch=0)
    elapsed = time.time() - ts
    h = t.get_hardware_report()
    assert "peak_gpu_mem_mb" in h and "throughput_samples_per_step" in h
    out = Path("results/hardware_adaptation_test")
    rg = ReportGenerator(out)
    rg.save_json("hardware_report.json", {"metrics": m, "hardware": h, "elapsed_sec": elapsed})
    assert (out / "hardware_report.json").exists()


def test_compat_with_layer4_layer5_min_chain() -> None:
    out = FullExperimentSuite(results_root=Path("results/hardware_adaptation_protocol"), max_retries=0).run_48h_schedule()
    assert (out / "summary.json").exists()


def test_reproducibility_same_seed() -> None:
    _set_seed(42)
    cfg = RTX4050Config().to_training_config({"seed": 42, "lr": 1e-3, "device": "cpu"})
    t1 = CRSMedicalTrainer(TinyModel(), MemoryOptimizer.optimize_config(cfg))
    m1 = t1.train_epoch(TinyLoader(batches=2, bsz=16), epoch=0)

    _set_seed(42)
    t2 = CRSMedicalTrainer(TinyModel(), MemoryOptimizer.optimize_config(cfg))
    m2 = t2.train_epoch(TinyLoader(batches=2, bsz=16), epoch=0)
    assert _sig(m1) == _sig(m2)


def test_smoke_batch16_no_oom() -> None:
    _set_seed(42)
    cfg = RTX4050Config(batch_size=16).to_training_config({"seed": 42, "lr": 1e-3, "device": "cpu"})
    t = CRSMedicalTrainer(TinyModel(), MemoryOptimizer.optimize_config(cfg))
    m = t.train_epoch(TinyLoader(batches=1, bsz=16), epoch=0)
    assert m["avg_loss"] >= 0.0


if __name__ == "__main__":
    test_imports_layer6()
    test_feature_switches_effective()
    test_resource_metrics_and_outputs()
    test_compat_with_layer4_layer5_min_chain()
    test_reproducibility_same_seed()
    test_smoke_batch16_no_oom()
    print("HARDWARE_ADAPTATION_TESTS_OK")

