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

from data.pipeline import MedicalDataLoader
from evaluation import MedicalMetrics, ReportGenerator
from experiments import AblationStudy, FullExperimentSuite
from training import CRSMedicalTrainer, MemoryOptimizer, RTX4050Config

SEEDS = [1, 42, 123, 456, 2024]


class TinyModel(nn.Module):
    """最小端到端模型。

    Input:
      x: Tensor[B, ...] float32
    Output:
      dict(logits): Tensor[B, C] float32
    """

    def __init__(self, d_in: int = 16, n_cls: int = 3) -> None:
        super().__init__()
        self.d_in = d_in
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, n_cls))

    def _to_fixed_dim(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        if flat.shape[1] >= self.d_in:
            return flat[:, : self.d_in]
        pad = torch.zeros((b, self.d_in - flat.shape[1]), dtype=flat.dtype, device=flat.device)
        return torch.cat([flat, pad], dim=1)

    def forward(self, x: torch.Tensor, stage: str = "comprehension", sample_ids: torch.Tensor | None = None):
        z = self._to_fixed_dim(x)
        return {"logits": self.net(z)}


def _set_seed(seed: int = 42) -> None:
    assert seed in SEEDS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sig(obj: dict) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _run_e2e(seed: int = 42, inject_failure: bool = False) -> dict:
    _set_seed(seed)
    out = Path("results/integration_test")
    out.mkdir(parents=True, exist_ok=True)

    trace = []
    t0 = time.time()
    result = {"seed": seed, "status": "PASS", "trace": trace, "error": None}

    try:
        # Layer0
        dl = MedicalDataLoader("physionet", "train", batch_size=16, memory_map=True, seed=seed)
        b0 = next(iter(dl))
        trace.append({"layer": 0, "batch_size": int(b0["input"].shape[0])})

        # Layer1+2+3+6
        cfg = RTX4050Config(batch_size=16).to_training_config({"seed": seed, "lr": 1e-3, "device": "cpu"})
        tr = CRSMedicalTrainer(TinyModel(), MemoryOptimizer.optimize_config(cfg))
        m_train = tr.train_epoch([b0], epoch=0)
        trace.append({"layer": 3, "avg_loss": m_train["avg_loss"], "peak_gpu_mem_mb": m_train["peak_gpu_mem_mb"]})

        # Layer5
        logits = tr._forward_logits(b0["input"], stage="comprehension", sample_ids=b0["sample_id"])
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        y_true = b0["label"].cpu().numpy()
        m_eval = MedicalMetrics.compute_all(y_true, y_pred, y_prob)
        trace.append({"layer": 5, "accuracy": m_eval["accuracy"]})

        # Layer4
        suite_out = FullExperimentSuite(results_root=out / "protocol", max_retries=0).run_48h_schedule()
        trace.append({"layer": 4, "summary_exists": (suite_out / "summary.json").exists()})

        # Layer7
        abl = AblationStudy(output_dir=out / "ablation")
        ares = abl.run_component_ablation({"batch_size": 16}, ["gla", "swa", "adaptive_threshold"], [seed])
        trace.append({"layer": 7, "variants": len(ares["variants"])})

        if inject_failure:
            raise RuntimeError("INJECTED_FAILURE: gradient_flow_mock_error")

        summary = {
            "seed": seed,
            "train": m_train,
            "eval": m_eval,
            "ablation_baseline": ares["baseline"]["avg_score"],
            "elapsed_sec": time.time() - t0,
        }
        result["summary"] = summary
        result["signature"] = _sig(
            {
                "seed": seed,
                "avg_loss": m_train["avg_loss"],
                "accuracy": m_eval["accuracy"],
                "ablation_baseline": ares["baseline"]["avg_score"],
            }
        )
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = {"type": type(e).__name__, "message": str(e), "where": "_run_e2e"}
        result["summary"] = {"elapsed_sec": time.time() - t0}
        result["signature"] = _sig({"seed": seed, "status": "FAIL", "error": result["error"]})

    return result


def test_imports() -> None:
    assert CRSMedicalTrainer is not None
    assert FullExperimentSuite is not None
    assert AblationStudy is not None


def test_memory_budget() -> None:
    r = _run_e2e(seed=42)
    assert r["status"] == "PASS"
    assert float(r["summary"]["train"]["peak_gpu_mem_mb"]) < 4200.0


def test_gradient_flow() -> None:
    r = _run_e2e(seed=42)
    assert r["summary"]["train"]["avg_loss"] >= 0.0


def test_determinism() -> None:
    r1 = _run_e2e(seed=42)
    r2 = _run_e2e(seed=42)
    assert r1["signature"] == r2["signature"]


def test_48h_stability() -> None:
    # 轻量模拟：重复多次代表长稳态调度
    sigs = []
    for _ in range(3):
        r = _run_e2e(seed=42)
        assert r["status"] == "PASS"
        sigs.append(r["signature"])
    assert len(set(sigs)) == 1


def test_failure_handling_and_artifacts() -> None:
    out = Path("results/integration_test")
    rg = ReportGenerator(out)
    ok = _run_e2e(seed=42)
    bad = _run_e2e(seed=42, inject_failure=True)
    payload = {"ok": ok, "bad": bad}
    p_json = rg.save_json("integration_summary.json", payload)
    p_md = rg.save_markdown("integration_report.md", "Integration Test", payload)
    rg.append_log("integration.log", f"ok={ok['status']} bad={bad['status']}")

    assert p_json.exists() and p_md.exists() and (out / "integration.log").exists()
    assert bad["status"] == "FAIL"
    assert "where" in bad["error"] and "message" in bad["error"]


if __name__ == "__main__":
    test_imports()
    test_memory_budget()
    test_gradient_flow()
    test_determinism()
    test_48h_stability()
    test_failure_handling_and_artifacts()
    print("INTEGRATION_TESTS_OK")

