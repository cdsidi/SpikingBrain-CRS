from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from data.pipeline import MedicalDataLoader
from evaluation import MedicalMetrics
from training import CRSMedicalTrainer, MemoryOptimizer, RTX4050Config


class TinyModel(nn.Module):
    def __init__(self, d_in: int = 16, n_cls: int = 3) -> None:
        super().__init__()
        self.d_in = d_in
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, n_cls))

    def _to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = x.reshape(b, -1)
        if z.shape[1] >= self.d_in:
            return z[:, : self.d_in]
        pad = torch.zeros((b, self.d_in - z.shape[1]), dtype=z.dtype, device=z.device)
        return torch.cat([z, pad], dim=1)

    def forward(self, x: torch.Tensor, stage: str = "comprehension", sample_ids: torch.Tensor | None = None) -> dict:
        return {"logits": self.net(self._to_fixed(x))}


def run_once(seed: int, out_json: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = RTX4050Config(batch_size=16).to_training_config({"seed": seed, "lr": 1e-3, "device": "cpu"})
    trainer = CRSMedicalTrainer(TinyModel(), MemoryOptimizer.optimize_config(cfg))
    batch = next(iter(MedicalDataLoader("lc25000", "train", batch_size=16, seed=seed)))

    t0 = time.perf_counter()
    train_metrics = trainer.train_epoch([batch], epoch=0)
    elapsed = time.perf_counter() - t0

    logits = trainer._forward_logits(batch["input"], stage="comprehension", sample_ids=batch["sample_id"])
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
    y_true = batch["label"].cpu().numpy()
    eval_metrics = MedicalMetrics.compute_all(y_true, y_pred, y_prob)

    ckpt = Path("checkpoints") / f"e2e_smoke_seed{seed}.pt"
    trainer.save_checkpoint(ckpt, epoch=0, metrics=train_metrics)
    loaded_epoch = trainer.load_checkpoint(ckpt)

    stable = {
        "seed": seed,
        "avg_loss": train_metrics["avg_loss"],
        "accuracy": eval_metrics["accuracy"],
        "peak_gpu_mem_mb": train_metrics["peak_gpu_mem_mb"],
        "throughput": train_metrics["throughput_samples_per_step"],
        "batch_size": 16,
    }
    signature = hashlib.md5(json.dumps(stable, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    payload = {
        "elapsed_sec": elapsed,
        "train": train_metrics,
        "eval": eval_metrics,
        "signature": signature,
        "checkpoint": str(ckpt),
        "epoch_loaded": loaded_epoch,
        "config": {"seed": seed, "batch_size": 16, "device": "cpu"},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    out = Path("results/e2e_validation/smoke_metrics_seed42.json")
    r = run_once(42, out)
    print("E2E_SMOKE_OK", round(r["elapsed_sec"], 6), r["signature"], Path(r["checkpoint"]).exists())

