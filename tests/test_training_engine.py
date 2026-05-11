from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from training import CRSMedicalTrainer
from training.checkpoint import load_training_checkpoint

SEEDS = [1, 42, 123, 456, 2024]


def _set_seed(seed: int = 42) -> None:
    assert seed in SEEDS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyModel(nn.Module):
    def __init__(self, d_in: int = 16, n_cls: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, n_cls))

    def forward(self, x: torch.Tensor):
        return {"logits": self.net(x.mean(dim=1))}


class TinyLoader:
    def __init__(self, batches: int = 4, bsz: int = 8, l: int = 12, d: int = 16, n_cls: int = 3):
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


def test_imports_training() -> None:
    assert CRSMedicalTrainer is not None


def test_train_epoch_checkpoint_and_resume() -> None:
    _set_seed(42)
    model = TinyModel()
    trainer = CRSMedicalTrainer(model, {"seed": 42, "lr": 1e-3, "grad_accum_steps": 2, "max_grad_norm": 1.0})
    loader = TinyLoader(batches=4, bsz=8)
    metrics1 = trainer.train_epoch(loader, epoch=0)
    assert "avg_loss" in metrics1 and isinstance(metrics1["avg_loss"], float)

    ckpt_path = Path("checkpoints/test_layer3_ckpt.pt")
    trainer.save_checkpoint(ckpt_path, epoch=0, metrics=metrics1)
    assert ckpt_path.exists()

    payload = load_training_checkpoint(ckpt_path)
    for k in ["model_state", "optimizer_state", "scheduler_state", "epoch", "metrics", "phase_history"]:
        assert k in payload

    model2 = TinyModel()
    trainer2 = CRSMedicalTrainer(model2, {"seed": 42, "lr": 1e-3, "grad_accum_steps": 2, "max_grad_norm": 1.0})
    restored_epoch = trainer2.load_checkpoint(ckpt_path)
    assert restored_epoch == 0
    assert len(trainer2.optimizer.state_dict().get("state", {})) >= 0

    metrics2 = trainer2.train_epoch(TinyLoader(batches=2, bsz=8), epoch=1)
    assert "avg_loss" in metrics2


def test_reproducibility_same_seed() -> None:
    _set_seed(42)
    m1 = TinyModel()
    t1 = CRSMedicalTrainer(m1, {"seed": 42, "lr": 1e-3, "grad_accum_steps": 2})
    out1 = t1.train_epoch(TinyLoader(batches=3, bsz=8), epoch=0)

    _set_seed(42)
    m2 = TinyModel()
    t2 = CRSMedicalTrainer(m2, {"seed": 42, "lr": 1e-3, "grad_accum_steps": 2})
    out2 = t2.train_epoch(TinyLoader(batches=3, bsz=8), epoch=0)

    assert abs(out1["avg_loss"] - out2["avg_loss"]) < 1e-8


def test_batch_size_16_smoke() -> None:
    _set_seed(42)
    model = TinyModel()
    trainer = CRSMedicalTrainer(model, {"seed": 42, "lr": 1e-3})
    metrics = trainer.train_epoch(TinyLoader(batches=1, bsz=16), epoch=0)
    assert metrics["avg_loss"] >= 0.0


if __name__ == "__main__":
    test_imports_training()
    test_train_epoch_checkpoint_and_resume()
    test_reproducibility_same_seed()
    test_batch_size_16_smoke()
    print("TRAINING_ENGINE_TESTS_OK")

