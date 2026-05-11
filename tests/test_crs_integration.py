from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.crs import (
    ComprehensionPhase,
    ErrorCorrectionPhase,
    FSRSScheduler,
    MetacognitiveMonitor,
    RecallPhase,
    SpacedReviewPhase,
    SynthesisPhase,
)

SEEDS = [1, 42, 123, 456, 2024]


def _set_seed(seed: int = 42) -> None:
    assert seed in SEEDS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DummyModel(nn.Module):
    def __init__(self, d_in: int = 16, n_cls: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, n_cls)

    def forward(self, x: torch.Tensor, stage: str = "", sample_ids: torch.Tensor | None = None):
        pooled = x.mean(dim=1)
        return {"logits": self.fc(pooled)}


def _batch(b: int = 8, l: int = 12, d: int = 16, n_cls: int = 3) -> dict:
    return {
        "input": torch.randn(b, l, d),
        "label": torch.randint(0, n_cls, (b,)),
        "sample_id": torch.arange(0, b, dtype=torch.long),
    }


def test_imports() -> None:
    assert ComprehensionPhase and RecallPhase and SynthesisPhase
    assert SpacedReviewPhase and ErrorCorrectionPhase
    assert FSRSScheduler and MetacognitiveMonitor


def test_five_phases_forward_loss() -> None:
    _set_seed(42)
    m = DummyModel()
    b = _batch(b=8)
    fsrs = FSRSScheduler(request_retention=0.9)

    c_loss = ComprehensionPhase().forward(m, b)
    r_loss = RecallPhase().forward(m, b, None)
    s_loss = SynthesisPhase().forward(m, b)
    sr_out = SpacedReviewPhase().forward(m, b, epoch=0, fsrs=fsrs)
    e_loss = ErrorCorrectionPhase().forward(m, b)

    # SpacedReviewPhase 约定返回 (loss, due_count, total_count)
    assert isinstance(sr_out, tuple) and len(sr_out) == 3
    sr_loss, due_count, total_count = sr_out
    assert isinstance(due_count, int)
    assert isinstance(total_count, int)
    assert 0 <= due_count <= total_count

    losses = [c_loss, r_loss, s_loss, sr_loss, e_loss]
    for loss in losses:
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()


def test_fsrs_due_update_reproducible() -> None:
    _set_seed(42)
    sid = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    f1 = FSRSScheduler(request_retention=0.9)
    due1 = f1.get_due_samples(epoch=0, sample_ids=sid)
    assert due1.numel() == 4
    correct = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    conf = torch.tensor([0.9, 0.8, 0.55, 0.4], dtype=torch.float32)
    f1.update_intervals(sid, correct, conf)
    intervals_1 = [f1.card_states[int(x)]["interval"] for x in sid.tolist()]

    _set_seed(42)
    f2 = FSRSScheduler(request_retention=0.9)
    due2 = f2.get_due_samples(epoch=0, sample_ids=sid)
    f2.update_intervals(sid, correct, conf)
    intervals_2 = [f2.card_states[int(x)]["interval"] for x in sid.tolist()]

    assert due1.tolist() == due2.tolist()
    assert intervals_1 == intervals_2
    assert intervals_1[0] >= intervals_1[1]


def test_metacognitive_gap_shape_values() -> None:
    mon = MetacognitiveMonitor(d_model=16)
    mc = torch.tensor([0.9, 0.7, 0.2], dtype=torch.float32)
    sc = torch.tensor([0.5, 0.8, 0.2], dtype=torch.float32)
    gap = mon.calculate_confidence_gap(mc, sc)
    assert gap.shape == mc.shape
    assert torch.allclose(gap, torch.tensor([0.4, -0.1, 0.0]))


def test_batch16_smoke() -> None:
    _set_seed(42)
    m = DummyModel(d_in=16, n_cls=4)
    b = _batch(b=16, l=8, d=16, n_cls=4)
    fsrs = FSRSScheduler()
    out = SpacedReviewPhase().forward(m, b, epoch=0, fsrs=fsrs)
    assert isinstance(out, tuple) and len(out) == 3
    loss, due_count, total_count = out
    assert loss.ndim == 0
    assert isinstance(due_count, int)
    assert isinstance(total_count, int)


if __name__ == "__main__":
    test_imports()
    test_five_phases_forward_loss()
    test_fsrs_due_update_reproducible()
    test_metacognitive_gap_shape_values()
    test_batch16_smoke()
    print("CRS_INTEGRATION_TESTS_OK")

