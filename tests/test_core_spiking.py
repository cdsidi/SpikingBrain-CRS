from __future__ import annotations

import random

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.core import AdaptiveThresholdSpiking, GatedLinearAttention, SlidingWindowAttention, SpikingFFN

SEEDS = [1, 42, 123, 456, 2024]


def _set_seed(seed: int = 42) -> None:
    assert seed in SEEDS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_imports_core() -> None:
    assert GatedLinearAttention is not None
    assert SlidingWindowAttention is not None
    assert AdaptiveThresholdSpiking is not None
    assert SpikingFFN is not None


def test_shapes_and_constraints() -> None:
    _set_seed(42)
    b, l, d, dk, dff = 4, 32, 64, 16, 128
    x = torch.randn(b, l, d)

    gla = GatedLinearAttention(d_model=d, d_k=dk, modality="pathology")
    y, s = gla(x, None)
    assert y.shape == (b, l, d)
    assert s.shape == (b, l, dk)

    swa = SlidingWindowAttention(d_model=d, window_size=4)
    y2 = swa(x)
    assert y2.shape == (b, l, d)

    ats = AdaptiveThresholdSpiking(k_base=1.0, k_medical={"pathology": 1.2})
    so, si, thr = ats(torch.randn(b, l, dff))
    assert so.shape == (b, l, dff)
    assert si.shape == (b, l, dff)
    assert isinstance(thr, float)
    assert int(si.min()) >= -8 and int(si.max()) <= 8

    sffn = SpikingFFN(d_model=d, d_ff=dff, blank_ratio=0.2)
    y3, si2 = sffn(x, stage="blank")
    assert y3.shape == (b, l, d)
    assert si2.shape == (b, l, dff)
    assert int(si2.min()) >= -8 and int(si2.max()) <= 8


def test_gla_invalid_dims() -> None:
    _set_seed(42)
    bad = GatedLinearAttention(d_model=65, d_k=16, modality="ecg")
    x = torch.randn(2, 8, 65)
    ok = False
    try:
        bad(x, None)
    except ValueError:
        ok = True
    assert ok, "GLA 非法维度未触发 ValueError"


def test_fp16_smoke_and_batch_limit() -> None:
    _set_seed(42)
    b, l, d, dk, dff = 16, 16, 64, 16, 128
    xh = torch.randn(b, l, d).half()
    gla = GatedLinearAttention(d_model=d, d_k=dk, modality="xray").half()
    swa = SlidingWindowAttention(d_model=d, window_size=3).half()
    ats = AdaptiveThresholdSpiking(k_base=1.0, k_medical={"xray": 1.0})
    sffn = SpikingFFN(d_model=d, d_ff=dff, blank_ratio=0.1).half()

    y, s = gla(xh, None)
    y2 = swa(y)
    so, si, _ = ats(torch.randn(b, l, dff).half())
    y3, si2 = sffn(y2, stage="non_blank")

    assert y3.shape == (b, l, d)
    assert s.shape[-1] == dk
    assert so.shape == (b, l, dff)
    assert int(si.min()) >= -8 and int(si.max()) <= 8
    assert int(si2.min()) >= -8 and int(si2.max()) <= 8


if __name__ == "__main__":
    test_imports_core()
    test_shapes_and_constraints()
    test_gla_invalid_dims()
    test_fp16_smoke_and_batch_limit()
    print("CORE_SPIKING_TESTS_OK")

