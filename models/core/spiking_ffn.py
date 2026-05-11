"""SpikingFFN 实现。"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .adaptive_threshold import AdaptiveThresholdSpiking


class SpikingFFN(nn.Module):
    """脉冲前馈网络。

    输入:
        x: Tensor, shape [B, L, d_model], dtype float16/float32
        stage: str, 控制 blank/非blank 路径
    输出:
        output: Tensor, shape [B, L, d_model], dtype 与 x 相同
        spike_integer: Tensor, shape [B, L, d_ff], dtype int32
    """

    def __init__(self, d_model: int, d_ff: int, blank_ratio: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.blank_ratio = float(blank_ratio)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.spike = AdaptiveThresholdSpiking(k_base=1.0, k_medical={"generic": 1.0})

    def forward(self, x: torch.Tensor, stage: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """stage 含 blank 时应用随机空白门控。"""
        in_dtype = x.dtype
        target_dtype = self.fc1.weight.dtype
        h = x.to(target_dtype)
        ff = torch.relu(self.fc1(h))
        if "blank" in stage.lower() and self.blank_ratio > 0.0:
            keep = (torch.rand_like(ff) > self.blank_ratio).float()
            ff = ff * keep
        spike_out, spike_integer, _ = self.spike(ff)
        out = self.fc2(spike_out.to(target_dtype))
        return out.to(in_dtype), spike_integer
