"""自适应阈值脉冲编码实现。"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class AdaptiveThresholdSpiking(nn.Module):
    """自适应阈值脉冲层。

    输入:
        x: Tensor, shape [..., dim], dtype float16/float32
    输出:
        spike_output: Tensor, shape [..., dim], dtype float16/float32
        spike_integer: Tensor, shape [..., dim], dtype int32，范围[-8, 8]
        threshold: float, 当前批次阈值标量
    """

    def __init__(self, k_base: float, k_medical: Dict[str, float]) -> None:
        super().__init__()
        self.k_base = float(k_base)
        self.k_medical = k_medical

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """将激活量化为脉冲整数并裁剪。"""
        in_dtype = x.dtype
        h = x.float()
        mod_scale = float(sum(self.k_medical.values()) / max(len(self.k_medical), 1)) if self.k_medical else 1.0
        thr_t = h.abs().mean() * self.k_base * mod_scale + 1e-6
        spike_float = h / thr_t
        spike_integer = torch.clamp(torch.round(spike_float), min=-8, max=8).to(torch.int32)
        spike_output = (spike_integer.float() * thr_t).to(in_dtype)
        return spike_output, spike_integer, float(thr_t.detach().item())
