"""元认知监控实现。"""

from __future__ import annotations

import torch
import torch.nn as nn


class MetacognitiveMonitor(nn.Module):
    """元认知监控器。

    Args:
        d_model: 特征维度，仅保留接口一致性。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.conf_head = nn.Linear(d_model, 1)

    def calculate_confidence_gap(self, model_confidence: torch.Tensor, spike_confidence: torch.Tensor) -> torch.Tensor:
        """计算置信度差值。

        Args:
            model_confidence: Tensor, shape [B] or [B, 1], dtype float16/float32.
            spike_confidence: Tensor, shape [B] or [B, 1] or scalar, dtype float16/float32.
        Returns:
            gap: Tensor, shape broadcast 后与 model_confidence 对齐, dtype float32.
        """
        mc = model_confidence.to(torch.float32)
        sc = spike_confidence.to(torch.float32)
        gap = mc - sc
        return gap
