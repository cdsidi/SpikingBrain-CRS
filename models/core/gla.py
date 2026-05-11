"""GLA 组件实现。

资源约束提示（01_PRD）：
- 显存预算目标 < 4.2GB
- Layer 1 保持轻量，避免大参数膨胀（全模型目标 < 2.5M）
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GatedLinearAttention(nn.Module):
    """门控线性注意力。

    输入:
        x: Tensor, shape [B, L, d_model], dtype float16/float32
        state: Optional[Tensor], shape [B, L, d_k], dtype float16/float32
    输出:
        output: Tensor, shape [B, L, d_model], dtype 与 x 相同
        new_state: Tensor, shape [B, L, d_k], dtype 与 x 相同
    """

    def __init__(self, d_model: int, d_k: int, modality: str) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.modality = modality
        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_k, bias=False)
        self.gate_proj = nn.Linear(d_model, d_k, bias=True)
        self.out_proj = nn.Linear(d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向计算。

        Raises:
            ValueError: 当 d_model 不是 d_k 的整数倍时触发。
        """
        if self.d_model % self.d_k != 0:
            raise ValueError(f"d_model({self.d_model}) 必须是 d_k({self.d_k}) 的整数倍")
        in_dtype = x.dtype
        target_dtype = self.q_proj.weight.dtype
        work_x = x.to(target_dtype)
        q = self.q_proj(work_x)
        k = self.k_proj(work_x)
        v = self.v_proj(work_x)
        gate = torch.sigmoid(self.gate_proj(work_x))
        if state is None:
            prev = torch.zeros_like(k)
        else:
            prev = state.to(target_dtype)
        new_state = gate * k + (1.0 - gate) * prev
        scores = torch.matmul(q, new_state.transpose(-1, -2)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        out = self.out_proj(ctx)
        return out.to(in_dtype), new_state.to(in_dtype)
