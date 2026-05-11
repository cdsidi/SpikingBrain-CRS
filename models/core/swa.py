"""SWA 组件实现（滑动窗口掩码）。"""

import torch
import torch.nn as nn


class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力。

    输入:
        x: Tensor, shape [B, L, d_model], dtype float16/float32
    输出:
        Tensor, shape [B, L, d_model], dtype 与 x 相同
    """

    def __init__(self, d_model: int, window_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.window_size = max(1, window_size)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """仅允许窗口内 token 可见。"""
        in_dtype = x.dtype
        target_dtype = self.q_proj.weight.dtype
        h = x.to(target_dtype)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        l = x.shape[1]
        idx = torch.arange(l, device=x.device)
        dist = (idx[:, None] - idx[None, :]).abs()
        mask = dist <= self.window_size
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)
        return out.to(in_dtype)
