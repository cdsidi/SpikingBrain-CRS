"""稀疏度分析实现。"""

from __future__ import annotations

from typing import Dict

import torch


class SparsityAnalyzer:
    """模型稀疏度分析器。"""

    def analyze_model(self, model: object, dataloader: object = None) -> Dict[str, float]:
        """返回每层与全局参数稀疏度。

        Args:
            model: 具有 named_parameters() 的 PyTorch 模型。
            dataloader: 预留参数，当前最小实现不依赖。
        """
        total = 0
        zeros = 0
        layer_stats: Dict[str, float] = {}

        if not hasattr(model, "named_parameters"):
            return {"global_sparsity": 0.0}

        for name, p in model.named_parameters():
            if not isinstance(p, torch.Tensor):
                continue
            numel = int(p.numel())
            if numel == 0:
                continue
            z = int((p.detach() == 0).sum().item())
            s = float(z / numel)
            layer_stats[f"layer::{name}"] = s
            total += numel
            zeros += z

        global_sparsity = float(zeros / total) if total > 0 else 0.0
        out: Dict[str, float] = {"global_sparsity": global_sparsity, "param_count": float(total)}
        out.update(layer_stats)
        return out
