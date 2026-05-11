"""CRS 五阶段实现（Layer 2）。

保持轻量，不扩展训练引擎；遵守 PRD 资源约束（显存目标 < 4.2GB）。
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F


def _model_logits(model: Any, x: torch.Tensor, stage: str, sample_ids: torch.Tensor | None = None) -> torch.Tensor:
    """统一获取 logits。

    兼容 model 返回 Tensor 或 Dict[str, Tensor]。
    """
    if callable(model):
        try:
            out = model(x, stage=stage, sample_ids=sample_ids)
        except TypeError:
            out = model(x)
    else:
        out = x
    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "output" in out:
            return out["output"]
        first = next(iter(out.values()))
        return first
    return out


def _ce_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        logits = logits.view(logits.size(0), -1)
    weight = None
    if class_weights is not None:
        weight = class_weights.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, labels.long(), weight=weight)


class ComprehensionPhase:
    def forward(self, model: Any, batch: dict, class_weights: torch.Tensor | None = None) -> torch.Tensor:
        """精讲阶段。

        batch:
            input: Tensor [B, ...], float16/float32
            label: Tensor [B], int64
        returns:
            loss: Tensor scalar, 可反传
        """
        logits = _model_logits(model, batch["input"], stage="comprehension", sample_ids=batch.get("sample_id"))
        return _ce_loss(logits, batch["label"], class_weights=class_weights)


class RecallPhase:
    def forward(self, model: Any, batch: dict, difficulty_scheduler: Any, class_weights: torch.Tensor | None = None) -> torch.Tensor:
        """回忆阶段，加入轻量遮挡构造合意困难。"""
        x = batch["input"]
        ratio = 0.2
        if difficulty_scheduler is not None and hasattr(difficulty_scheduler, "get_difficulty"):
            ratio = float(getattr(difficulty_scheduler, "get_difficulty")())
            ratio = max(0.1, min(0.5, ratio))
        mask = (torch.rand_like(x) > ratio).to(x.dtype)
        logits = _model_logits(model, x * mask, stage="recall", sample_ids=batch.get("sample_id"))
        return _ce_loss(logits, batch["label"], class_weights=class_weights)


class SynthesisPhase:
    def forward(self, model: Any, batch: dict, class_weights: torch.Tensor | None = None) -> torch.Tensor:
        """合成阶段（blank 路径）。"""
        logits = _model_logits(model, batch["input"], stage="synthesis_blank", sample_ids=batch.get("sample_id"))
        ce = _ce_loss(logits, batch["label"], class_weights=class_weights)
        reg = 1e-4 * (batch["input"].float().pow(2).mean())
        return ce + reg


class SpacedReviewPhase:
    def forward(self, model: Any, batch: dict, epoch: int, fsrs: Any, class_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, int, int]:
        """间隔复习阶段：只对到期样本计算loss并更新FSRS状态。"""
        sample_ids = batch["sample_id"]
        due_positions = fsrs.get_due_samples(epoch=epoch, sample_ids=sample_ids)
        total_count = int(sample_ids.numel())
        due_count = int(due_positions.numel())
        if due_count == 0:
            return batch["input"].float().mean() * 0.0, 0, total_count
        x_due = batch["input"].index_select(0, due_positions)
        y_due = batch["label"].index_select(0, due_positions)
        sid_due = sample_ids.index_select(0, due_positions)
        logits = _model_logits(model, x_due, stage="spaced_review", sample_ids=sid_due)
        loss = _ce_loss(logits, y_due, class_weights=class_weights)
        with torch.no_grad():
            prob = torch.softmax(logits.float(), dim=-1)
            conf, pred = prob.max(dim=-1)
            correct = (pred == y_due).to(torch.float32)
            fsrs.update_intervals(sample_ids=sid_due, correct=correct, confidence=conf)
        return loss, due_count, total_count


class ErrorCorrectionPhase:
    def forward(self, model: Any, batch: dict, class_weights: torch.Tensor | None = None) -> torch.Tensor:
        """即时纠错阶段：交叉熵 + 轻量置信惩罚。"""
        logits = _model_logits(model, batch["input"], stage="error_correction", sample_ids=batch.get("sample_id"))
        ce = _ce_loss(logits, batch["label"], class_weights=class_weights)
        conf = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
        penalty = torch.relu(conf - 0.9).mean()
        return ce + 0.1 * penalty
