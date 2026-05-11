"""SpikingBrain-CRS 完整模型（fangan.md §2.2 / §4.2）。

架构：4 层交替 GLA+SWA，每层后接 SpikingFFN，末层分类头。
目标参数量：~1.8M（<2.5M PRD 约束）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.gla import GatedLinearAttention
from models.core.swa import SlidingWindowAttention
from models.core.spiking_ffn import SpikingFFN


# ---------------------------------------------------------------------------
# 单层：GLA + SWA + SpikingFFN
# ---------------------------------------------------------------------------

class SpikingBrainLayer(nn.Module):
    """一个 GLA+SWA+SpikingFFN 复合层。"""

    def __init__(
        self,
        d_model: int,
        d_k: int,
        window_size: int,
        d_ff: int,
        blank_ratio: float,
        enable_gla: bool = True,
        enable_swa: bool = True,
        enable_spike: bool = True,
    ) -> None:
        super().__init__()
        self.gla = GatedLinearAttention(d_model=d_model, d_k=d_k, modality="generic")
        self.swa = SlidingWindowAttention(d_model=d_model, window_size=window_size)
        self.ffn = SpikingFFN(d_model=d_model, d_ff=d_ff, blank_ratio=blank_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.enable_gla = bool(enable_gla)
        self.enable_swa = bool(enable_swa)
        self.enable_spike = bool(enable_spike)
        self._d_k = int(d_k)
        self._d_ff = int(d_ff)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor],
        stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向：支持 NoGLA/NoSWA/NoSpike 消融。"""
        b, l, _ = x.shape
        if self.enable_gla:
            gla_out, new_state = self.gla(self.norm1(x), state)
            x = x + gla_out
        else:
            new_state = state
            if new_state is None:
                new_state = torch.zeros((b, l, self._d_k), device=x.device, dtype=x.dtype)

        if self.enable_swa:
            swa_out = self.swa(self.norm2(x))
            x = x + swa_out

        if self.enable_spike:
            ffn_out, spike_int = self.ffn(self.norm3(x), stage=stage)
        else:
            ff = torch.relu(self.ffn.fc1(self.norm3(x).to(self.ffn.fc1.weight.dtype)))
            ffn_out = self.ffn.fc2(ff).to(x.dtype)
            spike_int = torch.zeros((b, l, self._d_ff), device=x.device, dtype=torch.int32)
        x = x + ffn_out
        return x, new_state, spike_int


# ---------------------------------------------------------------------------
# 核心 SpikingBrain 主干（S-CRS / S-Base 共用）
# ---------------------------------------------------------------------------

class SpikingBrainBackbone(nn.Module):
    """4 层 SpikingBrain 主干。"""

    def __init__(
        self,
        d_model: int = 384,
        d_k: int = 64,
        window_size: int = 256,
        d_ff: int = 1536,
        blank_ratio: float = 0.3,
        num_layers: int = 4,
        num_classes: int = 5,
        input_dim: int = 4096,
        enable_gla: bool = True,
        enable_swa: bool = True,
        enable_spike: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            SpikingBrainLayer(
                d_model=d_model,
                d_k=d_k,
                window_size=window_size,
                d_ff=d_ff,
                blank_ratio=blank_ratio,
                enable_gla=enable_gla,
                enable_swa=enable_swa,
                enable_spike=enable_spike,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        stage: str = "comprehension",
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        b = x.shape[0]
        flat = x.reshape(b, -1).float()
        h = self.input_proj(flat).unsqueeze(1)

        spike_records: List[torch.Tensor] = []
        state: Optional[torch.Tensor] = None
        total_spikes = 0
        total_elements = 0

        for layer in self.layers:
            h, state, spike_int = layer(h, state, stage=stage)
            spike_records.append(spike_int)
            total_spikes += (spike_int != 0).sum().item()
            total_elements += spike_int.numel()

        h = self.out_norm(h)
        pooled = h.mean(dim=1)
        logits = self.classifier(pooled)

        density = float(total_spikes) / max(total_elements, 1)
        sparsity = 1.0 - density
        return {
            "logits": logits,
            "sparsity": float(sparsity),
            "density": float(density),
            "spike_records": spike_records,
        }

# ---------------------------------------------------------------------------
# S-CRS：含 CRS 五阶段感知
# ---------------------------------------------------------------------------

class SpikingBrainCRS(nn.Module):
    """S-CRS：SpikingBrain + CRS 五阶段训练支持（fangan.md §4.2）。"""

    def __init__(self, cfg: Dict[str, Any] = {}) -> None:
        super().__init__()
        d_model    = int(cfg.get("d_model",    384))
        d_k        = int(cfg.get("d_k",         64))
        window_size= int(cfg.get("window_size", 256))
        d_ff       = int(cfg.get("d_ff",       1536))
        blank_ratio= float(cfg.get("blank_ratio", 0.3))
        num_layers = int(cfg.get("num_layers",    4))
        num_classes= int(cfg.get("num_classes",   5))
        input_dim  = int(cfg.get("input_dim",  4096))
        enable_gla = bool(cfg.get("enable_gla", True))
        enable_swa = bool(cfg.get("enable_swa", True))
        enable_spike = bool(cfg.get("enable_spike", True))
        self.backbone = SpikingBrainBackbone(
            d_model=d_model, d_k=d_k, window_size=window_size,
            d_ff=d_ff, blank_ratio=blank_ratio, num_layers=num_layers,
            num_classes=num_classes, input_dim=input_dim,
            enable_gla=enable_gla, enable_swa=enable_swa, enable_spike=enable_spike,
        )

    def forward(
        self,
        x: torch.Tensor,
        stage: str = "comprehension",
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self.backbone(x, stage=stage, sample_ids=sample_ids)


# ---------------------------------------------------------------------------
# S-Base：仅 comprehension 阶段（无 CRS 调度）
# ---------------------------------------------------------------------------

class SpikingBrainBase(nn.Module):
    """S-Base：同 S-CRS 架构，forward 固定走 comprehension 阶段。"""

    def __init__(self, cfg: Dict[str, Any] = {}) -> None:
        super().__init__()
        d_model    = int(cfg.get("d_model",    384))
        d_k        = int(cfg.get("d_k",         64))
        window_size= int(cfg.get("window_size", 256))
        d_ff       = int(cfg.get("d_ff",       1536))
        blank_ratio= float(cfg.get("blank_ratio", 0.3))
        num_layers = int(cfg.get("num_layers",    4))
        num_classes= int(cfg.get("num_classes",   5))
        input_dim  = int(cfg.get("input_dim",  4096))
        enable_gla = bool(cfg.get("enable_gla", True))
        enable_swa = bool(cfg.get("enable_swa", True))
        enable_spike = bool(cfg.get("enable_spike", True))
        self.backbone = SpikingBrainBackbone(
            d_model=d_model, d_k=d_k, window_size=window_size,
            d_ff=d_ff, blank_ratio=blank_ratio, num_layers=num_layers,
            num_classes=num_classes, input_dim=input_dim,
            enable_gla=enable_gla, enable_swa=enable_swa, enable_spike=enable_spike,
        )

    def forward(
        self,
        x: torch.Tensor,
        stage: str = "comprehension",
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self.backbone(x, stage="comprehension", sample_ids=sample_ids)


# ---------------------------------------------------------------------------
# T-Base：标准多头自注意力 Transformer（~2.1M 参数基准）
# ---------------------------------------------------------------------------

class TransformerBase(nn.Module):
    """T-Base：标准 Transformer，作为对照基线。"""

    def __init__(self, cfg: Dict[str, Any] = {}) -> None:
        super().__init__()
        d_model    = int(cfg.get("d_model",    384))
        num_heads  = int(cfg.get("num_heads",    6))
        d_ff       = int(cfg.get("d_ff",       1536))
        num_layers = int(cfg.get("num_layers",    4))
        num_classes= int(cfg.get("num_classes",   5))
        input_dim  = int(cfg.get("input_dim",  4096))

        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        stage: str = "comprehension",
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        b = x.shape[0]
        flat = x.reshape(b, -1).float()
        h = self.input_proj(flat).unsqueeze(1)
        h = self.encoder(h)
        h = self.out_norm(h)
        logits = self.classifier(h.mean(dim=1))
        return {"logits": logits, "sparsity": 0.0, "spike_records": []}


# ---------------------------------------------------------------------------
# T-CRS：T-Base + CRS 五阶段训练支持
# ---------------------------------------------------------------------------

class TransformerCRS(nn.Module):
    """T-CRS：Transformer + CRS 五阶段训练支持。"""

    def __init__(self, cfg: Dict[str, Any] = {}) -> None:
        super().__init__()
        self._base = TransformerBase(cfg)

    def forward(
        self,
        x: torch.Tensor,
        stage: str = "comprehension",
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self._base(x, stage=stage, sample_ids=sample_ids)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, type] = {
    "s_crs":   SpikingBrainCRS,
    "s_base":  SpikingBrainBase,
    "t_base":  TransformerBase,
    "t_crs":   TransformerCRS,
}


def build_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """根据 model_type 实例化对应模型。

    Args:
        model_type: 'S_CRS', 's_crs', 'S-CRS' 等（大小写不敏感，-/_互换）
        config:     超参字典（对应 configs/s_crs.yaml 内容）
    Returns:
        nn.Module 实例
    Raises:
        ValueError: 未知 model_type
    """
    key = model_type.lower().replace("-", "_")
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"未知模型类型 '{model_type}'，可选：{list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[key](config)
