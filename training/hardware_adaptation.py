"""Layer 6: RTX 4050 硬件适配工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RTX4050Config:
    """RTX 4050 6GB 推荐配置。

    约束: 显存目标 < 4.2GB, batch_size <= 16。
    """

    batch_size: int = 16
    enable_amp: bool = True
    enable_grad_checkpoint: bool = True
    enable_cpu_offload: bool = False
    optimizer: str = "adam8bit"
    grad_accum_steps: int = 4

    def to_training_config(self, base: Dict | None = None) -> Dict:
        """合并为训练配置字典。"""
        cfg = dict(base or {})
        cfg.update(
            {
                "hardware_profile": "rtx4050",
                "batch_size": min(16, int(self.batch_size)),
                "enable_amp": bool(self.enable_amp),
                "enable_grad_checkpoint": bool(self.enable_grad_checkpoint),
                "enable_cpu_offload": bool(self.enable_cpu_offload),
                "optimizer": str(self.optimizer),
                "grad_accum_steps": int(self.grad_accum_steps),
            }
        )
        return cfg


class MemoryOptimizer:
    """内存优化策略包装器（最小实现）。"""

    @staticmethod
    def optimize_config(config: Dict) -> Dict:
        """对训练配置做安全收敛（不越界改核心逻辑）。"""
        cfg = dict(config)
        cfg["batch_size"] = min(16, int(cfg.get("batch_size", 16)))
        cfg.setdefault("hardware_profile", "auto")
        cfg.setdefault("enable_amp", True)
        cfg.setdefault("enable_grad_checkpoint", False)
        cfg.setdefault("enable_cpu_offload", False)
        cfg.setdefault("optimizer", "adamw")
        cfg.setdefault("grad_accum_steps", 4)
        return cfg

