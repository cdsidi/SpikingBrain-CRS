"""Layer 3 检查点工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_training_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    """保存训练检查点。

    Args:
        path: checkpoint 文件路径。
        payload: 需可被 torch.save 序列化，至少建议包含 model_state/optimizer_state/epoch。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    """加载训练检查点。

    Returns:
        dict: checkpoint 字典。
    """
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint 格式错误，需为 dict")
    return ckpt

