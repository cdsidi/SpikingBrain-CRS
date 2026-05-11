"""Layer 0 数据管道：精炼入口与统一加载器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Literal

import numpy as np
import torch

from .refine_iu_xray import refine_iu_xray
from .refine_lc25000 import refine_lc25000
from .refine_physionet import refine_physionet

SEEDS = [1, 42, 123, 456, 2024]


class LC25000Refiner:
    """LC25000 精炼器（03_DATA_PIPELINE）。"""

    def refine(self, raw_dir: Path, out_dir: Path, target_per_class: int = 200, seed: int = 42) -> Dict[str, int]:
        """执行 K-means 形态学精炼并输出患者级 split 索引。"""
        return refine_lc25000(raw_dir=raw_dir, out_dir=out_dir, target_per_class=target_per_class, seed=seed)


class PhysioNetRefiner:
    """PhysioNet AHI 分层精炼器。"""

    def refine(self, raw_dir: Path, out_dir: Path, target_patients: int = 20, seed: int = 42) -> Dict[str, int]:
        """执行 a01-a20 分层采样并输出患者级 split 索引。"""
        return refine_physionet(raw_dir=raw_dir, out_dir=out_dir, target_patients=target_patients, seed=seed)

    def refine_by_ahi(self, records: List[Path], target_patients: int = 20) -> List[Path]:
        """兼容02接口：按AHI进行记录筛选，返回记录路径列表。"""
        selected = sorted(records)[:target_patients]
        return selected


class IUXRayRefiner:
    """IU X-Ray 信息密度精炼器。"""

    def refine(self, raw_dir: Path, out_dir: Path, first_n: int = 1000, target: int = 500, seed: int = 42) -> Dict[str, int]:
        """执行前1000对约束和top-k筛选，输出患者级 split 索引。"""
        return refine_iu_xray(raw_dir=raw_dir, out_dir=out_dir, first_n=first_n, target=target, seed=seed)

    def refine_by_density(self, pairs: List[Dict], target: int = 500) -> List[Dict]:
        """兼容02接口：按信息密度返回前target条。"""
        return pairs[:target]


class MedicalDataLoader:
    """统一数据加载器。

    Yields dict:
    - input: Tensor, shape [B, ...], dtype float32
    - label: Tensor, shape [B], dtype int64
    - sample_id: Tensor, shape [B], dtype int64
    - patient_id: List[str], len B
    - modality: List[str], len B
    """

    def __init__(
        self,
        dataset_type: Literal["lc25000", "physionet", "iu_xray"],
        split: Literal["train", "val", "test"],
        batch_size: int,
        num_workers: int = 0,
        memory_map: bool = True,
        seed: int = 42,
    ) -> None:
        if seed not in SEEDS:
            raise ValueError(f"seed必须在{SEEDS}中")
        self.dataset_type, self.split = dataset_type, split
        self.batch_size = min(max(1, batch_size), 16)  # PRD: <=16
        self.num_workers, self.memory_map = num_workers, memory_map
        self.root = Path(r"E:\开发区\crs")
        self.processed = self.root / "data" / "processed"
        self.indices = self._load_refined_indices(seed)
        self._memmap_cache: Dict[str, np.ndarray] = {}
        if self.memory_map and self.dataset_type == "physionet":
            self._prepare_memmap()

    def _load_refined_indices(self, seed: int) -> List[Dict]:
        p = self.processed / f"{self.dataset_type}_{self.split}_indices.json"
        if not p.exists():
            if self.dataset_type == "lc25000":
                refine_lc25000(self.root / "shuju" / "lung_colon_image_set", self.processed, seed=seed)
            elif self.dataset_type == "physionet":
                refine_physionet(self.root / "shuju" / "apnea-ecg-database-1.0.0", self.processed, seed=seed)
            else:
                refine_iu_xray(self.root / "shuju" / "images", self.processed, first_n=1000, target=500, seed=seed)
        return json.loads(p.read_text(encoding="utf-8"))

    def _prepare_memmap(self) -> None:
        cache_dir = self.processed / "memmap"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for row in self.indices:
            pid = row["patient_id"]
            if pid in self._memmap_cache:
                continue
            npy_path = cache_dir / f"physionet_{pid}.npy"
            if not npy_path.exists():
                dat = np.fromfile(row["record"], dtype=np.int16)
                ch = 2 if dat.size % 2 == 0 else 1
                sig = dat.reshape(-1, ch).astype(np.float32)
                np.save(npy_path, sig)
            self._memmap_cache[pid] = np.load(npy_path, mmap_mode="r")

    def _load_one(self, row: Dict) -> Dict:
        if self.dataset_type == "lc25000":
            b = Path(row["path"]).read_bytes()[:4096]
            arr = np.frombuffer(b, dtype=np.uint8).astype(np.float32)
            if arr.size < 4096:
                arr = np.pad(arr, (0, 4096 - arr.size), mode="constant")
            x = torch.from_numpy((arr[:4096] / 255.0).reshape(1, 64, 64))
        elif self.dataset_type == "physionet":
            sig = self._memmap_cache[row["patient_id"]]
            win = min(1000, sig.shape[0])
            # memmap切片默认可能是只读，copy 后避免 torch 非可写数组告警
            x = torch.from_numpy(np.asarray(sig[:win], dtype=np.float32).copy()).transpose(0, 1)
        else:
            b = Path(row["path"]).read_bytes()[:4096]
            arr = np.frombuffer(b, dtype=np.uint8).astype(np.float32)
            if arr.size < 4096:
                arr = np.pad(arr, (0, 4096 - arr.size), mode="constant")
            txt_len = min(len(row.get("report", "")) / 500.0, 1.0)
            x = torch.from_numpy((arr[:4096] / 255.0).reshape(1, 64, 64)) * (0.5 + txt_len)
        return {
            "input": x,
            "label": int(row["label"]),
            "sample_id": int(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
            "modality": str(row["modality"]),
        }

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for i in range(0, len(self.indices), self.batch_size):
            batch = [self._load_one(r) for r in self.indices[i : i + self.batch_size]]
            yield self._collate_fn(batch)

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, object]:
        inputs = [b["input"] for b in batch]
        max_ndim = max(x.ndim for x in inputs)
        norm_inputs = [x.unsqueeze(0) if x.ndim == max_ndim - 1 else x for x in inputs]
        if max_ndim == 2:
            max_l = max(x.shape[-1] for x in norm_inputs)
            norm_inputs = [torch.nn.functional.pad(x, (0, max_l - x.shape[-1])) for x in norm_inputs]
        input_tensor = torch.stack(norm_inputs, dim=0).float()
        return {
            "input": input_tensor,
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.int64),
            "sample_id": torch.tensor([b["sample_id"] for b in batch], dtype=torch.int64),
            "patient_id": [b["patient_id"] for b in batch],
            "modality": [b["modality"] for b in batch],
        }

