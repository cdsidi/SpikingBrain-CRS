from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

SEEDS = [1, 42, 123, 456, 2024]
CLASSES = ["lung_aca", "lung_scc", "lung_n", "colon_aca", "colon_n"]


def _kmeans(x: np.ndarray, k: int, seed: int = 42, iters: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=min(k, len(x)), replace=False)]
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        new_centers = np.array([x[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(len(centers))])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def _feat(p: Path) -> np.ndarray:
    """提取图像特征。优先灰度32x32像素特征；PIL不可用时退化为字节特征。"""
    if Image is not None:
        img = Image.open(p).convert("L").resize((32, 32))
        arr = np.asarray(img, dtype=np.float32).reshape(-1)
        return arr / 255.0
    data = p.read_bytes()
    arr = np.frombuffer(data[:1024], dtype=np.uint8).astype(np.float32)
    if arr.size < 1024:
        arr = np.pad(arr, (0, 1024 - arr.size), mode="constant", constant_values=0)
    return arr[:1024] / 255.0


def _patient_id(p: Path) -> str:
    """提取伪患者ID。

    兼容03文档“文件名前缀患者信息”原则：
    - 优先取下划线前前缀（常见如 lungaca12_001）
    - 若不可用，回退前三字符
    """
    head = p.stem.split("_")[0].strip()
    return head if head else (p.stem[:3] if len(p.stem) >= 3 else p.stem)


def _patient_split(patient_ids: List[str], seed: int = 42) -> Dict[str, set]:
    ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(n * 0.6), int(n * 0.8)
    return {"train": set(ids[:a]), "val": set(ids[a:b]), "test": set(ids[b:])}


def refine_lc25000(raw_dir: Path, out_dir: Path, target_per_class: int = 200, seed: int = 42) -> Dict[str, int]:
    if seed not in SEEDS:
        raise ValueError(f"seed必须在{SEEDS}中")
    out_dir.mkdir(parents=True, exist_ok=True)
    base1, base2 = raw_dir / "lung_image_sets", raw_dir / "colon_image_sets"
    class_dirs = {
        "lung_aca": base1 / "lung_aca",
        "lung_scc": base1 / "lung_scc",
        "lung_n": base1 / "lung_n",
        "colon_aca": base2 / "colon_aca",
        "colon_n": base2 / "colon_n",
    }
    selected = []
    label_map = {c: i for i, c in enumerate(CLASSES)}
    rng = np.random.default_rng(seed)
    for cname in CLASSES:
        files = sorted([*class_dirs[cname].glob("*.jpeg"), *class_dirs[cname].glob("*.jpg"), *class_dirs[cname].glob("*.png")])
        if not files:
            continue
        feats = np.stack([_feat(p) for p in files])
        labels = _kmeans(feats, k=20, seed=seed)
        keep = []
        for k in sorted(set(labels.tolist())):
            idx = np.where(labels == k)[0]
            center = feats[idx].mean(0)
            dist = ((feats[idx] - center) ** 2).sum(1)
            keep.extend([idx[dist.argmin()], idx[dist.argmax()]])
        keep = sorted(set(keep))
        if len(keep) < target_per_class:
            rest = [i for i in range(len(files)) if i not in keep]
            extra = rng.choice(rest, size=min(target_per_class - len(keep), len(rest)), replace=False).tolist()
            keep.extend(extra)
        for i in keep[:target_per_class]:
            p = files[i]
            selected.append({"path": str(p), "label": label_map[cname], "patient_id": _patient_id(p), "modality": "pathology"})
    splits = _patient_split([x["patient_id"] for x in selected], seed=seed)
    sid = 0
    counts = {}
    for split, pset in splits.items():
        rows = []
        for r in selected:
            if r["patient_id"] in pset:
                rows.append({**r, "sample_id": sid})
                sid += 1
        (out_dir / f"lc25000_{split}_indices.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        counts[split] = len(rows)
    return counts


if __name__ == "__main__":
    root = Path(r"E:\开发区\crs\shuju\lung_colon_image_set")
    out = Path(r"E:\开发区\crs\data\processed")
    print(refine_lc25000(root, out, target_per_class=200, seed=42))

