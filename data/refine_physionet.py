from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SEEDS = [1, 42, 123, 456, 2024]


def _parse_duration_hours(hea_path: Path) -> float:
    if not hea_path.exists():
        return 1.0
    first = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip().split()
    fs = float(first[2]) if len(first) > 2 else 100.0
    n_samples = float(first[3]) if len(first) > 3 else fs * 3600
    return max(n_samples / fs / 3600.0, 1e-6)


def _count_apnea_events(apn_path: Path) -> int:
    if not apn_path.exists():
        return 0
    # PhysioNet apnea annotation files in this dataset are binary-like records.
    # Byte value 0x04 consistently marks apnea events for a01-a20.
    raw = apn_path.read_bytes()
    cnt = raw.count(4)
    if cnt > 0:
        return int(cnt)

    # Fallback for text-encoded variants.
    text = raw.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    txt_cnt = 0
    for ln in lines:
        toks = ln.split()
        if len(toks) >= 3 and toks[-1].upper() in {"A", "APNEA"}:
            txt_cnt += 1
        elif " APNEA" in f" {ln.upper()}":
            txt_cnt += 1
    return int(txt_cnt)


def _severity(ahi: float) -> str:
    if ahi < 5:
        return "normal"
    if ahi < 15:
        return "mild"
    if ahi < 30:
        return "moderate"
    return "severe"


def _stratified_pid_split(rows: List[Dict], rng: np.random.Generator) -> Tuple[set, set, set]:
    pos = [r["patient_id"] for r in rows if int(r["ahi"] >= 15.0) == 1]
    neg = [r["patient_id"] for r in rows if int(r["ahi"] >= 15.0) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_one(items: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(items)
        if n <= 2:
            return items[:1], items[1:2], items[2:]
        n_train = max(1, int(round(0.6 * n)))
        n_val = max(1, int(round(0.2 * n)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    p_tr, p_va, p_te = split_one(pos)
    n_tr, n_va, n_te = split_one(neg)
    train = set(p_tr + n_tr)
    val = set(p_va + n_va)
    test = set(p_te + n_te)
    return train, val, test


def refine_physionet(raw_dir: Path, out_dir: Path, target_patients: int = 20, seed: int = 42) -> Dict[str, int]:
    """精炼 a01-a20 并生成患者级 split 索引（带标签分层）。"""
    if seed not in SEEDS:
        raise ValueError(f"seed必须在{SEEDS}中")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records: List[Dict] = []
    for i in range(1, 21):
        pid = f"a{i:02d}"
        dat = raw_dir / f"{pid}.dat"
        hea = raw_dir / f"{pid}.hea"
        apn = raw_dir / f"{pid}.apn"
        if not dat.exists() or not hea.exists():
            continue
        hours = _parse_duration_hours(hea)
        apnea_events = _count_apnea_events(apn)
        ahi = apnea_events / hours
        records.append({
            "patient_id": pid,
            "record": str(dat),
            "hea": str(hea),
            "apn": str(apn),
            "ahi": float(ahi),
            "severity": _severity(ahi),
            "duration_hours": float(hours),
            "modality": "ecg",
        })

    records = sorted(records, key=lambda x: x["patient_id"])[:target_patients]
    train, val, test = _stratified_pid_split(records, rng)

    sid = 0
    stats: Dict[str, int] = {}
    for split, pset in [("train", train), ("val", val), ("test", test)]:
        rows = []
        for r in records:
            if r["patient_id"] in pset:
                rows.append({**r, "label": int(r["ahi"] >= 15.0), "sample_id": sid})
                sid += 1
        (out_dir / f"physionet_{split}_indices.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        stats[split] = len(rows)
        stats[f"{split}_pos"] = int(sum(int(x["label"]) for x in rows))
        stats[f"{split}_neg"] = int(len(rows) - stats[f"{split}_pos"])
    return stats


if __name__ == "__main__":
    root = Path(r"E:\开发区\crs\shuju\apnea-ecg-database-1.0.0")
    out = Path(r"E:\开发区\crs\data\processed")
    print(refine_physionet(root, out, target_patients=20, seed=42))

