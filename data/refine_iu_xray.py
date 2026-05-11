from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SEEDS = [1, 42, 123, 456, 2024]
KEYWORDS = ["opacity", "consolidation", "effusion", "nodule", "mass", "infiltrate", "pneumonia", "cardiomegaly"]


def _read_reports(reports_csv: Path) -> Dict[str, str]:
    reports: Dict[str, str] = {}
    with open(reports_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get("uid", "")).strip()
            findings = (row.get("findings") or "").replace("\n", " ").strip()
            impression = (row.get("impression") or "").replace("\n", " ").strip()
            txt = f"{findings} {impression}".strip()
            if uid and txt:
                reports[uid] = txt
    return reports


def _score_report(img_path: Path, report: str) -> float:
    words = [w for w in report.lower().split() if w]
    if not words:
        return 0.0
    entity_score = sum(1 for w in words if w in KEYWORDS) / len(words)
    keyword_score = sum(1 for kw in KEYWORDS if kw in report.lower()) / len(KEYWORDS)
    completeness = min(report.count(".") + 1, 5) / 5.0
    size_kb = img_path.stat().st_size / 1024.0 if img_path.exists() else 0.0
    align = 1.0 if (100.0 < size_kb < 5000.0 and 80 < len(report) < 1200) else 0.5
    return 0.3 * entity_score + 0.3 * keyword_score + 0.2 * completeness + 0.2 * align


def _label_from_report(report: str) -> int:
    txt = report.lower()
    return int(any(k in txt for k in ["pneumonia", "effusion", "cardiomegaly", "nodule", "mass"]))


def _stratified_pid_split(pid_to_label: Dict[str, int], rng: np.random.Generator) -> Tuple[set, set, set]:
    pos = [pid for pid, lb in pid_to_label.items() if int(lb) == 1]
    neg = [pid for pid, lb in pid_to_label.items() if int(lb) == 0]
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
    return set(p_tr + n_tr), set(p_va + n_va), set(p_te + n_te)


def refine_iu_xray(raw_dir: Path, out_dir: Path, first_n: int = 1000, target: int = 500, seed: int = 42) -> Dict[str, int]:
    """只取前1000图文对，再按信息密度选前500（带标签分层切分）。"""
    if seed not in SEEDS:
        raise ValueError(f"seed必须在{SEEDS}中")
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = _read_reports(raw_dir / "indiana_reports.csv")
    pairs: List[Dict] = []
    with open(raw_dir / "indiana_projections.csv", "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row.get("uid", "")).strip()
            fn = str(row.get("filename", "")).strip()
            if not uid or not fn or uid not in reports:
                continue
            img_path = raw_dir / "images_normalized" / fn
            if not img_path.exists():
                continue
            pairs.append({"uid": uid, "path": str(img_path), "report": reports[uid], "patient_id": uid, "modality": "xray_text"})
            if len(pairs) >= first_n:
                break

    scored_pos = []
    scored_neg = []
    for p in pairs:
        s = _score_report(Path(p["path"]), p["report"])
        if _label_from_report(p["report"]) == 1:
            scored_pos.append((s, p))
        else:
            scored_neg.append((s, p))
    scored_pos.sort(key=lambda x: x[0], reverse=True)
    scored_neg.sort(key=lambda x: x[0], reverse=True)

    target_total = min(target, len(pairs))
    target_neg = min(len(scored_neg), max(50, target_total // 5))
    target_pos = min(len(scored_pos), target_total - target_neg)
    if target_pos + target_neg < target_total:
        # backfill whichever side still has headroom
        rem = target_total - (target_pos + target_neg)
        extra_pos = min(rem, len(scored_pos) - target_pos)
        target_pos += extra_pos
        rem -= extra_pos
        if rem > 0:
            extra_neg = min(rem, len(scored_neg) - target_neg)
            target_neg += extra_neg

    top = [x[1] for x in scored_pos[:target_pos]] + [x[1] for x in scored_neg[:target_neg]]
    rng.shuffle(top)

    pid_to_label: Dict[str, int] = {}
    for x in top:
        pid = x["patient_id"]
        lb = _label_from_report(x["report"])
        if pid not in pid_to_label:
            pid_to_label[pid] = lb
    train, val, test = _stratified_pid_split(pid_to_label, rng)

    sid = 0
    stats: Dict[str, int] = {"first_n_pairs": len(pairs), "selected_pairs": len(top)}
    for split, pset in [("train", train), ("val", val), ("test", test)]:
        rows = []
        for p in top:
            if p["patient_id"] in pset:
                label = _label_from_report(p["report"])
                rows.append({**p, "label": label, "sample_id": sid})
                sid += 1
        (out_dir / f"iu_xray_{split}_indices.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        stats[split] = len(rows)
        stats[f"{split}_pos"] = int(sum(int(x["label"]) for x in rows))
        stats[f"{split}_neg"] = int(len(rows) - stats[f"{split}_pos"])
    return stats


if __name__ == "__main__":
    root = Path(r"E:\开发区\crs\shuju\images")
    out = Path(r"E:\开发区\crs\data\processed")
    print(refine_iu_xray(root, out, first_n=1000, target=500, seed=42))

