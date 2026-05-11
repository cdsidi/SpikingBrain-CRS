from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.pipeline import IUXRayRefiner, LC25000Refiner, MedicalDataLoader, PhysioNetRefiner

ROOT = Path(r"E:\开发区\crs")
PROCESSED = ROOT / "data" / "processed"


def _md5_first10(json_path: Path, key: str = "path") -> str:
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    vals = [str(r.get(key, "")) for r in rows[:10]]
    return hashlib.md5("\n".join(vals).encode("utf-8")).hexdigest()


def run() -> None:
    lc = LC25000Refiner().refine(ROOT / "shuju" / "lung_colon_image_set", PROCESSED, target_per_class=200, seed=42)
    phy = PhysioNetRefiner().refine(ROOT / "shuju" / "apnea-ecg-database-1.0.0", PROCESSED, target_patients=20, seed=42)
    iu = IUXRayRefiner().refine(ROOT / "shuju" / "images", PROCESSED, first_n=1000, target=500, seed=42)
    print("[REFINE_COUNTS]", {"lc25000": lc, "physionet": phy, "iu_xray": iu})

    lc_train = PROCESSED / "lc25000_train_indices.json"
    md5_1 = _md5_first10(lc_train)
    LC25000Refiner().refine(ROOT / "shuju" / "lung_colon_image_set", PROCESSED, target_per_class=200, seed=42)
    md5_2 = _md5_first10(lc_train)
    print("[REPRODUCIBLE]", md5_1 == md5_2, md5_1, md5_2)

    tr_loader = MedicalDataLoader("lc25000", "train", batch_size=8, seed=42)
    te_loader = MedicalDataLoader("lc25000", "test", batch_size=8, seed=42)
    tr_pids, te_pids = set(), set()
    for b in tr_loader:
        tr_pids.update(b["patient_id"])
    for b in te_loader:
        te_pids.update(b["patient_id"])
    leak = len(tr_pids & te_pids)
    print("[PATIENT_LEAK]", leak)

    p_loader = MedicalDataLoader("physionet", "train", batch_size=4, seed=42, memory_map=True)
    start = time.time()
    for i, _ in enumerate(p_loader):
        if i >= 99:
            break
    elapsed = time.time() - start
    print("[LOAD_100_BATCH_TIME]", round(elapsed, 4))

    iu_rows = json.loads((PROCESSED / "iu_xray_train_indices.json").read_text(encoding="utf-8"))
    iu_rows += json.loads((PROCESSED / "iu_xray_val_indices.json").read_text(encoding="utf-8"))
    iu_rows += json.loads((PROCESSED / "iu_xray_test_indices.json").read_text(encoding="utf-8"))
    unique_first_n = len({(r["patient_id"], r["path"]) for r in iu_rows})
    print("[IU_SELECTED_TOTAL]", unique_first_n, "(must be <=500 and sourced from first 1000 pairs)")


if __name__ == "__main__":
    run()

