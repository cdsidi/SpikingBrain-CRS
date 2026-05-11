from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """Layer12 数据准备脚本（离线友好）。

    说明：当前工程使用本地 `shuju/` 数据目录。
    本脚本执行目录检查与清单落盘，作为“数据下载步骤”的可运行替代入口。
    """
    root = Path(__file__).resolve().parents[1]
    data_root = root / "shuju"
    required = [
        data_root / "lung_colon_image_set",
        data_root / "apnea-ecg-database-1.0.0",
        data_root / "images",
    ]
    status = {str(p.relative_to(root)): p.exists() for p in required}

    out_dir = root / "results" / "execution_guide" / "data_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "scripts/download_data.py",
        "mode": "offline_check",
        "all_ready": all(status.values()),
        "items": status,
    }
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DATA_PREP_OK", "ready=" + str(manifest["all_ready"]))


if __name__ == "__main__":
    main()

