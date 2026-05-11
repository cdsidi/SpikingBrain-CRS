"""基线实验编排（真实训练驱动）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from training.train_crs import run_train


class BaselineExperiment:
    """执行基线模型实验。

    固定模型组: T-Base / S-Base。
    """

    MODEL_MAP = {
        "T-Base": "t_base",
        "S-Base": "s_base",
    }

    def run(
        self,
        model_type: str,
        dataset: str,
        n_folds: int = 5,
        seeds: List[int] | None = None,
        output_root: Path | None = None,
        quick_smoke: bool = True,
        max_batches: int = 1,
    ) -> Dict:
        """执行多折真实训练评估并返回汇总。"""
        seeds = seeds or [1, 42, 123, 456, 2024]
        use_folds = min(n_folds, len(seeds))
        model_key = self.MODEL_MAP.get(model_type, model_type.lower().replace("-", "_"))
        output_root = output_root or (Path("results") / "experiment_protocol_real" / "baseline")

        records = []
        for i in range(use_folds):
            seed = seeds[i]
            fold = i + 1
            fold_dir = output_root / dataset / model_type.replace("-", "_") / f"fold{fold}_seed{seed}"
            try:
                out = run_train(
                    dataset=dataset,
                    model_type=model_key,
                    epochs=1,
                    batch_size=4,
                    seed=seed,
                    output_dir=fold_dir,
                    quick_smoke=quick_smoke,
                    fold_info=(fold, use_folds),
                    device_override="cpu",
                    max_batches=max_batches,
                )
                metrics = out.get("eval", {})
                records.append(
                    {
                        "model": model_type,
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "accuracy": float(metrics.get("accuracy", 0.0)),
                        "f1": float(metrics.get("f1", 0.0)),
                        "auc_roc": float(metrics.get("auc_roc", 0.0)),
                        "checkpoint": out.get("checkpoint", ""),
                        "summary_path": str(fold_dir / "train_summary.json"),
                    }
                )
            except Exception as e:
                records.append(
                    {
                        "model": model_type,
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "error": str(e),
                    }
                )

        valid = [r for r in records if "accuracy" in r]
        avg_score = sum(r["accuracy"] for r in valid) / max(1, len(valid))
        return {
            "kind": "baseline",
            "model": model_type,
            "dataset": dataset,
            "n_folds": use_folds,
            "records": records,
            "avg_score": float(avg_score),
            "failures": [r for r in records if "error" in r],
        }
