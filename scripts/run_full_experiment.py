from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation import ReportGenerator
from experiments import AblationStudy, FullExperimentSuite
from training.train_crs import run_train


def main() -> None:
    """Layer12 主执行入口（Python 版，一键运行）。"""
    p = argparse.ArgumentParser(description="Run full CRS pipeline in smoke mode")
    p.add_argument("--dataset", default="lc25000", choices=["lc25000", "physionet", "iu_xray"])
    p.add_argument("--model", default="s_crs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick-smoke", action="store_true", help="run smoke-sized protocol for tests")
    p.add_argument("--output-dir", default="results/execution_guide/full")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train = run_train(
        args.dataset,
        args.model,
        args.epochs,
        args.batch_size,
        args.seed,
        out_root / "single_model",
        quick_smoke=bool(args.quick_smoke),
        mode=("smoke" if args.quick_smoke else "full"),
        max_batches=(1 if args.quick_smoke else None),
    )
    protocol_dir = FullExperimentSuite(results_root=out_root / "protocol", max_retries=0).run_48h_schedule(
        quick_smoke=bool(args.quick_smoke),
        max_batches=(1 if args.quick_smoke else 1),
    )
    ablation = AblationStudy(output_dir=out_root / "ablation").run_component_ablation(
        {"batch_size": min(max(1, int(args.batch_size)), 16)}, ["gla", "swa", "adaptive_threshold"], [args.seed]
    )

    summary = {
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": min(max(1, int(args.batch_size)), 16),
            "seed": args.seed,
        },
        "train_signature": train["signature"],
        "protocol_summary": str(protocol_dir / "summary.json"),
        "ablation_baseline": ablation["baseline"]["avg_score"],
    }
    stable_obj = {
        "config": summary["config"],
        "train_signature": summary["train_signature"],
        "ablation_baseline": summary["ablation_baseline"],
    }
    sig = hashlib.md5(json.dumps(stable_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    summary["signature"] = sig

    rg = ReportGenerator(out_root)
    rg.save_json("full_summary.json", summary)
    rg.save_markdown("full_report.md", "Layer12 Full Execution Report", summary)
    rg.append_log("full.log", f"seed={args.seed} signature={sig}")
    print("FULL_EXPERIMENT_OK", sig)


if __name__ == "__main__":
    main()

