from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> int:
    print("\n[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        print(f"[FAIL] return_code={p.returncode}")
        return p.returncode
    print("[OK]")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="One-click acceptance pipeline")
    p.add_argument("--dataset", default="lc25000", choices=["lc25000", "physionet", "iu_xray"])
    p.add_argument("--model", default="s_crs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", default="results/acceptance")
    p.add_argument("--skip-full", action="store_true", help="skip 3x5 full experiments")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]

    steps = [
        [sys.executable, "-c", "import models,training,evaluation; print('IMPORT_OK')"],
        [sys.executable, "-m", "tests.test_training_engine"],
        [sys.executable, "-m", "tests.test_hardware_adaptation"],
        [sys.executable, "scripts/e2e_smoke_validation.py"],
        [
            sys.executable,
            "-m",
            "training.train_crs",
            "--dataset",
            args.dataset,
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--folds",
            "1",
            "--output-dir",
            f"{args.output_dir}/single_fold",
        ],
        [
            sys.executable,
            "-m",
            "experiments.ablation_study",
            "--base-config",
            "configs/s_crs.yaml",
            "--seed",
            str(args.seed),
            "--output-dir",
            f"{args.output_dir}/ablation",
        ],
    ]

    if not args.skip_full:
        steps.append(
            [
                sys.executable,
                "-m",
                "training.train_crs",
                "--all-datasets",
                "--model",
                args.model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--folds",
                "5",
                "--output-dir",
                f"{args.output_dir}/full_3x5",
            ]
        )

    for cmd in steps:
        rc = run(cmd, root)
        if rc != 0:
            sys.exit(rc)

    print("\nACCEPTANCE_PIPELINE_OK")


if __name__ == "__main__":
    main()

