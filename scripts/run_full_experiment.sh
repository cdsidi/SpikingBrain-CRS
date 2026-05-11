#!/usr/bin/env bash
set -euo pipefail

# Layer12 一键入口（与 sheji/12 命令示例对齐）
# Windows/PowerShell 用户可直接执行: python scripts/run_full_experiment.py ...

python scripts/run_full_experiment.py "$@"

