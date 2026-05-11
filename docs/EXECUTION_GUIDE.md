# Layer 12 执行指南（12_EXECUTION_GUIDE）

## 1. 环境安装命令
> 约束：batch_size <= 16，显存目标 < 4.2GB（PRD/ARCH 基线）

```bash
cd E:/开发区/crs
python -m venv .venv
# Windows
.venv/Scripts/activate
pip install -U pip
pip install -r requirements.txt
# 如需指定 CUDA 版本可覆盖安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 2. 数据准备脚本
当前仓库为离线数据模式，使用目录检查脚本代替在线下载：

```bash
python scripts/download_data.py
```

产物：
- `results/execution_guide/data_prep/data_manifest.json`

## 3. 主执行入口（完整实验）
与文档示例保持一致：

```bash
bash scripts/run_full_experiment.sh --dataset lc25000 --model s_crs --epochs 1 --batch-size 16 --seed 42
# 等价（Windows推荐）
python scripts/run_full_experiment.py --dataset lc25000 --model s_crs --epochs 1 --batch-size 16 --seed 42
```

产物：
- `results/execution_guide/full/full_summary.json`
- `results/execution_guide/full/full_report.md`
- `results/execution_guide/full/full.log`

## 4. 分步骤入口
### 4.1 单模型测试
```bash
python -m training.train_crs --dataset lc25000 --model s_crs --epochs 1 --batch-size 16 --seed 42 --output-dir results/execution_guide/single_model
```

### 4.2 消融实验
```bash
python -m experiments.ablation_study --base-config configs/s_crs.yaml --seed 42 --output-dir results/execution_guide/ablation
```

## 5. 参数说明
- `--dataset`: `lc25000 | physionet | iu_xray`
- `--model`: 模型标识（默认 `s_crs`）
- `--epochs`: 训练轮次（执行指南建议 smoke=1）
- `--batch-size`: 自动裁剪到 `<=16`
- `--seed`: 固定种子，推荐 `[1,42,123,456,2024]`
- `--output-dir`: 结果目录

## 6. 结果解读指南
- `signature`：同 seed/同配置下应一致（可复现证据）。
- `peak_gpu_mem_mb`：显存峰值，目标 `<4200`。
- `accuracy / avg_loss`：用于快速健康检查，不作为最终SOTA结论。
- `ablation_baseline`：消融基线分数，观察变体差值排序。

## 7. 常见问题与故障排查
1. `ModuleNotFoundError`：请在项目根目录执行命令。
2. 无 GPU 或 `nvidia-smi` 不可用：报告中显存可能为 `0.0`，属降级路径。
3. `matplotlib` 缺失：消融图会生成占位 PNG，不影响流程验收。
4. Windows 无 `bash`：直接使用 `python scripts/run_full_experiment.py`。
5. 输出不一致：核对 seed、参数、输出目录是否一致。

## 8. 一键验收脚本（import/test/smoke/full/ablation/eval/report）

```bash
# 快速模式（跳过 3数据集×5折 full）
python scripts/run_acceptance.py --skip-full --dataset lc25000 --epochs 20 --batch-size 4 --seed 1

# 完整模式（包含 3数据集×5折 full）
python scripts/run_acceptance.py --dataset lc25000 --epochs 20 --batch-size 4 --seed 1
```

默认执行顺序：
1. import 健康检查
2. 单元测试（training_engine / hardware_adaptation）
3. smoke 验证（scripts/e2e_smoke_validation.py）
4. 单折 20epoch 抗坍缩回归（含 class-weight + warmup + balanced sampler）
5. 消融实验
6. full（3数据集×5折）
7. 汇总报告产物检查

主要产物目录：
- `results/acceptance/single_fold/`：单折抗坍缩回归结果（含 `eval_diagnostics.json`、`epoch_prediction_trend.json`）
- `results/acceptance/ablation/`：消融报告
- `results/acceptance/full_3x5/`：3数据集×5折全量结果与 `global_summary.json`
