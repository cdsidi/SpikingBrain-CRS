# CRS 项目详细说明（SpikingBrain-CRS）

[中文](README.md) | [English](README_EN.md)

> 一个面向医疗任务的神经形态学习实验框架，融合 SpikingBrain 与 CRS（课程化复习）机制，支持从数据处理、训练、评估到实验验收的完整流程。

---

## 1. 项目简介

本项目用于验证 `fangan.md` 与 `sheji/01~12` 中定义的系统方案是否可以在工程上完整落地。核心特点：

- **模型体系完整**：支持 `t_base / t_crs / s_base / s_crs` 四类模型。
- **CRS 五阶段训练**：`comprehension / recall / synthesis / spaced_review / error_correction`。
- **FSRS 记忆调度**：内置 19 参数权重，支持间隔复习策略。
- **全链路可运行**：数据→训练→checkpoint→评估→实验汇总。
- **可复现性**：支持 folds + repeats、seed 管理、标准化结果目录。

---

## 2. 设计文档基准（验收依据）

运行与验收请以以下文档为唯一基准：

- `fangan.md`
- `sheji/01_PRD.txt`
- `sheji/02_ARCHITECTURE.txt`
- `sheji/03_DATA_PIPELINE.txt`
- `sheji/04_CORE_SPiking.txt`
- `sheji/05_CRS_INTEGRATION.txt`
- `sheji/06_TRAINING_ENGINE.txt`
- `sheji/07_EXPERIMENT_PROTOCOL.txt`
- `sheji/08_EVALUATION_SUITE.txt`
- `sheji/09_HARDWARE_ADAPTATION.txt`
- `sheji/10_ABLATION_FRAMEWORK.txt`
- `sheji/11_INTEGRATION_TEST.txt`
- `sheji/12_EXECUTION_GUIDE.txt`

---

## 3. 仓库结构

```text
configs/            默认配置、模型配置
checkpoints/        训练保存权重
data/               数据预处理、pipeline、数据集适配
models/             核心网络与 CRS 子模块
training/           Trainer、训练入口、阶段调度
evaluation/         指标、统计检验、硬件监控
experiments/        主实验、消融、长期记忆实验
scripts/            验收脚本、全量实验脚本
tests/              单元与集成测试
results/            训练/评估结果产物
docs/               额外执行文档
```

---

## 4. 环境要求与安装

### 4.1 环境建议

- Python 3.10+
- 建议使用 Conda 或 venv 隔离环境
- 建议具备 CUDA 环境（CPU 也可运行但速度较慢）

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

### 4.3 平台说明（Windows）

在 PowerShell 中请用 `;` 分隔命令，不要使用 `&&`。

---

## 5. 快速开始（从 0 到可验收）

### Step 1：跑关键测试

```bash
python -m tests.test_imports
python -m tests.test_crs_integration
python -m tests.test_execution_guide
```

### Step 2：跑最小验收链路

```bash
python scripts/run_acceptance.py --dataset lc25000 --model s_crs --epochs 1 --batch-size 2 --seed 1 --output-dir results/acceptance_audit --skip-full
```

成功标志：终端出现 `ACCEPTANCE_PIPELINE_OK`。

### Step 3：跑四模型全协议（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all_models_serial.ps1 -Epochs 20 -BatchSize 4 -Folds 5 -Repeats 5 -BaseOutput results
```

或逐模型运行（示例）：

```bash
python -m training.train_crs --all-datasets --model s_crs --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_s_crs
```

### Step 4：汇总绘图数据（先不画图）

```bash
python scripts/collect_plot_data.py --results-dir results --out-dir results/plot_data
python scripts/generate_data_summary_md.py
```

产物：`results/plot_data/DATA_SUMMARY.md`

---

## 6. 训练与实验命令手册

### 6.1 单数据集训练

```bash
python -m training.train_crs --dataset lc25000 --model s_crs --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/train_s_crs_lc25000
```

### 6.2 全数据集训练（推荐协议）

```bash
python -m training.train_crs --all-datasets --model t_base --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_t_base
python -m training.train_crs --all-datasets --model t_crs  --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_t_crs
python -m training.train_crs --all-datasets --model s_base --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_s_base
python -m training.train_crs --all-datasets --model s_crs  --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_s_crs
```

### 6.3 全流程实验脚本

```bash
python scripts/run_full_experiment.py --help
```

如需快速检查链路，可使用 quick-smoke 参数（若脚本支持）。

---

## 7. 测试矩阵说明

### 7.1 基础测试

- `tests.test_imports`：模块导入完整性
- `tests.test_core_spiking`：核心神经元/算子逻辑
- `tests.test_training_engine`：训练引擎正确性

### 7.2 CRS 与执行链路

- `tests.test_crs_integration`：CRS 五阶段协议、loss 输出契约
- `tests.test_execution_guide`：按执行指南验证脚本链路
- `tests.test_integration_test`：全链路组合验证

### 7.3 实验与评估能力

- `tests.test_experiment_protocol`
- `tests.test_evaluation_suite`
- `tests.test_ablation_framework`
- `tests.test_hardware_adaptation`

---

## 8. 输出产物与结果解读

### 8.1 当前权威数据入口（优先使用）

- `results/full_protocol_{t_base|t_crs|s_base|s_crs}/global_summary.json`：四模型三数据集全协议汇总（每数据集 25 条 fold_results）
- `results/plot_data/DATA_SUMMARY.md`：绘图数据总览（完整性检查 + Table.1 均值汇总 + 可得性清单）
- `results/long_term_memory_real/long_term_memory_curve.json`：Fig.3 真实训练评估驱动曲线
- `results/ablation_framework_final/ablation_results.json`：Fig.4 消融贡献主数据

### 8.2 绘图数据导出文件

- `results/plot_data/fig2_dataset_comparison.csv`
- `results/plot_data/fig4_ablation_waterfall.csv`
- `results/plot_data/fig5_pareto.csv`
- `results/plot_data/fig6_hardware_radar.csv`
- `results/plot_data/table1_model_comparison.csv`
- `results/plot_data/missing_items.json`

### 8.3 指标解读建议

- **效果**：Accuracy / F1 / AUC
- **推理**：P50/P95/P99 延迟、吞吐
- **资源**：显存峰值、稀疏度、（建议补）功耗
- **稳定性**：是否 NaN/Inf、是否异常中断、日志完整性

说明：`1 epoch` 或 `quick-smoke` 仅用于流程验证，不代表最终性能上限。

---

## 9. 当前完成状态与缺口

### 9.1 已完成

- 四模型 × 三数据集 × 5折 × 5重复 的结果文件完整。
- Fig.3 已升级为**真实训练评估驱动**（非模拟占位）。
- Fig.4 消融实验已可稳定产出 JSON + PNG。
- 绘图数据汇总链路已打通：`collect_plot_data.py` + `generate_data_summary_md.py`。

### 9.2 仍需补齐

- Fig.6：`power_w` 尚未统一采集。
- Fig.7：CRS 各阶段注意力热力图缺少统一导出脚本。
- Table.2：项目内缺少 SOTA 基线对照源文件（需外部对照录入）。

---

## 10. 常见问题（FAQ）

**Q1：如何确认 `--all-datasets` 真的跑了三数据集？**
- 查看 `results/full_protocol_xxx/` 下是否存在 `lc25000 / physionet / iu_xray`；
- 或检查 `global_summary.json` 中三数据集条目及 `fold_results` 数量。

**Q2：为什么有些图的数据还是“部分可得”？**
- 项目已覆盖大多数训练指标；
- 但功耗、注意力热力图、SOTA外部对照属于额外采集链路，需单独补脚本/数据源。

**Q3：如何最快复现实验数据汇总？**
- 先跑全协议；再执行：
  `python scripts/collect_plot_data.py --results-dir results --out-dir results/plot_data`
  `python scripts/generate_data_summary_md.py`

---

## 11. 一页式命令清单

```bash
# 关键测试
python -m tests.test_imports
python -m tests.test_crs_integration
python -m tests.test_execution_guide

# 四模型串行全协议
powershell -ExecutionPolicy Bypass -File scripts/run_all_models_serial.ps1 -Epochs 20 -BatchSize 4 -Folds 5 -Repeats 5 -BaseOutput results

# Fig.3 真实训练评估驱动（示例）
python -m experiments.long_term_memory_test --model s_crs --dataset-a lc25000 --dataset-b physionet --cycles 10 --epochs-per-cycle 1 --batch-size 4 --quick-smoke --output-dir results/long_term_memory_real

# Fig.4 消融实验
python -m experiments.ablation_study --base-config configs/s_crs.yaml --seed 42 --output-dir results/ablation_framework_final

# 汇总绘图数据（先不画图）
python scripts/collect_plot_data.py --results-dir results --out-dir results/plot_data
python scripts/generate_data_summary_md.py
```

---

更多文档：
- `README_EN.md`
- `docs/ACCEPTANCE_TEMPLATE.md`
- `results/plot_data/ESTIMATED_FINAL_DATA.md`