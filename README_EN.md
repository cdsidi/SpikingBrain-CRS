# CRS Project Detailed Guide (SpikingBrain-CRS)

[中文](README.md) | [English](README_EN.md)

> A neuromorphic learning framework for medical tasks, integrating SpikingBrain and CRS (Curriculum Review Scheduling), covering the full path from data processing and training to evaluation and acceptance.

---

## 1. Project Overview

This project validates whether the system defined in `fangan.md` and `sheji/01~12` is fully implementable in engineering practice. Core features:

- **Complete model family**: `t_base / t_crs / s_base / s_crs`
- **CRS 5-phase training**: `comprehension / recall / synthesis / spaced_review / error_correction`
- **FSRS memory scheduling**: built-in 19-parameter weights
- **End-to-end runnable pipeline**: data → training → checkpoints → evaluation → experiment summary
- **Reproducibility**: folds + repeats, seed management, standardized result directories

---

## 2. Design Baseline (Acceptance References)

Use the following files as the only acceptance baseline:

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

## 3. Repository Layout

```text
configs/            default configs and model configs
checkpoints/        saved model weights
data/               data preprocessing, pipeline, dataset adapters
models/             core networks and CRS submodules
training/           trainer, training entrypoints, phase scheduler
evaluation/         metrics, statistical tests, hardware monitoring
experiments/        main experiments, ablation, long-term memory tests
scripts/            acceptance scripts and full experiment scripts
tests/              unit and integration tests
results/            training/evaluation artifacts
docs/               extra execution documents
```

---

## 4. Environment Requirements & Setup

### 4.1 Recommended environment

- Python 3.10+
- Use Conda or venv for isolation
- CUDA environment is recommended (CPU is supported but slower)

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Platform note (Windows)

In PowerShell, use `;` to separate commands (avoid `&&`).

---

## 5. Quick Start (from zero to acceptance-ready)

### Step 1: Run key tests

```bash
python -m tests.test_imports
python -m tests.test_crs_integration
python -m tests.test_execution_guide
```

### Step 2: Run minimal acceptance pipeline

```bash
python scripts/run_acceptance.py --dataset lc25000 --model s_crs --epochs 1 --batch-size 2 --seed 1 --output-dir results/acceptance_audit --skip-full
```

Success marker: terminal prints `ACCEPTANCE_PIPELINE_OK`.

### Step 3: Run 4-model full protocol (recommended)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all_models_serial.ps1 -Epochs 20 -BatchSize 4 -Folds 5 -Repeats 5 -BaseOutput results
```

Or run per model (example):

```bash
python -m training.train_crs --all-datasets --model s_crs --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_s_crs
```

### Step 4: Aggregate plot data (without plotting)

```bash
python scripts/collect_plot_data.py --results-dir results --out-dir results/plot_data
python scripts/generate_data_summary_md.py
```

Artifact: `results/plot_data/DATA_SUMMARY.md`

---

## 6. Training & Experiment Command Manual

### 6.1 Single-dataset training

```bash
python -m training.train_crs --dataset lc25000 --model s_crs --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/train_s_crs_lc25000
```

### 6.2 All-dataset training (recommended protocol)

```bash
python -m training.train_crs --all-datasets --model t_base --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_t_base
python -m training.train_crs --all-datasets --model t_crs  --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_t_crs
python -m training.train_crs --all-datasets --model s_base --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_s_base
python -m training.train_crs --all-datasets --model s_crs  --epochs 20 --batch-size 4 --folds 5 --repeats 5 --output-dir results/full_protocol_s_crs
```

### 6.3 Full experiment script

```bash
python scripts/run_full_experiment.py --help
```

For fast pipeline checks, use `quick-smoke` when supported.

---

## 7. Test Matrix

### 7.1 Core tests

- `tests.test_imports`: module import integrity
- `tests.test_core_spiking`: neuron/operator correctness
- `tests.test_training_engine`: training engine correctness

### 7.2 CRS and execution chain

- `tests.test_crs_integration`: CRS 5-phase protocol and loss output contract
- `tests.test_execution_guide`: execution-guide chain validation
- `tests.test_integration_test`: end-to-end integration verification

### 7.3 Experiment and evaluation capability

- `tests.test_experiment_protocol`
- `tests.test_evaluation_suite`
- `tests.test_ablation_framework`
- `tests.test_hardware_adaptation`

---

## 8. Output Artifacts and Result Interpretation

### 8.1 Canonical data entry points (preferred)

- `results/full_protocol_{t_base|t_crs|s_base|s_crs}/global_summary.json`: 4-model, 3-dataset protocol summaries (25 fold_results per dataset)
- `results/plot_data/DATA_SUMMARY.md`: plot-data overview (completeness check + Table.1 means + availability list)
- `results/long_term_memory_real/long_term_memory_curve.json`: Fig.3 real train-eval curve
- `results/ablation_framework_final/ablation_results.json`: Fig.4 ablation source

### 8.2 Plot-data export files

- `results/plot_data/fig2_dataset_comparison.csv`
- `results/plot_data/fig4_ablation_waterfall.csv`
- `results/plot_data/fig5_pareto.csv`
- `results/plot_data/fig6_hardware_radar.csv`
- `results/plot_data/table1_model_comparison.csv`
- `results/plot_data/missing_items.json`

### 8.3 Metric interpretation

- **Effectiveness**: Accuracy / F1 / AUC
- **Inference**: P50/P95/P99 latency, throughput
- **Resources**: peak memory, sparsity, (recommended) power
- **Stability**: NaN/Inf checks, abnormal interruption checks, log completeness

Note: `1 epoch` or `quick-smoke` is for pipeline validation only, not final performance.

---

## 9. Current Status and Gaps

### 9.1 Completed

- Result files are complete for 4 models × 3 datasets × 5 folds × 5 repeats.
- Fig.3 has been upgraded to **real train-eval driven** mode (not placeholder simulation).
- Fig.4 ablation experiment stably produces JSON + PNG outputs.
- Plot-data aggregation pipeline is complete: `collect_plot_data.py` + `generate_data_summary_md.py`.

### 9.2 Remaining gaps

- Fig.6: `power_w` is not yet uniformly collected.
- Fig.7: missing unified export pipeline for CRS stage attention heatmaps.
- Table.2: no in-repo SOTA baseline source file yet (needs external baseline intake).

---

## 10. FAQ

**Q1: How can I confirm `--all-datasets` actually uses all three datasets?**
- Check `results/full_protocol_xxx/` contains `lc25000 / physionet / iu_xray`.
- Or inspect `global_summary.json` and verify dataset entries and `fold_results` counts.

**Q2: Why are some figure data still only partially available?**
- Most training metrics are covered.
- Power, attention heatmaps, and external SOTA comparison require extra scripts or external data sources.

**Q3: Fastest way to reproduce the data summary?**
- Run full protocol first, then:
  `python scripts/collect_plot_data.py --results-dir results --out-dir results/plot_data`
  `python scripts/generate_data_summary_md.py`

---

## 11. One-Page Command Sheet

```bash
# Key tests
python -m tests.test_imports
python -m tests.test_crs_integration
python -m tests.test_execution_guide

# 4-model serial full protocol
powershell -ExecutionPolicy Bypass -File scripts/run_all_models_serial.ps1 -Epochs 20 -BatchSize 4 -Folds 5 -Repeats 5 -BaseOutput results

# Fig.3 real train-eval driver (example)
python -m experiments.long_term_memory_test --model s_crs --dataset-a lc25000 --dataset-b physionet --cycles 10 --epochs-per-cycle 1 --batch-size 4 --quick-smoke --output-dir results/long_term_memory_real

# Fig.4 ablation
python -m experiments.ablation_study --base-config configs/s_crs.yaml --seed 42 --output-dir results/ablation_framework_final

# Aggregate plot data (without plotting)
python scripts/collect_plot_data.py --results-dir results --out-dir results/plot_data
python scripts/generate_data_summary_md.py
```

---

More docs:
- `README.md`
- `docs/ACCEPTANCE_TEMPLATE.md`
- `results/plot_data/ESTIMATED_FINAL_DATA.md` (non-measured estimate draft)

