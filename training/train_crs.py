"""SpikingBrain-CRS 训练入口。

支持：
- GPU 自动检测（CUDA 优先）
- 3 数据集 × 5 折交叉验证（--all-datasets --folds 5）
- 原地刷新进度条
- 完整 FSRS 19 参数
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from data.pipeline import MedicalDataLoader
from evaluation import MedicalMetrics, ReportGenerator
from models import build_model
from training import CRSMedicalTrainer, MemoryOptimizer, RTX4050Config

SEEDS = [1, 42, 123, 456, 2024]
ALL_DATASETS = ["lc25000", "physionet", "iu_xray"]
NUM_CLASSES   = {"lc25000": 5, "physionet": 2, "iu_xray": 2}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_yaml_config(path: Path) -> Dict[str, Any]:
    """极简 YAML 加载；有 pyyaml 时直接使用。"""
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    cfg: Dict[str, Any] = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line.lstrip().startswith("#"):
            i += 1; continue
        if ":" in line and not line.startswith(" "):
            key, _, rest = line.partition(":")
            rest = rest.strip()
            if rest.startswith("["):
                items = [v.strip() for v in rest.strip("[]").split(",") if v.strip()]
                cfg[key.strip()] = [float(v) if "." in v else int(v) for v in items]
            elif rest == "":
                nested: Dict[str, Any] = {}
                i += 1
                while i < len(lines):
                    sub = lines[i].rstrip()
                    if not sub or sub.lstrip().startswith("#"):
                        i += 1; continue
                    if not sub.startswith(" "):
                        break
                    sk, _, sv = sub.strip().partition(":")
                    sv = sv.strip()
                    if sv.startswith("["):
                        items2 = [v.strip() for v in sv.strip("[]").split(",") if v.strip()]
                        nested[sk.strip()] = [float(v) if "." in v else int(v) for v in items2]
                    else:
                        nested[sk.strip()] = _parse_scalar(sv)
                    i += 1
                cfg[key.strip()] = nested
                continue
            else:
                cfg[key.strip()] = _parse_scalar(rest)
        i += 1
    return cfg


def _parse_scalar(s: str) -> Any:
    if "#" in s:
        s = s[:s.index("#")].rstrip()
    s = s.strip()
    if s.lower() in ("true", "yes"):  return True
    if s.lower() in ("false", "no"): return False
    try:
        return float(s) if ("." in s or "e" in s.lower()) else int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_device() -> str:
    """自动选择：CUDA > CPU。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _peak_memory_mb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def _fsrs_summary(scheduler) -> Dict[str, Any]:
    """提取 FSRS 状态摘要用于日志。"""
    w = scheduler.w
    return {
        "w_count": len(w),
        "w_sample": [round(v, 4) for v in w[:5]],  # 前 5 个初始稳定性参数
        "card_count": len(scheduler.card_states),
        "request_retention": scheduler.request_retention,
    }


def _mean_std_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    n = int(arr.size)
    if n <= 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(max(1, n))
    ci = 1.96 * sem
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - ci),
        "ci95_high": float(mean + ci),
        "n": n,
    }



def _iqr_outlier_ratio(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size < 4:
        return 0.0
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    if iqr <= 1e-12:
        return 0.0
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    outliers = np.logical_or(arr < lo, arr > hi)
    return float(np.mean(outliers.astype(np.float32)))


def _stability_summary(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"max_min": 0.0, "outlier_ratio_iqr": 0.0}
    return {
        "max_min": float(np.max(arr) - np.min(arr)),
        "outlier_ratio_iqr": _iqr_outlier_ratio(values),
    }


def _binary_ppv_npv(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.astype(np.int64)
    yp = y_pred.astype(np.int64)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    return {"ppv": ppv, "npv": npv}


def _loss_slope_last_k(losses: List[float], k: int = 5) -> float:
    if not losses:
        return 0.0
    kk = max(2, min(int(k), len(losses)))
    y = np.array(losses[-kk:], dtype=float)
    x = np.arange(kk, dtype=float)
    if np.allclose(y, y[0]):
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _converge_epoch(losses: List[float], min_delta: float = 1e-3, patience: int = 3) -> int:
    if not losses:
        return 0
    best = float(losses[0])
    stale = 0
    for i in range(1, len(losses)):
        cur = float(losses[i])
        if (best - cur) > float(min_delta):
            best = cur
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                return int(i + 1)
    return int(len(losses))


def _fit_temperature_binary(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """Binary temperature scaling on validation logits."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        return {"ok": False, "method": "temperature", "reason": "invalid_logits", "temperature": 1.0}
    z = (logits[:, 1] - logits[:, 0]).astype(np.float32)
    y = y_true.astype(np.float32)
    z_t = torch.tensor(z)
    y_t = torch.tensor(y)
    log_t = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(120):
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.5, 5.0)
        loss = bce(z_t / t, y_t)
        loss.backward()
        opt.step()
    t_final = float(torch.exp(log_t).clamp(0.5, 5.0).item())
    p = 1.0 / (1.0 + np.exp(-(z / max(t_final, 1e-6))))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"ok": True, "method": "temperature", "temperature": t_final, "p_pos": p.tolist()}


def _fit_platt_binary(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """Binary Platt scaling on validation logits."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        return {"ok": False, "method": "platt", "reason": "invalid_logits", "a": 1.0, "b": 0.0}
    z = (logits[:, 1] - logits[:, 0]).astype(np.float32)
    y = y_true.astype(np.float32)
    z_t = torch.tensor(z)
    y_t = torch.tensor(y)
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(160):
        opt.zero_grad()
        loss = bce(a * z_t + b, y_t)
        loss.backward()
        opt.step()
    a_f = float(a.item())
    b_f = float(b.item())
    p = 1.0 / (1.0 + np.exp(-(a_f * z + b_f)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"ok": True, "method": "platt", "a": a_f, "b": b_f, "p_pos": p.tolist()}


def _calibrate_binary_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_logits: np.ndarray,
    method: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Return calibrated [N,2] probs and calibration metadata."""
    if method == "none":
        return y_prob, {"ok": False, "method": "none", "reason": "disabled"}
    if y_prob.ndim != 2 or y_prob.shape[1] < 2:
        return y_prob, {"ok": False, "method": method, "reason": "invalid_prob"}

    if method == "temperature":
        info = _fit_temperature_binary(y_logits, y_true)
    elif method == "platt":
        info = _fit_platt_binary(y_logits, y_true)
    else:
        return y_prob, {"ok": False, "method": method, "reason": "unknown_method"}

    if not info.get("ok", False):
        return y_prob, info

    p_pos = np.array(info.get("p_pos", []), dtype=float)
    if p_pos.size != y_prob.shape[0]:
        return y_prob, {"ok": False, "method": method, "reason": "bad_calibrated_size"}
    y_prob_cal = y_prob.copy()
    y_prob_cal[:, 1] = p_pos
    y_prob_cal[:, 0] = 1.0 - p_pos
    return y_prob_cal, info


def _pick_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    center: float = 0.5,
    reg_lambda: float = 0.0,
) -> tuple[float, np.ndarray, Dict[str, Any]]:
    if num_classes != 2:
        y_pred = np.argmax(y_prob, axis=1)
        return 0.5, y_pred, {"optimized": False, "reason": "non_binary"}
    pos_prob = y_prob[:, 1]
    best_thr = 0.5
    best_score = -1.0
    best_pred = (pos_prob >= 0.5).astype(np.int64)
    lo = max(0.01, min(0.99, float(threshold_min)))
    hi = max(lo + 0.01, min(0.99, float(threshold_max)))
    grid = np.arange(lo, hi + 1e-9, 0.02).tolist()
    hist: List[Dict[str, float]] = []
    for thr in grid:
        pred = (pos_prob >= thr).astype(np.int64)
        m = MedicalMetrics.compute_all(y_true, pred, y_prob)
        f1 = float(m.get("f1", 0.0))
        penalty = float(reg_lambda) * abs(float(thr) - float(center))
        score = f1 - penalty
        hist.append({
            "threshold": float(thr),
            "f1": f1,
            "acc": float(m.get("accuracy", 0.0)),
            "penalty": penalty,
            "regularized_score": score,
        })
        if (score > best_score) or (abs(score - best_score) < 1e-12 and abs(float(thr) - float(center)) < abs(best_thr - float(center))):
            best_score = score
            best_thr = float(thr)
            best_pred = pred
    return best_thr, best_pred, {
        "optimized": True,
        "objective": "f1_minus_lambda_abs_thr_center",
        "best_regularized_score": float(best_score),
        "regularization": {
            "lambda": float(reg_lambda),
            "center": float(center),
            "threshold_min": float(lo),
            "threshold_max": float(hi),
        },
        "grid": hist,
    }


def _failure_reasons(eval_m: Dict[str, Any], fsrs_m: Dict[str, Any], threshold: float) -> List[str]:
    out: List[str] = []
    if float(eval_m.get("f1", 0.0)) <= 0.05:
        out.append("f1_low")
    if float(eval_m.get("auc_roc", 0.5)) <= 0.51:
        out.append("auc_near_random")
    if float(fsrs_m.get("spaced_review_due_ratio", 0.0)) < 0.005:
        out.append("spaced_review_due_low")
    if abs(float(threshold) - 0.5) >= 0.25:
        out.append("threshold_far_from_default")
    return out


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def run_train(
    dataset: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
    quick_smoke: bool = False,
    fold_info: Optional[tuple] = None,   # (fold_idx_1based, total_folds)
    device_override: Optional[str] = None,
    max_batches: Optional[int] = None,
    mode: Optional[str] = None,
    optimize_threshold: bool = False,
    physionet_weight_scale: float = 1.0,
    physionet_sampler_pos_ratio: float = 0.5,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_reg_lambda: float = 0.0,
    calibration_method: str = "none",
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.2,
    rank_neg_pos_ratio: float = 1.0,
    rank_loss_type: str = "margin",
    repeat_idx: int = 1,
) -> dict:
    _set_seed(int(seed) + max(0, int(repeat_idx) - 1) * 10000)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bs = min(max(1, int(batch_size)), 16)
    device = device_override or _auto_device()

    # --- Config ---
    cfg_path = Path("configs") / f"{model_type.lower()}.yaml"
    if not cfg_path.exists():
        cfg_path = Path("configs/s_crs.yaml")
    cfg = _load_yaml_config(cfg_path)
    cfg["num_classes"] = NUM_CLASSES.get(dataset, 5)
    # 每个数据集的拍平输入维度
    INPUT_DIMS = {"lc25000": 4096, "physionet": 2000, "iu_xray": 4096}
    cfg["input_dim"] = INPUT_DIMS.get(dataset, 4096)

    # --- Model ---
    model = build_model(model_type, cfg)
    num_params = _count_params(model)
    fold_str = f" fold {fold_info[0]}/{fold_info[1]}" if fold_info else ""
    print(f"\n[{dataset}{fold_str}] {model_type}: {num_params/1e6:.3f}M params  device={device}")

    # --- FSRS config（完整 19 参数）---
    fsrs_cfg = cfg.get("fsrs", {
        "request_retention": 0.9,
        "desired_retention": 0.9,
        "sm2_initial_interval": 1,
        "sm2_easiness_factor": 2.5,
        "w": [0.4, 0.6, 2.4, 5.8, 4.9, 0.8, 1.6, 0.2, 1.3, 0.14,
              0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61, 0.11, 0.31],
    })
    # 确保 w 有完整 19 个参数
    w = list(fsrs_cfg.get("w", []))
    if len(w) < 19:
        w.extend([0.1] * (19 - len(w)))
    fsrs_cfg["w"] = w

    # 模式固化：full 与 smoke 明确分离
    run_mode = mode or ("smoke" if quick_smoke else "full")
    run_epochs = 1 if quick_smoke else max(1, int(epochs))

    # --- Data ---
    train_loader = MedicalDataLoader(dataset, "train", batch_size=bs, memory_map=True, seed=seed)
    val_loader = MedicalDataLoader(dataset, "val", batch_size=bs, memory_map=True, seed=seed)
    batches = list(train_loader)
    val_batches = list(val_loader)

    # 类别权重：按 train split 频率反比（mean=1 归一化）
    class_counts = Counter(int(x["label"]) for x in getattr(train_loader, "indices", []))
    num_classes = int(NUM_CLASSES.get(dataset, 2))
    inv = []
    for i in range(num_classes):
        c = max(1, int(class_counts.get(i, 0)))
        inv.append(1.0 / float(c))
    inv_mean = float(np.mean(inv)) if inv else 1.0
    class_weights = [float(v / max(1e-12, inv_mean)) for v in inv]
    if dataset == "physionet" and num_classes == 2:
        class_weights[1] = float(class_weights[1] * max(1.0, physionet_weight_scale))

    # --- Trainer ---
    base_lr = float(cfg.get("lr", 1e-4))
    train_lr = base_lr if quick_smoke else min(base_lr, 5e-5)
    warmup_epochs = 0 if quick_smoke else min(5, max(2, run_epochs // 6))
    progress_path = output_dir / "train_progress.json"
    rank_loss_enabled = (
        dataset == "physionet"
        and int(num_classes) == 2
        and float(rank_loss_weight) > 0.0
        and run_mode == "full"
    )

    train_cfg = RTX4050Config(batch_size=bs).to_training_config({
        "seed": seed,
        "lr": train_lr,
        "device": device,
        "grad_accum_steps": int(cfg.get("grad_accum_steps", 4)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "progress_output_path": str(progress_path),
        "fsrs": fsrs_cfg,
        "enable_amp": device == "cuda",
        "class_weights": class_weights,
        "warmup_epochs": warmup_epochs,
        "lr_decay_gamma": float(cfg.get("lr_decay_gamma", 0.995)),
        "rank_loss_weight": float(rank_loss_weight) if rank_loss_enabled else 0.0,
        "rank_margin": float(rank_margin),
        "rank_neg_pos_ratio": float(rank_neg_pos_ratio),
        "rank_loss_type": str(rank_loss_type).lower(),
    })
    trainer = CRSMedicalTrainer(model, MemoryOptimizer.optimize_config(train_cfg))
    trainer.total_epochs = run_epochs
    trainer.run_label = f"{dataset}{fold_str}"
    if max_batches is not None:
        batches = batches[: max(1, int(max_batches))]
    if not batches:
        raise RuntimeError(f"数据集 {dataset} 无可用批次，请先运行数据精炼")
    if not val_batches:
        raise RuntimeError(f"数据集 {dataset} 无可用验证批次，请先运行数据精炼")
    print(f"  训练批次数: {len(batches)} × bs={bs} = {len(batches)*bs} 样本")
    print(f"  验证批次数: {len(val_batches)} × bs={bs} = {len(val_batches)*bs} 样本")
    print(f"  FSRS: w[0:5]={[round(v,3) for v in fsrs_cfg['w'][:5]]}  retention={fsrs_cfg['request_retention']}")

    # --- Training ---
    epoch_times: list[float] = []
    epoch_losses: list[float] = []
    last_train: dict = {}
    epoch_pred_trend: list[Dict[str, Any]] = []

    # balanced sampler（按类重采样；仅 full 模式开启）
    use_balanced_sampler = (run_mode == "full")

    def _build_balanced_epoch_batches(epoch_seed: int) -> list[Dict[str, object]]:
        if not use_balanced_sampler:
            return batches
        idx_rows = list(getattr(train_loader, "indices", []))
        by_class: Dict[int, list[Dict[str, Any]]] = {i: [] for i in range(num_classes)}
        for row in idx_rows:
            by_class[int(row["label"])].append(row)
        rng = random.Random((int(seed) + max(0, int(repeat_idx) - 1) * 10000) * 1000 + int(epoch_seed))
        sampled_rows: list[Dict[str, Any]] = []
        if dataset == "physionet" and num_classes == 2:
            pos_ratio = min(0.95, max(0.05, float(physionet_sampler_pos_ratio)))
            p1 = by_class.get(1) or idx_rows
            p0 = by_class.get(0) or idx_rows
            for _ in range(len(idx_rows)):
                if rng.random() < pos_ratio:
                    sampled_rows.append(p1[rng.randrange(len(p1))])
                else:
                    sampled_rows.append(p0[rng.randrange(len(p0))])
        else:
            for _ in range(len(idx_rows)):
                cls = rng.randrange(num_classes)
                pool = by_class.get(cls) or idx_rows
                sampled_rows.append(pool[rng.randrange(len(pool))])
        sampled_rows = sampled_rows[: len(idx_rows)]
        out_batches: list[Dict[str, object]] = []
        for i in range(0, len(sampled_rows), bs):
            pack = [train_loader._load_one(r) for r in sampled_rows[i : i + bs]]
            out_batches.append(train_loader._collate_fn(pack))
        return out_batches

    actual_epochs_run = 0
    for ep in range(run_epochs):
        t0 = time.time()
        train_batches_ep = _build_balanced_epoch_batches(ep)
        last_train = trainer.train_epoch(train_batches_ep, epoch=ep)
        epoch_times.append(time.time() - t0)
        epoch_losses.append(float(last_train.get("avg_loss", 0.0)))
        actual_epochs_run = ep + 1

        # 每个 epoch 导出预测分布与 class-wise recall（防坍缩）
        model.eval()
        with torch.no_grad():
            ep_true_all: list[np.ndarray] = []
            ep_pred_all: list[np.ndarray] = []
            for vb in val_batches:
                inp = vb["input"].to(device)
                sid = vb.get("sample_id")
                sid = sid.to(device) if sid is not None else None
                logits = trainer._forward_logits(inp, stage="comprehension", sample_ids=sid)
                ep_pred_all.append(torch.argmax(logits, dim=1).cpu().numpy())
                ep_true_all.append(vb["label"].cpu().numpy())
        model.train()

        ep_true = np.concatenate(ep_true_all, axis=0)
        ep_pred = np.concatenate(ep_pred_all, axis=0)
        pred_dist_ep = {str(i): int(np.sum(ep_pred == i)) for i in range(num_classes)}
        class_recall_ep = {}
        for i in range(num_classes):
            denom = int(np.sum(ep_true == i))
            hit = int(np.sum((ep_true == i) & (ep_pred == i)))
            class_recall_ep[str(i)] = float(hit / denom) if denom > 0 else 0.0
        epoch_pred_trend.append({
            "epoch": int(ep + 1),
            "pred_distribution": pred_dist_ep,
            "class_recall": class_recall_ep,
            "lr": float(last_train.get("lr", train_lr)),
        })

        if trainer.should_stop:
            print(f"  Early stopping triggered at epoch {actual_epochs_run}/{run_epochs}")
            break

    # --- Checkpoint ---
    fold_tag = f"_fold{fold_info[0]}" if fold_info else ""
    finished_epochs = max(1, actual_epochs_run)
    ckpt_suffix = "smoke" if quick_smoke else f"ep{finished_epochs}"
    ckpt_name = f"{model_type}_{dataset}{fold_tag}_{ckpt_suffix}_seed{seed}.pt"
    ckpt_path = ckpt_dir / ckpt_name
    trainer.save_checkpoint(ckpt_path, finished_epochs - 1, last_train)
    print(f"  Checkpoint -> {ckpt_path}")

    # --- Eval on full validation split ---
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    y_logits_all: list[np.ndarray] = []
    first_val_input: Optional[torch.Tensor] = None
    inference_batch_ms: list[float] = []
    inference_batch_samples: list[int] = []
    with torch.no_grad():
        for vb in val_batches:
            inp = vb["input"].to(device)
            sid = vb.get("sample_id")
            sid = sid.to(device) if sid is not None else None
            t_infer_0 = time.perf_counter()
            logits = trainer._forward_logits(inp, stage="comprehension", sample_ids=sid)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t_infer_0) * 1000.0
            inference_batch_ms.append(float(dt_ms))
            inference_batch_samples.append(int(inp.shape[0]))
            if first_val_input is None:
                first_val_input = inp
            y_pred_all.append(torch.argmax(logits, dim=1).cpu().numpy())
            y_prob_all.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            y_logits_all.append(logits.detach().cpu().numpy())
            y_true_all.append(vb["label"].cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_logits = np.concatenate(y_logits_all, axis=0)

    infer_p50 = float(np.percentile(np.array(inference_batch_ms, dtype=float), 50)) if inference_batch_ms else 0.0
    infer_p95 = float(np.percentile(np.array(inference_batch_ms, dtype=float), 95)) if inference_batch_ms else 0.0
    infer_p99 = float(np.percentile(np.array(inference_batch_ms, dtype=float), 99)) if inference_batch_ms else 0.0
    infer_total_s = float(np.sum(np.array(inference_batch_ms, dtype=float)) / 1000.0) if inference_batch_ms else 0.0
    infer_total_samples = int(np.sum(np.array(inference_batch_samples, dtype=np.int64))) if inference_batch_samples else 0
    infer_throughput = float(infer_total_samples / max(infer_total_s, 1e-9)) if infer_total_samples > 0 else 0.0
    n_classes = int(NUM_CLASSES.get(dataset, int(y_prob.shape[1] if y_prob.ndim == 2 else 2)))
    selected_threshold = 0.5
    threshold_search: Dict[str, Any] = {"optimized": False, "reason": "disabled"}
    calibration_info: Dict[str, Any] = {"ok": False, "method": "none", "reason": "disabled"}

    if optimize_threshold and dataset == "physionet":
        y_prob_for_thr = y_prob
        if n_classes == 2 and calibration_method.lower() != "none":
            y_prob_for_thr, calibration_info = _calibrate_binary_probs(
                y_true=y_true,
                y_prob=y_prob,
                y_logits=y_logits,
                method=calibration_method.lower(),
            )
        selected_threshold, y_pred, threshold_search = _pick_best_threshold(
            y_true=y_true,
            y_prob=y_prob_for_thr,
            num_classes=n_classes,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            center=0.5,
            reg_lambda=threshold_reg_lambda,
        )
        threshold_search["calibration"] = calibration_info

    eval_metrics = MedicalMetrics.compute_all(y_true, y_pred, y_prob)
    eval_metrics["selected_threshold"] = float(selected_threshold)
    eval_metrics["threshold_optimized"] = bool(threshold_search.get("optimized", False))
    if n_classes == 2:
        ppv_npv = _binary_ppv_npv(y_true, y_pred)
        eval_metrics["ppv"] = float(ppv_npv.get("ppv", eval_metrics.get("precision", 0.0)))
        eval_metrics["ppv_alias_precision"] = float(eval_metrics.get("precision", ppv_npv.get("ppv", 0.0)))
        eval_metrics["npv"] = float(ppv_npv.get("npv", 0.0))

    # diagnostics: label/pred 分布 + confusion matrix
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        ti, pi = int(t), int(p)
        if 0 <= ti < n_classes and 0 <= pi < n_classes:
            conf[ti, pi] += 1
    label_dist = {str(i): int(np.sum(y_true == i)) for i in range(n_classes)}
    pred_dist = {str(i): int(np.sum(y_pred == i)) for i in range(n_classes)}
    class_recall = {}
    for i in range(n_classes):
        denom = int(conf[i, :].sum())
        hit = int(conf[i, i])
        class_recall[str(i)] = float(hit / denom) if denom > 0 else 0.0
    diag = {
        "mode": run_mode,
        "dataset": dataset,
        "fold": fold_info[0] if fold_info else None,
        "seed": seed,
        "num_classes": n_classes,
        "support": int(len(y_true)),
        "label_distribution": label_dist,
        "pred_distribution": pred_dist,
        "class_recall": class_recall,
        "confusion_matrix": conf.tolist(),
        "class_weights": class_weights,
        "selected_threshold": float(selected_threshold),
        "threshold_search": threshold_search,
        "calibration": calibration_info,
    }
    (output_dir / "eval_diagnostics.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "epoch_prediction_trend.json").write_text(
        json.dumps(epoch_pred_trend, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Sparsity
    probe_inp = first_val_input if first_val_input is not None else batches[0]["input"].to(device)
    with torch.no_grad():
        out = model(probe_inp)
    sparsity = out.get("sparsity", 0.0)

    # Performance summary
    avg_epoch_time = float(np.mean(epoch_times))
    total_epochs_cfg = int(cfg.get("epochs", epochs))
    estimated_total_s = avg_epoch_time * total_epochs_cfg
    peak_mem = _peak_memory_mb(device)
    total_samples = sum(len(b["label"]) for b in batches) * run_epochs
    throughput = total_samples / max(sum(epoch_times), 1e-6)

    converge_epoch = _converge_epoch(epoch_losses, min_delta=1e-3, patience=3)
    loss_slope_last5 = _loss_slope_last_k(epoch_losses, k=5)

    sig_obj = {"seed": seed, "dataset": dataset, "model": model_type,
               "avg_loss": last_train.get("avg_loss", 0.0),
               "accuracy": eval_metrics.get("accuracy", 0.0)}
    signature = hashlib.md5(
        json.dumps(sig_obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

    fail_reasons = _failure_reasons(eval_metrics, _fsrs_summary(trainer.fsrs_scheduler), selected_threshold)
    result = {
        "config": {"dataset": dataset, "model": model_type, "epochs_run": finished_epochs,
                   "batch_size": bs, "seed": seed, "repeat_idx": int(repeat_idx),
                   "repeat_seed": int(seed + max(0, int(repeat_idx) - 1) * 10000), "device": device,
                   "mode": run_mode,
                   "lr": train_lr,
                   "warmup_epochs": int(warmup_epochs),
                   "class_weights": class_weights,
                   "balanced_sampler": bool(use_balanced_sampler),
                   "fold": fold_info[0] if fold_info else None,
                   "total_folds": fold_info[1] if fold_info else None,
                   "optimize_threshold": bool(optimize_threshold and dataset == "physionet"),
                   "physionet_weight_scale": float(physionet_weight_scale),
                   "physionet_sampler_pos_ratio": float(physionet_sampler_pos_ratio),
                   "threshold_min": float(threshold_min),
                   "threshold_max": float(threshold_max),
                   "threshold_reg_lambda": float(threshold_reg_lambda),
                   "calibration_method": str(calibration_method).lower(),
                   "rank_loss_enabled": bool(rank_loss_enabled),
                   "rank_loss_weight": float(rank_loss_weight) if rank_loss_enabled else 0.0,
                   "rank_margin": float(rank_margin),
                   "rank_neg_pos_ratio": float(rank_neg_pos_ratio),
                   "rank_loss_type": str(rank_loss_type).lower()},
        "fsrs": _fsrs_summary(trainer.fsrs_scheduler),
        "performance": {
            "num_params": num_params,
            "num_params_M": round(num_params / 1e6, 3),
            "peak_memory_MB": round(peak_mem, 1),
            "sparsity": round(sparsity, 4),
            "throughput_samples_per_s": round(throughput, 2),
            "avg_epoch_time_s": round(avg_epoch_time, 2),
            "estimated_total_train_s": round(estimated_total_s, 1),
            "estimated_total_train_min": round(estimated_total_s / 60, 1),
            "inference_latency_ms": {
                "p50": round(infer_p50, 3),
                "p95": round(infer_p95, 3),
                "p99": round(infer_p99, 3),
                "num_batches": int(len(inference_batch_ms)),
            },
            "inference_throughput_samples_per_s": round(infer_throughput, 2),
        },
        "train": {
            **last_train,
            "converge_epoch": int(converge_epoch),
            "loss_slope_last_k": float(loss_slope_last5),
            "loss_slope_last_k_window": 5,
            "loss_curve": [float(x) for x in epoch_losses],
        },
        "eval": eval_metrics,
        "failure_reasons": fail_reasons,
        "checkpoint": str(ckpt_path),
        "signature": signature,
    }

    rg = ReportGenerator(output_dir)
    rg.save_json("train_summary.json", result)
    rg.save_markdown("train_report.md", "CRS Train Report", result)
    rg.append_log("train.log", f"dataset={dataset} fold={fold_info} model={model_type} "
                               f"seed={seed} acc={eval_metrics.get('accuracy',0):.4f} sig={signature}")

    p = result["performance"]
    print(f"  参数量={p['num_params_M']}M  峰值显存={p['peak_memory_MB']}MB  "
          f"稀疏度={p['sparsity']:.4f}  throughput={p['throughput_samples_per_s']:.1f}samp/s  "
          f"预估总训练={p['estimated_total_train_min']:.1f}min({total_epochs_cfg}ep)  "
          f"acc={eval_metrics.get('accuracy',0):.4f}")
    return result


# ---------------------------------------------------------------------------
# 3 datasets × k folds
# ---------------------------------------------------------------------------

def run_all_folds(
    datasets: List[str],
    model_type: str,
    epochs: int,
    batch_size: int,
    folds: int,
    output_dir: Path,
    quick_smoke: bool,
    device_override: Optional[str],
    repeats: int = 1,
    optimize_threshold: bool = False,
    physionet_weight_scale: float = 1.0,
    physionet_sampler_pos_ratio: float = 0.5,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_reg_lambda: float = 0.0,
    calibration_method: str = "none",
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.2,
    rank_neg_pos_ratio: float = 1.0,
    rank_loss_type: str = "margin",
    summary_name: str = "global_summary.json",
) -> Dict[str, Any]:
    """在指定数据集列表上跑 k 折交叉验证，汇总结果。"""
    seeds_to_use = SEEDS[:folds]
    all_results: Dict[str, Any] = {}

    repeats = max(1, int(repeats))
    total_runs = len(datasets) * folds * repeats
    run_num = 0

    print(f"\n{'='*65}")
    print(f"  SpikingBrain-CRS  |  {model_type}  |  "
          f"{len(datasets)} datasets × {folds} folds × {repeats} repeats  |  "
          f"device={device_override or _auto_device()}")
    print(f"{'='*65}")

    for ds in datasets:
        fold_results = []
        for rep in range(repeats):
            for fold_i, seed in enumerate(seeds_to_use):
                run_num += 1
                run_seed = int(seed)
                fold_dir = output_dir / ds / f"repeat{rep+1}" / f"fold{fold_i+1}_seed{run_seed}_rep{rep+1}"
                result = run_train(
                    dataset=ds,
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=run_seed,
                    output_dir=fold_dir,
                    quick_smoke=quick_smoke,
                    fold_info=(fold_i + 1, folds),
                    device_override=device_override,
                    mode=("smoke" if quick_smoke else "full"),
                    optimize_threshold=optimize_threshold,
                    physionet_weight_scale=physionet_weight_scale,
                    physionet_sampler_pos_ratio=physionet_sampler_pos_ratio,
                    threshold_min=threshold_min,
                    threshold_max=threshold_max,
                    threshold_reg_lambda=threshold_reg_lambda,
                    calibration_method=calibration_method,
                    rank_loss_weight=rank_loss_weight,
                    rank_margin=rank_margin,
                    rank_neg_pos_ratio=rank_neg_pos_ratio,
                    rank_loss_type=rank_loss_type,
                    repeat_idx=rep + 1,
                )
                result["repeat"] = rep + 1
                result["repeat_seed"] = int(seed + rep * 10000)
                fold_results.append(result)

        accs = [r["eval"].get("accuracy", 0.0) for r in fold_results]
        f1s = [r["eval"].get("f1", 0.0) for r in fold_results]
        aucs = [r["eval"].get("auc_roc", 0.5) for r in fold_results]
        losses = [r["train"].get("avg_loss", 0.0) for r in fold_results]
        mems = [r["performance"].get("peak_memory_MB", 0.0) for r in fold_results]
        times = [r["performance"].get("avg_epoch_time_s", 0.0) for r in fold_results]
        thresholds = [r["eval"].get("selected_threshold", 0.5) for r in fold_results]
        npvs = [r["eval"].get("npv", 0.0) for r in fold_results]
        ppv_alias = [r["eval"].get("ppv_alias_precision", r["eval"].get("precision", 0.0)) for r in fold_results]
        params = [r["performance"].get("num_params", 0.0) for r in fold_results]
        infer_p50s = [r["performance"].get("inference_latency_ms", {}).get("p50", 0.0) for r in fold_results]
        infer_p95s = [r["performance"].get("inference_latency_ms", {}).get("p95", 0.0) for r in fold_results]
        infer_p99s = [r["performance"].get("inference_latency_ms", {}).get("p99", 0.0) for r in fold_results]
        infer_tps = [r["performance"].get("inference_throughput_samples_per_s", 0.0) for r in fold_results]
        conv_epochs = [r["train"].get("converge_epoch", 0.0) for r in fold_results]
        loss_slopes = [r["train"].get("loss_slope_last_k", 0.0) for r in fold_results]
        fail_hist: Dict[str, int] = {}
        for r in fold_results:
            for reason in r.get("failure_reasons", []):
                fail_hist[reason] = int(fail_hist.get(reason, 0) + 1)

        acc_stat = _mean_std_ci95(accs)
        f1_stat = _mean_std_ci95(f1s)
        auc_stat = _mean_std_ci95(aucs)
        loss_stat = _mean_std_ci95(losses)
        th_stat = _mean_std_ci95(thresholds)
        npv_stat = _mean_std_ci95(npvs)
        ppv_alias_stat = _mean_std_ci95(ppv_alias)
        conv_stat = _mean_std_ci95(conv_epochs)
        slope_stat = _mean_std_ci95(loss_slopes)

        all_results[ds] = {
            "fold_results": fold_results,
            "repeats": repeats,
            "folds": folds,
            "n_runs": len(fold_results),
            "accuracy": acc_stat,
            "f1": f1_stat,
            "auc_roc": auc_stat,
            "loss": loss_stat,
            "ppv_alias_precision": ppv_alias_stat,
            "npv": npv_stat,
            "stability": {
                "accuracy": _stability_summary(accs),
                "f1": _stability_summary(f1s),
                "auc_roc": _stability_summary(aucs),
                "loss": _stability_summary(losses),
                "selected_threshold": _stability_summary(thresholds),
            },
            "resource_efficiency": {
                "param_count": {
                    "mean": float(np.mean(np.array(params, dtype=float))) if params else 0.0,
                    "min": float(np.min(np.array(params, dtype=float))) if params else 0.0,
                    "max": float(np.max(np.array(params, dtype=float))) if params else 0.0,
                },
                "mean_peak_memory_MB": float(np.mean(mems)) if mems else 0.0,
                "mean_epoch_time_s": float(np.mean(times)) if times else 0.0,
                "inference_latency_ms": {
                    "p50": _mean_std_ci95(infer_p50s),
                    "p95": _mean_std_ci95(infer_p95s),
                    "p99": _mean_std_ci95(infer_p99s),
                },
                "inference_throughput_samples_per_s": _mean_std_ci95(infer_tps),
            },
            "convergence": {
                "converge_epoch": conv_stat,
                "loss_slope_last_k": slope_stat,
                "loss_slope_last_k_window": 5,
            },
            "mean_peak_memory_MB": float(np.mean(mems)) if mems else 0.0,
            "mean_epoch_time_s": float(np.mean(times)) if times else 0.0,
            "threshold_distribution": {
                "mean": th_stat["mean"],
                "std": th_stat["std"],
                "ci95_low": th_stat["ci95_low"],
                "ci95_high": th_stat["ci95_high"],
                "p25": float(np.percentile(np.array(thresholds, dtype=float), 25)) if thresholds else 0.5,
                "p50": float(np.percentile(np.array(thresholds, dtype=float), 50)) if thresholds else 0.5,
                "p75": float(np.percentile(np.array(thresholds, dtype=float), 75)) if thresholds else 0.5,
                "values": [float(x) for x in thresholds],
            },
            "failure_reason_histogram": fail_hist,
            "failure_details": [
                {
                    "repeat": int(r.get("repeat", 1)),
                    "fold": int(r.get("config", {}).get("fold") or 0),
                    "seed": int(r.get("config", {}).get("seed") or 0),
                    "reasons": r.get("failure_reasons", []),
                    "f1": float(r.get("eval", {}).get("f1", 0.0)),
                    "auc": float(r.get("eval", {}).get("auc_roc", 0.5)),
                    "threshold": float(r.get("eval", {}).get("selected_threshold", 0.5)),
                }
                for r in fold_results
            ],
        }
        print(f"\n  [{ds}] {folds}-fold × {repeats} repeats: "
              f"acc={acc_stat['mean']:.4f} ± {acc_stat['std']:.4f}  f1={f1_stat['mean']:.4f}  auc={auc_stat['mean']:.4f}")

    print(f"\n{'='*65}")
    print("  全局汇总:")
    for ds, r in all_results.items():
        print(f"    {ds:12s}  acc={r['accuracy']['mean']:.4f} ± {r['accuracy']['std']:.4f}")
    print(f"{'='*65}\n")

    global_summary_path = output_dir / summary_name
    global_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out = {ds: r for ds, r in all_results.items()}
    global_summary_path.write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="SpikingBrain-CRS 训练入口")
    p.add_argument("--dataset", default="lc25000", choices=ALL_DATASETS)
    p.add_argument("--model", default="s_crs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/smoke_test")
    p.add_argument("--quick-smoke", action="store_true",
                   help="仅跑 1 epoch + 输出预估时间")
    p.add_argument("--all-datasets", action="store_true",
                   help="在全部 3 个数据集上训练")
    p.add_argument("--folds", type=int, default=1,
                   help="k 折交叉验证折数（默认 1；--all-datasets 时自动变为 5）")
    p.add_argument("--device", default=None,
                   help="强制指定设备（cuda/cpu），默认自动检测")
    p.add_argument("--repeats", type=int, default=1,
                   help="重复次数（每个 fold 使用不同 seed 偏移）")
    p.add_argument("--optimize-threshold", action="store_true",
                   help="二分类任务在验证集上搜索最佳阈值（当前用于 physionet）")
    p.add_argument("--physionet-weight-scale", type=float, default=1.5,
                   help="physionet 正类权重缩放系数")
    p.add_argument("--rank-loss-weight", type=float, default=0.0,
                   help="v3.4: 排序损失权重（仅physionet二分类 full模式生效）")
    p.add_argument("--rank-margin", type=float, default=0.2,
                   help="v3.4: margin ranking 的间隔参数")
    p.add_argument("--rank-neg-pos-ratio", type=float, default=1.0,
                   help="v3.4: 每个正样本配对负样本比例")
    p.add_argument("--rank-loss-type", default="margin", choices=["margin", "logistic"],
                   help="v3.4: 排序损失类型")
    p.add_argument("--physionet-sampler-pos-ratio", type=float, default=0.6,
                   help="physionet 采样时正类目标比例")
    p.add_argument("--summary-name", default="global_summary.json",
                   help="聚合汇总文件名")
    p.add_argument("--threshold-min", type=float, default=0.1,
                   help="阈值搜索下界（physionet二分类）")
    p.add_argument("--threshold-max", type=float, default=0.9,
                   help="阈值搜索上界（physionet二分类）")
    p.add_argument("--threshold-reg-lambda", type=float, default=0.0,
                   help="阈值正则系数：score = f1 - lambda*|thr-0.5|")
    p.add_argument("--calibration-method", default="none", choices=["none", "temperature", "platt"],
                   help="阈值搜索前概率校准方法（physionet二分类）")
    p.add_argument("--compare-v2", default=None,
                   help="可选：传入 v2 的 global_summary.json 路径，输出增量对比")
    args = p.parse_args()

    if int(args.repeats) < 1:
        raise ValueError("repeats 必须 >= 1")

    datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
    # --all-datasets 默认 5 折；否则按 --folds 参数（最少 1）
    folds = min(max(1, args.folds if args.folds > 1 else (5 if args.all_datasets else 1)), len(SEEDS))

    if len(datasets) == 1 and folds == 1 and int(args.repeats) == 1:
        # 单数据集单折：直接调用
        out = run_train(
            dataset=datasets[0],
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            quick_smoke=args.quick_smoke,
            device_override=args.device,
            mode=("smoke" if args.quick_smoke else "full"),
            optimize_threshold=args.optimize_threshold,
            physionet_weight_scale=args.physionet_weight_scale,
            physionet_sampler_pos_ratio=args.physionet_sampler_pos_ratio,
            threshold_min=float(args.threshold_min),
            threshold_max=float(args.threshold_max),
            threshold_reg_lambda=float(args.threshold_reg_lambda),
            calibration_method=str(args.calibration_method).lower(),
            rank_loss_weight=float(args.rank_loss_weight),
            rank_margin=float(args.rank_margin),
            rank_neg_pos_ratio=float(args.rank_neg_pos_ratio),
            rank_loss_type=str(args.rank_loss_type).lower(),
        )
        print("TRAIN_CRS_OK", out["signature"])
    else:
        all_out = run_all_folds(
            datasets=datasets,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            folds=folds,
            output_dir=Path(args.output_dir),
            quick_smoke=args.quick_smoke,
            device_override=args.device,
            repeats=int(args.repeats),
            optimize_threshold=args.optimize_threshold,
            physionet_weight_scale=float(args.physionet_weight_scale),
            physionet_sampler_pos_ratio=float(args.physionet_sampler_pos_ratio),
            threshold_min=float(args.threshold_min),
            threshold_max=float(args.threshold_max),
            threshold_reg_lambda=float(args.threshold_reg_lambda),
            calibration_method=str(args.calibration_method).lower(),
            rank_loss_weight=float(args.rank_loss_weight),
            rank_margin=float(args.rank_margin),
            rank_neg_pos_ratio=float(args.rank_neg_pos_ratio),
            rank_loss_type=str(args.rank_loss_type).lower(),
            summary_name=str(args.summary_name),
        )
        if args.compare_v2:
            p = Path(args.compare_v2)
            if p.exists():
                old = json.loads(p.read_text(encoding="utf-8"))
                delta: Dict[str, Any] = {}
                for ds, cur in all_out.items():
                    prev = old.get(ds, {})
                    prev_acc = float(prev.get("mean_accuracy", prev.get("accuracy", {}).get("mean", 0.0)))
                    prev_f1 = float(prev.get("mean_f1", prev.get("f1", {}).get("mean", 0.0)))
                    prev_auc = float(prev.get("mean_auc_roc", prev.get("auc_roc", {}).get("mean", 0.5)))
                    delta[ds] = {
                        "delta_accuracy": float(cur["accuracy"]["mean"] - prev_acc),
                        "delta_f1": float(cur["f1"]["mean"] - prev_f1),
                        "delta_auc_roc": float(cur["auc_roc"]["mean"] - prev_auc),
                        "cur_std_accuracy": float(cur["accuracy"]["std"]),
                    }
                (Path(args.output_dir) / "v2_delta_comparison.json").write_text(
                    json.dumps(delta, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        sigs = [r["signature"]
                for ds_res in all_out.values()
                for r in ds_res["fold_results"]]
        combined = hashlib.md5("".join(sigs).encode()).hexdigest()
        print("TRAIN_CRS_ALL_OK", combined)


if __name__ == "__main__":
    main()
