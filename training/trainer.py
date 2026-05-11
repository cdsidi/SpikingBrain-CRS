"""Layer 3 训练引擎最小可运行实现。"""

from __future__ import annotations

import json
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.crs import (
    ComprehensionPhase,
    ErrorCorrectionPhase,
    FSRSScheduler,
    RecallPhase,
    SpacedReviewPhase,
    SynthesisPhase,
)
from training.phase_scheduler import PhaseScheduler


class CRSMedicalTrainer:
    """CRS医学训练器（最小版本）。

    约束说明（01_PRD）：
    - batch_size <= 16（由数据层控制）
    - 轻量实现，显存目标 < 4.2GB

    Args:
        model: 可调用模型，输入 x: Tensor[B, ...], 输出 logits 或 {'logits': Tensor[B, C]}。
        config: 训练配置，如 lr/grad_accum_steps/max_grad_norm/seed。
    """

    def __init__(self, model: object, config: Dict) -> None:
        self.model = model
        self.config = dict(config)
        self.device = torch.device(self.config.get("device", "cpu"))
        self.model.to(self.device)
        self.seed = int(self.config.get("seed", 42))
        self._set_seed(self.seed)

        # Layer 6: RTX 4050 适配开关（最小侵入）
        self.hw_profile = self.config.get("hardware_profile", "auto")
        self.enable_amp = bool(self.config.get("enable_amp", self.device.type == "cuda"))
        self.enable_grad_checkpoint = bool(self.config.get("enable_grad_checkpoint", False))
        self.enable_cpu_offload = bool(self.config.get("enable_cpu_offload", False))
        self.optimizer_name = str(self.config.get("optimizer", "adamw")).lower()
        self.batch_size = int(self.config.get("batch_size", 16))
        if self.batch_size > 16:
            self.batch_size = 16  # 基线约束
        if self.enable_grad_checkpoint and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.grad_accum_steps = int(self.config.get("grad_accum_steps", 4))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))

        # 09 文档含 8-bit Adam；在无 bitsandbytes 时回退 AdamW，保持可运行
        self.optimizer = self._build_optimizer(lr=float(self.config.get("lr", 1e-3)))
        warmup_epochs = int(self.config.get("warmup_epochs", 0))
        decay_gamma = float(self.config.get("lr_decay_gamma", 0.99))

        def _lr_lambda(ep_idx: int) -> float:
            if warmup_epochs > 0 and ep_idx < warmup_epochs:
                return float(ep_idx + 1) / float(max(1, warmup_epochs))
            decay_steps = max(0, ep_idx - warmup_epochs)
            return float(decay_gamma ** decay_steps)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
        scaler_enabled = bool(self.enable_amp and self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        else:
            self.scaler = None

        self.phase_history: list[Dict[str, Any]] = []
        self.fsrs_state: Dict[str, Any] = {}
        self.progress_output_path = self.config.get("progress_output_path")
        self.last_progress: list[Dict[str, float | int]] = []

        self.phase_scheduler = PhaseScheduler()
        self.phase_names = ["comprehension", "recall", "synthesis", "spaced_review", "error_correction"]
        self.fsrs_scheduler = FSRSScheduler(self.config.get("fsrs", {}))
        self.class_weights: torch.Tensor | None = None
        cw = self.config.get("class_weights")
        if isinstance(cw, (list, tuple)) and len(cw) > 0:
            self.class_weights = torch.tensor([float(v) for v in cw], dtype=torch.float32, device=self.device)

        # v3.4: 可选 rank-aware 二分类排序损失（仅在外层配置开启时生效）
        self.rank_loss_weight = float(self.config.get("rank_loss_weight", 0.0))
        self.rank_margin = float(self.config.get("rank_margin", 0.2))
        self.rank_neg_pos_ratio = float(self.config.get("rank_neg_pos_ratio", 1.0))
        self.rank_loss_type = str(self.config.get("rank_loss_type", "margin")).lower()

        # v5: ablation 开关（阶段/FSRS）
        self.ablation_name = str(self.config.get("ablation_name", "none")).lower()
        self.phase_ablation_flags = {
            "no_recall": bool(self.config.get("disable_recall", False)),
            "no_synthesis": bool(self.config.get("disable_synthesis", False)),
            "no_spaced": bool(self.config.get("disable_spaced_review", False)),
            "no_errorcorr": bool(self.config.get("disable_error_correction", False)),
        }

        self._phase_ops = {
            "comprehension": ComprehensionPhase(),
            "recall": RecallPhase(),
            "synthesis": SynthesisPhase(),
            "spaced_review": SpacedReviewPhase(),
            "error_correction": ErrorCorrectionPhase(),
        }
        self.early_stopping_patience = int(self.config.get("early_stopping_patience", 20))
        self.checkpoint_every_minutes = float(self.config.get("checkpoint_every_minutes", 30.0))
        self._best_loss = float("inf")
        self._no_improve_epochs = 0
        self.should_stop = False
        self.last_checkpoint_walltime = time.time()
        signal.signal(signal.SIGINT, self._handle_interrupt)

        self._hw_samples: list[Dict[str, float]] = []

        # 进度条上下文（由外部 run_train 设置）
        self.total_epochs: int = 1      # 总 epoch 数
        self.run_label: str = ""        # 如 "lc25000 fold 1/5"
        self._epoch_start_time: float = 0.0

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def _handle_interrupt(self, signum, frame) -> None:
        """处理 Ctrl+C：标记提前停止，由外层训练循环感知。"""
        self.should_stop = True

    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """构建优化器。

        优先: 8-bit Adam（09 文档要求）；不可用时回退 AdamW。
        """
        if self.optimizer_name in {"adam8bit", "8bit_adam", "adamw8bit"}:
            try:
                import bitsandbytes as bnb  # type: ignore

                return bnb.optim.Adam8bit(self.model.parameters(), lr=lr)
            except Exception:
                pass
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _sample_hw_stats(self, step: int, samples: int) -> None:
        """采集硬件统计样本（显存/吞吐）。"""
        if self.device.type == "cuda" and torch.cuda.is_available():
            mem_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))
        else:
            mem_mb = 0.0
        self._hw_samples.append({"step": float(step), "samples": float(samples), "peak_gpu_mem_mb": mem_mb})

    def get_hardware_report(self) -> Dict[str, float | bool | str]:
        """汇总 Layer 6 硬件适配报告。"""
        if not self._hw_samples:
            peak_mem = 0.0
            throughput = 0.0
        else:
            peak_mem = max(s["peak_gpu_mem_mb"] for s in self._hw_samples)
            throughput = float(self._hw_samples[-1]["samples"] / max(1.0, self._hw_samples[-1]["step"]))
        return {
            "hardware_profile": str(self.hw_profile),
            "enable_amp": bool(self.enable_amp and self.device.type == "cuda"),
            "enable_grad_checkpoint": bool(self.enable_grad_checkpoint),
            "enable_cpu_offload": bool(self.enable_cpu_offload),
            "optimizer": self.optimizer.__class__.__name__,
            "effective_batch_size": int(self.batch_size),
            "peak_gpu_mem_mb": float(peak_mem),
            "within_target": bool(peak_mem < 4200.0),
            "throughput_samples_per_step": float(throughput),
        }

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        s = max(0, int(seconds))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m"
        if m:
            return f"{m}m{sec:02d}s"
        return f"{sec}s"

    def _emit_progress(self, item: Dict[str, float | int], is_final_step: bool = False) -> None:
        """原地刷新进度条（\r 覆写）。epoch 结束时打印换行并显示摘要。"""
        cur_ep   = int(item["epoch"]) + 1          # 1-based 显示
        tot_ep   = max(1, self.total_epochs)
        step     = int(item["step"])
        total    = int(item["total_steps"])
        pct      = max(0.0, min(100.0, float(item["progress_pct"])))
        loss     = float(item["loss"])
        lr       = float(item["lr"])
        mem      = float(item["peak_gpu_mem_mb"])
        eta      = float(item["eta_sec"])

        # ── Epoch 外层进度条（窄）──────────────────────────────────
        EP_W = 10
        ep_filled = int((cur_ep / tot_ep) * EP_W)
        ep_bar = "#" * ep_filled + "-" * (EP_W - ep_filled)

        # ── 步骤内层进度条（宽）──────────────────────────────────
        ST_W = 25
        st_filled = int((pct / 100.0) * ST_W)
        st_bar = "=" * st_filled + "." * (ST_W - st_filled)

        label = f"[{self.run_label}] " if self.run_label else ""
        mem_str = f" VRAM={mem:.0f}MB" if mem > 0 else ""
        eta_str = self._fmt_eta(eta)

        line = (
            f"\r{label}"
            f"Ep {cur_ep:2d}/{tot_ep} |{ep_bar}|  "
            f"Step {step:4d}/{total} |{st_bar}| {pct:5.1f}%  "
            f"loss={loss:.4f} lr={lr:.1e}{mem_str}  ETA={eta_str}   "
        )
        sys.stdout.write(line)
        sys.stdout.flush()

        if is_final_step:
            # 打印换行 + epoch 结束摘要
            elapsed = time.perf_counter() - self._epoch_start_time
            sys.stdout.write(
                f"\n  Epoch {cur_ep}/{tot_ep} done | "
                f"avg_loss={loss:.4f} | elapsed={self._fmt_eta(elapsed)} | "
                f"FSRS cards={len(self.fsrs_scheduler.card_states)}\n"
            )
            sys.stdout.flush()

        if self.progress_output_path:
            p = Path(str(self.progress_output_path))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.last_progress, ensure_ascii=False, indent=2), encoding="utf-8")

    def _forward_logits(self, x: torch.Tensor, stage: str = "comprehension", sample_ids: torch.Tensor | None = None) -> torch.Tensor:
        """统一前向适配。

        优先调用约定接口 model(input, stage=..., sample_ids=...)；
        若模型不支持该签名则回退到 model(input)。
        """
        try:
            out = self.model(x, stage=stage, sample_ids=sample_ids)
        except TypeError:
            out = self.model(x)
        if isinstance(out, dict):
            if "logits" in out:
                return out["logits"]
            return next(iter(out.values()))
        return out

    def _binary_rank_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """二分类 pairwise rank loss（margin / logistic）。

        仅当 batch 同时含正负样本时有效，否则返回 0。
        """
        if logits.ndim != 2 or logits.shape[1] != 2:
            return logits.sum() * 0.0

        pos_mask = (y == 1)
        neg_mask = (y == 0)
        if int(pos_mask.sum().item()) == 0 or int(neg_mask.sum().item()) == 0:
            return logits.sum() * 0.0

        pos_scores = logits[pos_mask, 1]
        neg_scores = logits[neg_mask, 1]

        # 控制 pair 数量，避免 O(P*N) 过大
        max_neg = max(1, int(round(len(pos_scores) * max(0.1, self.rank_neg_pos_ratio))))
        if len(neg_scores) > max_neg:
            idx = torch.randperm(len(neg_scores), device=neg_scores.device)[:max_neg]
            neg_scores = neg_scores[idx]

        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)

        if self.rank_loss_type == "logistic":
            # softplus(-diff) = log(1 + exp(-diff))
            return F.softplus(-diff).mean()

        # default: margin ranking, 期望 pos - neg >= margin
        return F.relu(self.rank_margin - diff).mean()

    def train_epoch(self, dataloader: object, epoch: int) -> Dict[str, float]:
        """训练单个 epoch（真实五阶段加权损失融合）。"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        batches = list(dataloader)
        total_steps = len(batches)
        total_loss, total_samples = 0.0, 0
        total_nonzero, total_elems = 0.0, 0.0
        phase_counts = {"comprehension": 0, "recall": 0, "synthesis": 0, "spaced_review": 0, "error_correction": 0}
        phase_loss_sums = {"comprehension": 0.0, "recall": 0.0, "synthesis": 0.0, "spaced_review": 0.0, "error_correction": 0.0}
        phase_weights = self.phase_scheduler.get_phase_weights(epoch)
        if self.phase_ablation_flags.get("no_recall", False):
            phase_weights["recall"] = 0.0
        if self.phase_ablation_flags.get("no_synthesis", False):
            phase_weights["synthesis"] = 0.0
        if self.phase_ablation_flags.get("no_spaced", False):
            phase_weights["spaced_review"] = 0.0
        if self.phase_ablation_flags.get("no_errorcorr", False):
            phase_weights["error_correction"] = 0.0
        due_hits, due_total = 0, 0
        step_count = 0
        step_times: list[float] = []
        self.last_progress = []
        self._epoch_start_time = time.perf_counter()

        for step, batch in enumerate(batches):
            step_count = step + 1
            _step_t0 = time.perf_counter()
            x = batch["input"].to(self.device)
            y = batch["label"].to(self.device).long()
            sid = batch.get("sample_id")
            if sid is not None:
                sid = sid.to(self.device)
            phase_batch = {"input": x, "label": y, "sample_id": sid}

            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (self.enable_amp and self.device.type == "cuda") else nullcontext()
            with amp_ctx:
                loss_map: Dict[str, torch.Tensor] = {}
                loss_map["comprehension"] = self._phase_ops["comprehension"].forward(
                    self.model, phase_batch, class_weights=self.class_weights
                )
                if self.phase_ablation_flags.get("no_recall", False):
                    loss_map["recall"] = loss_map["comprehension"] * 0.0
                else:
                    loss_map["recall"] = self._phase_ops["recall"].forward(
                        self.model, phase_batch, difficulty_scheduler=None, class_weights=self.class_weights
                    )

                if self.phase_ablation_flags.get("no_synthesis", False):
                    loss_map["synthesis"] = loss_map["comprehension"] * 0.0
                else:
                    loss_map["synthesis"] = self._phase_ops["synthesis"].forward(
                        self.model, phase_batch, class_weights=self.class_weights
                    )

                if self.phase_ablation_flags.get("no_spaced", False):
                    loss_map["spaced_review"] = loss_map["comprehension"] * 0.0
                elif sid is not None:
                    sr_loss, sr_due, sr_total = self._phase_ops["spaced_review"].forward(
                        self.model,
                        phase_batch,
                        epoch=epoch,
                        fsrs=self.fsrs_scheduler,
                        class_weights=self.class_weights,
                    )
                    due_hits += int(sr_due)
                    due_total += int(sr_total)
                    loss_map["spaced_review"] = sr_loss
                else:
                    loss_map["spaced_review"] = loss_map["comprehension"] * 0.0

                if self.phase_ablation_flags.get("no_errorcorr", False):
                    loss_map["error_correction"] = loss_map["comprehension"] * 0.0
                else:
                    loss_map["error_correction"] = self._phase_ops["error_correction"].forward(
                        self.model, phase_batch, class_weights=self.class_weights
                    )
                loss = sum(phase_weights[p] * loss_map[p] for p in self.phase_names)

                if self.rank_loss_weight > 0.0:
                    logits_comp = self._forward_logits(x, stage="comprehension", sample_ids=sid)
                    rank_loss = self._binary_rank_loss(logits_comp, y)
                    loss = loss + float(self.rank_loss_weight) * rank_loss

            scaled_loss = loss / self.grad_accum_steps
            scaler_enabled = bool(self.scaler is not None and self.scaler.is_enabled())
            if scaler_enabled:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step_count % self.grad_accum_steps == 0:
                if scaler_enabled:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if scaler_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            bs = int(y.shape[0])
            total_loss += float(loss.detach().item()) * bs
            total_samples += bs
            total_nonzero += float((x != 0).float().sum().item())
            total_elems += float(x.numel())
            for p in self.phase_names:
                phase_counts[p] += 1
                phase_loss_sums[p] += float(loss_map[p].detach().item())
            self._sample_hw_stats(step=step_count, samples=total_samples)

            if self.enable_cpu_offload and self.device.type == "cuda":
                torch.cuda.empty_cache()
            step_elapsed = time.perf_counter() - _step_t0
            step_times.append(step_elapsed)
            avg_step_time_sec = float(sum(step_times) / len(step_times))
            eta_sec = float(avg_step_time_sec * max(0, total_steps - step_count))
            progress_pct = float((step_count / max(1, total_steps)) * 100.0)
            hw_now = self.get_hardware_report()
            progress_item = {
                "epoch": int(epoch),
                "step": int(step_count),
                "total_steps": int(total_steps),
                "progress_pct": progress_pct,
                "loss": float(loss.detach().item()),
                "lr": float(self.scheduler.get_last_lr()[0]),
                "throughput_samples_per_step": float(hw_now["throughput_samples_per_step"]),
                "peak_gpu_mem_mb": float(hw_now["peak_gpu_mem_mb"]),
                "avg_step_time_sec": avg_step_time_sec,
                "eta_sec": eta_sec,
            }
            self.last_progress.append(progress_item)
            self._emit_progress(progress_item, is_final_step=(step_count == total_steps))

            if (time.time() - self.last_checkpoint_walltime) >= self.checkpoint_every_minutes * 60.0:
                auto_dir = Path("checkpoints") / "auto"
                auto_dir.mkdir(parents=True, exist_ok=True)
                auto_path = auto_dir / f"auto_ep{epoch}_step{step_count}.pt"
                self.save_checkpoint(auto_path, epoch=epoch, metrics={"avg_loss": float(loss.detach().item())})
                self.last_checkpoint_walltime = time.time()

        if total_samples == 0:
            raise ValueError("dataloader 为空，无法训练")

        if step_count % self.grad_accum_steps != 0:
            scaler_enabled = bool(self.scaler is not None and self.scaler.is_enabled())
            if scaler_enabled:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if scaler_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.scheduler.step()
        avg_loss = total_loss / total_samples
        sparsity = 1.0 - (total_nonzero / max(1.0, total_elems))
        density = 1.0 - sparsity
        total_phase_steps = sum(phase_counts.values())
        phase_distribution = {k: (v / total_phase_steps if total_phase_steps > 0 else 0.0) for k, v in phase_counts.items()}
        phase_losses = {k: (phase_loss_sums[k] / max(1, phase_counts[k])) for k in self.phase_names}
        hw = self.get_hardware_report()
        final_avg_step_time = float(sum(step_times) / len(step_times)) if step_times else 0.0
        due_hit_ratio = float(due_hits / due_total) if due_total > 0 else 0.0
        metrics = {
            "avg_step_time_sec": final_avg_step_time,
            "eta_sec": 0.0,
            "avg_loss": float(avg_loss),
            "sparsity": float(sparsity),
            "density": float(density),
            "phase_distribution": phase_distribution,
            "phase_losses": phase_losses,
            "spaced_review_due_hits": int(due_hits),
            "spaced_review_due_total": int(due_total),
            "spaced_review_due_ratio": due_hit_ratio,
            "lr": float(self.scheduler.get_last_lr()[0]),
            "peak_gpu_mem_mb": float(hw["peak_gpu_mem_mb"]),
            "within_target": bool(hw["within_target"]),
            "throughput_samples_per_step": float(hw["throughput_samples_per_step"]),
        }
        self.phase_history.append({"epoch": int(epoch), **metrics})

        if avg_loss < self._best_loss - 1e-12:
            self._best_loss = float(avg_loss)
            self._no_improve_epochs = 0
        else:
            self._no_improve_epochs += 1
        self.should_stop = self._no_improve_epochs >= self.early_stopping_patience
        return metrics

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict) -> None:
        """保存检查点（模型/优化器/调度器/epoch/metrics/历史）。"""
        payload = {
            "epoch": int(epoch),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": dict(metrics),
            "phase_history": self.phase_history,
            "fsrs_state": self.fsrs_state,
            "seed": self.seed,
            "config": self.config,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> int:
        """加载检查点并恢复状态，返回保存时 epoch。"""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.phase_history = list(ckpt.get("phase_history", []))
        self.fsrs_state = dict(ckpt.get("fsrs_state", {}))
        return int(ckpt["epoch"])
