"""FSRS 调度器实现（参数化轻量版）。

说明：
- 保留轻量可运行特性；
- 支持完整参数向量 w（默认 19 个参数），用于近似 FSRS 更新；
- 兼容旧接口（仅传 request_retention 也可工作）。
"""

from __future__ import annotations

from typing import Dict, List

import torch


class FSRSScheduler:
    """间隔重复调度器（参数化）。"""

    def __init__(self, config: Dict | None = None, request_retention: float = 0.9) -> None:
        cfg = dict(config or {})
        self.request_retention = float(cfg.get("request_retention", request_retention))
        self.desired_retention = float(cfg.get("desired_retention", self.request_retention))
        self.max_interval = int(cfg.get("max_interval", 365))
        self.current_epoch = 0
        self.sm2_initial_interval = int(cfg.get("sm2_initial_interval", 1))
        self.sm2_easiness_factor = float(cfg.get("sm2_easiness_factor", 2.5))
        self.w: List[float] = list(cfg.get("w", [
            0.4, 0.6, 2.4, 5.8, 4.9, 0.8, 1.6, 0.2, 1.3, 0.14,
            0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61, 0.11, 0.31,
        ]))
        if len(self.w) < 19:
            self.w.extend([0.1] * (19 - len(self.w)))
        self.card_states: Dict[int, Dict[str, float | int]] = {}

    def get_due_samples(self, epoch: int, sample_ids: torch.Tensor) -> torch.Tensor:
        self.current_epoch = int(epoch)
        due_positions = []
        for pos, sid in enumerate(sample_ids.detach().cpu().tolist()):
            sid = int(sid)
            if sid not in self.card_states:
                self.card_states[sid] = {
                    "difficulty": 5.0,
                    "stability": float(self.sm2_initial_interval),
                    "last_review": -1,
                    "interval": self.sm2_initial_interval,
                }
                due_positions.append(pos)
                continue
            s = self.card_states[sid]
            elapsed = int(epoch) - int(s["last_review"])
            if elapsed >= int(s["interval"]):
                due_positions.append(pos)
        return torch.tensor(due_positions, dtype=torch.long, device=sample_ids.device)

    def update_intervals(
        self,
        sample_ids: torch.Tensor,
        correct: torch.Tensor,
        confidence: torch.Tensor,
        ratings: torch.Tensor | None = None,
    ) -> None:
        ids = sample_ids.detach().cpu().tolist()
        corr = correct.detach().to(torch.float32).cpu().tolist()
        confs = confidence.detach().to(torch.float32).cpu().tolist()
        rates = ratings.detach().to(torch.float32).cpu().tolist() if ratings is not None else None

        for i, (sid, c, cf) in enumerate(zip(ids, corr, confs)):
            sid = int(sid)
            state = self.card_states.get(
                sid,
                {"difficulty": 5.0, "stability": float(self.sm2_initial_interval), "last_review": -1, "interval": self.sm2_initial_interval},
            )
            old_s = max(0.1, float(state["stability"]))
            old_d = min(10.0, max(1.0, float(state["difficulty"])))
            cf = max(0.0, min(1.0, float(cf)))
            rating = float(rates[i]) if rates is not None else (4.0 if c >= 0.5 else 2.0)

            if c >= 0.5:
                gain = self.w[8] + self.w[9] * cf + self.w[10] * (rating - 3.0)
                new_s = old_s * (1.0 + max(0.05, gain))
                new_d = old_d - (self.w[6] * (rating - 3.0) + self.w[7] * cf)
            else:
                penalty = self.w[11] + self.w[12] * (1.0 - cf) + self.w[13] * (3.0 - rating)
                new_s = max(0.1, old_s * (1.0 - min(0.95, penalty)))
                new_d = old_d + (self.w[14] * (3.0 - rating) + self.w[15] * (1.0 - cf))

            new_d = min(10.0, max(1.0, new_d))
            interval = int(max(1, round(new_s / max(1e-6, (1.0 - self.desired_retention)))))
            state["stability"] = float(new_s)
            state["difficulty"] = float(new_d)
            state["last_review"] = int(self.current_epoch)
            state["interval"] = int(min(self.max_interval, interval))
            self.card_states[sid] = state
