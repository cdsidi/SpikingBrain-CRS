"""硬件监控实现（轻量，无重依赖）。"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List


class HardwareMonitor:
    """nvidia-smi 监控封装。

    记录时间、显存占用（MB）和简单吞吐估计。
    """

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = int(log_interval)
        self._records: List[Dict] = []
        self._started = False
        self._start_t = 0.0

    def start_monitoring(self) -> None:
        """开始监控会话。"""
        self._started = True
        self._start_t = time.time()
        self._records = []

    def sample(self, steps: int = 0, samples: int = 0) -> Dict:
        """采样一次硬件状态并记录。"""
        mem_mb = self._query_gpu_mem_mb()
        rec = {
            "t": time.time(),
            "elapsed_sec": time.time() - self._start_t if self._started else 0.0,
            "gpu_mem_mb": mem_mb,
            "steps": int(steps),
            "samples": int(samples),
        }
        self._records.append(rec)
        return rec

    def get_report(self) -> Dict:
        """返回监控报告。"""
        if not self._records:
            return {"samples": 0, "peak_gpu_mem_mb": 0.0, "avg_gpu_mem_mb": 0.0, "throughput_samples_per_sec": 0.0}
        mems = [float(r["gpu_mem_mb"]) for r in self._records]
        elapsed = max(1e-6, float(self._records[-1]["elapsed_sec"]))
        total_samples = int(self._records[-1].get("samples", 0))
        return {
            "samples": len(self._records),
            "peak_gpu_mem_mb": max(mems),
            "avg_gpu_mem_mb": sum(mems) / len(mems),
            "throughput_samples_per_sec": total_samples / elapsed,
            "target_vram_mb": 4200.0,
            "within_target": max(mems) < 4200.0,
        }

    def save_report(self, path: Path) -> None:
        """保存监控报告为 JSON。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.get_report(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _query_gpu_mem_mb() -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.0,
            )
            vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
            return max(vals) if vals else 0.0
        except Exception:
            return 0.0
