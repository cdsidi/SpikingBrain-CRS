"""完整实验套件（Layer 4 最小编排实现）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from .baseline_experiment import BaselineExperiment
from .crs_experiment import CRSExperiment


class FullExperimentSuite:
    """执行 4模型×3数据集×5折×5种子 的实验协议。

    功能: 计划加载、执行、失败重试、状态落盘、断点续跑、结果汇总。
    """

    def __init__(self, results_root: Path | None = None, max_retries: int = 1) -> None:
        self.results_root = results_root or Path("results") / "experiment_protocol"
        self.max_retries = max(0, int(max_retries))
        self.allowed_seeds = [1, 42, 123, 456, 2024]
        self.models = ["T-Base", "T-CRS", "S-Base", "S-CRS"]
        self.datasets = ["lc25000", "physionet", "iu_xray"]
        self.n_folds = 5

    def _state_path(self) -> Path:
        return self.results_root / "state.json"

    def _summary_path(self) -> Path:
        return self.results_root / "summary.json"

    def _plan(self) -> List[Dict]:
        jobs = []
        for model in self.models:
            for dataset in self.datasets:
                jobs.append({"model": model, "dataset": dataset, "n_folds": self.n_folds})
        return jobs

    def run_48h_schedule(self, quick_smoke: bool = True, max_batches: int = 1) -> Path:
        """运行实验计划并返回结果目录。"""
        self.results_root.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        jobs = self._plan()
        logs = []
        for idx, job in enumerate(jobs):
            key = f"{job['model']}|{job['dataset']}"
            if key in state.get("done", {}):
                logs.append({"job": key, "status": "skipped_resumed"})
                continue
            result = self._run_with_retry(job, quick_smoke=quick_smoke, max_batches=max_batches)
            state.setdefault("done", {})[key] = result
            state["last_index"] = idx
            self._save_state(state)
            logs.append({
                "job": key,
                "status": "done",
                "avg_score": float(result.get("avg_score", 0.0)),
                "failures": len(result.get("failures", [])),
            })
        summary = {
            "protocol": "07_EXPERIMENT_PROTOCOL",
            "mode": "real_training",
            "quick_smoke": bool(quick_smoke),
            "timestamp": int(time.time()),
            "jobs_total": len(jobs),
            "jobs_done": len(state.get("done", {})),
            "results": state.get("done", {}),
            "logs": logs,
        }
        self._summary_path().write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.results_root

    def _run_with_retry(self, job: Dict, quick_smoke: bool = True, max_batches: int = 1) -> Dict:
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                if "CRS" in job["model"]:
                    return CRSExperiment().run(
                        job["model"],
                        job["dataset"],
                        job["n_folds"],
                        self.allowed_seeds,
                        output_root=self.results_root / "runs",
                        quick_smoke=quick_smoke,
                        max_batches=max_batches,
                    )
                return BaselineExperiment().run(
                    job["model"],
                    job["dataset"],
                    job["n_folds"],
                    self.allowed_seeds,
                    output_root=self.results_root / "runs",
                    quick_smoke=quick_smoke,
                    max_batches=max_batches,
                )
            except Exception as e:  # pragma: no cover
                last_err = str(e)
        return {
            "model": job["model"],
            "dataset": job["dataset"],
            "avg_score": 0.0,
            "failures": [{"error": last_err}],
            "error": last_err,
        }

    def _load_state(self) -> Dict:
        p = self._state_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"done": {}, "last_index": -1}

    def _save_state(self, state: Dict) -> None:
        self._state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
