"""五阶段调度器（最小可运行实现）。"""

from typing import Dict


class PhaseScheduler:
    """按 epoch 返回五阶段权重。"""

    def __init__(self) -> None:
        self.base_weights = {
            "comprehension": 0.3,
            "recall": 0.25,
            "synthesis": 0.2,
            "spaced_review": 0.15,
            "error_correction": 0.1,
        }

    def get_phase_weights(self, epoch: int) -> Dict[str, float]:
        e = int(epoch)
        if e < 5:
            weights = {
                "comprehension": 0.5,
                "recall": 0.3,
                "synthesis": 0.2,
                "spaced_review": 0.0,
                "error_correction": 0.0,
            }
        elif e < 15:
            weights = dict(self.base_weights)
        else:
            weights = {
                "comprehension": 0.1,
                "recall": 0.1,
                "synthesis": 0.1,
                "spaced_review": 0.4,
                "error_correction": 0.3,
            }
        s = sum(weights.values())
        return {k: (v / s if s > 0 else 0.0) for k, v in weights.items()}
