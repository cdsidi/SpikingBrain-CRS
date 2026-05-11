"""实验模块（Layer 4 + Layer 7）。"""

from .baseline_experiment import BaselineExperiment
from .crs_experiment import CRSExperiment
from .full_experiment_suite import FullExperimentSuite

__all__ = [
    "BaselineExperiment",
    "CRSExperiment",
    "AblationStudy",
    "ComponentMask",
    "LongTermMemoryTest",
    "FullExperimentSuite",
]


def __getattr__(name: str):
    """延迟导入 Layer7，避免 `python -m experiments.xxx` 运行时 runpy 告警。"""
    if name in {"AblationStudy", "ComponentMask"}:
        from .ablation_study import AblationStudy, ComponentMask

        return {"AblationStudy": AblationStudy, "ComponentMask": ComponentMask}[name]
    if name == "LongTermMemoryTest":
        from .long_term_memory_test import LongTermMemoryTest

        return LongTermMemoryTest
    raise AttributeError(f"module 'experiments' has no attribute {name!r}")
