"""最小导入冒烟测试（仅验证骨架可导入）。"""

from data import LC25000Refiner, PhysioNetRefiner, IUXRayRefiner, MedicalDataLoader
from models.core import GatedLinearAttention, SlidingWindowAttention, AdaptiveThresholdSpiking, SpikingFFN
from models.crs import (
    ComprehensionPhase,
    RecallPhase,
    SynthesisPhase,
    SpacedReviewPhase,
    ErrorCorrectionPhase,
    FSRSScheduler,
    MetacognitiveMonitor,
)
from training import CRSMedicalTrainer, PhaseScheduler
from evaluation import MedicalMetrics, HardwareMonitor, SparsityAnalyzer, StatisticalTesting
from experiments import BaselineExperiment, AblationStudy, LongTermMemoryTest, FullExperimentSuite


def test_imports() -> None:
    assert LC25000Refiner is not None
    assert GatedLinearAttention is not None
    assert ComprehensionPhase is not None
    assert CRSMedicalTrainer is not None
    assert MedicalMetrics is not None
    assert FullExperimentSuite is not None
