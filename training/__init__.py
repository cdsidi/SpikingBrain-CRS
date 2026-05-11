"""训练模块（含 Layer 6 硬件适配）。"""

from .trainer import CRSMedicalTrainer
from .phase_scheduler import PhaseScheduler
from .checkpoint import save_training_checkpoint, load_training_checkpoint
from .hardware_adaptation import MemoryOptimizer, RTX4050Config

__all__ = [
    "CRSMedicalTrainer",
    "PhaseScheduler",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "MemoryOptimizer",
    "RTX4050Config",
]
