from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models import build_model
from training.train_crs import _load_yaml_config


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_s_crs_param_budget_under_2_5m() -> None:
    cfg = _load_yaml_config(Path("configs/s_crs.yaml"))
    cfg["num_classes"] = 5
    cfg["input_dim"] = 4096
    model = build_model("s_crs", cfg)
    num_params = _count_params(model)
    assert num_params <= 2_500_000, f"s_crs params too high: {num_params}"


if __name__ == "__main__":
    test_s_crs_param_budget_under_2_5m()
    print("PARAM_BUDGET_TEST_OK")

