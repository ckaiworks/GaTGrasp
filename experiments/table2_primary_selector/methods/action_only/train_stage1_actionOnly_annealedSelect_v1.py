#!/usr/bin/env python3
"""Current deterministic Stage-1 loss applied to a clean Action-Only model."""

import importlib.util
import sys
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
CORE = load_module(
    "stage1_fullcorridor_core_for_action_only",
    HERE.parents[3]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_old12_fullcorridor_annealedSelect_v1.py",
)
ACTION_ONLY = load_module(
    "stage1_action_only_base_v1",
    HERE / "action_only_base_v1.py",
)


def action_only_forward(model, batch):
    return model(batch["action_geom"])


# The training/loss/checkpoint logic remains the current final implementation;
# only its dataset/model/forward interface is replaced by the clean ablation.
CORE.BASE = ACTION_ONLY
CORE.forward = action_only_forward


if __name__ == "__main__":
    CORE.main()
