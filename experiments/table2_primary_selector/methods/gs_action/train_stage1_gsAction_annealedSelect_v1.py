#!/usr/bin/env python3
"""Current deterministic Stage-1 loss applied to clean GS+Action inputs."""

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
    "stage1_fullcorridor_core_for_gs_action",
    HERE.parents[3]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_old12_fullcorridor_annealedSelect_v1.py",
)
GS_ACTION = load_module(
    "stage1_gs_action_base_v1",
    HERE / "gs_action_base_v1.py",
)


def gs_action_forward(model, batch):
    return GS_ACTION.model_forward(model, batch)


CORE.BASE = GS_ACTION
CORE.forward = gs_action_forward


if __name__ == "__main__":
    CORE.main()
