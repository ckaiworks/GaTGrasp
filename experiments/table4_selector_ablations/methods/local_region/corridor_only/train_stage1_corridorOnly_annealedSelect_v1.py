#!/usr/bin/env python3
"""Current deterministic Stage-1 loss for clean Corridor-Only GS input."""

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
    "stage1_fullcorridor_core_for_corridor_only",
    HERE.parents[4]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_old12_fullcorridor_annealedSelect_v1.py",
)
CORRIDOR_ONLY = load_module(
    "stage1_corridor_only_base_v1",
    HERE / "corridor_only_base_v1.py",
)


def corridor_only_forward(model, batch):
    return CORRIDOR_ONLY.model_forward(model, batch)


CORE.BASE = CORRIDOR_ONLY
CORE.forward = corridor_only_forward


if __name__ == "__main__":
    CORE.main()
