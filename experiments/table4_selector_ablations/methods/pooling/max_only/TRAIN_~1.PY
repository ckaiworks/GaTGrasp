"""Deterministic Stage-1 entry for the clean Max-Only pooling ablation."""
import importlib.util
import sys
from pathlib import Path


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
CORE = _load(
    "stage1_full_annealed_core_for_max_ablation",
    Path(__file__).resolve().parents[5] / "src" / "stage1" / "final" / "train_stage1_old12_fullcorridor_annealedSelect_v1.py",
)
BASE = _load("stage1_max_only_base", Path(__file__).with_name("max_only_base_v1.py"))
forward = CORE.forward
CORE.BASE = BASE

if __name__ == "__main__":
    CORE.main()
