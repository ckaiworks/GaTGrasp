#!/usr/bin/env python3
"""Run an existing Stage-1 entry under a strict deterministic contract."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def parse_wrapper_args() -> tuple[Path, int, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--deterministic_seed", required=True, type=int)
    args, remaining = parser.parse_known_args()
    return Path(args.entry).resolve(), args.deterministic_seed, remaining


entry, run_seed, target_argv = parse_wrapper_args()
if not entry.is_file():
    raise RuntimeError(f"MISSING_ENTRY={entry}")

# cuBLAS reads this setting when the CUDA runtime is initialized, so it must be
# present before importing torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(run_seed)

import numpy as np
import torch
import torch.utils.data

random.seed(run_seed)
np.random.seed(run_seed)
torch.manual_seed(run_seed)
torch.cuda.manual_seed_all(run_seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

original_dataloader = torch.utils.data.DataLoader
loader_index = 0


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def deterministic_dataloader(*args, **kwargs):
    global loader_index
    generator = torch.Generator()
    generator.manual_seed(run_seed + loader_index)
    loader_index += 1
    kwargs.setdefault("generator", generator)
    kwargs.setdefault("worker_init_fn", seed_worker)
    return original_dataloader(*args, **kwargs)


torch.utils.data.DataLoader = deterministic_dataloader

contract = {
    "entry": str(entry),
    "run_seed": run_seed,
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cudnn_allow_tf32": False,
    "cuda_matmul_allow_tf32": False,
    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "torch_num_threads": torch.get_num_threads(),
    "torch_num_interop_threads": torch.get_num_interop_threads(),
    "dataloader_generator": "run_seed_plus_loader_index",
    "dataloader_worker_seed": "torch.initial_seed modulo 2**32",
}
print("DETERMINISTIC_EXECUTION_CONTRACT=", contract, flush=True)

sys.argv = [str(entry), *target_argv]
runpy.run_path(str(entry), run_name="__main__")
