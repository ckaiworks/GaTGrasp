#!/usr/bin/env python3
"""Clean Action-Only data path and model for the current Stage-1 ablation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
FULL = load_module(
    "stage1_full_base_for_action_only",
    HERE.parents[3]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_strictsuccess_explicitLRC_v4b.py",
)
common = FULL.common
set_seed = FULL.set_seed
strict_label = FULL.strict_label
read_rows = FULL.read_rows


def prepare_stats(train_csv, out_path):
    rows = read_rows(train_csv)
    action = np.stack([common.build_action_geom(row) for row in rows], axis=0)
    np.savez(
        out_path,
        ag_mean=action.mean(axis=0).astype(np.float32),
        ag_std=(action.std(axis=0) + 1e-6).astype(np.float32),
    )
    print("[saved Action-Only stats]", out_path)
    print("action dims:", action.shape[1])


class LRCDataset(Dataset):
    """Dataset intentionally loading neither RGB nor any GS artifact."""

    def __init__(self, csv_path, stats_path, image_size=224):
        del image_size
        self.rows = read_rows(csv_path)
        stats = np.load(stats_path)
        self.ag_mean = stats["ag_mean"]
        self.ag_std = stats["ag_std"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        action = (common.build_action_geom(row) - self.ag_mean) / self.ag_std
        return {
            "action_geom": torch.from_numpy(action.astype(np.float32)),
            "label": torch.tensor(strict_label(row), dtype=torch.float32),
            "object_id": torch.tensor(int(row["object_id"]), dtype=torch.long),
        }


class ExplicitLRCModel(nn.Module):
    """Only the unchanged 13D action encoder and an action-only classifier."""

    def __init__(self, no_gs=False):
        super().__init__()
        del no_gs
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, action_geom):
        action = self.action_encoder(action_geom)
        return self.head(action).squeeze(1)


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    labels = []
    probabilities = []
    object_ids = []
    total = 0.0
    count = 0
    for batch in loader:
        moved = move(batch, device)
        logits = model(moved["action_geom"])
        loss = criterion(logits, moved["label"])
        probability = torch.sigmoid(logits)
        total += float(loss.item()) * len(probability)
        count += len(probability)
        labels.extend(moved["label"].cpu().numpy().tolist())
        probabilities.extend(probability.cpu().numpy().tolist())
        object_ids.extend(moved["object_id"].cpu().numpy().tolist())
    result = common.safe_metrics(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(object_ids, dtype=np.int64),
    )
    result["loss"] = total / max(count, 1)
    return result
