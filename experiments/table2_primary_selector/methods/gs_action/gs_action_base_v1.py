#!/usr/bin/env python3
"""Clean GS+Action base: no RGB data path and no RGB encoder."""

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
    "stage1_full_base_for_gs_action",
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
prepare_stats = FULL.prepare_stats
PatchEncoder = FULL.PatchEncoder


class LRCDataset(Dataset):
    """Load Action13 and all three GS regions, but never load RGB."""

    def __init__(self, csv_path, stats_path, image_size=224):
        del image_size
        self.rows = read_rows(csv_path)
        stats = np.load(stats_path)
        self.ag_mean = stats["ag_mean"]
        self.ag_std = stats["ag_std"]
        self.lrc_mean = stats["lrc_mean"]
        self.lrc_std = stats["lrc_std"]

    def __len__(self):
        return len(self.rows)

    def patch(self, data, feature_key, mask_key):
        mask = data[mask_key].astype(np.float32)
        feature = data[feature_key].astype(np.float32)
        feature = (feature - self.lrc_mean[None]) / self.lrc_std[None]
        feature *= mask[:, None]
        return feature.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, index):
        row = self.rows[index]
        data = np.load(row["local_gs_path"], allow_pickle=False)
        left, left_mask = self.patch(data, "left_feat", "left_mask")
        right, right_mask = self.patch(data, "right_feat", "right_mask")
        corridor, corridor_mask = self.patch(
            data, "corridor_feat", "corridor_mask"
        )
        action = (common.build_action_geom(row) - self.ag_mean) / self.ag_std
        return {
            "action_geom": torch.from_numpy(action.astype(np.float32)),
            "left": torch.from_numpy(left),
            "right": torch.from_numpy(right),
            "corridor": torch.from_numpy(corridor),
            "left_mask": torch.from_numpy(left_mask),
            "right_mask": torch.from_numpy(right_mask),
            "corridor_mask": torch.from_numpy(corridor_mask),
            "label": torch.tensor(strict_label(row), dtype=torch.float32),
            "object_id": torch.tensor(int(row["object_id"]), dtype=torch.long),
        }


class ExplicitLRCModel(nn.Module):
    """The final Action and GS branches with the RGB branch structurally absent."""

    def __init__(self, no_gs=False):
        super().__init__()
        if no_gs:
            raise RuntimeError("GS+Action ablation requires GS; do not pass --no_gs")
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.endpoint_encoder = PatchEncoder()
        self.corridor_encoder = PatchEncoder()
        self.relation = nn.Sequential(
            nn.Linear(64 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        action_geom,
        left,
        right,
        corridor,
        left_mask,
        right_mask,
        corridor_mask,
    ):
        left_code = self.endpoint_encoder(left, left_mask)
        right_code = self.endpoint_encoder(right, right_mask)
        corridor_code = self.corridor_encoder(corridor, corridor_mask)
        gs_code = self.relation(
            torch.cat(
                [
                    left_code,
                    right_code,
                    corridor_code,
                    torch.abs(left_code - right_code),
                    left_code * right_code,
                ],
                dim=1,
            )
        )
        action_code = self.action_encoder(action_geom)
        return self.head(torch.cat([action_code, gs_code], dim=1)).squeeze(1)


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def model_forward(model, batch):
    return model(
        batch["action_geom"],
        batch["left"],
        batch["right"],
        batch["corridor"],
        batch["left_mask"],
        batch["right_mask"],
        batch["corridor_mask"],
    )


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
        logits = model_forward(model, moved)
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
