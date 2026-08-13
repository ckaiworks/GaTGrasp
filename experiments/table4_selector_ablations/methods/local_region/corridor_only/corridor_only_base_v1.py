#!/usr/bin/env python3
"""Clean local-region ablation retaining only corridor GS nodes."""

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
    "stage1_full_base_for_corridor_only",
    HERE.parents[4]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_strictsuccess_explicitLRC_v4b.py",
)
common = FULL.common
set_seed = FULL.set_seed
strict_label = FULL.strict_label
read_rows = FULL.read_rows
PatchEncoder = FULL.PatchEncoder


def prepare_stats(train_csv, out_path):
    rows = read_rows(train_csv)
    action = np.stack([common.build_action_geom(row) for row in rows], axis=0)
    sums = np.zeros((12,), dtype=np.float64)
    squares = np.zeros((12,), dtype=np.float64)
    count = 0
    for row in rows:
        data = np.load(row["local_gs_path"], allow_pickle=False)
        feature = data["corridor_feat"].astype(np.float32)
        valid = data["corridor_mask"].astype(np.float32) > 0.5
        selected = feature[valid]
        sums += selected.sum(axis=0)
        squares += (selected ** 2).sum(axis=0)
        count += len(selected)
    mean = sums / max(count, 1)
    variance = squares / max(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(variance, 1e-12)) + 1e-6
    np.savez(
        out_path,
        ag_mean=action.mean(axis=0).astype(np.float32),
        ag_std=(action.std(axis=0) + 1e-6).astype(np.float32),
        lrc_mean=mean.astype(np.float32),
        lrc_std=std.astype(np.float32),
    )
    print("[saved Corridor-Only stats]", out_path)
    print("action/corridor dims:", action.shape[1], mean.shape[0])


class LRCDataset(Dataset):
    """Load RGB, Action13 and corridor GS; contact patches are never read."""

    def __init__(self, csv_path, stats_path, image_size=224):
        self.rows = read_rows(csv_path)
        self.image_size = image_size
        stats = np.load(stats_path)
        self.ag_mean = stats["ag_mean"]
        self.ag_std = stats["ag_std"]
        self.lrc_mean = stats["lrc_mean"]
        self.lrc_std = stats["lrc_std"]
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def rgb(self, path):
        image = FULL.Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), FULL.Image.BILINEAR
        )
        value = np.asarray(image).astype(np.float32) / 255.0
        value = (
            value - self.rgb_mean[None, None]
        ) / self.rgb_std[None, None]
        return value.transpose(2, 0, 1).astype(np.float32)

    def __getitem__(self, index):
        row = self.rows[index]
        data = np.load(row["local_gs_path"], allow_pickle=False)
        mask = data["corridor_mask"].astype(np.float32)
        corridor = data["corridor_feat"].astype(np.float32)
        corridor = (corridor - self.lrc_mean[None]) / self.lrc_std[None]
        corridor *= mask[:, None]
        action = (common.build_action_geom(row) - self.ag_mean) / self.ag_std
        return {
            "rgb": torch.from_numpy(self.rgb(row["vision_rgb_path"])),
            "action_geom": torch.from_numpy(action.astype(np.float32)),
            "corridor": torch.from_numpy(corridor.astype(np.float32)),
            "corridor_mask": torch.from_numpy(mask.astype(np.float32)),
            "label": torch.tensor(strict_label(row), dtype=torch.float32),
            "object_id": torch.tensor(int(row["object_id"]), dtype=torch.long),
        }


class ExplicitLRCModel(nn.Module):
    """RGB+Action+Corridor with contact encoders/relation structurally absent."""

    def __init__(self, no_gs=False):
        super().__init__()
        if no_gs:
            raise RuntimeError("Corridor-Only ablation requires corridor GS")
        self.rgb_encoder = common.RGBEncoder(out_dim=256)
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.corridor_encoder = PatchEncoder()
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, rgb, action_geom, corridor, corridor_mask):
        rgb_code = self.rgb_encoder(rgb)
        action_code = self.action_encoder(action_geom)
        corridor_code = self.corridor_encoder(corridor, corridor_mask)
        return self.head(
            torch.cat([rgb_code, action_code, corridor_code], dim=1)
        ).squeeze(1)


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def model_forward(model, batch):
    return model(
        batch["rgb"],
        batch["action_geom"],
        batch["corridor"],
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
