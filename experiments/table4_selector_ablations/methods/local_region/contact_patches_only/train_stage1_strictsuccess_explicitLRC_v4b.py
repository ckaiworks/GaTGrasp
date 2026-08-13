"""Clean Stage-1 RGB+Action+contact-GS base.

This variant structurally removes the closing-corridor branch.  It reuses only
the audited data/metric helpers from the frozen Old12 implementation.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


SOURCE = (
    Path(__file__).resolve().parents[5]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_strictsuccess_explicitLRC_v4b.py"
)
spec = importlib.util.spec_from_file_location("old12_frozen_reference", SOURCE)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)

common = legacy.common
set_seed = legacy.set_seed
strict_label = legacy.strict_label
read_rows = legacy.read_rows


def prepare_stats(train_csv, out_path):
    """Fit Action13 and Old12 statistics without reading corridor features."""
    rows = read_rows(train_csv)
    ag = np.stack([common.build_action_geom(row) for row in rows], axis=0)
    sums = np.zeros((12,), dtype=np.float64)
    sqs = np.zeros((12,), dtype=np.float64)
    count = 0
    for row in rows:
        data = np.load(row["local_gs_path"], allow_pickle=False)
        for feat_key, mask_key in (
            ("left_feat", "left_mask"),
            ("right_feat", "right_mask"),
        ):
            feat = data[feat_key].astype(np.float32)
            valid = data[mask_key].astype(np.float32) > 0.5
            take = feat[valid]
            sums += take.sum(axis=0)
            sqs += (take ** 2).sum(axis=0)
            count += len(take)
    mean = sums / max(count, 1)
    var = sqs / max(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12)) + 1e-6
    np.savez(
        out_path,
        ag_mean=ag.mean(axis=0).astype(np.float32),
        ag_std=(ag.std(axis=0) + 1e-6).astype(np.float32),
        lrc_mean=mean.astype(np.float32),
        lrc_std=std.astype(np.float32),
    )
    print("[saved clean contact-only stats]", out_path)
    print("action/LRC dims:", ag.shape[1], mean.shape[0])


class LRCDataset(Dataset):
    """Load RGB, Action13, and only the left/right GS contact patches."""

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
        return legacy.LRCDataset.rgb(self, path)

    def patch(self, data, feat_key, mask_key):
        mask = data[mask_key].astype(np.float32)
        feat = data[feat_key].astype(np.float32)
        feat = (feat - self.lrc_mean[None]) / self.lrc_std[None]
        feat *= mask[:, None]
        return feat.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, index):
        row = self.rows[index]
        data = np.load(row["local_gs_path"], allow_pickle=False)
        left, left_mask = self.patch(data, "left_feat", "left_mask")
        right, right_mask = self.patch(data, "right_feat", "right_mask")
        action = (common.build_action_geom(row) - self.ag_mean) / self.ag_std
        return {
            "rgb": torch.from_numpy(self.rgb(row["vision_rgb_path"])),
            "action_geom": torch.from_numpy(action.astype(np.float32)),
            "left": torch.from_numpy(left),
            "right": torch.from_numpy(right),
            "left_mask": torch.from_numpy(left_mask),
            "right_mask": torch.from_numpy(right_mask),
            "label": torch.tensor(strict_label(row), dtype=torch.float32),
            "object_id": torch.tensor(int(row["object_id"]), dtype=torch.long),
        }


PatchEncoder = legacy.PatchEncoder


class ExplicitLRCModel(nn.Module):
    """RGB256 + Action64 + structural contact-GS128; no corridor module."""

    def __init__(self, no_gs=False):
        super().__init__()
        if no_gs:
            raise RuntimeError("clean withGS source refuses --no_gs")
        self.no_gs = False
        self.rgb_encoder = common.RGBEncoder(out_dim=256)
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.endpoint_encoder = PatchEncoder()
        self.relation = nn.Sequential(
            nn.Linear(64 * 4, 128), nn.ReLU(inplace=True), nn.Dropout(0.25)
        )
        self.head = nn.Sequential(
            nn.Linear(448, 256), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, rgb, action_geom, left, right, left_mask, right_mask):
        left_code = self.endpoint_encoder(left, left_mask)
        right_code = self.endpoint_encoder(right, right_mask)
        gs = self.relation(torch.cat([
            left_code,
            right_code,
            torch.abs(left_code - right_code),
            left_code * right_code,
        ], dim=1))
        rgb_code = self.rgb_encoder(rgb)
        action_code = self.action_encoder(action_geom)
        return self.head(torch.cat([rgb_code, action_code, gs], dim=1)).squeeze(1)


move = legacy.move


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    ys, ps, objects = [], [], []
    total = 0.0
    count = 0
    for batch in loader:
        moved = move(batch, device)
        logit = model(
            moved["rgb"], moved["action_geom"],
            moved["left"], moved["right"],
            moved["left_mask"], moved["right_mask"],
        )
        loss = criterion(logit, moved["label"])
        probability = torch.sigmoid(logit)
        total += float(loss.item()) * len(probability)
        count += len(probability)
        ys.extend(moved["label"].cpu().numpy().tolist())
        ps.extend(probability.cpu().numpy().tolist())
        objects.extend(moved["object_id"].cpu().numpy().tolist())
    result = common.safe_metrics(
        np.asarray(ys, dtype=np.int64),
        np.asarray(ps, dtype=np.float64),
        np.asarray(objects, dtype=np.int64),
    )
    result["loss"] = total / max(count, 1)
    return result
