"""Clean Stage-1 RGB+Action base with every GS branch structurally removed."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_strictsuccess_explicitLRC_v4b.py"
)
spec = importlib.util.spec_from_file_location("old12_frozen_reference_nogs", SOURCE)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)

common = legacy.common
set_seed = legacy.set_seed
strict_label = legacy.strict_label
read_rows = legacy.read_rows


def prepare_stats(train_csv, out_path):
    """Fit Action13 statistics without opening a GS NPZ file."""
    rows = read_rows(train_csv)
    ag = np.stack([common.build_action_geom(row) for row in rows], axis=0)
    np.savez(
        out_path,
        ag_mean=ag.mean(axis=0).astype(np.float32),
        ag_std=(ag.std(axis=0) + 1e-6).astype(np.float32),
    )
    print("[saved clean noGS stats]", out_path)
    print("action dim:", ag.shape[1], "GS dim: absent")


class LRCDataset(Dataset):
    """Load only RGB and Action13; local_gs_path is never accessed."""

    def __init__(self, csv_path, stats_path, image_size=224):
        self.rows = read_rows(csv_path)
        self.image_size = image_size
        stats = np.load(stats_path)
        self.ag_mean = stats["ag_mean"]
        self.ag_std = stats["ag_std"]
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def rgb(self, path):
        return legacy.LRCDataset.rgb(self, path)

    def __getitem__(self, index):
        row = self.rows[index]
        action = (common.build_action_geom(row) - self.ag_mean) / self.ag_std
        return {
            "rgb": torch.from_numpy(self.rgb(row["vision_rgb_path"])),
            "action_geom": torch.from_numpy(action.astype(np.float32)),
            "label": torch.tensor(strict_label(row), dtype=torch.float32),
            "object_id": torch.tensor(int(row["object_id"]), dtype=torch.long),
        }


class ExplicitLRCModel(nn.Module):
    """RGB256 + Action64 only; no endpoint/corridor/relation GS modules."""

    def __init__(self, no_gs=False):
        super().__init__()
        if not no_gs:
            raise RuntimeError("clean noGS source requires --no_gs")
        self.no_gs = True
        self.rgb_encoder = common.RGBEncoder(out_dim=256)
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(320, 256), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, rgb, action_geom):
        rgb_code = self.rgb_encoder(rgb)
        action_code = self.action_encoder(action_geom)
        return self.head(torch.cat([rgb_code, action_code], dim=1)).squeeze(1)


move = legacy.move


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    ys, ps, objects = [], [], []
    total = 0.0
    count = 0
    for batch in loader:
        moved = move(batch, device)
        logit = model(moved["rgb"], moved["action_geom"])
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
