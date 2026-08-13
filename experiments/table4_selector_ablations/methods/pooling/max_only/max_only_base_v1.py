"""Clean Max-Only GS-node pooling ablation.

Everything outside PatchEncoder is inherited from the deterministic full model.
The only architectural change is replacing [masked mean, masked max] pooling
with masked max pooling and changing the following Linear input 128 -> 64.
"""
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FULL = _load(
    "stage1_full_pooling_base_for_max_ablation",
    Path(__file__).resolve().parents[5]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_strictsuccess_explicitLRC_v4b.py",
)

set_seed = FULL.set_seed
strict_label = FULL.strict_label
read_rows = FULL.read_rows
prepare_stats = FULL.prepare_stats
LRCDataset = FULL.LRCDataset
move = FULL.move
evaluate = FULL.evaluate
common = FULL.common


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

    def forward(self, feat, mask):
        b, n, c = feat.shape
        x = self.point(feat.reshape(b * n, c)).reshape(b, n, -1)
        m = mask.unsqueeze(-1)
        maximum = x.masked_fill(m < 0.5, -10000.0).max(dim=1).values
        return self.out(maximum)


class ExplicitLRCModel(nn.Module):
    def __init__(self, no_gs=False):
        super().__init__()
        self.no_gs = no_gs
        self.rgb_encoder = common.RGBEncoder(out_dim=256)
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.endpoint_encoder = PatchEncoder()
        self.corridor_encoder = PatchEncoder()
        self.relation = nn.Sequential(
            nn.Linear(64 * 5, 128), nn.ReLU(inplace=True), nn.Dropout(0.25)
        )
        self.head = nn.Sequential(
            nn.Linear(448, 256), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(
        self, rgb, action_geom, left, right, corridor,
        left_mask, right_mask, corridor_mask,
    ):
        if self.no_gs:
            left = torch.zeros_like(left)
            right = torch.zeros_like(right)
            corridor = torch.zeros_like(corridor)
            left_mask = torch.ones_like(left_mask)
            right_mask = torch.ones_like(right_mask)
            corridor_mask = torch.ones_like(corridor_mask)
        l = self.endpoint_encoder(left, left_mask)
        r = self.endpoint_encoder(right, right_mask)
        c = self.corridor_encoder(corridor, corridor_mask)
        gs = self.relation(
            torch.cat([l, r, c, torch.abs(l - r), l * r], dim=1)
        )
        rgb_feature = self.rgb_encoder(rgb)
        action_feature = self.action_encoder(action_geom)
        return self.head(
            torch.cat([rgb_feature, action_feature, gs], dim=1)
        ).squeeze(1)
