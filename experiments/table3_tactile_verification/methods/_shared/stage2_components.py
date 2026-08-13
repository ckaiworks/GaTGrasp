"""Shared data and network components for the released single-head Stage-2."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18


BASE_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_strictsuccess_explicitLRC_v4b.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module={path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("released_stage1_base", BASE_PATH)


class TouchLRCDataset(BASE.LRCDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.rows[index]
        available = float(row.get("touch_available", "0")) > 0.5
        if available:
            left = self.rgb(row["tactile_left_after_close_path"])
            right = self.rgb(row["tactile_right_after_close_path"])
        else:
            left = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            right = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        item["tactile_left"] = torch.from_numpy(left)
        item["tactile_right"] = torch.from_numpy(right)
        item["touch_available"] = torch.tensor(float(available), dtype=torch.float32)
        return item


class SharedTouchResNet18(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

    def forward(self, value):
        return self.proj(self.backbone(value))


class TouchOnceExplicitLRCModel(BASE.ExplicitLRCModel):
    """Current RGB+Action13+contact/corridor GS+Touch single-logit model."""

    def __init__(self, no_gs=False):
        super().__init__(no_gs=no_gs)
        self.touch_encoder = SharedTouchResNet18(out_dim=128)
        self.touch_relation = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def touch_features(self, tactile_left, tactile_right, touch_available):
        batch_size = tactile_left.shape[0]
        output = tactile_left.new_zeros((batch_size, 128))
        indices = torch.nonzero(touch_available > 0.5, as_tuple=False).flatten()
        if len(indices) == 0:
            return output
        left = self.touch_encoder(tactile_left.index_select(0, indices))
        right = self.touch_encoder(tactile_right.index_select(0, indices))
        relation = self.touch_relation(torch.cat([
            left,
            right,
            torch.abs(left - right),
            left * right,
        ], dim=1))
        return output.index_copy(0, indices, relation)

    def forward(
        self,
        rgb,
        action_geom,
        left,
        right,
        corridor,
        left_mask,
        right_mask,
        corridor_mask,
        tactile_left,
        tactile_right,
        touch_available,
    ):
        if self.no_gs:
            left, right, corridor = (
                torch.zeros_like(left),
                torch.zeros_like(right),
                torch.zeros_like(corridor),
            )
            left_mask, right_mask, corridor_mask = (
                torch.ones_like(left_mask),
                torch.ones_like(right_mask),
                torch.ones_like(corridor_mask),
            )
        left_feature = self.endpoint_encoder(left, left_mask)
        right_feature = self.endpoint_encoder(right, right_mask)
        corridor_feature = self.corridor_encoder(corridor, corridor_mask)
        gs_feature = self.relation(torch.cat([
            left_feature,
            right_feature,
            corridor_feature,
            torch.abs(left_feature - right_feature),
            left_feature * right_feature,
        ], dim=1))
        rgb_feature = self.rgb_encoder(rgb)
        action_feature = self.action_encoder(action_geom)
        touch_feature = self.touch_features(
            tactile_left, tactile_right, touch_available
        )
        return self.head(torch.cat([
            rgb_feature,
            action_feature,
            gs_feature,
            touch_feature,
        ], dim=1)).squeeze(1)


def available_indices(dataset):
    return [
        index
        for index, row in enumerate(dataset.rows)
        if float(row.get("touch_available", "0")) > 0.5
    ]
