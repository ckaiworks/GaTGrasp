#!/usr/bin/env python3
"""Current deterministic RGB+Action+Depth with the correct test interface."""

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
    "stage1_current_core_for_rgb_action_depth_v2",
    HERE.parents[3]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_old12_fullcorridor_annealedSelect_v1.py",
)
DEPTH = load_module(
    "stage1_rgb_action_depth_base_current_v2",
    HERE / "train_stage1_strictsuccess_explicitLRC_v4b.py",
)


def depth_forward(model, batch):
    return model(
        batch["rgb"],
        batch["action_geom"],
        batch["depth"],
        batch["left"],
        batch["right"],
        batch["corridor"],
        batch["left_mask"],
        batch["right_mask"],
        batch["corridor_mask"],
    )


@DEPTH.torch.no_grad()
def depth_evaluate(model, loader, device, criterion):
    """Same base evaluation, with depth passed through the current forward adapter."""
    model.eval()
    ys, ps, objs = [], [], []
    total, count = 0.0, 0
    for batch in loader:
        moved = DEPTH.move(batch, device)
        logit = depth_forward(model, moved)
        loss = criterion(logit, moved["label"])
        prob = DEPTH.torch.sigmoid(logit)
        total += float(loss.item()) * len(prob)
        count += len(prob)
        ys.extend(moved["label"].cpu().numpy().tolist())
        ps.extend(prob.cpu().numpy().tolist())
        objs.extend(moved["object_id"].cpu().numpy().tolist())
    out = DEPTH.common.safe_metrics(
        DEPTH.np.asarray(ys, dtype=DEPTH.np.int64),
        DEPTH.np.asarray(ps, dtype=DEPTH.np.float64),
        DEPTH.np.asarray(objs, dtype=DEPTH.np.int64),
    )
    out["loss"] = total / max(count, 1)
    return out


CORE.BASE = DEPTH
CORE.forward = depth_forward
DEPTH.evaluate = depth_evaluate


if __name__ == "__main__":
    CORE.main()
