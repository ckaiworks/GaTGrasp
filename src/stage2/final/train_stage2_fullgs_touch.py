#!/usr/bin/env python3
"""Deterministic Stage-2: exact Stage-1 old12 backbone plus Touch only.

RGB, Action13, left/right old12 GS, corridor old12 GS, their encoders, and
the GS relation are inherited unchanged from the selected Stage-1 model.
The only new modality is a shared bilateral Touch encoder and a 128-D touch
relation appended to the Stage-1 448-D representation.  One success logit is
trained from scratch with weighted BCE on rows that have real touch.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


CONTROL_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "table3_tactile_verification"
    / "methods"
    / "rgb_action_touch"
    / "train.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import control module={path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTROL = load_module("stage2_stage1backbone_control", CONTROL_PATH)
BASE = CONTROL.BASE
SOURCE = CONTROL.SOURCE
TouchLRCDataset = CONTROL.TouchLRCDataset
EXPECTED_FAMILY = "explicit_lrc_actionframe_orthonormal_old12_v1"


class Stage1BackboneWithTouch(SOURCE.TouchOnceExplicitLRCModel):
    """Exact ExplicitLRCModel common trunk with one bilateral Touch branch."""

    def __init__(self):
        super().__init__(no_gs=False)


def model_forward(model, batch):
    return model(
        batch["rgb"],
        batch["action_geom"],
        batch["left"],
        batch["right"],
        batch["corridor"],
        batch["left_mask"],
        batch["right_mask"],
        batch["corridor_mask"],
        batch["tactile_left"],
        batch["tactile_right"],
        batch["touch_available"],
    )


def gradient_sum(model, prefix):
    return sum(
        float(parameter.grad.abs().sum().item())
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    )


def validate_old12_contract(dataset, split_name):
    families = {row.get("gs_feature_family", "") for row in dataset.rows}
    if families != {EXPECTED_FAMILY}:
        raise RuntimeError(
            f"{split_name} GS family mismatch={sorted(families)}"
        )
    paths = {row["local_gs_path"] for row in dataset.rows}
    bad_paths = [
        path for path in paths
        if "shared_old12_orthonormal" not in Path(path).as_posix()
        or not Path(path).is_file()
    ]
    if bad_paths:
        raise RuntimeError(
            f"{split_name} old12 path contract failed count={len(bad_paths)} "
            f"first={bad_paths[0]}"
        )
    return len(paths)


def common_initialization_audit(seed, model):
    BASE.set_seed(seed)
    stage1 = BASE.ExplicitLRCModel(no_gs=False)
    stage1_state = stage1.state_dict()
    model_state = model.state_dict()
    prefixes = (
        "rgb_encoder.",
        "action_encoder.",
        "endpoint_encoder.",
        "corridor_encoder.",
        "relation.",
    )
    compared = 0
    maximum = 0.0
    for key, value in stage1_state.items():
        if not key.startswith(prefixes):
            continue
        if key not in model_state or model_state[key].shape != value.shape:
            raise RuntimeError(f"Stage-1 common parameter mismatch={key}")
        difference = float(torch.max(torch.abs(
            value.detach().cpu() - model_state[key].detach().cpu()
        )).item())
        maximum = max(maximum, difference)
        compared += 1
    if compared == 0 or maximum != 0.0:
        raise RuntimeError(
            f"Stage-1 common initialization failed compared={compared} "
            f"maximum={maximum}"
        )
    del stage1
    return {"tensors": compared, "maximum_abs_difference": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, type=Path)
    parser.add_argument("--test_csv", required=True, type=Path)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fixed_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.fixed_epoch != args.epochs:
        raise RuntimeError("fixed_epoch must equal epochs")
    if args.num_workers != 0:
        raise RuntimeError("strict reproducibility requires num_workers=0")
    if args.run_dir.exists():
        raise RuntimeError(f"refuse existing run_dir={args.run_dir}")

    BASE.set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)

    args.run_dir.mkdir(parents=True)
    stats_path = args.run_dir / "fullobject_feature_stats.npz"
    BASE.prepare_stats(args.train_csv, stats_path)
    train_set = TouchLRCDataset(args.train_csv, stats_path, args.image_size)
    test_set = TouchLRCDataset(args.test_csv, stats_path, args.image_size)
    unique_train_gs = validate_old12_contract(train_set, "train")
    unique_test_gs = validate_old12_contract(test_set, "test")

    train_indices = SOURCE.available_indices(train_set)
    if not train_indices:
        raise RuntimeError("NO_REAL_TOUCH_TRAIN_ROWS")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        Subset(train_set, train_indices),
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    train_labels = np.asarray([
        BASE.strict_label(train_set.rows[index])
        for index in train_indices
    ], dtype=np.int64)
    positive = int((train_labels == 1).sum())
    negative = int((train_labels == 0).sum())
    if positive == 0 or negative == 0:
        raise RuntimeError(
            f"TOUCH_TRAIN_CLASS_INVALID P/N={positive}/{negative}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positive_weight = torch.tensor(
        [negative / positive], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    # Re-seeding makes every common Stage-1 parameter bit-identical at
    # initialization; the Touch branch and 576-D head are the only additions.
    BASE.set_seed(args.seed)
    model = Stage1BackboneWithTouch()
    initialization_audit = common_initialization_audit(args.seed, model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    contract = {
        "experiment": "stage2_stage1backbone_old12_plus_touch_v1",
        "training_initialization": "fresh end-to-end",
        "output": "one success logit",
        "stage1_common_backbone": (
            "RGBEncoder256 + Action13Encoder64 + shared L/R PatchEncoder64 "
            "+ Corridor PatchEncoder64 + L/R/C relation128"
        ),
        "only_new_modality": (
            "shared bilateral ResNet18 Touch + 128-D bilateral relation"
        ),
        "fusion": "RGB256 + Action64 + old12GS128 + Touch128 -> head576",
        "inputs_used": [
            "RGB", "Action13", "old12 left contact GS",
            "old12 right contact GS", "old12 corridor GS",
            "bilateral real Touch",
        ],
        "gs_feature_family": EXPECTED_FAMILY,
        "objective": "weighted BCE only",
        "positive_weight": negative / positive,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "deterministic_algorithms": "strict",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
        "torch_num_threads": 1,
        "num_workers": args.num_workers,
        "train_real_touch_rows": len(train_indices),
        "train_real_touch_positive": positive,
        "train_real_touch_negative": negative,
        "missing_touch_rows_not_used_for_gradient": (
            len(train_set) - len(train_indices)
        ),
        "unique_train_old12_gs": unique_train_gs,
        "unique_test_old12_gs": unique_test_gs,
        "common_initialization_audit": initialization_audit,
        "test_selection": "none; fixed final epoch only",
    }
    (args.run_dir / "run_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    print(json.dumps(contract, indent=2), flush=True)

    if args.smoke:
        model.train()
        batch = BASE.move(next(iter(train_loader)), device)
        logits = model_forward(model, batch)
        loss = criterion(logits, batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        prefixes = (
            "rgb_encoder", "action_encoder", "endpoint_encoder",
            "corridor_encoder", "relation", "touch_encoder",
            "touch_relation", "head",
        )
        gradients = {
            prefix: gradient_sum(model, prefix) for prefix in prefixes
        }
        if any(value <= 0.0 for value in gradients.values()):
            raise RuntimeError(f"EXPECTED_BRANCH_GRADIENT_MISSING={gradients}")
        model.eval()
        with torch.inference_mode():
            reference = model_forward(model, batch)
            changed_gs = dict(batch)
            for key in ("left", "right", "corridor"):
                changed_gs[key] = torch.roll(batch[key], 1, 0)
            for key in ("left_mask", "right_mask", "corridor_mask"):
                changed_gs[key] = torch.roll(batch[key], 1, 0)
            gs_difference = float(torch.max(torch.abs(
                reference - model_forward(model, changed_gs)
            )).item())
            changed_touch = dict(batch)
            for key in ("tactile_left", "tactile_right"):
                changed_touch[key] = torch.roll(batch[key], 1, 0)
            touch_difference = float(torch.max(torch.abs(
                reference - model_forward(model, changed_touch)
            )).item())
        if gs_difference <= 0.0 or touch_difference <= 0.0:
            raise RuntimeError(
                f"INPUT_SENSITIVITY_FAILED GS={gs_difference} "
                f"Touch={touch_difference}"
            )
        print("STAGE2_STAGE1BACKBONE_OLD12_TOUCH_SMOKE_PASS", {
            "batch": len(batch["label"]),
            "loss": float(loss.item()),
            "branch_gradient_sums": gradients,
            "gs_sensitivity_max_diff": gs_difference,
            "touch_sensitivity_max_diff": touch_difference,
        }, flush=True)
        return

    log_path = args.run_dir / "train_log.csv"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        row_count = 0
        for batch in train_loader:
            moved = BASE.move(batch, device)
            logits = model_forward(model, moved)
            loss = criterion(logits, moved["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size = len(moved["label"])
            loss_total += float(loss.item()) * batch_size
            row_count += batch_size
        scheduler.step()
        log_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "weighted_bce": loss_total / max(row_count, 1),
        }
        with log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(log_row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(log_row)
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"weighted_bce={log_row['weighted_bce']:.4f}",
            flush=True,
        )

    checkpoint_name = f"checkpoint_epoch_{args.fixed_epoch:03d}.pt"
    checkpoint = args.run_dir / checkpoint_name
    torch.save({
        "model": model.state_dict(),
        "epoch": args.fixed_epoch,
        "selection_metric": "fixed_epoch_no_validation",
        "contract": contract,
    }, checkpoint)
    metrics = CONTROL.evaluate(
        model, test_set, test_loader, device, criterion, args.run_dir
    )
    (args.run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"DONE: {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
