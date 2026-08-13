#!/usr/bin/env python3
"""Strict deterministic Stage-2 Touch-Only weighted-BCE control."""

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
    Path(__file__).resolve().parent.parent
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


CONTROL = load_module("stage2_touchonly_control", CONTROL_PATH)
BASE = CONTROL.BASE
SOURCE = CONTROL.SOURCE
TouchLRCDataset = CONTROL.TouchLRCDataset
EXPECTED_PAIR_FAMILY = "explicit_lrc_actionframe_orthonormal_old12_v1"


def validate_paired_old12_index(dataset, split_name):
    families = {row.get("gs_feature_family", "") for row in dataset.rows}
    if families != {EXPECTED_PAIR_FAMILY}:
        raise RuntimeError(
            f"{split_name} paired-index family mismatch={sorted(families)}"
        )
    return len({row["raw_pair_id"] for row in dataset.rows})


class CleanTouchOnlyStage2(nn.Module):
    """Only bilateral real touch; RGB, Action, and GS branches do not exist."""

    def __init__(self):
        super().__init__()
        self.touch_encoder = SOURCE.SharedTouchResNet18(out_dim=128)
        self.touch_relation = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def touch_embeddings(
        self,
        tactile_left,
        tactile_right,
        touch_available,
    ):
        batch = tactile_left.shape[0]
        left_out = tactile_left.new_zeros((batch, 128))
        right_out = tactile_right.new_zeros((batch, 128))
        indices = torch.nonzero(
            touch_available > 0.5,
            as_tuple=False,
        ).flatten()
        if len(indices) > 0:
            left = self.touch_encoder(
                tactile_left.index_select(0, indices)
            )
            right = self.touch_encoder(
                tactile_right.index_select(0, indices)
            )
            left_out = left_out.index_copy(0, indices, left)
            right_out = right_out.index_copy(0, indices, right)
        return left_out, right_out

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
        del (
            rgb,
            action_geom,
            left,
            right,
            corridor,
            left_mask,
            right_mask,
            corridor_mask,
        )
        touch_left, touch_right = self.touch_embeddings(
            tactile_left,
            tactile_right,
            touch_available,
        )
        touch_feature = self.touch_relation(torch.cat([
            touch_left,
            touch_right,
            torch.abs(touch_left - touch_right),
            touch_left * touch_right,
        ], dim=1))
        return self.head(touch_feature).squeeze(1)


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
    unique_train_pairs = validate_paired_old12_index(
        train_set, "train"
    )
    unique_test_pairs = validate_paired_old12_index(
        test_set, "test"
    )
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
    BASE.set_seed(args.seed)
    model = CleanTouchOnlyStage2().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    contract = {
        "experiment": "stage2_touchonly_weighted_bce_deterministic_v1",
        "training_initialization": "fresh end-to-end",
        "output": "one success logit",
        "inputs_used": ["bilateral real Touch"],
        "inputs_not_used": [
            "RGB",
            "Action13",
            "left contact GS",
            "right contact GS",
            "corridor GS",
        ],
        "objective": "weighted BCE only",
        "positive_weight": negative / positive,
        "touch_encoder": (
            "shared bilateral ResNet18(weights=None), all parameters trainable"
        ),
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
        "paired_index_family": EXPECTED_PAIR_FAMILY,
        "unique_train_raw_pair_ids": unique_train_pairs,
        "unique_test_raw_pair_ids": unique_test_pairs,
        "train_real_touch_rows": len(train_indices),
        "train_real_touch_positive": positive,
        "train_real_touch_negative": negative,
        "missing_touch_rows_not_used_for_gradient": (
            len(train_set) - len(train_indices)
        ),
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
        gradients = {
            prefix: gradient_sum(model, prefix)
            for prefix in ("touch_encoder", "touch_relation", "head")
        }
        if any(value <= 0.0 for value in gradients.values()):
            raise RuntimeError(f"EXPECTED_BRANCH_GRADIENT_MISSING={gradients}")
        model.eval()
        with torch.inference_mode():
            reference = model_forward(model, batch)
            changed_non_touch = dict(batch)
            for key in ("rgb", "action_geom", "left", "right", "corridor"):
                changed_non_touch[key] = torch.roll(batch[key], 1, 0)
            for key in ("left_mask", "right_mask", "corridor_mask"):
                changed_non_touch[key] = torch.roll(batch[key], 1, 0)
            invariant = model_forward(model, changed_non_touch)
            non_touch_difference = float(torch.max(torch.abs(
                reference - invariant
            )).item())
            changed_touch = dict(batch)
            changed_touch["tactile_left"] = torch.roll(
                batch["tactile_left"], 1, 0
            )
            changed_touch["tactile_right"] = torch.roll(
                batch["tactile_right"], 1, 0
            )
            touch_output = model_forward(model, changed_touch)
            touch_difference = float(torch.max(torch.abs(
                reference - touch_output
            )).item())
        if non_touch_difference != 0.0 or touch_difference <= 0.0:
            raise RuntimeError(
                "TOUCHONLY_MODALITY_AUDIT_FAILED "
                f"non_touch={non_touch_difference} touch={touch_difference}"
            )
        print("DETERMINISTIC_TOUCHONLY_SMOKE_PASS", {
            "batch": len(batch["label"]),
            "loss": float(loss.item()),
            "branch_gradient_sums": gradients,
            "non_touch_invariance_max_diff": non_touch_difference,
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

    state = {
        "model": model.state_dict(),
        "epoch": args.fixed_epoch,
        "selection_metric": "fixed_epoch_50_no_validation",
        "contract": contract,
    }
    checkpoint = args.run_dir / "checkpoint_epoch_050.pt"
    torch.save(state, checkpoint)
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
