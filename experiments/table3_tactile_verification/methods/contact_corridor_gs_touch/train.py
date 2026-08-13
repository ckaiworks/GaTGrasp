#!/usr/bin/env python3
"""Deterministic Stage-2 with contact GS and closing-corridor GS.

This is the strict incremental control for the contact-GS verifier.  It first
constructs the complete contact-GS model, then adds only a zero-initialized
corridor residual.  Training uses one success logit and weighted BCE only.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


CONTACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "contact_gs_touch"
    / "train.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import noGS control={path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTACT = load_module("stage2_deterministic_contactgs_control", CONTACT_PATH)
NOGS = CONTACT.NOGS
BASE = CONTACT.BASE
SOURCE = CONTACT.SOURCE
TouchLRCDataset = CONTACT.TouchLRCDataset


class CleanSingleHeadStage2WithFullGS(
    CONTACT.CleanSingleHeadStage2WithContactGS
):
    """Contact-GS verifier plus a closing-corridor residual."""

    def __init__(self):
        # Construct the complete contact-GS model first.  With the same seed,
        # every common parameter is bit-identical to the contact-GS control.
        super().__init__()
        self.corridor_encoder = BASE.PatchEncoder()
        self.corridor_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.corridor_delta_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )
        # The full-GS model begins at the exact contact-GS function.
        nn.init.zeros_(self.corridor_delta_head[-1].weight)
        nn.init.zeros_(self.corridor_delta_head[-1].bias)

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
        rgb_feature = self.rgb_encoder(rgb)
        action_feature = self.action_encoder(action_geom)
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
        base_logit = self.head(torch.cat([
            rgb_feature,
            action_feature,
            touch_feature,
        ], dim=1)).squeeze(1)

        gs_left = self.endpoint_encoder(left, left_mask)
        gs_right = self.endpoint_encoder(right, right_mask)
        fused_left = self.side_fusion(torch.cat([
            touch_left,
            gs_left,
            action_feature,
        ], dim=1))
        fused_right = self.side_fusion(torch.cat([
            touch_right,
            gs_right,
            action_feature,
        ], dim=1))
        gs_touch = self.gs_touch_relation(torch.cat([
            fused_left,
            fused_right,
            torch.abs(fused_left - fused_right),
            fused_left * fused_right,
        ], dim=1))
        gs_delta = self.gs_delta_head(gs_touch).squeeze(1)
        corridor_feature = self.corridor_encoder(corridor, corridor_mask)
        corridor_context = self.corridor_fusion(torch.cat([
            gs_touch,
            corridor_feature,
            action_feature,
        ], dim=1))
        corridor_delta = self.corridor_delta_head(
            corridor_context
        ).squeeze(1)
        return base_logit + gs_delta + corridor_delta


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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.fixed_epoch != args.epochs:
        raise RuntimeError("fixed_epoch must equal epochs")
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
        raise RuntimeError(f"TOUCH_TRAIN_CLASS_INVALID P/N={positive}/{negative}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positive_weight = torch.tensor(
        [negative / positive], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    if args.smoke:
        # This comparison is independent of dataset construction and proves
        # the only initial difference is the new GS residual branch.
        BASE.set_seed(args.seed)
        reference = CONTACT.CleanSingleHeadStage2WithContactGS().to(device)
        BASE.set_seed(args.seed)
        model = CleanSingleHeadStage2WithFullGS().to(device)
        common_differences = []
        model_state = model.state_dict()
        for name, tensor in reference.state_dict().items():
            common_differences.append(float(torch.max(torch.abs(
                tensor - model_state[name]
            )).item()))
        common_max_diff = max(common_differences, default=0.0)
        if common_max_diff != 0.0:
            raise RuntimeError(f"COMMON_INITIALIZATION_DIFF={common_max_diff}")

        batch = BASE.move(next(iter(train_loader)), device)
        reference.eval()
        model.eval()
        with torch.inference_mode():
            initial_reference = CONTACT.model_forward(reference, batch)
            initial_withgs = model_forward(model, batch)
            initial_output_diff = float(torch.max(torch.abs(
                initial_reference - initial_withgs
            )).item())
        if initial_output_diff != 0.0:
            raise RuntimeError(f"INITIAL_FUNCTION_DIFF={initial_output_diff}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        # Step 1 learns the zero-initialized final residual layer.
        model.train()
        first_loss = criterion(model_forward(model, batch), batch["label"])
        optimizer.zero_grad(set_to_none=True)
        first_loss.backward()
        optimizer.step()
        # Step 2 must propagate through every common and GS module.
        second_loss = criterion(model_forward(model, batch), batch["label"])
        optimizer.zero_grad(set_to_none=True)
        second_loss.backward()
        prefixes = (
            "rgb_encoder", "action_encoder", "touch_encoder", "head",
            "endpoint_encoder", "side_fusion", "gs_touch_relation",
            "gs_delta_head", "corridor_encoder", "corridor_fusion",
            "corridor_delta_head",
        )
        gradients = {prefix: gradient_sum(model, prefix) for prefix in prefixes}
        if any(value <= 0.0 for value in gradients.values()):
            raise RuntimeError(f"EXPECTED_BRANCH_GRADIENT_MISSING={gradients}")
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            output = model_forward(model, batch)
            changed_gs = dict(batch)
            changed_gs["left"] = torch.roll(batch["left"], 1, 0)
            changed_gs["right"] = torch.roll(batch["right"], -1, 0)
            changed_gs["left_mask"] = torch.roll(batch["left_mask"], 1, 0)
            changed_gs["right_mask"] = torch.roll(batch["right_mask"], -1, 0)
            gs_sensitivity = float(torch.max(torch.abs(
                output - model_forward(model, changed_gs)
            )).item())
            changed_corridor = dict(batch)
            changed_corridor["corridor"] = torch.roll(batch["corridor"], 1, 0)
            changed_corridor["corridor_mask"] = torch.roll(
                batch["corridor_mask"], 1, 0
            )
            corridor_sensitivity = float(torch.max(torch.abs(
                output - model_forward(model, changed_corridor)
            )).item())
        if gs_sensitivity <= 0.0:
            raise RuntimeError(f"CONTACT_GS_NOT_READ={gs_sensitivity}")
        if corridor_sensitivity <= 0.0:
            raise RuntimeError(f"CORRIDOR_GS_NOT_READ={corridor_sensitivity}")
        print("DETERMINISTIC_SINGLEHEAD_FULLGS_SMOKE_PASS", {
            "common_initialization_max_diff": common_max_diff,
            "initial_output_max_diff": initial_output_diff,
            "branch_gradient_sums_after_opening_residual": gradients,
            "contact_gs_sensitivity_max_diff": gs_sensitivity,
            "corridor_gs_sensitivity_max_diff": corridor_sensitivity,
            "first_loss": float(first_loss.item()),
            "second_loss": float(second_loss.item()),
        }, flush=True)
        return

    model = CleanSingleHeadStage2WithFullGS().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    contract = {
        "experiment": (
            "stage2_deterministic_singlehead_rgb_action_touch_fullgs_"
            "weighted_bce_v1"
        ),
        "control_pair": (
            "stage2_deterministic_singlehead_rgb_action_touch_contactgs_"
            "weighted_bce_v1"
        ),
        "training_initialization": "fresh end-to-end",
        "output": "one success logit",
        "inputs_used": [
            "RGB", "Action13", "bilateral real Touch",
            "left/right contact-aligned GS", "closing-corridor GS",
        ],
        "inputs_not_used": [],
        "objective": "weighted BCE only",
        "positive_weight": negative / positive,
        "contact_gs_fusion": (
            "shared left/right PatchEncoder; Touch+GS+Action side fusion; "
            "bilateral relation; zero-initialized logit residual"
        ),
        "corridor_gs_fusion": (
            "corridor PatchEncoder; contact-GS relation+corridor+Action; "
            "zero-initialized incremental logit residual"
        ),
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
        "train_real_touch_rows": len(train_indices),
        "train_real_touch_positive": positive,
        "train_real_touch_negative": negative,
        "missing_touch_rows_not_used_for_gradient": len(train_set) - len(train_indices),
        "test_selection": "none; fixed final epoch only",
    }
    (args.run_dir / "run_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    print(json.dumps(contract, indent=2), flush=True)

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
            import csv
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
    metrics = NOGS.evaluate(
        model, test_set, test_loader, device, criterion, args.run_dir
    )
    (args.run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"DONE: {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
