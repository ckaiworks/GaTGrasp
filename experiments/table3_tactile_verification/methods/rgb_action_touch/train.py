#!/usr/bin/env python3
"""Deterministic clean single-output Stage-2 RGB+Action+Touch verifier."""

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


COMPONENTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "_shared"
    / "stage2_components.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import parent module={path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = load_module("stage2_components", COMPONENTS_PATH)
BASE = SOURCE.BASE
TouchLRCDataset = SOURCE.TouchLRCDataset


class CleanSingleHeadStage2(nn.Module):
    """One logit from RGB, Action13, and bilateral real touch."""

    def __init__(self):
        super().__init__()
        self.rgb_encoder = BASE.common.RGBEncoder(out_dim=256)
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.touch_encoder = SOURCE.SharedTouchResNet18(out_dim=128)
        self.touch_relation = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 64 + 128, 256),
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
        # These arguments are retained only for dataset/API compatibility.
        # The final verifier never reads reconstructed surface features.
        del left, right, corridor, left_mask, right_mask, corridor_mask
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
        return self.head(torch.cat([
            rgb_feature,
            action_feature,
            touch_feature,
        ], dim=1)).squeeze(1)


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


def binary_summary(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= 0.5
    positive = labels == 1
    negative = labels == 0
    tp = int((predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    return {
        "rows": int(len(labels)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "accuracy_at_0p5": float((predicted == positive).mean()),
        "specificity_at_0p5": float(tn / max(tn + fp, 1)),
        "sensitivity_at_0p5": float(tp / max(tp + fn, 1)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.inference_mode()
def evaluate(model, dataset, loader, device, criterion, run_dir):
    model.eval()
    labels = []
    probabilities = []
    objects = []
    prediction_rows = []
    loss_total = 0.0
    count = 0
    offset = 0
    for batch in loader:
        moved = BASE.move(batch, device)
        logits = model_forward(model, moved)
        batch_probabilities = torch.sigmoid(logits)
        available = moved["touch_available"] > 0.5
        if bool(available.any()):
            selected_labels = moved["label"][available]
            selected_logits = logits[available]
            selected_count = int(available.sum())
            loss_total += float(
                criterion(selected_logits, selected_labels).item()
            ) * selected_count
            count += selected_count

        batch_size = len(moved["label"])
        rows = dataset.rows[offset:offset + batch_size]
        offset += batch_size
        for row, label, probability, is_available in zip(
            rows,
            moved["label"].cpu().numpy(),
            batch_probabilities.cpu().numpy(),
            available.cpu().numpy(),
        ):
            prediction_rows.append({
                "object_id": int(row["object_id"]),
                "raw_pair_id": row["raw_pair_id"],
                "label": int(label),
                "touch_available": int(bool(is_available)),
                "probability_if_touch": float(probability),
            })
            if bool(is_available):
                labels.append(int(label))
                probabilities.append(float(probability))
                objects.append(int(row["object_id"]))
    if offset != len(dataset):
        raise RuntimeError(f"evaluation row mismatch={offset}/{len(dataset)}")
    with (run_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    objects_array = np.asarray(objects, dtype=np.int64)
    return {
        "loss": loss_total / max(count, 1),
        "metrics": BASE.common.safe_metrics(
            labels_array,
            probabilities_array,
            objects_array,
        ),
        "threshold_metrics": binary_summary(
            labels_array,
            probabilities_array,
        ),
    }


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
    model = CleanSingleHeadStage2().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    contract = {
        "experiment": "stage2_clean_singlehead_rgb_action_touch_weighted_bce_v1",
        "training_initialization": "fresh end-to-end",
        "output": "one success logit",
        "inputs_used": ["RGB", "Action13", "bilateral real Touch"],
        "inputs_not_used": [
            "left contact reconstructed surface",
            "right contact reconstructed surface",
            "closing corridor reconstructed surface",
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

    if args.smoke:
        model.train()
        batch = BASE.move(next(iter(train_loader)), device)
        logits = model_forward(model, batch)
        loss = criterion(logits, batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {
            prefix: sum(
                float(parameter.grad.abs().sum().item())
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            )
            for prefix in ("rgb_encoder", "action_encoder", "touch_encoder", "head")
        }
        if any(value <= 0.0 for value in gradients.values()):
            raise RuntimeError(f"EXPECTED_BRANCH_GRADIENT_MISSING={gradients}")
        model.eval()
        with torch.inference_mode():
            reference = model_forward(model, batch)
            changed = dict(batch)
            for key in ("left", "right", "corridor"):
                changed[key] = torch.roll(batch[key], 1, 0)
            for key in ("left_mask", "right_mask", "corridor_mask"):
                changed[key] = torch.roll(batch[key], 1, 0)
            changed_output = model_forward(model, changed)
            surface_invariance = float(torch.max(torch.abs(
                reference - changed_output
            )).item())
        if surface_invariance != 0.0:
            raise RuntimeError(
                f"FINAL_SINGLEHEAD_READS_RECONSTRUCTED_SURFACE={surface_invariance}"
            )
        print("CLEAN_SINGLEHEAD_STAGE2_SMOKE_PASS", {
            "batch": len(batch["label"]),
            "loss": float(loss.item()),
            "branch_gradient_sums": gradients,
            "reconstructed_surface_invariance_max_diff": surface_invariance,
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
    metrics = evaluate(model, test_set, test_loader, device, criterion, args.run_dir)
    (args.run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"DONE: {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
