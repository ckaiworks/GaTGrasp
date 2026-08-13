#!/usr/bin/env python3
"""Full RGB+Action+GS Stage-1 trained with weighted BCE only.

This is a true lambda=0 ablation: the per-object selection optimization pass is
not executed, so it cannot consume dropout RNG or perform zero-loss AdamW steps.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
    "stage1_fullcorridor_core_for_bce_only",
    HERE.parents[4]
    / "src"
    / "stage1"
    / "final"
    / "train_stage1_old12_fullcorridor_annealedSelect_v1.py",
)
BASE = CORE.BASE


def main():
    args = CORE.parse_args()
    if args.soft_top1_weight != 0.0:
        raise RuntimeError(
            "true BCE-only entry requires --soft_top1_weight 0"
        )
    if args.fixed_epoch > 0 and args.val_csv is not None:
        raise RuntimeError("--fixed_epoch is for final train/test mode; omit --val_csv")
    if args.fixed_epoch == 0 and args.val_csv is None:
        raise RuntimeError("selection mode requires --val_csv")
    if args.run_dir.exists():
        raise RuntimeError(f"refuse existing run_dir: {args.run_dir}")
    args.run_dir.mkdir(parents=True)

    BASE.set_seed(args.seed)
    stats_path = args.run_dir / "fullobject_feature_stats.npz"
    BASE.prepare_stats(args.train_csv, stats_path)
    train_set = BASE.LRCDataset(args.train_csv, stats_path, args.image_size)
    val_set = (
        BASE.LRCDataset(args.val_csv, stats_path, args.image_size)
        if args.val_csv
        else None
    )
    test_set = (
        BASE.LRCDataset(args.test_csv, stats_path, args.image_size)
        if args.test_csv
        else None
    )
    train_groups = CORE.group_indices(train_set)
    val_groups = CORE.group_indices(val_set) if val_set else None

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = (
        DataLoader(val_set, shuffle=False, **loader_args) if val_set else None
    )
    test_loader = (
        DataLoader(test_set, shuffle=False, **loader_args) if test_set else None
    )

    pos = sum(BASE.strict_label(row) == 1 for row in train_set.rows)
    neg = len(train_set.rows) - pos
    if pos == 0 or neg == 0:
        raise RuntimeError(f"invalid train P/N={pos}/{neg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = BASE.ExplicitLRCModel(no_gs=args.no_gs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print("device:", device, "no_gs:", args.no_gs)
    print("strict P/N:", pos, neg, "pos_weight:", float(pos_weight))
    print(
        "full-object groups:",
        len(train_groups),
        "sizes:",
        {str(key): len(value) for key, value in sorted(train_groups.items())},
    )
    print(
        "loss_contract:",
        {
            "loss": "weighted_BCE_only",
            "selection_weight": 0.0,
            "selection_training_pass_executed": False,
        },
    )

    best = -float("inf")
    best_path = args.run_dir / "best_fullobject_softtop1.pt"
    log_path = args.run_dir / "train_log.csv"
    for epoch in range(1, args.epochs + 1):
        model.train()
        bce_total = 0.0
        bce_count = 0
        for batch in train_loader:
            moved = BASE.move(batch, device)
            logits = CORE.forward(model, moved)
            loss = criterion(logits, moved["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bce_total += float(loss.item()) * len(logits)
            bce_count += len(logits)
        scheduler.step()

        temperature = CORE.cosine_select_temperature(
            epoch,
            args.epochs,
            args.select_temperature_start,
            args.select_temperature_end,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_bce": bce_total / max(bce_count, 1),
            "train_annealed_select_loss": 0.0,
            "select_temperature": temperature,
            "soft_top1_weight_effective": 0.0,
            "selection_training_pass_executed": 0,
        }
        if val_set is not None:
            val = BASE.evaluate(model, val_loader, device, criterion)
            val_soft = CORE.full_object_softtop1_score(
                model,
                val_set,
                val_groups,
                device,
                args.select_temperature_end,
            )
            score = val["mean_ap"]
            row.update(
                {
                    "val_soft_top1": val_soft,
                    "val_map": val["mean_ap"],
                    "val_top1": val["top1_success"],
                    "val_top5": val["top5_success"],
                    "val_auroc": val["auroc"],
                    "selection_score": score,
                }
            )
            if score > best:
                best = float(score)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_score": best,
                        "selection_metric": "val_mean_ap",
                    },
                    best_path,
                )
        else:
            if epoch == args.fixed_epoch:
                best = float(epoch)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_score": best,
                        "selection_metric": "fixed_epoch_from_nested_validation",
                    },
                    best_path,
                )

        with log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"bce={row['train_bce']:.4f} selectL=SKIPPED w=0.0000"
        )

    if not best_path.is_file():
        raise RuntimeError("no checkpoint was saved")
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    (args.run_dir / "best_val.json").write_text(
        json.dumps(
            {
                "epoch": int(state["epoch"]),
                "selection_metric": state["selection_metric"],
                "best_score": float(state["best_score"]),
                "used_outer_test_for_selection": False,
                "selection_training_pass_executed": False,
            },
            indent=2,
        )
    )
    if test_loader is not None:
        test = BASE.evaluate(model, test_loader, device, criterion)
        test_groups = CORE.group_indices(test_set)
        test["soft_top1_success"] = CORE.full_object_softtop1_score(
            model,
            test_set,
            test_groups,
            device,
            args.select_temperature_end,
        )
        (args.run_dir / "test_metrics.json").write_text(
            json.dumps(test, indent=2)
        )
        print("\n===== TEST =====")
        print(json.dumps(test, indent=2))
    print("DONE:", args.run_dir)


if __name__ == "__main__":
    main()
