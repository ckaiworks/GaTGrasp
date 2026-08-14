#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def same_npz(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(
        second, allow_pickle=False
    ) as b:
        if sorted(a.files) != sorted(b.files):
            return False
        return all(np.array_equal(a[key], b[key]) for key in a.files)


def same_checkpoint(first, second):
    a = torch.load(first, map_location="cpu", weights_only=False)
    b = torch.load(second, map_location="cpu", weights_only=False)
    if a.keys() != b.keys() or a["model"].keys() != b["model"].keys():
        return False
    if a["epoch"] != b["epoch"] or a["contract"] != b["contract"]:
        return False
    return all(
        torch.equal(a["model"][key], b["model"][key])
        for key in a["model"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    report = {"root": str(args.root), "variants": {}, "all_exact": True}
    for variant in ("touch_only", "nogs", "fullgs"):
        folds = {}
        for fold in range(5):
            a = args.root / "reference" / variant / f"fold_{fold}"
            b = args.root / "repeat" / variant / f"fold_{fold}"
            checks = {
                "train_log_exact": (
                    (a / "train_log.csv").read_bytes()
                    == (b / "train_log.csv").read_bytes()
                ),
                "metrics_exact": (
                    (a / "test_metrics.json").read_bytes()
                    == (b / "test_metrics.json").read_bytes()
                ),
                "contract_exact": (
                    (a / "run_contract.json").read_bytes()
                    == (b / "run_contract.json").read_bytes()
                ),
                "feature_stats_exact": same_npz(
                    a / "fullobject_feature_stats.npz",
                    b / "fullobject_feature_stats.npz",
                ),
                "checkpoint_tensors_exact": same_checkpoint(
                    a / "checkpoint_epoch_050.pt",
                    b / "checkpoint_epoch_050.pt",
                ),
            }
            checks["all_exact"] = all(checks.values())
            report["all_exact"] &= checks["all_exact"]
            folds[str(fold)] = checks
        report["variants"][variant] = folds
    output = args.root / "exact_reproduction_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_exact"]:
        raise RuntimeError(f"STAGE2_EXACT_REPRODUCTION_FAILED report={output}")
    print("STAGE2_THREE_VARIANT_EXACT_REPRODUCTION_PASS", flush=True)


if __name__ == "__main__":
    main()
