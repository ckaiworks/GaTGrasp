#!/usr/bin/env python3
"""Evaluate three retrained Stage-2 variants on one frozen Stage-1 selection.

The input selection contains exactly one action per held-out object and is
produced by the reproducible corridor-withGS Stage-1 model (17/19 physical
successes).  Stage 2 never selects or reranks actions here: each variant only
looks up and verifies those same 19 ``raw_pair_id`` values in its out-of-fold
epoch-50 predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SELECTION_SHA256 = (
    "fa445d853bc9531b784f1ee60a76b8b59366c70b8e566b84188fd3d9b062c636"
)

VARIANTS = {
    "touch_only": {
        "label": "Touch Only",
        "experiment": "stage2_touchonly_weighted_bce_deterministic_v1",
    },
    "nogs": {
        "label": "RGB+Action13+Touch",
        "experiment": "stage2_clean_singlehead_rgb_action_touch_weighted_bce_v1",
    },
    "fullgs": {
        "label": "RGB+Action13+ContactGS+CorridorGS+Touch",
        "experiment": "stage2_stage1backbone_old12_plus_touch_v1",
    },
}

EXPECTED_FOLD_ROWS = {0: 600, 1: 450, 2: 600, 3: 600, 4: 600}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument("--touch-only-run", required=True, type=Path)
    parser.add_argument("--nogs-run", required=True, type=Path)
    parser.add_argument("--fullgs-run", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def auroc(labels: list[int], scores: list[float]) -> float:
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    if not positives or not negatives:
        raise RuntimeError("AUROC requires both classes")
    credit = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return credit / (len(positives) * len(negatives))


def average_precision(labels: list[int], scores: list[float]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        raise RuntimeError("AUPRC requires positives")
    grouped: dict[float, list[int]] = {}
    for label, score in zip(labels, scores):
        grouped.setdefault(score, []).append(label)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def ece_equal_width(labels: list[int], scores: list[float], bins: int = 5) -> float:
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, score in enumerate(scores)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if members:
            confidence = sum(scores[position] for position in members) / len(members)
            accuracy = sum(labels[position] for position in members) / len(members)
            result += len(members) / len(labels) * abs(accuracy - confidence)
    return result


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    f1_positive = safe_div(2 * tp, 2 * tp + fp + fn)
    f1_negative = safe_div(2 * tn, 2 * tn + fp + fn)
    return {
        "n": len(labels),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "specificity": safe_div(tn, tn + fp),
        "sensitivity": safe_div(tp, tp + fn),
        "macro_f1": (f1_positive + f1_negative) / 2,
        "auroc": auroc(labels, scores),
        "auprc_average_precision": average_precision(labels, scores),
        "brier": sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels),
        "ece5_equal_width": ece_equal_width(labels, scores),
        "probability_mean": sum(scores) / len(scores),
    }


def load_selection(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing Stage-1 selection: {path}")
    actual_hash = sha256(path)
    if actual_hash != EXPECTED_SELECTION_SHA256:
        raise RuntimeError(
            f"Stage-1 selection hash mismatch: {actual_hash} != {EXPECTED_SELECTION_SHA256}"
        )
    rows = [
        {
            "fold": int(row["fold"]),
            "object_id": int(row["object_id"]),
            "raw_pair_id": row["raw_pair_id"],
            "label": int(float(row["label"])),
            "stage1_probability": float(row["stage1_withGS_probability"]),
        }
        for row in read_csv(path)
    ]
    if len(rows) != 19 or len({row["object_id"] for row in rows}) != 19:
        raise RuntimeError("Stage-1 selection must contain 19 unique objects")
    if len({row["raw_pair_id"] for row in rows}) != 19:
        raise RuntimeError("Stage-1 raw_pair_id values are not unique")
    if sum(row["label"] for row in rows) != 17:
        raise RuntimeError("Stage-1 selection is not the reproducible 17/19 version")
    return sorted(rows, key=lambda row: row["object_id"])


def validate_contract(path: Path, expected_experiment: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing run contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment": expected_experiment,
        "objective": "weighted BCE only",
        "seed": 42,
        "batch_size": 32,
        "epochs": 50,
        "num_workers": 0,
        "deterministic_algorithms": "strict",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
        "test_selection": "none; fixed final epoch only",
    }
    mismatches = {
        key: {"actual": contract.get(key), "expected": expected}
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"run contract mismatch {path}: {mismatches}")
    return contract


def prediction_index(run_root: Path, variant: str) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    expected_experiment = VARIANTS[variant]["experiment"]
    indexed: dict[str, dict[str, str]] = {}
    provenance: list[dict[str, Any]] = []
    for fold in range(5):
        fold_root = run_root / f"fold_{fold}"
        checkpoint = fold_root / "checkpoint_epoch_050.pt"
        predictions = fold_root / "test_predictions.csv"
        metrics = fold_root / "test_metrics.json"
        for path in (checkpoint, predictions, metrics):
            if not path.is_file():
                raise RuntimeError(f"missing {variant} artifact: {path}")
        contract = validate_contract(fold_root / "run_contract.json", expected_experiment)
        rows = read_csv(predictions)
        if len(rows) != EXPECTED_FOLD_ROWS[fold]:
            raise RuntimeError(
                f"{variant} fold_{fold} row count={len(rows)} expected={EXPECTED_FOLD_ROWS[fold]}"
            )
        for row in rows:
            pair_id = row["raw_pair_id"]
            if pair_id in indexed:
                raise RuntimeError(f"duplicate {variant} raw_pair_id: {pair_id}")
            normalized = dict(row)
            normalized["fold"] = str(fold)
            indexed[pair_id] = normalized
        provenance.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "predictions": str(predictions.resolve()),
                "predictions_sha256": sha256(predictions),
                "contract": contract,
            }
        )
    if len(indexed) != 2850:
        raise RuntimeError(f"{variant} OOF rows={len(indexed)} expected=2850")
    return indexed, provenance


def markdown_report(metrics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Same reproducible Stage-1 -> retrained deterministic Stage-2",
        "",
        "All variants evaluate exactly the same 19 Stage-1 selected actions.",
        "",
        "| Stage-2 input | Acc. | Specificity | Sensitivity | Macro-F1 | AUROC | AUPRC | Brier | ECE-5 | Confusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        item = metrics[variant]
        lines.append(
            f"| {VARIANTS[variant]['label']} | {item['accuracy']:.4f} | "
            f"{item['specificity']:.4f} | {item['sensitivity']:.4f} | "
            f"{item['macro_f1']:.4f} | {item['auroc']:.4f} | "
            f"{item['auprc_average_precision']:.4f} | {item['brier']:.4f} | "
            f"{item['ece5_equal_width']:.4f} | TP={item['tp']}, TN={item['tn']}, "
            f"FP={item['fp']}, FN={item['fn']} |"
        )
    lines.extend(
        [
            "",
            "- Stage-1 is frozen before Stage-2 lookup and succeeds on 17/19 objects.",
            "- Threshold is fixed at 0.5 and is not tuned on these 19 results.",
            "- The missing-touch negative uses P(success)=0; a missing-touch positive is refused.",
            "",
            "FIXED_STAGE1_REPRO17_RETRAINED_STAGE2_VARIANTS_PASS",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise RuntimeError("threshold must be between zero and one")
    if args.out_dir.exists():
        raise RuntimeError(f"refuse existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    selection = load_selection(args.selection_csv)
    run_roots = {
        "touch_only": args.touch_only_run.resolve(),
        "nogs": args.nogs_run.resolve(),
        "fullgs": args.fullgs_run.resolve(),
    }
    combined: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    selected_ids_by_variant: dict[str, list[str]] = {}

    for variant, run_root in run_roots.items():
        indexed, records = prediction_index(run_root, variant)
        provenance[variant] = {"run_root": str(run_root), "folds": records}
        variant_rows: list[dict[str, Any]] = []
        for selected in selection:
            pair_id = selected["raw_pair_id"]
            if pair_id not in indexed:
                raise RuntimeError(f"{variant} missing selected pair: {pair_id}")
            prediction = indexed[pair_id]
            observed = (
                int(prediction["fold"]),
                int(prediction["object_id"]),
                int(float(prediction["label"])),
            )
            expected = (selected["fold"], selected["object_id"], selected["label"])
            if observed != expected:
                raise RuntimeError(
                    f"{variant} metadata mismatch for {pair_id}: {observed} != {expected}"
                )
            touch_available = int(float(prediction["touch_available"]))
            raw_probability = float(prediction["probability_if_touch"])
            if touch_available:
                probability = raw_probability
                policy = "stage2_model"
            elif selected["label"] == 0:
                probability = 0.0
                policy = "no_touch_negative_probability_zero"
            else:
                raise RuntimeError(f"missing-touch positive is unsupported: {pair_id}")
            row = {
                "variant": variant,
                **selected,
                "touch_available": touch_available,
                "stage2_raw_probability": raw_probability,
                "stage2_effective_probability": probability,
                "prediction_at_0p5": int(probability >= args.threshold),
                "probability_policy": policy,
            }
            variant_rows.append(row)
            combined.append(row)
        selected_ids_by_variant[variant] = [row["raw_pair_id"] for row in variant_rows]
        labels = [row["label"] for row in variant_rows]
        scores = [row["stage2_effective_probability"] for row in variant_rows]
        metrics[variant] = binary_metrics(labels, scores, args.threshold)

    first_ids = next(iter(selected_ids_by_variant.values()))
    if any(ids != first_ids for ids in selected_ids_by_variant.values()):
        raise RuntimeError("Stage-2 variants did not evaluate identical Stage-1 actions")

    write_csv(args.out_dir / "predictions.csv", combined)
    payload = {
        "status": "PASS",
        "stage1": {
            "version": "current reproducible corridor_withgs 17/19",
            "selection_csv": str(args.selection_csv.resolve()),
            "selection_sha256": sha256(args.selection_csv),
            "physical_top1_success": "17/19",
        },
        "same_raw_pair_ids": True,
        "threshold": args.threshold,
        "missing_touch_objects": sorted(
            {
                row["object_id"]
                for row in combined
                if row["touch_available"] == 0
            }
        ),
        "metrics": metrics,
        "stage2_provenance": provenance,
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = args.out_dir / "report.md"
    report_path.write_text(markdown_report(metrics), encoding="utf-8")
    hash_path = args.out_dir / "sha256.txt"
    hash_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in (metrics_path, args.out_dir / "predictions.csv", report_path)
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
