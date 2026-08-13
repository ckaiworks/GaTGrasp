#!/usr/bin/env python3
"""Evaluate the fixed Stage-1 selection with Stage-2 out-of-fold predictions.

The selector is evaluated once and contributes exactly one ``raw_pair_id`` per
object.  This script then looks up those same actions in the independently
trained Stage-2 noGS, contact-GS, and full-GS fold predictions.  It never uses
the Stage-2 labels to choose an action.

The release contains one action without tactile observations (object 39).  It
is a physical failure, so deployment assigns P(success)=0.  A missing-touch
positive is rejected instead of being silently imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


VARIANTS = {
    "nogs": "RGB+Action13+Touch",
    "contactgs": "RGB+Action13+Touch+ContactGS",
    "fullgs": "RGB+Action13+Touch+ContactGS+CorridorGS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument(
        "--stage2-oof-root",
        required=True,
        type=Path,
        help="Root containing VARIANT/fold_0..fold_4/test_predictions.csv.",
    )
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
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise RuntimeError("AUROC requires both classes")
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            credit += float(positive > negative) + 0.5 * float(positive == negative)
    return credit / (len(positives) * len(negatives))


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise RuntimeError("AUPRC requires a positive sample")
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
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def ece_equal_width(
    labels: list[int], scores: list[float], bins: int = 5
) -> float:
    total = len(labels)
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, score in enumerate(scores)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        confidence = sum(scores[position] for position in members) / len(members)
        accuracy = sum(labels[position] for position in members) / len(members)
        result += len(members) / total * abs(accuracy - confidence)
    return result


def binary_metrics(
    labels: list[int], scores: list[float], threshold: float
) -> dict[str, Any]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
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
        "ece5_equal_width": ece_equal_width(labels, scores, bins=5),
        "probability_mean": sum(scores) / len(scores),
    }


def load_selection(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    normalized = [
        {
            "fold": int(row["fold"]),
            "object_id": int(row["object_id"]),
            "raw_pair_id": row["raw_pair_id"],
            "label": int(float(row["label"])),
            "stage1_probability": float(row["stage1_withGS_probability"]),
        }
        for row in rows
    ]
    if len(normalized) != 19:
        raise RuntimeError(f"expected 19 Stage-1 selections, got {len(normalized)}")
    if len({row["object_id"] for row in normalized}) != 19:
        raise RuntimeError("Stage-1 object IDs are not unique")
    if len({row["raw_pair_id"] for row in normalized}) != 19:
        raise RuntimeError("Stage-1 raw_pair_id values are not unique")
    return sorted(normalized, key=lambda row: row["object_id"])


def prediction_index(root: Path, variant: str) -> dict[str, dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fold in range(5):
        path = root / variant / f"fold_{fold}" / "test_predictions.csv"
        if not path.is_file():
            raise RuntimeError(f"missing Stage-2 OOF predictions: {path}")
        for row in read_csv(path):
            row = dict(row)
            row["fold"] = str(fold)
            rows.append(row)
    indexed = {row["raw_pair_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"duplicate raw_pair_id in Stage-2 {variant} predictions")
    return indexed


def markdown_report(metrics: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Fixed reproducible Stage-1 selection -> deterministic Stage-2 factorial",
        "",
        "All rows use exactly the same 19 Stage-1 selected `raw_pair_id` values.",
        "",
        "| Stage-2 input | Acc. | Specificity | Sensitivity | Macro-F1 | AUROC | AUPRC | Brier | ECE-5 | Confusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, label in VARIANTS.items():
        item = metrics[variant]
        lines.append(
            f"| {label} | {item['accuracy']:.4f} | {item['specificity']:.4f} | "
            f"{item['sensitivity']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['auroc']:.4f} | {item['auprc_average_precision']:.4f} | "
            f"{item['brier']:.4f} | {item['ece5_equal_width']:.4f} | "
            f"TP={item['tp']}, TN={item['tn']}, FP={item['fp']}, FN={item['fn']} |"
        )
    lines.extend([
        "",
        "- Threshold is fixed at 0.5; it is not tuned on the test set.",
        "- A missing-touch negative uses deployed P(success)=0.",
        "- A missing-touch positive is refused.",
        "",
        "CASCADE_STAGE1_REPRO17_TO_STAGE2_FACTORIAL_PASS",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise RuntimeError("threshold must be between zero and one")
    if args.out_dir.exists():
        raise RuntimeError(f"refuse existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    selection = load_selection(args.selection_csv)
    combined: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}

    for variant in VARIANTS:
        indexed = prediction_index(args.stage2_oof_root, variant)
        variant_rows: list[dict[str, Any]] = []
        for selected in selection:
            pair_id = selected["raw_pair_id"]
            if pair_id not in indexed:
                raise RuntimeError(f"Stage-2 {variant} missing selected pair {pair_id}")
            prediction = indexed[pair_id]
            label = int(float(prediction["label"]))
            object_id = int(prediction["object_id"])
            fold = int(prediction["fold"])
            if (label, object_id, fold) != (
                selected["label"], selected["object_id"], selected["fold"]
            ):
                raise RuntimeError(f"selection/Stage-2 metadata mismatch: {pair_id}")
            touch_available = int(float(prediction["touch_available"]))
            raw_probability = float(prediction["probability_if_touch"])
            if touch_available:
                effective_probability = raw_probability
                policy = "stage2_model"
            elif label == 0:
                effective_probability = 0.0
                policy = "no_touch_negative_probability_zero"
            else:
                raise RuntimeError(f"missing-touch positive is unsupported: {pair_id}")
            row = {
                "variant": variant,
                **selected,
                "touch_available": touch_available,
                "stage2_raw_probability": raw_probability,
                "stage2_effective_probability": effective_probability,
                "prediction_at_threshold": int(effective_probability >= args.threshold),
                "probability_policy": policy,
            }
            variant_rows.append(row)
            combined.append(row)
        labels = [row["label"] for row in variant_rows]
        scores = [row["stage2_effective_probability"] for row in variant_rows]
        metrics[variant] = binary_metrics(labels, scores, args.threshold)

    write_csv(args.out_dir / "predictions.csv", combined)
    payload = {
        "status": "PASS",
        "selection_csv": str(args.selection_csv.resolve()),
        "selection_sha256": sha256(args.selection_csv),
        "stage1_physical_top1_success": f"{sum(row['label'] for row in selection)}/19",
        "same_raw_pair_ids": True,
        "threshold": args.threshold,
        "metrics": metrics,
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
