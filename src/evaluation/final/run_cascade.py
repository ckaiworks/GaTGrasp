#!/usr/bin/env python3
"""Run the final Stage-1 selector and optionally connect final Stage-2 models.

This is the only public cascade entry.  Stage 1 runs exactly once and writes
one Top-1 ``raw_pair_id`` per object.  Every requested Stage-2 variant then
evaluates those same 19 actions; Stage 2 never reranks the candidate pool.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from evaluate_fixed_stage1_to_retrained_stage2_variants import binary_metrics
from infer_stage1_to_stage2_fullgs import (
    configure_inference,
    label_of,
    load_checkpoint,
    load_module,
    move_batch,
    read_csv,
    require_file,
    resolve_data_root,
    resolve_device,
    sha256,
    write_csv,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE1_SCRIPT = HERE / "infer_stage1_fivefold.py"

VARIANTS = {
    "fullgs": {
        "entry": PROJECT_ROOT / "src/stage2/final/train_stage2_fullgs_touch.py",
        "model_class": "Stage1BackboneWithTouch",
        "experiment": "stage2_stage1backbone_old12_plus_touch_v1",
        "label": "RGB+Action13+ContactGS+CorridorGS+Touch",
    },
    "nogs": {
        "entry": PROJECT_ROOT / "experiments/table3_tactile_verification/methods/rgb_action_touch/train.py",
        "model_class": "CleanSingleHeadStage2",
        "experiment": "stage2_clean_singlehead_rgb_action_touch_weighted_bce_v1",
        "label": "RGB+Action13+Touch",
    },
    "touch_only": {
        "entry": PROJECT_ROOT / "experiments/table3_tactile_verification/methods/touch_only/train.py",
        "model_class": "CleanTouchOnlyStage2",
        "experiment": "stage2_touchonly_weighted_bce_deterministic_v1",
        "label": "Touch Only",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-data-root", required=True, type=Path)
    parser.add_argument("--stage1-run-root", required=True, type=Path)
    parser.add_argument("--stage2-data-root", type=Path)
    parser.add_argument("--fullgs-run-root", type=Path)
    parser.add_argument("--nogs-run-root", type=Path)
    parser.add_argument("--touch-only-run-root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--stage2-variants",
        default="fullgs",
        help="none, all, or comma-separated: fullgs,nogs,touch_only",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=50)
    parser.add_argument("--reference-selection-csv", type=Path)
    parser.add_argument("--require-reference-probabilities", action="store_true")
    return parser.parse_args()


def variants_of(value: str) -> list[str]:
    if value == "none":
        return []
    if value == "all":
        return ["touch_only", "nogs", "fullgs"]
    variants = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(variants) - set(VARIANTS))
    if invalid or len(set(variants)) != len(variants):
        raise RuntimeError(f"invalid or duplicate Stage-2 variants: {invalid or variants}")
    return variants


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_selection(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(require_file(path, "Stage-1 selection")):
        rows.append(
            {
                "fold": int(row["fold"]),
                "object_id": int(row["object_id"]),
                "raw_pair_id": row["raw_pair_id"],
                "label": int(float(row["label"])),
                "stage1_withGS_probability": float(row["stage1_withGS_probability"]),
            }
        )
    if len(rows) != 19 or len({row["object_id"] for row in rows}) != 19:
        raise RuntimeError("Stage-1 selection must contain 19 unique objects")
    if len({row["raw_pair_id"] for row in rows}) != 19:
        raise RuntimeError("Stage-1 raw_pair_id values must be unique")
    if sum(row["label"] for row in rows) != 17:
        raise RuntimeError("Stage-1 output is not the required 17/19 selection")
    return sorted(rows, key=lambda row: row["object_id"])


def validate_run_contract(path: Path, expected_experiment: str) -> dict[str, Any]:
    contract = json.loads(require_file(path, "Stage-2 run contract").read_text())
    expected = {
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
        key: {"actual": contract.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Stage-2 contract mismatch {path}: {mismatches}")
    return contract


def classification_outcome(label: int, prediction: int) -> str:
    """Return an unambiguous binary-classification outcome name."""
    if label == 1:
        return "TP" if prediction == 1 else "FN"
    return "FP" if prediction == 1 else "TN"


def positive_prediction_summary(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """Summarize how the actually positive selected actions were predicted.

    ``true_positive_count`` is deliberately not called merely "predicted
    success": it counts only rows whose ground-truth label is 1 *and* whose
    thresholded prediction is 1.  Probabilities are the effective Stage-2
    success probabilities used for the threshold decision.
    """
    positive_rows = [row for row in rows if int(row["label"]) == 1]
    true_positive_rows = [
        row for row in positive_rows if int(row["prediction_at_0p5"]) == 1
    ]
    false_negative_rows = [
        row for row in positive_rows if int(row["prediction_at_0p5"]) == 0
    ]
    true_positive_probabilities = [
        float(row["stage2_effective_probability"]) for row in true_positive_rows
    ]

    def sample_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "fold": int(row["fold"]),
            "object_id": int(row["object_id"]),
            "raw_pair_id": row["raw_pair_id"],
            "probability": float(row["stage2_effective_probability"]),
            "prediction": int(row["prediction_at_0p5"]),
            "outcome": row["classification_outcome"],
        }

    actual_positive_count = len(positive_rows)
    true_positive_count = len(true_positive_rows)
    return {
        "threshold": float(threshold),
        "actual_positive_count": actual_positive_count,
        "true_positive_count": true_positive_count,
        "false_negative_count": len(false_negative_rows),
        "true_positive_fraction": (
            true_positive_count / actual_positive_count
            if actual_positive_count else float("nan")
        ),
        "true_positive_probability_mean": (
            sum(true_positive_probabilities) / len(true_positive_probabilities)
            if true_positive_probabilities else float("nan")
        ),
        "true_positive_probability_min": (
            min(true_positive_probabilities)
            if true_positive_probabilities else float("nan")
        ),
        "true_positive_probability_max": (
            max(true_positive_probabilities)
            if true_positive_probabilities else float("nan")
        ),
        "true_positive_samples": [
            sample_record(row) for row in true_positive_rows
        ],
        "false_negative_samples": [
            sample_record(row) for row in false_negative_rows
        ],
    }


@torch.inference_mode()
def infer_variant(
    variant: str,
    selection: list[dict[str, Any]],
    data_root: Path,
    run_root: Path,
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = VARIANTS[variant]
    module = load_module(
        f"cascade_stage2_{variant}", require_file(config["entry"], "Stage-2 entry")
    )
    model_type = getattr(module, config["model_class"])
    expected_signature = (
        "(self, rgb, action_geom, left, right, corridor, left_mask, right_mask, "
        "corridor_mask, tactile_left, tactile_right, touch_available)"
    )
    actual_signature = str(inspect.signature(model_type.forward))
    if actual_signature != expected_signature:
        raise RuntimeError(f"{variant} forward mismatch: {actual_signature}")

    output_rows: list[dict[str, Any]] = []
    fold_provenance = []
    for fold in range(5):
        selected_fold = [row for row in selection if row["fold"] == fold]
        csv_path = require_file(
            data_root / "folds" / f"fold_{fold}" / "test.csv", "Stage-2 test CSV"
        )
        fold_root = run_root / f"fold_{fold}"
        stats_path = require_file(
            fold_root / "fullobject_feature_stats.npz", "Stage-2 feature stats"
        )
        contract = validate_run_contract(
            fold_root / "run_contract.json", config["experiment"]
        )
        source_rows = read_csv(csv_path)
        wanted = {(row["object_id"], row["raw_pair_id"]): row for row in selected_fold}
        indices = [
            index
            for index, row in enumerate(source_rows)
            if (int(row["object_id"]), row["raw_pair_id"]) in wanted
        ]
        if len(indices) != len(wanted):
            raise RuntimeError(
                f"{variant} fold_{fold} selected lookup={len(indices)}/{len(wanted)}"
            )

        dataset = module.TouchLRCDataset(csv_path, stats_path, args.image_size)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = model_type().to(device)
        checkpoint = load_checkpoint(
            model,
            fold_root / "checkpoint_epoch_050.pt",
            device,
            args.expected_checkpoint_epoch,
        )
        model.eval()
        probabilities: list[float] = []
        availability: list[int] = []
        for batch in loader:
            moved = move_batch(batch, device)
            probabilities.extend(
                torch.sigmoid(module.model_forward(model, moved))
                .reshape(-1)
                .cpu()
                .tolist()
            )
            availability.extend(
                (moved["touch_available"] > 0.5)
                .to(torch.int64)
                .reshape(-1)
                .cpu()
                .tolist()
            )
        inferred = {}
        for index, probability, touch_available in zip(
            indices, probabilities, availability
        ):
            source_row = source_rows[index]
            inferred[(int(source_row["object_id"]), source_row["raw_pair_id"])] = {
                "label": label_of(source_row),
                "touch_available": int(touch_available),
                "stage2_raw_probability": float(probability),
            }
        for stage1_row in selected_fold:
            key = (stage1_row["object_id"], stage1_row["raw_pair_id"])
            stage2_row = inferred[key]
            if stage2_row["label"] != stage1_row["label"]:
                raise RuntimeError(f"{variant} cross-stage label mismatch: {key}")
            if stage2_row["touch_available"]:
                effective_probability = stage2_row["stage2_raw_probability"]
                policy = "stage2_model"
            elif stage2_row["label"] == 0:
                effective_probability = 0.0
                policy = "no_touch_negative_probability_zero"
            else:
                raise RuntimeError(f"missing-touch positive unsupported: {key}")
            prediction = int(effective_probability >= args.threshold)
            outcome = classification_outcome(stage1_row["label"], prediction)
            output_rows.append(
                {
                    "variant": variant,
                    **stage1_row,
                    **stage2_row,
                    "stage2_effective_probability": effective_probability,
                    "prediction_at_0p5": prediction,
                    "positive_predicted_success": int(outcome == "TP"),
                    "classification_outcome": outcome,
                    "probability_policy": policy,
                }
            )
        fold_provenance.append(
            {
                "fold": fold,
                "checkpoint": checkpoint,
                "test_csv": str(csv_path.resolve()),
                "test_csv_sha256": sha256(csv_path),
                "feature_stats_sha256": sha256(stats_path),
                "contract": contract,
            }
        )
        print(
            f"{variant} fold_{fold}: rows={len(selected_fold)} "
            f"touch={sum(row['touch_available'] for row in output_rows if row['fold'] == fold)}"
            f"/{len(selected_fold)} PASS",
            flush=True,
        )
        del model, loader, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_rows.sort(key=lambda row: row["object_id"])
    if [row["raw_pair_id"] for row in output_rows] != [
        row["raw_pair_id"] for row in selection
    ]:
        raise RuntimeError(f"{variant} did not preserve the Stage-1 action list")
    metrics = binary_metrics(
        [row["label"] for row in output_rows],
        [row["stage2_effective_probability"] for row in output_rows],
        args.threshold,
    )
    positive_summary = positive_prediction_summary(output_rows, args.threshold)
    if positive_summary["true_positive_count"] != metrics["tp"]:
        raise RuntimeError(
            f"{variant} TP summary mismatch: "
            f"{positive_summary['true_positive_count']} != {metrics['tp']}"
        )
    out_dir.mkdir(parents=True)
    predictions_path = out_dir / "predictions.csv"
    write_csv(predictions_path, output_rows)
    payload = {
        "status": "PASS",
        "variant": variant,
        "input": config["label"],
        "selection_sha256": sha256(args._selection_path),
        "same_stage1_actions": True,
        "metrics": metrics,
        "positive_prediction_summary": positive_summary,
        "folds": fold_provenance,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def report_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Final checkpoint-level Stage-1 -> Stage-2 cascade",
        "",
        "Stage 1 runs once. Every Stage-2 row uses exactly the same selected actions.",
        "",
        "| Stage-2 input | Acc. | Macro-F1 | AUROC | AUPRC | Brier | ECE-5 | TP/actual positive | TP prob. mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, payload in results.items():
        item = payload["metrics"]
        positive = payload["positive_prediction_summary"]
        lines.append(
            f"| {payload['input']} | {item['accuracy']:.4f} | "
            f"{item['macro_f1']:.4f} | {item['auroc']:.4f} | "
            f"{item['auprc_average_precision']:.4f} | {item['brier']:.4f} | "
            f"{item['ece5_equal_width']:.4f} | "
            f"{positive['true_positive_count']}/{positive['actual_positive_count']} | "
            f"{positive['true_positive_probability_mean']:.4f} |"
        )
    lines.extend(["", "## Actually positive samples predicted successfully", ""])
    for payload in results.values():
        positive = payload["positive_prediction_summary"]
        probabilities = ", ".join(
            f"obj_{sample['object_id']}={sample['probability']:.6f}"
            for sample in positive["true_positive_samples"]
        )
        missed = ", ".join(
            f"obj_{sample['object_id']}={sample['probability']:.6f}"
            for sample in positive["false_negative_samples"]
        ) or "none"
        lines.extend(
            [
                f"### {payload['input']}",
                "",
                f"- True positives: {positive['true_positive_count']}/"
                f"{positive['actual_positive_count']}",
                f"- TP success probabilities: {probabilities}",
                f"- Missed actual positives (FN): {missed}",
                "",
            ]
        )
    lines.extend(["CASCADE_CHECKPOINT_INFERENCE_PASS", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.num_workers != 0:
        raise RuntimeError("final deterministic cascade requires num_workers=0")
    if not 0.0 < args.threshold < 1.0:
        raise RuntimeError("threshold must be between zero and one")
    variants = variants_of(args.stage2_variants)
    final_dir = args.out_dir.resolve()
    build_dir = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or build_dir.exists():
        raise RuntimeError(f"refuse existing output/build: {final_dir} {build_dir}")
    build_dir.mkdir(parents=True)

    stage1_out = build_dir / "stage1"
    stage1_command = [
        sys.executable,
        str(STAGE1_SCRIPT),
        "--data-root",
        str(args.stage1_data_root),
        "--run-root",
        str(args.stage1_run_root),
        "--out-dir",
        str(stage1_out),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--image-size",
        str(args.image_size),
        "--expected-successes",
        "17",
        "--expected-checkpoint-epoch",
        str(args.expected_checkpoint_epoch),
    ]
    if args.reference_selection_csv is not None:
        stage1_command += [
            "--reference-selection-csv",
            str(args.reference_selection_csv),
        ]
    if args.require_reference_probabilities:
        stage1_command.append("--require-reference-probabilities")
    run(stage1_command)

    selection_path = stage1_out / "selected_actions.csv"
    selection = load_selection(selection_path)
    args._selection_path = selection_path
    if not variants:
        build_dir.rename(final_dir)
        print("CASCADE_STAGE1_ONLY_PASS")
        return
    if args.stage2_data_root is None:
        raise RuntimeError("--stage2-data-root is required when Stage-2 is enabled")
    roots = {
        "fullgs": args.fullgs_run_root,
        "nogs": args.nogs_run_root,
        "touch_only": args.touch_only_run_root,
    }
    missing = [variant for variant in variants if roots[variant] is None]
    if missing:
        raise RuntimeError(f"missing Stage-2 run roots for: {missing}")

    device = resolve_device(args.device)
    configure_inference(device)
    data_root = resolve_data_root(args.stage2_data_root, "Stage-2 data root")
    results = {}
    for variant in variants:
        run_root = roots[variant].resolve()
        if not run_root.is_dir():
            raise RuntimeError(f"missing {variant} run root: {run_root}")
        results[variant] = infer_variant(
            variant,
            selection,
            data_root,
            run_root,
            build_dir / "stage2" / variant,
            device,
            args,
        )

    selection_hashes = {payload["selection_sha256"] for payload in results.values()}
    if selection_hashes != {sha256(selection_path)}:
        raise RuntimeError(f"Stage-2 selection hash mismatch: {selection_hashes}")
    payload = {
        "status": "PASS",
        "stage1_top1_success": "17/19",
        "stage1_selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "same_stage1_selection_for_all_stage2_variants": True,
        "stage2_variants": variants,
        "metrics": {variant: result["metrics"] for variant, result in results.items()},
        "positive_prediction_summary": {
            variant: result["positive_prediction_summary"]
            for variant, result in results.items()
        },
    }
    metrics_path = build_dir / "cascade_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = build_dir / "report.md"
    report_path.write_text(report_markdown(results), encoding="utf-8")
    (build_dir / "sha256.txt").write_text(
        f"{sha256(metrics_path)}  cascade_metrics.json\n"
        f"{sha256(report_path)}  report.md\n",
        encoding="utf-8",
    )
    report_text = report_path.read_text(encoding="utf-8")
    build_dir.rename(final_dir)
    print(report_text, end="")
    print(f"OUT={final_dir}")


if __name__ == "__main__":
    main()
