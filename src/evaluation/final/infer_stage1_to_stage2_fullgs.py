#!/usr/bin/env python3
"""Checkpoint-level Stage-1 -> Stage-2 full-GS cascade inference.

For every held-out object, the reproducible Stage-1 model scores all 150
candidate actions and selects exactly one Top-1 action.  The same
``(object_id, raw_pair_id)`` is then loaded from the Stage-2 test fold and
evaluated by the independently trained full-GS verifier:

    RGB + Action13 + bilateral Touch + left/right contact GS + corridor GS.

Stage 2 never reranks candidates, and neither Stage-1 nor Stage-2 labels are
used to choose an action.  One release action has no tactile observation and
is a physical negative; its deployed success probability is fixed to zero.
A missing-touch positive is rejected.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE1_ENTRY = (
    PROJECT_ROOT
    / "src/stage1/final/train_stage1_old12_fullcorridor_annealedSelect_v1.py"
)
DEFAULT_STAGE2_ENTRY = (
    PROJECT_ROOT
    / "src/stage2/final/train_stage2_fullgs_touch.py"
)
DEFAULT_METRICS_SOURCE = (
    Path(__file__).with_name("evaluate_fixed_stage1_to_stage2_factorial.py")
)

FOLD_OBJECTS = {
    0: [11, 39, 41, 42],
    1: [34, 35, 36],
    2: [32, 33, 93, 94],
    3: [22, 60, 84, 95],
    4: [64, 76, 77, 78],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Stage-1 Top-1 actions and infer those same actions with "
            "the deterministic Stage-2 contact+corridor full-GS verifier."
        )
    )
    parser.add_argument("--stage1-data-root", required=True, type=Path)
    parser.add_argument("--stage1-run-root", required=True, type=Path)
    parser.add_argument("--stage2-data-root", required=True, type=Path)
    parser.add_argument("--stage2-run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--stage1-entry", type=Path, default=DEFAULT_STAGE1_ENTRY)
    parser.add_argument("--stage2-entry", type=Path, default=DEFAULT_STAGE2_ENTRY)
    parser.add_argument("--metrics-source", type=Path, default=DEFAULT_METRICS_SOURCE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--expected-stage1-successes", type=int, default=17)
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=50)
    parser.add_argument("--reference-selection-csv", type=Path)
    parser.add_argument("--reference-cascade-csv", type=Path)
    parser.add_argument("--reference-probability-tolerance", type=float, default=1e-5)
    parser.add_argument("--require-reference-probabilities", action="store_true")
    return parser.parse_args()


def load_module(name: str, path: Path):
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def label_of(row: dict[str, str]) -> int:
    for key in ("label_twofinger_strict", "label"):
        value = row.get(key, "")
        if value not in ("", None):
            return int(float(value))
    raise RuntimeError(f"label missing from row with keys={sorted(row)}")


def resolve_data_root(path: Path, name: str) -> Path:
    path = path.resolve()
    if (path / "folds").is_dir():
        return path
    if (path / "withGS" / "folds").is_dir():
        return path / "withGS"
    raise RuntimeError(
        f"{name} must contain folds/ or withGS/folds/: {path}"
    )


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    return device


def configure_inference(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    expected_epoch: int,
) -> dict[str, Any]:
    path = require_file(path, "checkpoint")
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint is not a mapping: {path}")
    model_state = state.get("model", state.get("state_dict"))
    if model_state is None:
        raise RuntimeError(f"checkpoint has no model/state_dict key: {path}")
    model.load_state_dict(model_state, strict=True)
    epoch = int(state.get("epoch", -1))
    if epoch != expected_epoch:
        raise RuntimeError(
            f"checkpoint epoch mismatch: {path} epoch={epoch} "
            f"expected={expected_epoch}"
        )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": epoch,
        "state_key": "model" if "model" in state else "state_dict",
    }


def validate_model_interfaces(stage1_base, stage2_module) -> None:
    expected_stage1 = (
        "(self, rgb, action_geom, left, right, corridor, left_mask, "
        "right_mask, corridor_mask)"
    )
    expected_stage2 = (
        "(self, rgb, action_geom, left, right, corridor, left_mask, "
        "right_mask, corridor_mask, tactile_left, tactile_right, "
        "touch_available)"
    )
    actual_stage1 = str(inspect.signature(stage1_base.ExplicitLRCModel.forward))
    actual_stage2 = str(inspect.signature(
        stage2_module.Stage1BackboneWithTouch.forward
    ))
    if actual_stage1 != expected_stage1:
        raise RuntimeError(f"Stage-1 forward mismatch: {actual_stage1}")
    if actual_stage2 != expected_stage2:
        raise RuntimeError(f"Stage-2 forward mismatch: {actual_stage2}")

    stage1_model = stage1_base.ExplicitLRCModel(no_gs=False)
    stage2_model = stage2_module.Stage1BackboneWithTouch()
    if not hasattr(stage1_model, "corridor_encoder"):
        raise RuntimeError("selected Stage-1 model has no corridor encoder")
    for name in (
        "corridor_encoder", "touch_encoder", "touch_relation", "head"
    ):
        if not hasattr(stage2_model, name):
            raise RuntimeError(f"selected Stage-2 model has no {name}")
    del stage1_model, stage2_model
    print("STAGE1_STAGE2_FULLGS_INTERFACE_PASS", flush=True)


@torch.inference_mode()
def select_stage1_fold(
    fold: int,
    stage1_entry,
    data_root: Path,
    run_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    expected_epoch: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = require_file(
        data_root / "folds" / f"fold_{fold}" / "test.csv",
        f"Stage-1 fold_{fold} test CSV",
    )
    fold_run = (run_root / f"fold_{fold}").resolve()
    stats_path = require_file(
        fold_run / "fullobject_feature_stats.npz",
        f"Stage-1 fold_{fold} feature stats",
    )
    checkpoint_path = fold_run / "best_fullobject_softtop1.pt"
    rows = read_csv(csv_path)
    dataset = stage1_entry.BASE.LRCDataset(csv_path, stats_path, image_size)
    if len(rows) != len(dataset):
        raise RuntimeError(
            f"Stage-1 fold_{fold} row mismatch={len(rows)}/{len(dataset)}"
        )

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indices[int(row["object_id"])].append(index)
    if sorted(grouped_indices) != FOLD_OBJECTS[fold]:
        raise RuntimeError(
            f"Stage-1 fold_{fold} object mismatch={sorted(grouped_indices)}"
        )
    bad_counts = {
        object_id: len(indices)
        for object_id, indices in grouped_indices.items()
        if len(indices) != 150
    }
    if bad_counts:
        raise RuntimeError(f"Stage-1 fold_{fold} candidates != 150: {bad_counts}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = stage1_entry.BASE.ExplicitLRCModel(no_gs=False).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device, expected_epoch)
    model.eval()
    probabilities: list[float] = []
    for batch in loader:
        moved = move_batch(batch, device)
        logits = stage1_entry.forward(model, moved)
        probabilities.extend(
            torch.sigmoid(logits).reshape(-1).cpu().tolist()
        )
    if len(probabilities) != len(rows):
        raise RuntimeError(
            f"Stage-1 fold_{fold} probability mismatch="
            f"{len(probabilities)}/{len(rows)}"
        )

    selected: list[dict[str, Any]] = []
    for object_id in sorted(grouped_indices):
        candidates = [
            {
                "fold": fold,
                "object_id": object_id,
                "raw_pair_id": rows[index]["raw_pair_id"],
                "label": label_of(rows[index]),
                "stage1_probability": float(probabilities[index]),
                "stage1_row_index": index,
            }
            for index in grouped_indices[object_id]
        ]
        candidates.sort(
            key=lambda item: (
                -item["stage1_probability"],
                item["stage1_row_index"],
                item["raw_pair_id"],
            )
        )
        selected.append(candidates[0])

    metrics_path = require_file(
        fold_run / "test_metrics.json",
        f"Stage-1 fold_{fold} stored metrics",
    )
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    reproduced_top1 = sum(row["label"] for row in selected) / len(selected)
    if abs(reproduced_top1 - float(stored["top1_success"])) > 1e-6:
        raise RuntimeError(
            f"Stage-1 fold_{fold} Top-1 mismatch: reproduced={reproduced_top1} "
            f"stored={stored['top1_success']}"
        )
    print(
        f"Stage1 fold_{fold}: Top1={sum(x['label'] for x in selected)}/"
        f"{len(selected)} checkpoint_epoch={checkpoint['epoch']} PASS",
        flush=True,
    )

    del model, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return selected, {
        "fold": fold,
        "test_csv": str(csv_path),
        "test_csv_sha256": sha256(csv_path),
        "feature_stats": str(stats_path),
        "feature_stats_sha256": sha256(stats_path),
        "checkpoint": checkpoint,
    }


@torch.inference_mode()
def infer_stage2_fold(
    fold: int,
    selected: list[dict[str, Any]],
    stage2_module,
    data_root: Path,
    run_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    expected_epoch: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = require_file(
        data_root / "folds" / f"fold_{fold}" / "test.csv",
        f"Stage-2 fold_{fold} test CSV",
    )
    fold_run = (run_root / f"fold_{fold}").resolve()
    stats_path = require_file(
        fold_run / "fullobject_feature_stats.npz",
        f"Stage-2 fold_{fold} feature stats",
    )
    checkpoint_path = fold_run / "checkpoint_epoch_050.pt"
    rows = read_csv(csv_path)
    wanted = {
        (row["object_id"], row["raw_pair_id"]): row for row in selected
    }
    selected_indices: list[int] = []
    found: set[tuple[int, str]] = set()
    for index, row in enumerate(rows):
        key = (int(row["object_id"]), row["raw_pair_id"])
        if key not in wanted:
            continue
        if key in found:
            raise RuntimeError(f"duplicate Stage-2 selected action: {key}")
        found.add(key)
        selected_indices.append(index)
    if found != set(wanted):
        raise RuntimeError(
            f"Stage-2 fold_{fold} missing Stage-1 actions="
            f"{sorted(set(wanted) - found)}"
        )

    full_dataset = stage2_module.TouchLRCDataset(
        csv_path, stats_path, image_size
    )
    subset = Subset(full_dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = stage2_module.Stage1BackboneWithTouch().to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device, expected_epoch)
    model.eval()

    probabilities: list[float] = []
    availability: list[int] = []
    for batch in loader:
        moved = move_batch(batch, device)
        logits = stage2_module.model_forward(model, moved)
        probabilities.extend(
            torch.sigmoid(logits).reshape(-1).cpu().tolist()
        )
        availability.extend(
            (moved["touch_available"] > 0.5)
            .reshape(-1)
            .to(torch.int64)
            .cpu()
            .tolist()
        )

    if not (
        len(selected_indices) == len(probabilities) == len(availability)
    ):
        raise RuntimeError(
            f"Stage-2 fold_{fold} output mismatch: indices={len(selected_indices)} "
            f"probabilities={len(probabilities)} touch={len(availability)}"
        )

    inferred: dict[tuple[int, str], dict[str, Any]] = {}
    for index, probability, touch_available in zip(
        selected_indices, probabilities, availability
    ):
        row = rows[index]
        key = (int(row["object_id"]), row["raw_pair_id"])
        inferred[key] = {
            "label": label_of(row),
            "touch_available": int(touch_available),
            "stage2_raw_probability": float(probability),
            "stage2_row_index": index,
        }

    output: list[dict[str, Any]] = []
    for stage1_row in selected:
        key = (stage1_row["object_id"], stage1_row["raw_pair_id"])
        stage2_row = inferred[key]
        if stage2_row["label"] != stage1_row["label"]:
            raise RuntimeError(f"cross-stage label mismatch: {key}")
        if stage2_row["touch_available"]:
            effective_probability = stage2_row["stage2_raw_probability"]
            policy = "stage2_fullgs_model"
        elif stage2_row["label"] == 0:
            effective_probability = 0.0
            policy = "no_touch_negative_probability_zero"
        else:
            raise RuntimeError(f"missing-touch positive is unsupported: {key}")
        output.append({
            **stage1_row,
            **stage2_row,
            "stage2_effective_probability": effective_probability,
            "probability_policy": policy,
        })

    print(
        f"Stage2 fullGS fold_{fold}: rows={len(output)} "
        f"touch={sum(x['touch_available'] for x in output)}/{len(output)} "
        f"checkpoint_epoch={checkpoint['epoch']} PASS",
        flush=True,
    )
    del model, loader, subset, full_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, {
        "fold": fold,
        "test_csv": str(csv_path),
        "test_csv_sha256": sha256(csv_path),
        "feature_stats": str(stats_path),
        "feature_stats_sha256": sha256(stats_path),
        "checkpoint": checkpoint,
    }


def compare_reference_selection(
    generated: list[dict[str, Any]], path: Path, tolerance: float
) -> dict[str, Any]:
    reference = read_csv(require_file(path, "reference Stage-1 selection"))
    expected = {
        (int(row["object_id"]), row["raw_pair_id"]): row for row in reference
    }
    actual = {
        (int(row["object_id"]), row["raw_pair_id"]): row for row in generated
    }
    if set(actual) != set(expected):
        raise RuntimeError(
            "Stage-1 selected raw_pair_id values differ from the release "
            f"reference: missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    maximum_error = 0.0
    for key, row in actual.items():
        expected_row = expected[key]
        if row["fold"] != int(expected_row["fold"]):
            raise RuntimeError(f"reference Stage-1 fold mismatch: {key}")
        if row["label"] != int(float(expected_row["label"])):
            raise RuntimeError(f"reference Stage-1 label mismatch: {key}")
        maximum_error = max(
            maximum_error,
            abs(
                row["stage1_probability"]
                - float(expected_row["stage1_withGS_probability"])
            ),
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "same_actions": True,
        "maximum_probability_error": maximum_error,
        "probabilities_within_tolerance": maximum_error <= tolerance,
    }


def compare_reference_cascade(
    generated: list[dict[str, Any]], path: Path, tolerance: float
) -> dict[str, Any]:
    reference_rows = [
        row
        for row in read_csv(require_file(path, "reference cascade CSV"))
        if row.get("variant") == "fullgs"
    ]
    expected = {
        (int(row["object_id"]), row["raw_pair_id"]): row
        for row in reference_rows
    }
    actual = {
        (int(row["object_id"]), row["raw_pair_id"]): row for row in generated
    }
    if set(actual) != set(expected):
        raise RuntimeError("full-GS cascade actions differ from release reference")
    maximum_stage1_error = 0.0
    maximum_stage2_error = 0.0
    for key, row in actual.items():
        expected_row = expected[key]
        if row["label"] != int(float(expected_row["label"])):
            raise RuntimeError(f"reference cascade label mismatch: {key}")
        if row["touch_available"] != int(float(expected_row["touch_available"])):
            raise RuntimeError(f"reference cascade touch mismatch: {key}")
        maximum_stage1_error = max(
            maximum_stage1_error,
            abs(row["stage1_probability"] - float(expected_row["stage1_probability"])),
        )
        maximum_stage2_error = max(
            maximum_stage2_error,
            abs(
                row["stage2_raw_probability"]
                - float(expected_row["stage2_raw_probability"])
            ),
        )
    maximum_error = max(maximum_stage1_error, maximum_stage2_error)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "same_actions": True,
        "maximum_stage1_probability_error": maximum_stage1_error,
        "maximum_stage2_probability_error": maximum_stage2_error,
        "probabilities_within_tolerance": maximum_error <= tolerance,
    }


def markdown_report(
    stage1_successes: int,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    reference_checks: dict[str, Any],
) -> str:
    lines = [
        "# Checkpoint-level Stage-1 -> Stage-2 full-GS cascade",
        "",
        "Stage 1 ranks all 150 actions per object. Stage 2 receives only the "
        "same selected `raw_pair_id`; it never reranks alternatives.",
        "",
        f"- Stage-1 Top-1 physical success: {stage1_successes}/19",
        f"- Touch available: {sum(row['touch_available'] for row in rows)}/19",
        "- Stage-2 input: RGB + Action13 + bilateral Touch + contact GS + corridor GS",
        "- Stage-2 objective: weighted BCE only",
        "- Decision threshold: 0.5 (fixed; not tuned on the test set)",
        "",
        "| Acc. | Specificity | Sensitivity | Macro-F1 | AUROC | AUPRC | Brier | ECE-5 | Confusion |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {metrics['accuracy']:.4f} | {metrics['specificity']:.4f} | "
            f"{metrics['sensitivity']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['auroc']:.4f} | "
            f"{metrics['auprc_average_precision']:.4f} | "
            f"{metrics['brier']:.4f} | {metrics['ece5_equal_width']:.4f} | "
            f"TP={metrics['tp']}, TN={metrics['tn']}, "
            f"FP={metrics['fp']}, FN={metrics['fn']} |"
        ),
        "",
    ]
    if reference_checks:
        lines.extend([
            "## Frozen-reference checks",
            "",
            "```json",
            json.dumps(reference_checks, indent=2),
            "```",
            "",
        ])
    lines.extend([
        "A missing-touch negative uses P(success)=0. A missing-touch positive is refused.",
        "",
        "CHECKPOINT_STAGE1_TO_STAGE2_FULLGS_PASS",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise RuntimeError("threshold must be strictly between zero and one")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise RuntimeError("invalid DataLoader settings")
    if not 0 <= args.expected_stage1_successes <= 19:
        raise RuntimeError("expected Stage-1 successes must be in [0, 19]")

    stage1_entry_path = require_file(args.stage1_entry, "Stage-1 entry")
    stage2_entry_path = require_file(args.stage2_entry, "Stage-2 full-GS entry")
    metrics_source_path = require_file(args.metrics_source, "metrics source")
    stage1_data_root = resolve_data_root(args.stage1_data_root, "Stage-1 data root")
    stage2_data_root = resolve_data_root(args.stage2_data_root, "Stage-2 data root")
    stage1_run_root = args.stage1_run_root.resolve()
    stage2_run_root = args.stage2_run_root.resolve()
    if not stage1_run_root.is_dir() or not stage2_run_root.is_dir():
        raise RuntimeError("Stage-1 and Stage-2 run roots must exist")

    out_dir = args.out_dir.resolve()
    build_dir = out_dir.with_name(out_dir.name + ".building")
    if out_dir.exists() or build_dir.exists():
        raise RuntimeError(
            f"refuse existing output/build: final={out_dir} build={build_dir}"
        )

    device = resolve_device(args.device)
    configure_inference(device)
    stage1_entry = load_module("cascade_stage1_current_full", stage1_entry_path)
    stage2_module = load_module("cascade_stage2_current_fullgs", stage2_entry_path)
    metric_module = load_module("cascade_release_metrics", metrics_source_path)
    validate_model_interfaces(stage1_entry.BASE, stage2_module)
    print(f"device={device}", flush=True)

    build_dir.mkdir(parents=True)
    selected_all: list[dict[str, Any]] = []
    predicted_all: list[dict[str, Any]] = []
    stage1_folds: list[dict[str, Any]] = []
    stage2_folds: list[dict[str, Any]] = []

    for fold in range(5):
        selected, stage1_contract = select_stage1_fold(
            fold,
            stage1_entry,
            stage1_data_root,
            stage1_run_root,
            device,
            args.batch_size,
            args.num_workers,
            args.image_size,
            args.expected_checkpoint_epoch,
        )
        inferred, stage2_contract = infer_stage2_fold(
            fold,
            selected,
            stage2_module,
            stage2_data_root,
            stage2_run_root,
            device,
            args.batch_size,
            args.num_workers,
            args.image_size,
            args.expected_checkpoint_epoch,
        )
        selected_all.extend(selected)
        predicted_all.extend(inferred)
        stage1_folds.append(stage1_contract)
        stage2_folds.append(stage2_contract)

    if len(selected_all) != 19 or len(predicted_all) != 19:
        raise RuntimeError(
            f"expected 19 selected/inferred actions, got "
            f"{len(selected_all)}/{len(predicted_all)}"
        )
    if len({row["object_id"] for row in selected_all}) != 19:
        raise RuntimeError("Stage-1 selected object IDs are not unique")
    if len({row["raw_pair_id"] for row in selected_all}) != 19:
        raise RuntimeError("Stage-1 selected raw_pair_id values are not unique")

    stage1_successes = sum(row["label"] for row in selected_all)
    if stage1_successes != args.expected_stage1_successes:
        raise RuntimeError(
            f"Stage-1 Top-1 success mismatch={stage1_successes}/19 "
            f"expected={args.expected_stage1_successes}/19"
        )

    for row in predicted_all:
        row["prediction_at_threshold"] = int(
            row["stage2_effective_probability"] >= args.threshold
        )
    labels = [row["label"] for row in predicted_all]
    probabilities = [row["stage2_effective_probability"] for row in predicted_all]
    metrics = metric_module.binary_metrics(labels, probabilities, args.threshold)

    reference_checks: dict[str, Any] = {}
    if args.reference_selection_csv is not None:
        reference_checks["stage1_selection"] = compare_reference_selection(
            selected_all,
            args.reference_selection_csv,
            args.reference_probability_tolerance,
        )
    if args.reference_cascade_csv is not None:
        reference_checks["fullgs_cascade"] = compare_reference_cascade(
            predicted_all,
            args.reference_cascade_csv,
            args.reference_probability_tolerance,
        )
    if args.require_reference_probabilities:
        failed = [
            name
            for name, result in reference_checks.items()
            if not result["probabilities_within_tolerance"]
        ]
        if failed:
            raise RuntimeError(
                f"reference probability tolerance failed: {failed}"
            )

    selection_path = build_dir / "selected_stage1_top1.csv"
    predictions_path = build_dir / "stage2_fullgs_predictions.csv"
    write_csv(selection_path, selected_all)
    write_csv(predictions_path, predicted_all)
    contract = {
        "status": "PASS",
        "protocol": (
            "Stage-1 ranks 150 actions per object; Stage-2 full-GS predicts "
            "only the same selected raw_pair_id and never reranks."
        ),
        "stage1": {
            "variant": "RGB+Action13+contactGS+corridorGS",
            "entry": str(stage1_entry_path),
            "entry_sha256": sha256(stage1_entry_path),
            "run_root": str(stage1_run_root),
            "data_root": str(stage1_data_root),
            "top1_success": f"{stage1_successes}/19",
            "folds": stage1_folds,
        },
        "stage2": {
            "variant": "RGB+Action13+Touch+contactGS+corridorGS",
            "entry": str(stage2_entry_path),
            "entry_sha256": sha256(stage2_entry_path),
            "run_root": str(stage2_run_root),
            "data_root": str(stage2_data_root),
            "objective": "weighted BCE only",
            "threshold": args.threshold,
            "missing_touch_policy": (
                "negative -> P(success)=0; missing-touch positive refused"
            ),
            "folds": stage2_folds,
        },
        "same_19_raw_pair_ids": True,
        "reference_checks": reference_checks,
        "metrics": metrics,
    }
    contract_path = build_dir / "cascade_contract.json"
    metrics_path = build_dir / "metrics.json"
    report_path = build_dir / "report.md"
    write_json(contract_path, contract)
    write_json(metrics_path, metrics)
    report_path.write_text(
        markdown_report(stage1_successes, metrics, predicted_all, reference_checks),
        encoding="utf-8",
    )
    hash_path = build_dir / "sha256.txt"
    hash_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in (
                selection_path,
                predictions_path,
                contract_path,
                metrics_path,
                report_path,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(build_dir, out_dir)
    print((out_dir / "report.md").read_text(encoding="utf-8"), end="")
    print(f"CHECKPOINT_CASCADE_READY={out_dir}", flush=True)


if __name__ == "__main__":
    main()
