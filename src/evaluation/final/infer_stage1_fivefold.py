#!/usr/bin/env python3
"""Run only the final five-fold Stage-1 selector.

This entry scores all 150 candidates for every held-out object and writes one
Top-1 action per object.  It has no dependency on a Stage-2 model.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from infer_stage1_to_stage2_fullgs import (
    compare_reference_selection,
    configure_inference,
    load_module,
    resolve_data_root,
    resolve_device,
    select_stage1_fold,
    sha256,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENTRY = (
    PROJECT_ROOT
    / "src/stage1/final/train_stage1_old12_fullcorridor_annealedSelect_v1.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--expected-successes", type=int, default=17)
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=50)
    parser.add_argument("--reference-selection-csv", type=Path)
    parser.add_argument("--reference-probability-tolerance", type=float, default=1e-5)
    parser.add_argument("--require-reference-probabilities", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise RuntimeError("invalid DataLoader settings")
    data_root = resolve_data_root(args.data_root, "Stage-1 data root")
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise RuntimeError(f"missing Stage-1 run root: {run_root}")
    out_dir = args.out_dir.resolve()
    build_dir = out_dir.with_name(out_dir.name + ".building")
    if out_dir.exists() or build_dir.exists():
        raise RuntimeError(f"refuse existing output/build: {out_dir} {build_dir}")

    device = resolve_device(args.device)
    configure_inference(device)
    entry = load_module("standalone_stage1_final", args.entry)
    signature = str(inspect.signature(entry.BASE.ExplicitLRCModel.forward))
    expected_signature = (
        "(self, rgb, action_geom, left, right, corridor, left_mask, "
        "right_mask, corridor_mask)"
    )
    if signature != expected_signature:
        raise RuntimeError(f"Stage-1 forward mismatch: {signature}")

    build_dir.mkdir(parents=True)
    selected_all = []
    fold_provenance = []
    for fold in range(5):
        selected, provenance = select_stage1_fold(
            fold,
            entry,
            data_root,
            run_root,
            device,
            args.batch_size,
            args.num_workers,
            args.image_size,
            args.expected_checkpoint_epoch,
        )
        selected_all.extend(selected)
        fold_provenance.append(provenance)

    if len(selected_all) != 19:
        raise RuntimeError(f"Stage-1 selected rows={len(selected_all)} expected=19")
    if len({row["object_id"] for row in selected_all}) != 19:
        raise RuntimeError("Stage-1 object IDs are not unique")
    if len({row["raw_pair_id"] for row in selected_all}) != 19:
        raise RuntimeError("Stage-1 raw_pair_id values are not unique")
    successes = sum(row["label"] for row in selected_all)
    if successes != args.expected_successes:
        raise RuntimeError(
            f"Stage-1 success={successes}/19 expected={args.expected_successes}/19"
        )

    reference_check = None
    if args.reference_selection_csv is not None:
        reference_check = compare_reference_selection(
            selected_all,
            args.reference_selection_csv,
            args.reference_probability_tolerance,
        )
        if (
            args.require_reference_probabilities
            and not reference_check["probabilities_within_tolerance"]
        ):
            raise RuntimeError(f"Stage-1 reference probability mismatch: {reference_check}")

    rows = [
        {
            "fold": row["fold"],
            "object_id": row["object_id"],
            "raw_pair_id": row["raw_pair_id"],
            "label": row["label"],
            "stage1_withGS_probability": row["stage1_probability"],
        }
        for row in sorted(selected_all, key=lambda item: item["object_id"])
    ]
    selection_path = build_dir / "selected_actions.csv"
    write_csv(selection_path, rows)
    payload = {
        "status": "PASS",
        "model": "reproducible corridor-withGS Stage-1",
        "physical_top1_success": f"{successes}/19",
        "entry": str(args.entry.resolve()),
        "entry_sha256": sha256(args.entry),
        "run_root": str(run_root),
        "data_root": str(data_root),
        "selection_sha256": sha256(selection_path),
        "reference_check": reference_check,
        "folds": fold_provenance,
    }
    metrics_path = build_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = build_dir / "report.md"
    report_path.write_text(
        "# Standalone Stage-1 inference\n\n"
        f"- Top-1 physical success: {successes}/19\n"
        "- Candidates ranked per object: 150\n"
        "- Stage-2 invoked: no\n\n"
        "STAGE1_FIVEFOLD_INFERENCE_PASS\n",
        encoding="utf-8",
    )
    (build_dir / "sha256.txt").write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in (selection_path, metrics_path, report_path)
        )
        + "\n",
        encoding="utf-8",
    )
    build_dir.rename(out_dir)
    print(f"STAGE1_FIVEFOLD_INFERENCE_PASS success={successes}/19")
    print(f"SELECTION={out_dir / 'selected_actions.csv'}")


if __name__ == "__main__":
    main()
