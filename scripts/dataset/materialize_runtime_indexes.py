#!/usr/bin/env python3
"""Create absolute-path runtime CSVs from a portable dataset extraction.

The selected Stage-1 and Stage-2 models share the same old12 GS features and
action identities. Stage 2 adds the tactile fields from ``stage2_all.csv``.
The earlier canonicalV3 Stage-2 index is still materialized under
``stage2_canonical_v3`` for historical factorial experiments only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path, PurePosixPath


RUNTIME_PATH_FIELDS = {
    "vision_rgb_path",
    "local_gs_path",
    "local_gs_feature_path",
    "tactile_left_after_close_path",
    "tactile_right_after_close_path",
}
IDENTITY_FIELDS = ("object_id", "raw_pair_id", "label", "label_twofinger_strict")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def materialize(dataset_root: Path, source: Path, target: Path) -> dict[str, int]:
    fields, rows = read_csv(source)
    before = [tuple(row.get(field, "") for field in IDENTITY_FIELDS) for row in rows]
    changed: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        row["data_root"] = str(dataset_root)
        for field in RUNTIME_PATH_FIELDS:
            value = row.get(field, "")
            if not value:
                continue
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"non-portable source path row={row_index} field={field} value={value}")
            absolute = (dataset_root / Path(*relative.parts)).resolve()
            if not absolute.is_file():
                raise RuntimeError(f"missing runtime file row={row_index} field={field} path={absolute}")
            row[field] = str(absolute)
            changed[field] += 1
    after = [tuple(row.get(field, "") for field in IDENTITY_FIELDS) for row in rows]
    if before != after:
        raise RuntimeError(f"sample identity changed while materializing {source}")
    write_csv(target, fields, rows)
    return dict(changed)


def selected_stage2_rows(
    stage1_source: Path, stage2_source: Path
) -> tuple[list[str], list[dict[str, str]]]:
    """Bind Stage-2 touch fields to the exact selected Stage-1 old12 GS rows."""
    _, stage1 = read_csv(stage1_source)
    stage2_fields, stage2 = read_csv(stage2_source)
    key = lambda row: (int(row["object_id"]), row["raw_pair_id"])
    indexed_stage1 = {key(row): row for row in stage1}
    if len(indexed_stage1) != len(stage1) or len(stage1) != len(stage2):
        raise RuntimeError(
            f"Stage-1/Stage-2 identity count mismatch: {len(stage1)}/{len(stage2)}"
        )
    merged = []
    for row in stage2:
        identity = key(row)
        if identity not in indexed_stage1:
            raise RuntimeError(f"Stage-2 identity missing from Stage-1: {identity}")
        stage1_row = indexed_stage1[identity]
        for field in IDENTITY_FIELDS:
            if row.get(field, "") != stage1_row.get(field, ""):
                raise RuntimeError(
                    f"cross-stage identity mismatch identity={identity} field={field}"
                )
        output = dict(row)
        output["local_gs_path"] = stage1_row["local_gs_path"]
        if "local_gs_feature_path" in output:
            output["local_gs_feature_path"] = stage1_row.get(
                "local_gs_feature_path", stage1_row["local_gs_path"]
            )
        output["gs_feature_family"] = stage1_row["gs_feature_family"]
        merged.append(output)
    if {row["gs_feature_family"] for row in merged} != {
        "explicit_lrc_actionframe_orthonormal_old12_v1"
    }:
        raise RuntimeError("selected Stage-2 runtime index is not old12")
    return stage2_fields, merged


def materialize_selected_stage2(
    dataset_root: Path,
    stage1_source: Path,
    stage2_source: Path,
    target: Path,
) -> dict[str, int]:
    fields, rows = selected_stage2_rows(stage1_source, stage2_source)
    changed: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        row["data_root"] = str(dataset_root)
        for field in RUNTIME_PATH_FIELDS:
            value = row.get(field, "")
            if not value:
                continue
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    f"non-portable selected Stage-2 path row={row_index} "
                    f"field={field} value={value}"
                )
            absolute = (dataset_root / Path(*relative.parts)).resolve()
            if not absolute.is_file():
                raise RuntimeError(
                    f"missing selected Stage-2 file row={row_index} "
                    f"field={field} path={absolute}"
                )
            row[field] = str(absolute)
            changed[field] += 1
    write_csv(target, fields, rows)
    return dict(changed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    out = args.out.resolve()
    if not (root / "metadata" / "dataset_contract.json").is_file():
        raise RuntimeError(f"not an ActiveTouchGS dataset root: {root}")
    if out.exists():
        raise RuntimeError(f"refuse existing output: {out}")

    direct_jobs = [
        (root / "metadata" / "stage1_all.csv", out / "stage1" / "all.csv"),
        (
            root / "metadata" / "stage2_all.csv",
            out / "stage2_canonical_v3" / "all.csv",
        ),
    ]
    selected_stage2_jobs = [
        (
            root / "metadata" / "stage1_all.csv",
            root / "metadata" / "stage2_all.csv",
            out / "stage2" / "all.csv",
        )
    ]
    for fold in range(5):
        for split in ("train", "test"):
            stage1_source = (
                root / "metadata" / "stage1" / "folds"
                / f"fold_{fold}" / f"{split}.csv"
            )
            stage2_source = (
                root / "metadata" / "stage2" / "folds"
                / f"fold_{fold}" / f"{split}.csv"
            )
            direct_jobs.extend(
                [
                    (
                        stage1_source,
                        out / "stage1" / "folds" / f"fold_{fold}" / f"{split}.csv",
                    ),
                    (
                        stage2_source,
                        out / "stage2_canonical_v3" / "folds"
                        / f"fold_{fold}" / f"{split}.csv",
                    ),
                ]
            )
            selected_stage2_jobs.append(
                (
                    stage1_source,
                    stage2_source,
                    out / "stage2" / "folds" / f"fold_{fold}" / f"{split}.csv",
                )
            )

    results = {}
    for source, target in direct_jobs:
        if not source.is_file():
            raise RuntimeError(f"missing portable index: {source}")
        results[target.relative_to(out).as_posix()] = materialize(root, source, target)
    for stage1_source, stage2_source, target in selected_stage2_jobs:
        for source in (stage1_source, stage2_source):
            if not source.is_file():
                raise RuntimeError(f"missing portable index: {source}")
        results[target.relative_to(out).as_posix()] = materialize_selected_stage2(
            root, stage1_source, stage2_source, target
        )
    contract = {
        "schema_version": 2,
        "dataset_root": str(root),
        "indexes": len(direct_jobs) + len(selected_stage2_jobs),
        "selected_stage2_gs_family": "explicit_lrc_actionframe_orthonormal_old12_v1",
        "historical_stage2_index": "stage2_canonical_v3",
        "runtime_path_fields": sorted(RUNTIME_PATH_FIELDS),
        "changed_values": results,
    }
    (out / "runtime_index_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"runtime_indexes={len(direct_jobs) + len(selected_stage2_jobs)}")
    print(f"output={out}")
    print("PORTABLE_RUNTIME_INDEX_MATERIALIZATION_PASS")


if __name__ == "__main__":
    main()
