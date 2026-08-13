#!/usr/bin/env python3
"""Independently validate the portable ActiveTouchGS dataset package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


OBJECT_FOLDS = {
    0: (11, 39, 41, 42),
    1: (34, 35, 36),
    2: (32, 33, 93, 94),
    3: (22, 60, 84, 95),
    4: (64, 76, 77, 78),
}
EXPECTED_OBJECTS = tuple(sorted({obj for values in OBJECT_FOLDS.values() for obj in values}))
EXPECTED_ROWS = 2850
EXPECTED_POSITIVE = 1779
EXPECTED_NEGATIVE = 1071
EXPECTED_TOUCH = 2769
EXPECTED_BINARY_ROLES = {
    "vision_rgb": 19,
    "stage1_local_gs": 2850,
    "stage2_local_gs": 2850,
    "tactile_left": 2769,
    "tactile_right": 2769,
}
ACTION_FIELDS = (
    "estimated_width_m",
    "center_x", "center_y", "center_z",
    "contact1_x", "contact1_y", "contact1_z",
    "contact2_x", "contact2_y", "contact2_z",
    "closing_x", "closing_y", "closing_z",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def key(row: dict[str, str]) -> tuple[int, str]:
    return int(row["object_id"]), row["raw_pair_id"]


def label(row: dict[str, str]) -> int:
    return int(float(row.get("label_twofinger_strict", "") or row["label"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_relative_file(root: Path, value: str, context: str) -> None:
    if not value:
        raise RuntimeError(f"empty runtime path: {context}")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or posix.parts[:1] != ("data",):
        raise RuntimeError(f"non-portable runtime path: {context} value={value}")
    if not (root / Path(*posix.parts)).is_file():
        raise RuntimeError(f"missing runtime file: {context} value={value}")


def validate_task_rows(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metadata = root / "metadata"
    fields1, rows1 = read_csv(metadata / "stage1_all.csv")
    fields2, rows2 = read_csv(metadata / "stage2_all.csv")
    if len(rows1) != EXPECTED_ROWS or len(rows2) != EXPECTED_ROWS:
        raise RuntimeError(f"row count stage1={len(rows1)} stage2={len(rows2)}")
    if not set(fields1).issubset(fields2):
        raise RuntimeError("Stage-1 fields are not a subset of Stage-2 fields")

    for name, rows in (("stage1", rows1), ("stage2", rows2)):
        keys = [key(row) for row in rows]
        if len(set(keys)) != EXPECTED_ROWS:
            raise RuntimeError(f"duplicate task keys in {name}")
        objects = Counter(obj for obj, _ in keys)
        if tuple(sorted(objects)) != EXPECTED_OBJECTS or set(objects.values()) != {150}:
            raise RuntimeError(f"invalid object counts in {name}: {dict(objects)}")
        labels = Counter(label(row) for row in rows)
        if labels != Counter({1: EXPECTED_POSITIVE, 0: EXPECTED_NEGATIVE}):
            raise RuntimeError(f"invalid labels in {name}: {dict(labels)}")

    indexed1 = {key(row): row for row in rows1}
    indexed2 = {key(row): row for row in rows2}
    if set(indexed1) != set(indexed2):
        raise RuntimeError("Stage-1 and Stage-2 task identities differ")
    paired_fields = (
        "label", "label_twofinger_strict", *ACTION_FIELDS, "vision_rgb_path",
    )
    for pair in sorted(indexed1):
        row1, row2 = indexed1[pair], indexed2[pair]
        for field in paired_fields:
            if row1.get(field, "") != row2.get(field, ""):
                raise RuntimeError(f"paired value mismatch pair={pair} field={field}")
        if row1.get("local_gs_path", "") == row2.get("local_gs_path", ""):
            raise RuntimeError(f"Stage-1/Stage-2 GS paths unexpectedly identical pair={pair}")
        if row1.get("gs_feature_family", "") != "explicit_lrc_actionframe_orthonormal_old12_v1":
            raise RuntimeError(f"invalid Stage-1 GS family pair={pair}")
        if row2.get("gs_feature_family", "") != "explicit_lrc_actionframe_canonicalV3_worldNative12_v3":
            raise RuntimeError(f"invalid Stage-2 GS family pair={pair}")
        for row_name, row in (("stage1", row1), ("stage2", row2)):
            validate_relative_file(root, row["vision_rgb_path"], f"{row_name}:{pair}:vision")
            validate_relative_file(root, row["local_gs_path"], f"{row_name}:{pair}:gs")
        available = float(row2.get("touch_available", "0") or 0) > 0.5
        for field in ("tactile_left_after_close_path", "tactile_right_after_close_path"):
            if available:
                validate_relative_file(root, row2[field], f"stage2:{pair}:{field}")
            elif row2.get(field, ""):
                raise RuntimeError(f"touch path present when unavailable pair={pair} field={field}")

    if sum(float(row.get("touch_available", "0") or 0) > 0.5 for row in rows2) != EXPECTED_TOUCH:
        raise RuntimeError("invalid touch-availability count")

    for csv_path in (metadata / "stage1_all.csv", metadata / "stage2_all.csv", metadata / "all.csv"):
        _, csv_rows = read_csv(csv_path)
        leaked = Counter(
            field
            for row in csv_rows
            for field, value in row.items()
            if value and ("/root/" in value or ":\\" in value)
        )
        if leaked:
            examples = {
                field: next(
                    row[field]
                    for row in csv_rows
                    if row.get(field, "") and ("/root/" in row[field] or ":\\" in row[field])
                )
                for field in leaked
            }
            raise RuntimeError(
                f"absolute host paths leaked into {csv_path}: counts={dict(leaked)} examples={examples}"
            )
    return rows1, rows2


def validate_folds(root: Path, task: str, all_rows: list[dict[str, str]]) -> None:
    all_keys = {key(row) for row in all_rows}
    for fold, test_objects in OBJECT_FOLDS.items():
        fold_root = root / "metadata" / task / "folds" / f"fold_{fold}"
        _, train_rows = read_csv(fold_root / "train.csv")
        _, test_rows = read_csv(fold_root / "test.csv")
        train_keys = {key(row) for row in train_rows}
        test_keys = {key(row) for row in test_rows}
        expected_test = {item for item in all_keys if item[0] in set(test_objects)}
        if train_keys & test_keys or train_keys | test_keys != all_keys or test_keys != expected_test:
            raise RuntimeError(f"invalid {task} fold_{fold} split")


def validate_file_manifest(root: Path) -> None:
    _, rows = read_csv(root / "metadata" / "files.csv")
    roles = Counter(row["role"] for row in rows)
    if roles != Counter(EXPECTED_BINARY_ROLES):
        raise RuntimeError(f"invalid file roles: {dict(roles)}")
    paths = [row["path"] for row in rows]
    if len(paths) != sum(EXPECTED_BINARY_ROLES.values()) or len(set(paths)) != len(paths):
        raise RuntimeError("binary manifest count or uniqueness failure")
    for row in rows:
        path = root / Path(*PurePosixPath(row["path"]).parts)
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"binary manifest mismatch: {row['path']}")


def validate_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / Path(*PurePosixPath(relative).parts)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
        checked += 1
    actual = sum(1 for path in root.rglob("*") if path.is_file() and path != checksum_path)
    if checked != actual:
        raise RuntimeError(f"SHA256SUMS coverage mismatch checked={checked} actual={actual}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"missing dataset root: {root}")

    contract = json.loads((root / "metadata" / "dataset_contract.json").read_text(encoding="utf-8"))
    expected_contract = {
        "schema_version": 2,
        "rows": EXPECTED_ROWS,
        "positive": EXPECTED_POSITIVE,
        "negative": EXPECTED_NEGATIVE,
        "touch_available": EXPECTED_TOUCH,
        "binary_files": sum(EXPECTED_BINARY_ROLES.values()),
        "content_verified": True,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise RuntimeError(f"contract mismatch field={field} value={contract.get(field)}")

    rows1, rows2 = validate_task_rows(root)
    validate_folds(root, "stage1", rows1)
    validate_folds(root, "stage2", rows2)
    validate_file_manifest(root)
    checksums = validate_checksums(root) if args.verify_sha256 else 0
    print(f"rows={EXPECTED_ROWS} positive={EXPECTED_POSITIVE} negative={EXPECTED_NEGATIVE}")
    print(f"objects={len(EXPECTED_OBJECTS)} touch_available={EXPECTED_TOUCH}")
    print(f"binary_files={sum(EXPECTED_BINARY_ROLES.values())} sha256_checked={checksums}")
    print("UNIFIED_FINAL_DATASET_INDEPENDENT_ACCEPTANCE_PASS")


if __name__ == "__main__":
    main()
