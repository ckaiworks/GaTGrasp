#!/usr/bin/env python3
"""Evaluate the non-neural Table-2 selector baselines.

The candidate CSV must contain 150 candidates for each object.  Random and
Oracle need only labels.  Geometry needs the three frozen candidate-pool
cosines.  Gaussian quality needs the frozen score column produced by the
original Gaussian-quality extractor; no new definition is inferred here.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def label_of(row):
    value = row.get("label_twofinger_strict", row.get("label", "0"))
    return int(float(value) >= 0.5)


def score_of(row, method, quality_field):
    if method == "geometry":
        names = (
            "a_endpoint1_inward_cos",
            "b_endpoint2_outward_cos",
            "c_normal_opposition_cos",
        )
        return sum(float(row[name]) for name in names) / 3.0
    if method == "gaussian_quality":
        if quality_field not in row or row[quality_field] == "":
            raise RuntimeError(
                "Gaussian-quality reproduction requires the frozen score "
                f"column --quality-field {quality_field!r}."
            )
        return float(row[quality_field])
    raise ValueError(method)


def exact_random_group(n, positives):
    if positives <= 0:
        return 0.0, 0.0, 0.0, 0.0
    s1 = positives / n
    miss5 = math.comb(n - positives, 5) / math.comb(n, 5) if n - positives >= 5 else 0.0
    miss10 = math.comb(n - positives, 10) / math.comb(n, 10) if n - positives >= 10 else 0.0
    mrr = 0.0
    for rank in range(1, n - positives + 2):
        after_n = n - rank
        after_k = positives - 1
        ways = math.comb(after_n, after_k) if after_n >= after_k >= 0 else 0
        mrr += ways / math.comb(n, positives) / rank
    return s1, 1.0 - miss5, 1.0 - miss10, mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument(
        "--method",
        required=True,
        choices=("random", "geometry", "gaussian_quality", "oracle"),
    )
    parser.add_argument("--quality-field", default="gaussian_quality_score")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["object_id"])].append(row)
    if not groups or any(len(group) != 150 for group in groups.values()):
        raise RuntimeError("expected exactly 150 candidates per object")

    per_object = []
    for object_id, group in sorted(groups.items()):
        positives = sum(label_of(row) for row in group)
        if args.method == "random":
            s1, h5, h10, mrr = exact_random_group(len(group), positives)
            rank = None
        else:
            if args.method == "oracle":
                ordered = sorted(group, key=label_of, reverse=True)
            else:
                ordered = sorted(
                    group,
                    key=lambda row: score_of(row, args.method, args.quality_field),
                    reverse=True,
                )
            rank = next(
                (index for index, row in enumerate(ordered, 1) if label_of(row)),
                None,
            )
            s1 = float(rank == 1)
            h5 = float(rank is not None and rank <= 5)
            h10 = float(rank is not None and rank <= 10)
            mrr = 0.0 if rank is None else 1.0 / rank
        per_object.append(
            {"object_id": object_id, "rank": rank, "s1": s1, "h5": h5, "h10": h10, "mrr": mrr}
        )

    count = len(per_object)
    metrics = {
        "method": args.method,
        "objects": count,
        "s_at_1": sum(item["s1"] for item in per_object) / count,
        "h_at_5": sum(item["h5"] for item in per_object) / count,
        "h_at_10": sum(item["h10"] for item in per_object) / count,
        "mrr": sum(item["mrr"] for item in per_object) / count,
        "per_object": per_object,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
