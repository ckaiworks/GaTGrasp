#!/usr/bin/env python3
"""Validate the final Stage-1 paper tables against archived source reports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"


def load(name):
    return json.loads((SOURCES / name).read_text(encoding="utf-8"))


def aggregate(report, key):
    return report["results"][key]["aggregate"]


def expected_row(agg):
    return (
        round(100.0 * int(agg["s1_count"]) / 19, 2),
        round(100.0 * int(agg["h5_count"]) / 19, 2),
        round(100.0 * int(agg["h10_count"]) / 19, 2),
        round(float(agg["mrr"]), 4),
    )


factorial = load("factorial_results.json")
completed = load("clean_completed_ablation_results.json")
corridor_only = load("corridor_only_results.json")
pooling = load("mean_max_pooling_results.json")
lambdas = load("lambda0p5_lambda2_results.json")
schedule = load("temperature_warmup_results.json")

for report in (factorial, completed, corridor_only, pooling, lambdas, schedule):
    assert report["status"] == "PASS", report

full = aggregate(factorial, "corridor_withgs")
mapping = {
    "Action Only": aggregate(completed, "clean_action_only"),
    "RGB+Action": aggregate(factorial, "nocorridor_nogs_clean"),
    "GS+Action": aggregate(completed, "clean_gs_action"),
    "RGB+Action+GS (full)": full,
    "Left/right contact patches only": aggregate(
        factorial, "nocorridor_withgs_clean"
    ),
    "Closing corridor only": corridor_only["aggregate"],
    "Contact patches + corridor (full)": full,
    "Mean Only": aggregate(pooling, "mean_only"),
    "Max Only": aggregate(pooling, "max_only"),
    "Mean + Max (full)": aggregate(pooling, "mean_plus_max_full"),
    "Weighted BCE only (lambda=0)": aggregate(completed, "true_bce_only"),
    "Weighted BCE + lambda=0.5": aggregate(lambdas, "lambda_0p5"),
    "Weighted BCE + lambda=1.0 (selected)": aggregate(
        lambdas, "baseline_lambda_1"
    ),
    "Weighted BCE + lambda=2.0": aggregate(lambdas, "lambda_2"),
    "Fixed T=1.0, warm-up 5": aggregate(schedule, "temperature_fixed_1"),
    "Cosine T 1.0 to 0.1, warm-up 10": aggregate(schedule, "warmup_10"),
    "Cosine T 1.0 to 0.1, warm-up 5 (full)": aggregate(
        schedule, "baseline"
    ),
}

with (ROOT / "table4_stage1_clean_ablations.csv").open(
    encoding="utf-8", newline=""
) as handle:
    rows = list(csv.DictReader(handle))

assert len(rows) == len(mapping), (len(rows), len(mapping))
for row in rows:
    variant = row["Variant"]
    assert variant in mapping, variant
    observed = tuple(
        float(row[key]) for key in ("S@1 (%)", "H@5 (%)", "H@10 (%)", "MRR")
    )
    expected = expected_row(mapping[variant])
    for actual, target in zip(observed, expected):
        assert math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-9), (
            variant,
            observed,
            expected,
        )

with (ROOT / "table2_primary_selector_comparison.csv").open(
    encoding="utf-8", newline=""
) as handle:
    primary = {row["Method"]: row for row in csv.DictReader(handle)}

for method, agg in {
    "Action Only": mapping["Action Only"],
    "RGB+Action": mapping["RGB+Action"],
    "Proposed Selector": full,
}.items():
    row = primary[method]
    observed = tuple(
        float(row[key]) for key in ("S@1 (%)", "H@5 (%)", "H@10 (%)", "MRR")
    )
    expected = expected_row(agg)
    assert observed == expected, (method, observed, expected)

for method, expected in {
    "RGB+Action+Depth": (73.68, 89.47, 94.74, 0.8100),
    "RGB+Action+Point Cloud": (78.95, 84.21, 94.74, 0.8207),
}.items():
    row = primary[method]
    observed = tuple(
        float(row[key]) for key in ("S@1 (%)", "H@5 (%)", "H@10 (%)", "MRR")
    )
    assert observed == expected, (method, observed, expected)
    assert row["Status"] == "current clean deterministic", row

print("CURRENT_STAGE1_TABLE2_TABLE4_VALIDATION_PASS")
