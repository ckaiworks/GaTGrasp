"""Shared, non-executable components for the released Stage-1 models.

This module contains only the data helpers, RGB encoder, and evaluation
metrics used by the current single-logit Stage-1 implementation.  It is
intentionally not a historical training entry point.
"""

import csv
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def get_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None, "None"):
        return float(default)
    return float(value)


def build_action_geom(row):
    """Return the released 13-D action vector in its canonical order."""
    values = [
        get_float(row, "estimated_width_m"),
        get_float(row, "center_x"),
        get_float(row, "center_y"),
        get_float(row, "center_z"),
        get_float(row, "contact1_x"),
        get_float(row, "contact1_y"),
        get_float(row, "contact1_z"),
        get_float(row, "contact2_x"),
        get_float(row, "contact2_y"),
        get_float(row, "contact2_z"),
        get_float(row, "closing_x"),
        get_float(row, "closing_y"),
        get_float(row, "closing_z"),
    ]
    return np.asarray(values, dtype=np.float32)


class RGBEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, value):
        return self.net(value).flatten(1)


def auroc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    positive = y_true == 1
    negative = y_true == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    auc = (
        ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    return float(auc)


def compute_metrics(labels, probabilities):
    labels = np.asarray(labels).astype(np.int64)
    probabilities = np.asarray(probabilities).astype(np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    accuracy = float((predictions == labels).mean())
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision_positive = tp / max(tp + fp, 1)
    recall_positive = tp / max(tp + fn, 1)
    f1_positive = 2 * precision_positive * recall_positive / max(
        precision_positive + recall_positive, 1e-12
    )
    precision_negative = tn / max(tn + fn, 1)
    recall_negative = tn / max(tn + fp, 1)
    f1_negative = 2 * precision_negative * recall_negative / max(
        precision_negative + recall_negative, 1e-12
    )
    return {
        "acc": accuracy,
        "macro_f1": float(0.5 * (f1_positive + f1_negative)),
        "f1_pos": float(f1_positive),
        "f1_neg": float(f1_negative),
        "auroc": float(auroc_score(labels, probabilities)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pos_ratio": float(labels.mean()),
        "prob_mean": float(probabilities.mean()),
    }


def compute_ranking_metrics(labels, probabilities, object_ids, top_k=5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    object_ids = np.asarray(object_ids, dtype=np.int64)
    if not len(labels) == len(probabilities) == len(object_ids):
        raise RuntimeError("ranking metric input lengths differ")
    top1, top5, reciprocal_ranks, average_precisions = [], [], [], []
    for object_id in sorted(np.unique(object_ids).tolist()):
        indices = np.flatnonzero(object_ids == object_id)
        object_labels = labels[indices]
        object_probabilities = probabilities[indices]
        order = np.argsort(-object_probabilities, kind="mergesort")
        object_labels = object_labels[order]
        top1.append(float(object_labels[0] == 1))
        top5.append(float(np.any(object_labels[: min(top_k, len(object_labels))] == 1)))
        positive = np.flatnonzero(object_labels == 1)
        if len(positive) == 0:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / float(positive[0] + 1))
            precision = np.cumsum(object_labels) / np.arange(1, len(object_labels) + 1)
            average_precisions.append(
                float((precision * object_labels).sum() / int(object_labels.sum()))
            )
    return {
        "ranking_object_count": int(len(top1)),
        "top1_success": float(np.mean(top1)),
        "top5_success": float(np.mean(top5)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_ap": float(np.mean(average_precisions)),
    }


def safe_metrics(labels, probabilities, object_ids):
    if len(labels) == 0:
        return {}
    metrics = compute_metrics(labels, probabilities)
    metrics.update(compute_ranking_metrics(labels, probabilities, object_ids))
    return metrics
