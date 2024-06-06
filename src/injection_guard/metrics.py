"""Dependency-free binary classification metrics."""

from __future__ import annotations

import numpy as np


def roc_auc_score(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have the same non-zero length")
    targets = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both classes")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("labels must be binary")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[targets == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def confusion_counts(labels: list[int], predictions: list[bool]) -> dict[str, int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    pairs = list(zip(labels, predictions, strict=True))
    return {
        "tp": sum(bool(label) and prediction for label, prediction in pairs),
        "fp": sum(not bool(label) and prediction for label, prediction in pairs),
        "tn": sum(not bool(label) and not prediction for label, prediction in pairs),
        "fn": sum(bool(label) and not prediction for label, prediction in pairs),
    }


def precision_recall(counts: dict[str, int]) -> tuple[float, float]:
    predicted_positive = counts["tp"] + counts["fp"]
    actual_positive = counts["tp"] + counts["fn"]
    precision = counts["tp"] / predicted_positive if predicted_positive else 0.0
    recall = counts["tp"] / actual_positive if actual_positive else 0.0
    return precision, recall
