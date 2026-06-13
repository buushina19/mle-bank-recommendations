from __future__ import annotations

import numpy as np


def map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 7) -> float:
    """Mean Average Precision@k для multi-label ranking."""
    aps = []
    for true_row, score_row in zip(y_true, y_score):
        order = np.argsort(-score_row)
        hits = 0
        precision_sum = 0.0
        relevant = int(true_row.sum())
        if relevant == 0:
            continue
        for rank, idx in enumerate(order[:k], start=1):
            if true_row[idx]:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / min(relevant, k))
    return float(np.mean(aps)) if aps else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 7) -> float:
    hits = []
    for true_row, score_row in zip(y_true, y_score):
        top = np.argsort(-score_row)[:k]
        hits.append(true_row[top].sum() / k)
    return float(np.mean(hits))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 7) -> float:
    recalls = []
    for true_row, score_row in zip(y_true, y_score):
        relevant = int(true_row.sum())
        if relevant == 0:
            continue
        top = np.argsort(-score_row)[:k]
        recalls.append(true_row[top].sum() / relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def coverage_at_k(scores: np.ndarray, k: int = 7) -> float:
    top_products = set()
    for score_row in scores:
        top_products.update(np.argsort(-score_row)[:k].tolist())
    return len(top_products) / scores.shape[1]
