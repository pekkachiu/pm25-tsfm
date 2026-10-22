"""Point, interval and exceedance metrics. NaN pairs are ignored."""

import numpy as np


def _valid_pairs(*arrays: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"shape mismatch: {[a.shape for a in arrays]}")
    mask = np.all([~np.isnan(a) for a in arrays], axis=0)
    return [a[mask] for a in arrays]


def mae(y_true, y_pred) -> float:
    y, p = _valid_pairs(y_true, y_pred)
    return float(np.mean(np.abs(p - y))) if y.size else float("nan")


def rmse(y_true, y_pred) -> float:
    y, p = _valid_pairs(y_true, y_pred)
    return float(np.sqrt(np.mean((p - y) ** 2))) if y.size else float("nan")


def interval_coverage(y_true, lower, upper) -> float:
    y, lo, hi = _valid_pairs(y_true, lower, upper)
    return float(np.mean((y >= lo) & (y <= hi))) if y.size else float("nan")


def exceedance_scores(y_true, y_pred, threshold: float) -> dict:
    """Precision/recall/F1 treating hours with PM2.5 >= threshold as positives."""
    y, p = _valid_pairs(y_true, y_pred)
    actual, predicted = y >= threshold, p >= threshold
    tp = int(np.sum(actual & predicted))
    precision = tp / predicted.sum() if predicted.sum() else 0.0
    recall = tp / actual.sum() if actual.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "support": int(actual.sum())}
