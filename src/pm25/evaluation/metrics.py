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


def weighted_quantile_loss(y_true, quantile_preds: dict[float, np.ndarray]) -> float:
    """Mean over quantile levels of 2 * sum(pinball loss) / sum(|y|) (GluonTS wQL).

    Hours where y or any quantile is NaN are dropped for all levels alike.
    """
    levels = sorted(quantile_preds)
    y, *preds = _valid_pairs(y_true, *(quantile_preds[q] for q in levels))
    scale = np.sum(np.abs(y))
    if not y.size or scale == 0:
        return float("nan")
    losses = [np.sum(np.maximum(q * (y - p), (q - 1) * (y - p))) for q, p in zip(levels, preds)]
    return float(np.mean([2 * loss / scale for loss in losses]))


def interval_score(y_true, lower, upper, alpha: float) -> float:
    """Gneiting-Raftery interval score of a central (1 - alpha) interval; lower is better."""
    y, lo, hi = _valid_pairs(y_true, lower, upper)
    if not y.size:
        return float("nan")
    penalty = (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return float(np.mean(hi - lo + penalty))
