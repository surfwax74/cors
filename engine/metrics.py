"""Evaluation metrics for anomaly scores against ground-truth labels."""

from __future__ import annotations

import math

import numpy as np


def evaluate(scores: np.ndarray, y: np.ndarray | None) -> dict:
    """Score quality vs. ground truth.

    Returns a dict with:
      * ``corr``     - Pearson correlation between scores and labels.
      * ``hit_rate`` - fraction of the top-K scored points that are true
                       anomalies, where K is the number of positives.

    For unsupervised datasets (``y is None``) both metrics are ``NaN``.
    """
    if y is None:
        return {"corr": math.nan, "hit_rate": math.nan}

    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)

    if np.std(scores) == 0 or np.std(y) == 0:
        # Correlation is undefined when either series is constant.
        return {"corr": 0.0, "hit_rate": 0.0}

    corr = float(np.corrcoef(scores, y)[0, 1])

    K = int(y.sum())
    if K == 0:
        return {"corr": corr, "hit_rate": 0.0}

    top_k_idx = np.argsort(scores)[-K:]
    hit_rate = float(y[top_k_idx].sum() / K)
    return {"corr": corr, "hit_rate": hit_rate}
