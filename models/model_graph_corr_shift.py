"""GraphCorrShift: L1 shift of the absolute-correlation adjacency matrix.

Treats the absolute correlation matrix as a weighted graph adjacency and
scores each window by the total absolute change in edge weights relative to
the baseline graph.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel


class GraphCorrShiftModel(BaseModel):
    name = "GraphCorrShift"

    def __init__(self, window: int = 12):
        super().__init__(window=window)
        self.window = window
        self.base_adj = None

    def fit(self, X_train: np.ndarray) -> "GraphCorrShiftModel":
        base_corr = np.corrcoef(X_train, rowvar=False)
        self.base_adj = np.abs(base_corr)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        window = self.window
        scores = []
        for i in range(len(X)):
            if i < window:
                scores.append(0.0)
                continue
            block = X[i - window:i]
            adj = np.abs(np.corrcoef(block, rowvar=False))
            scores.append(np.sum(np.abs(adj - self.base_adj)))
        return np.array(scores)
