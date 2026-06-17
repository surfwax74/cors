"""RollingCorr: Frobenius-norm shift of the rolling correlation matrix.

Compares the correlation matrix of each rolling window against the baseline
correlation matrix learned from the training data.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel


class RollingCorrModel(BaseModel):
    name = "RollingCorr"

    def __init__(self, window: int = 12):
        super().__init__(window=window)
        self.window = window
        self.base_corr = None

    def fit(self, X_train: np.ndarray) -> "RollingCorrModel":
        self.base_corr = np.corrcoef(X_train, rowvar=False)
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
            corr = np.corrcoef(block, rowvar=False)
            diff = corr - self.base_corr
            scores.append(np.linalg.norm(diff, ord="fro"))
        return np.array(scores)
