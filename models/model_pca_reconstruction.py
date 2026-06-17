"""PCA reconstruction-error model.

Fits a PCA subspace on the training data, then scores each rolling window by
the mean squared error of reconstructing it from that subspace. Windows whose
correlation structure no longer fits the trained subspace reconstruct poorly
and score high.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import BaseModel


class PcaReconstructionModel(BaseModel):
    name = "PCA_Reconstruction"

    def __init__(self, window: int = 12, k: int = 20):
        super().__init__(window=window, k=k)
        self.window = window
        self.k = k
        self.pca = None

    def fit(self, X_train: np.ndarray) -> "PcaReconstructionModel":
        k_eff = min(self.k, self.window, X_train.shape[1])
        self.pca = PCA(n_components=k_eff)
        self.pca.fit(X_train)
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
            Z = self.pca.transform(block)
            X_hat = self.pca.inverse_transform(Z)
            scores.append(np.mean((block - X_hat) ** 2))
        return np.array(scores)
