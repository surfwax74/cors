"""MahalanobisCorr: Mahalanobis distance in correlation-eigenvalue space.

Pipeline:
  1. For each training window, take the top-``k`` eigenvalues of its
     correlation matrix.
  2. Reduce those eigenvalue vectors with PCA (to ``pca_dim``).
  3. Fit an empirical covariance in that PCA space.
  4. Score each test window by its Mahalanobis distance from the baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

from .base import BaseModel


class MahalanobisCorrModel(BaseModel):
    name = "MahalanobisCorr"

    def __init__(self, window: int = 12, k: int = 50, pca_dim: int = 20, stride: int = 10):
        super().__init__(window=window, k=k, pca_dim=pca_dim, stride=stride)
        self.window = window
        self.k = k
        self.pca_dim = pca_dim
        self.stride = stride

        self.pca = None
        self.cov = None
        self.base_pca = None

    def _top_eigs(self, block: np.ndarray) -> np.ndarray:
        corr = np.corrcoef(block, rowvar=False)
        return np.linalg.eigvalsh(corr)[-self.k:]

    def fit(self, X_train: np.ndarray) -> "MahalanobisCorrModel":
        window, stride = self.window, max(1, self.stride)

        base_corr = np.corrcoef(X_train, rowvar=False)
        base_vals = np.linalg.eigvalsh(base_corr)[-self.k:]

        eig_vecs = [
            self._top_eigs(X_train[i - window:i])
            for i in range(window, len(X_train), stride)
        ]
        eig_vecs = np.array(eig_vecs)

        # PCA dimension is bounded by both the eigenvalue count and the number
        # of training windows we collected.
        pca_dim_eff = min(self.pca_dim, eig_vecs.shape[1], eig_vecs.shape[0])
        self.pca = PCA(n_components=pca_dim_eff)
        eig_vecs_pca = self.pca.fit_transform(eig_vecs)

        self.base_pca = self.pca.transform(base_vals.reshape(1, -1))[0]
        self.cov = EmpiricalCovariance().fit(eig_vecs_pca)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        window = self.window
        scores = []
        for i in range(len(X)):
            if i < window:
                scores.append(0.0)
                continue
            vals = self._top_eigs(X[i - window:i])
            vals_pca = self.pca.transform(vals.reshape(1, -1))[0]
            diff = vals_pca - self.base_pca
            scores.append(self.cov.mahalanobis(diff.reshape(1, -1))[0])
        return np.array(scores)
