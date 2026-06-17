"""PCA eigen-shift models.

Each model learns the explained-variance spectrum of the training data, then
scores every rolling window by how far its spectrum drifts from that baseline
(L2 norm of the eigenvalue difference).

Three solvers are exposed as separate registry entries because they have
slightly different numerical behaviour:

  * ``PCA_Full``        - exact full SVD
  * ``PCA_Randomized``  - randomized SVD (faster, approximate)
  * ``PCA_Incremental`` - incremental PCA (mini-batch friendly)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from .base import BaseModel

# solver key -> (registry name)
_SOLVER_NAMES = {
    "full": "PCA_Full",
    "randomized": "PCA_Randomized",
    "incremental": "PCA_Incremental",
}


class PcaEigenShiftModel(BaseModel):
    """Eigenvalue-spectrum drift detector, parametrised by PCA solver."""

    name = "PCA_EigenShift"

    def __init__(self, solver: str = "full", window: int = 12, k: int = 20):
        if solver not in _SOLVER_NAMES:
            raise ValueError(f"Unknown solver {solver!r}; choose from {list(_SOLVER_NAMES)}")
        super().__init__(solver=solver, window=window, k=k)
        self.solver = solver
        self.window = window
        self.k = k
        self.name = _SOLVER_NAMES[solver]

        self.k_eff = None
        self.base = None

    def _make_pca(self, n_components: int):
        if self.solver == "incremental":
            return IncrementalPCA(n_components=n_components)
        return PCA(n_components=n_components, svd_solver=self.solver)

    def _eigs(self, block: np.ndarray) -> np.ndarray:
        pca = self._make_pca(self.k_eff)
        pca.fit(block)
        return pca.explained_variance_

    def fit(self, X_train: np.ndarray) -> "PcaEigenShiftModel":
        self.k_eff = min(self.k, self.window, X_train.shape[1])
        self.base = self._eigs(X_train)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        window = self.window
        scores = []
        for i in range(len(X)):
            if i < window:
                scores.append(0.0)
                continue
            eigs = self._eigs(X[i - window:i])
            scores.append(np.linalg.norm(eigs - self.base))
        return np.array(scores)
