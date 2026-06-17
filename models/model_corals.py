"""CorALS: placeholder for a future correlation-anomaly latent-structure model.

Currently returns all-zero scores so it shows up in result tables as a wired-up
slot. Replace :meth:`fit` / :meth:`score` with a real implementation when ready.
"""

from __future__ import annotations

import numpy as np

from .base import BaseModel


class CoralsModel(BaseModel):
    name = "CorALS"

    def __init__(self):
        super().__init__()

    def fit(self, X_train: np.ndarray) -> "CoralsModel":
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))
