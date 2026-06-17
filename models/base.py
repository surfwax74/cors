"""Base class shared by every correlation-detection algorithm.

Every algorithm in this project is a *model* that follows the same small
contract so the train/test engine can drive them interchangeably:

    model.fit(X_train)   # learn / calibrate a baseline from training data
    model.score(X)       # return one anomaly score per row of X

Scores are "higher == more anomalous". The engine compares those scores
against ground-truth labels (see ``engine.metrics``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract base for all correlation / anomaly detection models.

    Subclasses set a class-level ``name`` (used by the registry and result
    tables) and implement :meth:`fit` and :meth:`score`.
    """

    #: Human-readable identifier, unique across the registry.
    name: str = "base"

    def __init__(self, **params):
        # Keep an explicit copy of the hyper-parameters so the engine can hash
        # them for caching and print them in result tables.
        self.params = dict(params)
        self.fitted = False

    # ------------------------------------------------------------------ #
    # Hyper-parameter helpers
    # ------------------------------------------------------------------ #
    def get_params(self) -> dict:
        """Return the model's hyper-parameters as a plain dict."""
        return dict(self.params)

    def set_params(self, **params) -> "BaseModel":
        """Update hyper-parameters in place and return ``self``."""
        self.params.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------ #
    # Core contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "BaseModel":
        """Learn a baseline from ``X_train`` (shape ``[n_samples, n_features]``).

        Must return ``self`` so calls can be chained.
        """

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return a 1-D array of anomaly scores, one per row of ``X``."""

    def fit_score(self, X_train: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Convenience: ``fit(X_train)`` then ``score(X)``."""
        return self.fit(X_train).score(X)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.params})"
