"""Ground-truth label strategies for the test split.

A label strategy turns the test DataFrame into a 1-D ``y_test`` array (or
``None`` for unsupervised datasets). Keeping this pluggable means a synthetic
dataset with known injected anomaly windows and a live dataset with a labelled
column can both flow through the same engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class LabelStrategy(ABC):
    """Builds ``y_test`` from the test split."""

    #: If set, names a DataFrame column that holds labels (excluded from features).
    requires_column: str | None = None

    @abstractmethod
    def build(self, df_test: pd.DataFrame, bins_per_day: int) -> np.ndarray | None:
        """Return integer labels aligned to ``df_test`` rows, or ``None``."""


class NoLabels(LabelStrategy):
    """Unsupervised: no ground truth available."""

    def build(self, df_test, bins_per_day):
        return None


class DailyWindowLabels(LabelStrategy):
    """Anomaly window of fixed length within each (selected) test day.

    Mirrors the synthetic generator, which injects a ``window_minutes`` anomaly
    at ``offset_minutes`` into each anomaly day. Keep these parameters in sync
    with the generator's ``anomaly_window_seconds`` / ``anomaly_offset_seconds``
    / ``anomaly_days`` so labels line up with the injected anomalies.

    Args:
        test_days: number of test days.
        window_minutes: anomaly window length.
        bin_minutes: feature bin size (to convert minutes -> bins).
        offset_minutes: offset from the start of each day to the window.
        days: which test days (0-based) carry an anomaly; None => all of them.
    """

    def __init__(
        self,
        test_days: int,
        window_minutes: int,
        bin_minutes: int,
        offset_minutes: int = 0,
        days: tuple[int, ...] | None = None,
    ):
        self.test_days = test_days
        self.window_minutes = window_minutes
        self.bin_minutes = bin_minutes
        self.offset_minutes = offset_minutes
        self.days = days

    def build(self, df_test, bins_per_day):
        n = len(df_test)
        y = np.zeros(n, dtype=int)
        window_bins = self.window_minutes // self.bin_minutes
        offset_bins = self.offset_minutes // self.bin_minutes
        days = self.days if self.days is not None else range(self.test_days)
        for d in days:
            start = d * bins_per_day + offset_bins
            end = min(start + window_bins, n)
            if 0 <= start < n:
                y[start:end] = 1
        return y


class ColumnLabels(LabelStrategy):
    """Read labels directly from a column in the data."""

    def __init__(self, column: str):
        self.column = column
        self.requires_column = column

    def build(self, df_test, bins_per_day):
        return df_test[self.column].astype(int).values
