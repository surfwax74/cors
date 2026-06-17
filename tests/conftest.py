"""Shared pytest fixtures.

Builds a tiny in-memory dataset so tests run fast and never depend on the large
bundled CSVs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the project root importable when running pytest from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import DatasetSpec, load_dataset  # noqa: E402
from data.labels import DailyWindowLabels  # noqa: E402


N_FEATURES = 6
TRAIN_ROWS = 200
TEST_ROWS = 80
BIN_MINUTES = 10


@pytest.fixture
def small_arrays():
    """Deterministic train/test feature matrices."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(TRAIN_ROWS, N_FEATURES))
    X_test = rng.normal(0, 1, size=(TEST_ROWS, N_FEATURES))
    return X_train, X_test


@pytest.fixture
def small_csv(tmp_path):
    """Write a tiny feature CSV and return its path + row count."""
    rng = np.random.default_rng(1)
    n = TRAIN_ROWS + TEST_ROWS
    ts = pd.date_range("2025-01-01", periods=n, freq=f"{BIN_MINUTES}min")
    data = {f"feat_{i}": rng.normal(0, 1, n) for i in range(N_FEATURES)}
    df = pd.DataFrame(data)
    df.insert(0, "timestamp", ts)
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return str(path), n


@pytest.fixture
def small_spec(small_csv):
    path, _ = small_csv
    # bins_per_day at 10min = 144; use a small index split instead of days.
    return DatasetSpec(
        name="tiny",
        path=path,
        bin_minutes=BIN_MINUTES,
        split_kind="index",
        train_index=TRAIN_ROWS,
        label=DailyWindowLabels(test_days=1, window_minutes=60, bin_minutes=BIN_MINUTES),
    )


@pytest.fixture
def small_loaded(small_spec):
    return load_dataset(small_spec)
