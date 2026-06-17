"""Tests for dataset loading, splitting, and labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.dataset import DatasetSpec, load_dataset
from data.labels import ColumnLabels, DailyWindowLabels, NoLabels


def test_index_split_shapes(small_loaded):
    d = small_loaded
    assert len(d.X_train) == 200
    assert len(d.X_test) == 80
    assert d.X_train.shape[1] == len(d.feature_cols)
    assert "timestamp" not in d.feature_cols


def test_daily_window_labels(small_loaded):
    d = small_loaded
    assert d.y_test is not None
    assert d.y_test.shape == (len(d.X_test),)
    # 60min window at 10min bins = 6 positives at the start.
    assert d.y_test[:6].sum() == 6
    assert d.y_test[6:].sum() == 0


def test_daily_window_labels_offset_and_days(tmp_path):
    import pandas as pd

    bins_per_day = 144  # 10-min bins
    n = 3 * bins_per_day
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n + bins_per_day, freq="10min"),
            "a": range(n + bins_per_day),
        }
    )
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)

    # Train = day 0; test = days 1..3. Anomaly only on test-day index 1,
    # offset 60 min (6 bins) in, window 60 min (6 bins).
    spec = DatasetSpec(
        name="multi",
        path=str(path),
        bin_minutes=10,
        split_kind="index",
        train_index=bins_per_day,
        label=DailyWindowLabels(
            test_days=3, window_minutes=60, bin_minutes=10,
            offset_minutes=60, days=(1,),
        ),
    )
    d = load_dataset(spec)
    y = d.y_test
    # Day 1 starts at bins_per_day; offset 6 bins; 6 positives.
    start = bins_per_day + 6
    assert y[start:start + 6].sum() == 6
    assert y.sum() == 6  # only one day labelled


def test_no_labels_strategy(small_csv):
    path, _ = small_csv
    spec = DatasetSpec(
        name="tiny", path=path, split_kind="index", train_index=200, label=NoLabels()
    )
    d = load_dataset(spec)
    assert d.y_test is None


def test_fraction_split(small_csv):
    path, n = small_csv
    spec = DatasetSpec(name="tiny", path=path, split_kind="fraction", train_fraction=0.5)
    d = load_dataset(spec)
    assert len(d.X_train) == n // 2


def test_column_labels_excluded_from_features(tmp_path):
    n = 50
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="10min"),
            "a": np.arange(n),
            "b": np.arange(n),
            "is_anom": ([0] * 40) + ([1] * 10),
        }
    )
    path = tmp_path / "labelled.csv"
    df.to_csv(path, index=False)

    spec = DatasetSpec(
        name="labelled",
        path=str(path),
        split_kind="index",
        train_index=40,
        label=ColumnLabels("is_anom"),
    )
    d = load_dataset(spec)
    assert "is_anom" not in d.feature_cols
    assert d.feature_cols == ["a", "b"]
    assert d.y_test is not None
    assert d.y_test.sum() == 10
