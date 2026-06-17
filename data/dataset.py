"""Parametrized dataset loading and train/test splitting.

A :class:`DatasetSpec` is a pure description of *where* the data is and *how* to
split and label it. :func:`load_dataset` turns a spec into a
:class:`LoadedDataset` holding the train/test frames, feature matrices, and
ground-truth labels that the engine consumes.

This indirection is what makes it cheap to add a new dataset: register another
``DatasetSpec`` (see ``data/registry.py``) and nothing in the engine or models
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .labels import LabelStrategy, NoLabels


@dataclass
class DatasetSpec:
    """Declarative description of a dataset."""

    name: str
    path: str
    timestamp_col: str = "timestamp"
    loader: str = "csv"  # "csv" or "feather"
    bin_minutes: int = 10

    # Split configuration.
    split_kind: str = "days"  # "days" | "fraction" | "index"
    train_days: int = 30
    train_fraction: float = 0.8
    train_index: int | None = None
    test_days: int = 5

    # Optional explicit feature columns; defaults to "everything but timestamp".
    feature_cols: list[str] | None = None

    label: LabelStrategy = field(default_factory=NoLabels)
    description: str = ""

    # Optional name of a generator preset (in data.generate.PRESETS) that
    # produces this dataset's file. Lets the CLI (re)generate it on demand.
    generator_preset: str | None = None


@dataclass
class LoadedDataset:
    """Materialised, split dataset ready for the engine."""

    spec: DatasetSpec
    df: pd.DataFrame
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    feature_cols: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray | None
    bins_per_day: int


def _read_frame(spec: DatasetSpec) -> pd.DataFrame:
    if spec.loader == "feather":
        df = pd.read_feather(spec.path)
    elif spec.loader == "csv":
        parse = [spec.timestamp_col]
        df = pd.read_csv(spec.path, parse_dates=parse)
    else:
        raise ValueError(f"Unknown loader {spec.loader!r} (use 'csv' or 'feather')")
    return df


def _train_end_index(spec: DatasetSpec, n_rows: int, bins_per_day: int) -> int:
    if spec.split_kind == "days":
        return min(spec.train_days * bins_per_day, n_rows)
    if spec.split_kind == "fraction":
        return int(round(spec.train_fraction * n_rows))
    if spec.split_kind == "index":
        if spec.train_index is None:
            raise ValueError("split_kind='index' requires train_index")
        return spec.train_index
    raise ValueError(f"Unknown split_kind {spec.split_kind!r}")


def load_dataset(spec: DatasetSpec) -> LoadedDataset:
    """Load, split, and label a dataset described by ``spec``."""
    df = _read_frame(spec)

    if spec.timestamp_col in df.columns:
        df = df.sort_values(spec.timestamp_col).reset_index(drop=True)

    bins_per_day = int(24 * 60 / spec.bin_minutes)
    train_end = _train_end_index(spec, len(df), bins_per_day)

    df_train = df.iloc[:train_end].copy()
    df_test = df.iloc[train_end:].reset_index(drop=True)

    # Resolve feature columns: explicit list, or everything except timestamp and
    # any label column we should not feed to the models.
    if spec.feature_cols is not None:
        feature_cols = list(spec.feature_cols)
    else:
        exclude = {spec.timestamp_col}
        if spec.label.requires_column:
            exclude.add(spec.label.requires_column)
        feature_cols = [c for c in df.columns if c not in exclude]

    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values

    y_test = spec.label.build(df_test, bins_per_day)

    return LoadedDataset(
        spec=spec,
        df=df,
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        bins_per_day=bins_per_day,
    )
