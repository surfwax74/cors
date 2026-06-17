"""Tests for the run.py CLI helpers (auto-generation of datasets)."""

from __future__ import annotations

import os

import pytest

from data.dataset import DatasetSpec
from run import ensure_dataset


def test_existing_file_is_left_alone(small_csv):
    path, _ = small_csv
    spec = DatasetSpec(name="x", path=path, generator_preset=None)
    # No preset, but the file exists and we're not regenerating -> no error.
    ensure_dataset(spec, regenerate=False, autogen=True, verbose=False)
    assert os.path.exists(path)


def test_missing_without_preset_raises(tmp_path):
    spec = DatasetSpec(name="x", path=str(tmp_path / "nope.csv"), generator_preset=None)
    with pytest.raises(FileNotFoundError):
        ensure_dataset(spec, regenerate=False, autogen=True, verbose=False)


def test_missing_with_no_autogen_raises(tmp_path):
    spec = DatasetSpec(
        name="x", path=str(tmp_path / "nope.csv"), generator_preset="fast"
    )
    with pytest.raises(FileNotFoundError):
        ensure_dataset(spec, regenerate=False, autogen=False, verbose=False)


def test_missing_with_preset_generates(tmp_path, monkeypatch):
    # Generate into a temp dir so we don't touch real artifacts.
    monkeypatch.chdir(tmp_path)
    target = "signals_fast_features.csv"
    spec = DatasetSpec(name="synthetic_fast", path=target, generator_preset="fast")

    assert not os.path.exists(target)
    ensure_dataset(spec, regenerate=False, autogen=True, verbose=False)
    assert os.path.exists(target)
