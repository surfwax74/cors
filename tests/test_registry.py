"""Tests for the model and dataset registries."""

from __future__ import annotations

from data.registry import get_dataset, list_datasets
from models.base import BaseModel
from models.registry import get_model, list_models


def test_expected_models_registered():
    names = set(list_models())
    expected = {
        "RollingCorr",
        "MahalanobisCorr",
        "PCA_Full",
        "PCA_Randomized",
        "PCA_Incremental",
        "PCA_Reconstruction",
        "GraphCorrShift",
        "DeepGraph_GNN",
        "CorALS",
    }
    assert expected <= names


def test_every_model_instantiates_as_basemodel():
    for name in list_models():
        model = get_model(name)
        assert isinstance(model, BaseModel)
        assert model.name


def test_synthetic_dataset_registered():
    assert "synthetic_10min" in list_datasets()
    spec = get_dataset("synthetic_10min")
    assert spec.path.endswith(".csv")
    assert spec.bin_minutes == 10


def test_fast_dataset_registered():
    assert "synthetic_fast" in list_datasets()
    spec = get_dataset("synthetic_fast")
    assert spec.train_days == 4
    assert spec.test_days == 2
    assert spec.label.test_days == 2
