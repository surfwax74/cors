"""Smoke + contract tests for every registered model."""

from __future__ import annotations

import numpy as np
import pytest

from models.registry import get_model, list_models
from models.model_deepgraph_gnn import DeepGraphModel

# Models with optional heavy deps are tested conditionally.
OPTIONAL = {"DeepGraph_GNN"}


@pytest.mark.parametrize("name", [m for m in list_models() if m not in OPTIONAL])
def test_model_fit_score_contract(name, small_arrays):
    X_train, X_test = small_arrays
    model = get_model(name)

    returned = model.fit(X_train)
    assert returned is model, "fit() must return self"
    assert model.fitted is True

    scores = model.score(X_test)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(X_test),)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("name", [m for m in list_models() if m not in OPTIONAL])
def test_model_fit_score_helper(name, small_arrays):
    X_train, X_test = small_arrays
    scores = get_model(name).fit_score(X_train, X_test)
    assert scores.shape == (len(X_test),)


def test_corals_returns_zeros(small_arrays):
    X_train, X_test = small_arrays
    scores = get_model("CorALS").fit_score(X_train, X_test)
    assert np.all(scores == 0)


def test_param_override_is_recorded():
    model = get_model("RollingCorr", window=7)
    assert model.window == 7
    assert model.get_params()["window"] == 7


def test_pca_solvers_have_distinct_names():
    names = {
        get_model("PCA_Full").name,
        get_model("PCA_Randomized").name,
        get_model("PCA_Incremental").name,
    }
    assert names == {"PCA_Full", "PCA_Randomized", "PCA_Incremental"}


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("DoesNotExist")


@pytest.mark.skipif(not DeepGraphModel.is_available(), reason="torch_geometric not installed")
def test_deepgraph_runs_when_available(small_arrays):
    X_train, X_test = small_arrays
    scores = get_model("DeepGraph_GNN", epochs=1, num_train_windows=3).fit_score(
        X_train, X_test
    )
    assert scores.shape == (len(X_test),)
    assert np.all(np.isfinite(scores))
