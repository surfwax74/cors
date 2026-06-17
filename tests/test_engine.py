"""Tests for the train/test engine runner."""

from __future__ import annotations

from engine.runner import Engine, ModelResult


def test_engine_runs_selected_models(small_loaded, tmp_path):
    engine = Engine(small_loaded, cache_dir=str(tmp_path), verbose=False)
    results = engine.run(model_names=["RollingCorr", "CorALS"])

    assert len(results) == 2
    assert all(isinstance(r, ModelResult) for r in results)
    by_name = {r.name: r for r in results}
    assert by_name["RollingCorr"].status == "ok"
    assert by_name["RollingCorr"].scores is not None
    assert "corr" in by_name["RollingCorr"].metrics


def test_engine_unknown_model_recorded_as_error(small_loaded, tmp_path):
    engine = Engine(small_loaded, cache_dir=str(tmp_path), verbose=False)
    (result,) = engine.run(model_names=["NoSuchModel"])
    assert result.status == "error"


def test_format_results_includes_model_names(small_loaded, tmp_path):
    engine = Engine(small_loaded, cache_dir=str(tmp_path), verbose=False)
    results = engine.run(model_names=["RollingCorr"])
    text = engine.format_results(results)
    assert "RollingCorr" in text


def test_param_overrides_applied(small_loaded, tmp_path):
    engine = Engine(small_loaded, cache_dir=str(tmp_path), verbose=False)
    results = engine.run(
        model_names=["RollingCorr"], param_overrides={"RollingCorr": {"window": 5}}
    )
    assert results[0].status == "ok"
