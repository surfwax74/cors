"""Tests for the on-disk score cache."""

from __future__ import annotations

import numpy as np

from engine.cache import array_hash, cached_compute, make_key, params_hash


def test_cache_roundtrip(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.array([1, 2, 3])

    key = "demo"
    first = cached_compute(str(tmp_path), key, compute, use_cache=True)
    second = cached_compute(str(tmp_path), key, compute, use_cache=True)

    assert np.array_equal(first, second)
    assert calls["n"] == 1, "second call should hit the cache"


def test_no_cache_always_recomputes(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 42

    cached_compute(str(tmp_path), "k", compute, use_cache=False)
    cached_compute(str(tmp_path), "k", compute, use_cache=False)
    assert calls["n"] == 2


def test_keys_depend_on_params_and_data():
    base = make_key("M", {"a": 1}, "datahash")
    other_param = make_key("M", {"a": 2}, "datahash")
    other_data = make_key("M", {"a": 1}, "otherhash")
    assert base != other_param
    assert base != other_data


def test_param_hash_is_order_independent():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_array_hash_changes_with_content():
    assert array_hash(np.array([1, 2])) != array_hash(np.array([1, 3]))
