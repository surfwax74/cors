"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import numpy as np

from engine.metrics import evaluate


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    m = evaluate(scores, y)
    assert m["hit_rate"] == 1.0
    assert m["corr"] > 0.9


def test_all_zero_scores():
    y = np.array([0, 1, 0, 1])
    m = evaluate(np.zeros(4), y)
    assert m["corr"] == 0.0
    assert m["hit_rate"] == 0.0


def test_unsupervised_returns_nan():
    m = evaluate(np.array([1.0, 2.0, 3.0]), None)
    assert math.isnan(m["corr"])
    assert math.isnan(m["hit_rate"])


def test_no_positive_labels():
    m = evaluate(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]))
    assert m["hit_rate"] == 0.0
