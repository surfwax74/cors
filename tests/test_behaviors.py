"""Tests for nominal behavioral noise injectors."""

from __future__ import annotations

import numpy as np
import pytest

from data.behaviors import (
    BehaviorSpec,
    apply_behaviors,
    default_behaviors,
    generate_behavior,
    list_behaviors,
)

FS = 10.0  # seconds per bin
N = 8640  # one day at 10s


@pytest.mark.parametrize("kind", list_behaviors())
def test_each_behavior_shape_and_finite(kind):
    rng = np.random.default_rng(0)
    out = generate_behavior(kind, N, FS, rng, amplitude=1.0)
    assert out.shape == (N,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("kind", list_behaviors())
def test_each_behavior_is_nonzero(kind):
    # Use a high event rate so sparse behaviors definitely fire.
    rng = np.random.default_rng(1)
    params = {"rate_per_hour": 50} if kind in ("thruster", "reaction_wheel") else {}
    out = generate_behavior(kind, N, FS, rng, amplitude=1.0, **params)
    assert np.any(out != 0)


def test_static_noise_amplitude_scales():
    rng = np.random.default_rng(2)
    small = generate_behavior("gaussian", N, FS, rng, amplitude=0.1)
    big = generate_behavior("gaussian", N, FS, rng, amplitude=5.0)
    assert big.std() > small.std()


def test_step_wave_uses_discrete_levels():
    rng = np.random.default_rng(3)
    out = generate_behavior("step_wave", N, FS, rng, amplitude=2.0, n_levels=3, min_seconds=100, max_seconds=200)
    # Few distinct values relative to length => piecewise constant.
    assert len(np.unique(out)) <= 3


def test_unknown_behavior_raises():
    rng = np.random.default_rng(4)
    with pytest.raises(KeyError):
        generate_behavior("nope", N, FS, rng, amplitude=1.0)


def test_apply_behaviors_modifies_subset_in_place():
    rng = np.random.default_rng(5)
    signals = np.zeros((20, N))
    specs = [BehaviorSpec("gaussian", fraction=0.5, amplitude=1.0, name="noise")]
    assignments = apply_behaviors(signals, specs, FS, rng)

    touched = assignments["noise"]
    assert len(touched) == 10  # 50% of 20
    # Touched channels changed; untouched channels stayed zero.
    for ch in range(20):
        changed = np.any(signals[ch] != 0)
        assert changed == (ch in touched)


def test_apply_behaviors_respects_exclude():
    rng = np.random.default_rng(6)
    signals = np.zeros((20, N))
    specs = [BehaviorSpec("gaussian", fraction=1.0, amplitude=1.0, name="noise")]
    assignments = apply_behaviors(signals, specs, FS, rng, exclude_channels=range(5))
    touched = set(assignments["noise"])
    assert touched.isdisjoint({0, 1, 2, 3, 4})


def test_apply_behaviors_is_deterministic():
    specs = default_behaviors()
    a = np.zeros((30, N))
    b = np.zeros((30, N))
    asn_a = apply_behaviors(a, specs, FS, np.random.default_rng(7))
    asn_b = apply_behaviors(b, specs, FS, np.random.default_rng(7))
    assert asn_a == asn_b
    assert np.array_equal(a, b)


def test_default_behaviors_all_registered():
    valid = set(list_behaviors())
    for spec in default_behaviors():
        assert spec.kind in valid
