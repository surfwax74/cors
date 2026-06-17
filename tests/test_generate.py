"""Tests for the parametrized synthetic data generator."""

from __future__ import annotations

import pytest

from data.generate import (
    PRESETS,
    GenConfig,
    aggregate_signals,
    fast_config,
    generate,
    get_preset,
)


def test_generate_small_in_memory():
    # Tiny config: 2 channels, 1 train day + 1 test day, coarse resolution.
    cfg = GenConfig(
        raw_freq_seconds=600,  # 10-minute raw bins -> fast
        train_days=1,
        test_days=1,
        num_channels=12,
        num_clusters=2,
        feature_bin_minutes=60,
    )
    data = generate(cfg, write=False)

    assert "timestamp" in data.raw_df.columns
    assert data.raw_df.shape[1] == cfg.num_channels + 1  # +timestamp
    assert len(data.features_df) > 0
    assert len(data.test_windows_raw) == cfg.test_days


def test_generate_is_deterministic():
    cfg = GenConfig(raw_freq_seconds=600, train_days=1, test_days=1, num_channels=8)
    a = generate(cfg, write=False).raw_df["telem_000"].values
    b = generate(cfg, write=False).raw_df["telem_000"].values
    assert (a == b).all()


def test_aggregate_produces_expected_feature_families():
    cfg = GenConfig(raw_freq_seconds=600, train_days=1, test_days=1, num_channels=4)
    raw = generate(cfg, write=False).raw_df
    feats = aggregate_signals(raw, bin_minutes=60)
    cols = "".join(feats.columns)
    for fam in ("_min", "_max", "_mean", "_std", "_median", "_fft_centroid"):
        assert fam in cols


def test_behaviors_are_applied_and_reported():
    cfg = GenConfig(raw_freq_seconds=600, train_days=2, test_days=1, num_channels=50)
    data = generate(cfg, write=False)
    assert data.behavior_assignments  # non-empty report
    # At least one behavior touched at least one channel.
    assert any(len(chs) > 0 for chs in data.behavior_assignments.values())


def test_reserved_channels_clean_when_fraction_zero():
    cfg = GenConfig(
        raw_freq_seconds=600, train_days=2, test_days=1,
        num_channels=60, reserved_channels=11, reserved_behavior_fraction=0.0,
    )
    data = generate(cfg, write=False)
    for chs in data.behavior_assignments.values():
        assert all(ch >= 11 for ch in chs)


def test_some_reserved_channels_become_shifty():
    cfg = GenConfig(
        raw_freq_seconds=600, train_days=2, test_days=1,
        num_channels=60, reserved_channels=11, reserved_behavior_fraction=1.0,
    )
    data = generate(cfg, write=False)
    touched = {ch for chs in data.behavior_assignments.values() for ch in chs}
    # With fraction=1.0 every reserved channel is eligible; expect some touched.
    assert any(ch < 11 for ch in touched)


def test_behaviors_can_be_disabled():
    cfg = GenConfig(raw_freq_seconds=600, train_days=1, test_days=1, num_channels=20, behaviors=[])
    data = generate(cfg, write=False)
    assert data.behavior_assignments == {}


def test_fast_preset_is_small_and_distinct():
    cfg = fast_config()
    full = GenConfig()
    assert cfg.num_channels < full.num_channels
    assert (cfg.train_days + cfg.test_days) < (full.train_days + full.test_days)
    # Fast variant writes to its own files so it never clobbers the full dataset.
    assert cfg.features_out != full.features_out
    assert cfg.raw_out != full.raw_out


def test_fast_preset_generates_quickly_in_memory():
    data = generate(fast_config(), write=False)
    # 6 days at 10-min features = 6 * 144 rows.
    assert len(data.features_df) == 6 * 144
    assert data.raw_df.shape[1] == fast_config().num_channels + 1


def test_get_preset_names():
    assert set(PRESETS) == {"full", "fast"}
    assert isinstance(get_preset("fast"), GenConfig)
    with pytest.raises(KeyError):
        get_preset("nope")


def test_anomaly_offset_and_day_selection():
    # 3 test days, but only day index 1 gets an anomaly, offset 1 hour in.
    cfg = GenConfig(
        raw_freq_seconds=600, train_days=1, test_days=3, num_channels=12,
        anomaly_window_seconds=3600, anomaly_offset_seconds=3600,
        anomaly_days=(1,),
    )
    data = generate(cfg, write=False)
    assert len(data.test_windows_raw) == 1

    raw_bins_per_day = int(24 * 3600 / cfg.raw_freq_seconds)
    offset_bins = cfg.anomaly_offset_seconds // cfg.raw_freq_seconds
    expected_start = (cfg.train_days + 1) * raw_bins_per_day + offset_bins
    assert data.test_windows_raw[0][0] == expected_start
