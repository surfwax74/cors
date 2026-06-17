"""Parametrized synthetic telemetry generator.

Refactor of the original ``gen.py`` into a configurable function so different
synthetic datasets (more/fewer channels, longer horizons, different anomaly
shapes) can be produced without editing code.

Produces:
  * a raw high-resolution feather file, and
  * an aggregated feature CSV (min/max/mean/std/median + FFT centroid per bin).

Run as a script to regenerate the bundled ``synthetic_10min`` dataset:

    python -m data.generate
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .behaviors import BehaviorSpec, apply_behaviors, default_behaviors


@dataclass
class GenConfig:
    """Knobs controlling synthetic data generation."""

    raw_freq_seconds: int = 10

    # --- Training window & anomaly target period ---
    # Length of the nominal training window, in days.
    train_days: int = 30
    # Number of test days following the training window.
    test_days: int = 5
    # Duration of each injected anomaly.
    anomaly_window_seconds: int = 3600
    # Offset from the start of each test day at which the anomaly begins.
    anomaly_offset_seconds: int = 0
    # Which test days (0-based) receive an anomaly; None => every test day.
    anomaly_days: tuple[int, ...] | None = None

    num_channels: int = 500
    num_clusters: int = 5
    start: str = "2025-01-01"
    seed: int = 42

    feature_bin_minutes: int = 10
    raw_out: str = "signals_raw_10s.feather"
    features_out: str = "signals_10min_features.csv"

    # Channels reserved for hidden correlation effects (kept out of clusters).
    reserved_channels: int = 11

    # Nominal behavioral noise (daily product usage). Applied across the full
    # timeline to a fraction of channels so detectors learn it as normal.
    behaviors: list[BehaviorSpec] = field(default_factory=default_behaviors)
    # Use a separate RNG stream so behaviors don't perturb base/anomaly signals.
    behavior_seed_offset: int = 1000
    # Fraction of the reserved anomaly-carrier channels (0..reserved-1) that also
    # subscribe to the nominal behavior signatures. 0.0 keeps them all clean;
    # 1.0 makes every anomaly channel shifty too.
    reserved_behavior_fraction: float = 0.5


@dataclass
class GeneratedData:
    raw_df: pd.DataFrame
    features_df: pd.DataFrame
    config: GenConfig
    test_windows_raw: list = field(default_factory=list)
    behavior_assignments: dict = field(default_factory=dict)


def fast_config() -> GenConfig:
    """A small, quick-to-generate dataset for agile development/iteration.

    ~40 channels over 6 days at 1-minute raw resolution: generates in seconds and
    keeps the per-window models fast, while preserving the full structure
    (clusters, behaviors, shifty anomaly channels, injected anomalies).
    """
    return GenConfig(
        raw_freq_seconds=60,
        train_days=4,
        test_days=2,
        num_channels=40,
        num_clusters=4,
        feature_bin_minutes=10,
        raw_out="signals_fast_raw.feather",
        features_out="signals_fast_features.csv",
    )


#: Named generation presets (each a factory returning a fresh GenConfig).
PRESETS = {
    "full": GenConfig,
    "fast": fast_config,
}


def get_preset(name: str) -> GenConfig:
    """Return a fresh GenConfig for a named preset."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {', '.join(PRESETS)}")
    return PRESETS[name]()


def _clean_reserved_channels(cfg: GenConfig, rng: np.random.Generator) -> list[int]:
    """Reserved channels that should NOT receive behaviors.

    A ``reserved_behavior_fraction`` of the reserved (anomaly-carrier) channels
    are allowed to be "shifty" like the rest; the remainder are excluded.
    """
    reserved = list(range(cfg.reserved_channels))
    if not reserved:
        return []
    n_shifty = int(round(cfg.reserved_behavior_fraction * len(reserved)))
    n_shifty = min(n_shifty, len(reserved))
    shifty = set()
    if n_shifty > 0:
        shifty = set(int(c) for c in rng.choice(reserved, size=n_shifty, replace=False))
    return [c for c in reserved if c not in shifty]


def _build_signals(cfg: GenConfig, rng: np.random.Generator):
    raw_bins_per_day = int(24 * 3600 / cfg.raw_freq_seconds)
    total_days = cfg.train_days + cfg.test_days
    total_raw_bins = raw_bins_per_day * total_days
    t = np.arange(total_raw_bins)

    periods = {
        "1m": max(int(60 / cfg.raw_freq_seconds), 2),
        "15m": max(int(900 / cfg.raw_freq_seconds), 2),
        "1h": max(int(3600 / cfg.raw_freq_seconds), 2),
        "1d": raw_bins_per_day,
        "1mo": raw_bins_per_day * 30,
        "1yr": raw_bins_per_day * 365,
    }
    period_values = list(periods.values())

    signals = np.zeros((cfg.num_channels, total_raw_bins))
    for i in range(cfg.num_channels):
        base = rng.normal(0, 0.5, total_raw_bins)
        osc = np.zeros(total_raw_bins)
        for _ in range(rng.integers(3, 7)):
            period = rng.choice(period_values)
            amp = 0.1 + 10 * rng.random()
            phase = rng.random() * 2 * np.pi
            osc += amp * np.sin(2 * np.pi * t / period + phase)
        signals[i] = 10 + base + osc

    # Drift clusters (exclude reserved special channels).
    cluster_sizes = rng.integers(5, 21, size=cfg.num_clusters)
    remaining = set(range(cfg.reserved_channels, cfg.num_channels))
    cluster_members = []
    for size in cluster_sizes:
        size = min(int(size), len(remaining))
        if size <= 0:
            cluster_members.append(np.array([], dtype=int))
            continue
        members = rng.choice(list(remaining), size=size, replace=False)
        cluster_members.append(members)
        remaining -= set(members.tolist())

    drifts = [
        2 * np.sin(2 * np.pi * t / raw_bins_per_day),
        0.0005 * t,
        np.cumsum(rng.normal(0, 0.01, total_raw_bins)),
        3 * np.sin(2 * np.pi * t / (7 * raw_bins_per_day)),
        1.5 * np.sin(2 * np.pi * t / (48 * 3600 / cfg.raw_freq_seconds))
        + np.cumsum(rng.normal(0, 0.005, total_raw_bins)),
    ]
    for drift, members in zip(drifts, cluster_members):
        if len(members):
            signals[members] += drift

    # Nominal behavioral noise (daily product usage), layered across the whole
    # timeline before anomalies are injected on top. A configurable fraction of
    # the reserved anomaly channels also subscribe, so anomaly carriers can be
    # "shifty" too; the rest stay clean.
    behavior_rng = np.random.default_rng(cfg.seed + cfg.behavior_seed_offset)
    exclude = _clean_reserved_channels(cfg, behavior_rng)
    behavior_assignments = apply_behaviors(
        signals, cfg.behaviors, cfg.raw_freq_seconds, behavior_rng, exclude_channels=exclude
    )

    # Anomaly target period: one window per selected test day, placed at a
    # configurable offset from the start of that day.
    anomaly_window_raw_bins = cfg.anomaly_window_seconds // cfg.raw_freq_seconds
    offset_raw_bins = cfg.anomaly_offset_seconds // cfg.raw_freq_seconds
    anomaly_days = cfg.anomaly_days if cfg.anomaly_days is not None else range(cfg.test_days)
    test_windows_raw = []
    for d in anomaly_days:
        start = (cfg.train_days + d) * raw_bins_per_day + offset_raw_bins
        test_windows_raw.append((start, start + anomaly_window_raw_bins))

    _inject_anomalies(cfg, signals, t, test_windows_raw, rng)

    return signals, total_raw_bins, raw_bins_per_day, test_windows_raw, behavior_assignments


def _inject_anomalies(cfg, signals, t, test_windows_raw, rng):
    n = cfg.num_channels
    hidden_extreme = np.arange(1, min(6, n))
    hidden_fftshift = np.arange(6, min(11, n))

    for ch in hidden_extreme:
        sign = -1 if rng.random() < 0.5 else 1
        for start, end in test_windows_raw:
            signals[ch, start:end] *= 1.0 + sign * (0.1 + rng.random() * 0.4)

    for start, end in test_windows_raw:
        hf = 2 * np.sin(2 * np.pi * t / 3)
        if len(hidden_fftshift):
            signals[hidden_fftshift, start:end] += hf[start:end]
        for ch, add in ((2, 5), (4, 5), (7, 5)):
            if ch < n:
                signals[ch, start:end] += add
        for ch in (3, 8):
            if ch < n:
                signals[ch, start:end] *= 0.7

    # Primary anomaly on channel 0.
    for start, end in test_windows_raw:
        period_bins = max(int(600 / cfg.raw_freq_seconds), 2)
        osc = 5 * np.sin(2 * np.pi * np.arange(end - start) / period_bins)
        signals[0, start:end] += osc
        signals[0, start:end] *= 1.5


def aggregate_signals(df: pd.DataFrame, bin_minutes: int, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Aggregate raw signals into per-bin features (stats + FFT centroid)."""
    df = df.set_index(timestamp_col)
    rule = f"{bin_minutes}min"

    groups = df.resample(rule)

    agg_df = groups.agg(["min", "max", "mean", "std", "median"])
    agg_df.columns = ["_".join(col) for col in agg_df.columns]

    # Spectral centroid per bin, computed on the full (rows x channels) block.
    centroid_rows = []
    centroid_index = []
    for ts, block in groups:
        x = block.values  # (rows_in_bin, n_channels)
        if x.shape[0] == 0:
            continue
        spectrum = np.abs(np.fft.rfft(x, axis=0))
        freqs = np.fft.rfftfreq(x.shape[0], d=1)
        denom = spectrum.sum(axis=0)
        denom[denom == 0] = 1e-12
        centroid_rows.append((freqs[:, None] * spectrum).sum(axis=0) / denom)
        centroid_index.append(ts)

    fftc_df = pd.DataFrame(
        np.vstack(centroid_rows),
        columns=[f"{col}_fft_centroid" for col in df.columns],
        index=centroid_index,
    ).reindex(agg_df.index)

    return pd.concat([agg_df, fftc_df], axis=1).reset_index()


def generate(cfg: GenConfig | None = None, write: bool = True) -> GeneratedData:
    """Generate synthetic raw + feature data. Optionally write files to disk."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)

    signals, total_raw_bins, _, test_windows_raw, behavior_assignments = _build_signals(cfg, rng)

    time_index = pd.date_range(
        cfg.start, periods=total_raw_bins, freq=f"{cfg.raw_freq_seconds}s"
    )
    channels = [f"telem_{i:03d}" for i in range(cfg.num_channels)]

    raw_df = pd.DataFrame(signals.T, columns=channels)
    raw_df.insert(0, "timestamp", time_index)

    features_df = aggregate_signals(raw_df, cfg.feature_bin_minutes)

    if write:
        raw_df.to_feather(cfg.raw_out)
        features_df.to_csv(cfg.features_out, index=False)

    return GeneratedData(
        raw_df=raw_df,
        features_df=features_df,
        config=cfg,
        test_windows_raw=test_windows_raw,
        behavior_assignments=behavior_assignments,
    )


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic telemetry data")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="full",
        help="Generation preset: 'full' (large, realistic) or 'fast' (small, for iteration)",
    )
    args = parser.parse_args(argv)

    cfg = get_preset(args.preset)
    data = generate(cfg)
    print(
        f"[{args.preset}] Generated raw data ({len(data.raw_df)} rows) and "
        f"{cfg.feature_bin_minutes}min features ({len(data.features_df)} rows) "
        f"-> {cfg.features_out}"
    )


if __name__ == "__main__":
    _main()
