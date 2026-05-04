import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

np.random.seed(42)

# Raw resolution: 10 seconds
RAW_FREQ_SECONDS = 10
RAW_BINS_PER_DAY = int(24 * 3600 / RAW_FREQ_SECONDS)  # 8640 bins/day

TRAIN_DAYS = 30
TEST_DAYS = 5
TOTAL_DAYS = TRAIN_DAYS + TEST_DAYS

TOTAL_RAW_BINS = RAW_BINS_PER_DAY * TOTAL_DAYS
TIME_INDEX = pd.date_range("2025-01-01", periods=TOTAL_RAW_BINS, freq=f"{RAW_FREQ_SECONDS}s")

NUM_CHANNELS = 500
CHANNELS = np.array([f"telem_{i:03d}" for i in range(NUM_CHANNELS)])

# 1-hour anomaly window per test day
ANOMALY_WINDOW_SECONDS = 3600
ANOMALY_WINDOW_RAW_BINS = ANOMALY_WINDOW_SECONDS // RAW_FREQ_SECONDS

TEST_DAY_STARTS_RAW = [
    TRAIN_DAYS * RAW_BINS_PER_DAY + d * RAW_BINS_PER_DAY
    for d in range(TEST_DAYS)
]

TEST_WINDOWS_RAW = [
    (start, start + ANOMALY_WINDOW_RAW_BINS)
    for start in TEST_DAY_STARTS_RAW
]

# ============================================================
# BASE SIGNAL GENERATION (FAST)
# ============================================================

t = np.arange(TOTAL_RAW_BINS)

# Oscillation periods in raw bins
periods = {
    "1m": max(int(60 / RAW_FREQ_SECONDS), 2),
    "15m": max(int(900 / RAW_FREQ_SECONDS), 2),
    "1h": max(int(3600 / RAW_FREQ_SECONDS), 2),
    "1d": RAW_BINS_PER_DAY,
    "1mo": RAW_BINS_PER_DAY * 30,
    "1yr": RAW_BINS_PER_DAY * 365
}

signals = np.zeros((NUM_CHANNELS, TOTAL_RAW_BINS))

for i in range(NUM_CHANNELS):
    base = np.random.normal(0, 0.5, TOTAL_RAW_BINS)

    num_osc = np.random.randint(3, 7)
    osc = np.zeros(TOTAL_RAW_BINS)

    for _ in range(num_osc):
        period = np.random.choice(list(periods.values()))
        amp = 0.1 + 10 * np.random.random()
        phase = np.random.random() * 2 * np.pi
        osc += amp * np.sin(2 * np.pi * t / period + phase)

    signals[i] = 10 + base + osc

# ============================================================
# DRIFT CLUSTERS (5 clusters, 5–20 telems each)
# ============================================================

NUM_CLUSTERS = 5
cluster_sizes = np.random.randint(5, 21, size=NUM_CLUSTERS)

remaining = set(range(11, NUM_CHANNELS))  # exclude special telems 0–10
cluster_members = []

for size in cluster_sizes:
    size = min(size, len(remaining))
    members = np.random.choice(list(remaining), size=size, replace=False)
    cluster_members.append(members)
    remaining -= set(members)

# Drift patterns
drifts = [
    2 * np.sin(2 * np.pi * t / (24 * 3600 / RAW_FREQ_SECONDS)),
    0.0005 * t,
    np.cumsum(np.random.normal(0, 0.01, TOTAL_RAW_BINS)),
    3 * np.sin(2 * np.pi * t / (7 * RAW_BINS_PER_DAY)),
    1.5 * np.sin(2 * np.pi * t / (48 * 3600 / RAW_FREQ_SECONDS))
    + np.cumsum(np.random.normal(0, 0.005, TOTAL_RAW_BINS))
]

for drift, members in zip(drifts, cluster_members):
    signals[members] += drift

# ============================================================
# HIDDEN CORRELATION TELEMETRIES
# ============================================================

hidden_extreme = np.arange(1, 6)
hidden_fftshift = np.arange(6, 11)

# Extreme mins/maxes during anomaly windows
for ch in hidden_extreme:
    for start, end in TEST_WINDOWS_RAW:
        signals[ch, start:end] += np.random.normal(0, 4, end - start)

# FFT centroid shift (add high-frequency component)
hf = 2 * np.sin(2 * np.pi * t / 3)
signals[hidden_fftshift] += hf

# Step shifts + scale shifts
signals[2] += 5
signals[4] += 5
signals[7] += 5
signals[3] *= 0.7
signals[8] *= 0.7

# ============================================================
# PRIMARY ANOMALY (telem_000)
# ============================================================

for start, end in TEST_WINDOWS_RAW:
    period_bins = max(int(600 / RAW_FREQ_SECONDS), 2)  # 10-minute oscillation
    osc = 5 * np.sin(2 * np.pi * np.arange(end - start) / period_bins)
    signals[0, start:end] += osc
    signals[0, start:end] *= 1.5

# ============================================================
# SAVE RAW SIGNALS TO CSV
# ============================================================

raw_df = pd.DataFrame(signals.T, columns=CHANNELS)
raw_df.insert(0, "timestamp", TIME_INDEX)

# raw_df.to_csv(    "signals_raw_10s.csv",    index=False,    float_format="%.3f")
raw_df.to_feather("signals_raw_10s.feather")

# ============================================================
# AGGREGATION FUNCTION (ANY BIN SIZE)
# ============================================================

def aggregate_signals(df, bin_minutes):
    df = df.set_index("timestamp")
    rule = f"{bin_minutes}min"

    # Basic stats
    agg_df = df.resample(rule).agg(["min", "max", "mean", "std", "median"])
    agg_df.columns = ["_".join(col) for col in agg_df.columns]

    # FFT centroid per bin (vectorized)
    def fft_centroid_block(block):
        x = block.values  # shape: (window_size, num_channels)
        spectrum = np.abs(np.fft.rfft(x, axis=0))
        freqs = np.fft.rfftfreq(x.shape[0], d=1)
        return (freqs[:, None] * spectrum).sum(axis=0) / spectrum.sum(axis=0)

    # Instead of apply(), use resample().apply() but reshape manually
    fftc_series = df.resample(rule).apply(fft_centroid_block)

    # fftc_series is a Series of arrays → convert to 2D array
    fftc_matrix = np.vstack(fftc_series.values)

    # Build DataFrame with correct column names
    fftc_df = pd.DataFrame(
        fftc_matrix,
        columns=[f"{col}_fft_centroid" for col in df.columns],
        index=fftc_series.index
    )

    # Combine
    return pd.concat([agg_df, fftc_df], axis=1).reset_index()


# Example: generate 10-minute features
df_10min = aggregate_signals(raw_df, 10)
df_10min.to_csv("signals_10min_features.csv", index=False)

print("Generated raw 10s data and 10min aggregated features.")
