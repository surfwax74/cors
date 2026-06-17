"""Dataset registry.

Register a new dataset by adding a :class:`DatasetSpec` to ``DATASETS``. The
CLI (``run.py``) and the engine refer to datasets by name, so a registered
dataset can be selected with ``--dataset <name>``.
"""

from __future__ import annotations

from .dataset import DatasetSpec
from .labels import DailyWindowLabels

# Shared config for the bundled synthetic dataset. Keep these aligned with the
# GenConfig used to produce the data so labels match the injected anomalies.
_SYNTH_TRAIN_DAYS = 30
_SYNTH_TEST_DAYS = 5
_SYNTH_BIN_MINUTES = 10
_SYNTH_ANOMALY_WINDOW_MINUTES = 60
_SYNTH_ANOMALY_OFFSET_MINUTES = 0       # matches GenConfig.anomaly_offset_seconds=0
_SYNTH_ANOMALY_DAYS = None              # matches GenConfig.anomaly_days=None (all days)

DATASETS: dict[str, DatasetSpec] = {
    "synthetic_10min": DatasetSpec(
        name="synthetic_10min",
        path="signals_10min_features.csv",
        timestamp_col="timestamp",
        loader="csv",
        bin_minutes=_SYNTH_BIN_MINUTES,
        split_kind="days",
        train_days=_SYNTH_TRAIN_DAYS,
        test_days=_SYNTH_TEST_DAYS,
        label=DailyWindowLabels(
            test_days=_SYNTH_TEST_DAYS,
            window_minutes=_SYNTH_ANOMALY_WINDOW_MINUTES,
            bin_minutes=_SYNTH_BIN_MINUTES,
            offset_minutes=_SYNTH_ANOMALY_OFFSET_MINUTES,
            days=_SYNTH_ANOMALY_DAYS,
        ),
        generator_preset="full",
        description=(
            "500-channel synthetic telemetry aggregated to 10-minute features, "
            "with a 1-hour injected anomaly at the start of each test day."
        ),
    ),
    # Small/fast variant for agile development. Matches data/generate.py
    # fast_config(): 40 channels, 4 train + 2 test days, 10-minute features.
    # Generate it with:  python -m data.generate --preset fast
    "synthetic_fast": DatasetSpec(
        name="synthetic_fast",
        path="signals_fast_features.csv",
        timestamp_col="timestamp",
        loader="csv",
        bin_minutes=_SYNTH_BIN_MINUTES,
        split_kind="days",
        train_days=4,
        test_days=2,
        label=DailyWindowLabels(
            test_days=2,
            window_minutes=_SYNTH_ANOMALY_WINDOW_MINUTES,
            bin_minutes=_SYNTH_BIN_MINUTES,
        ),
        generator_preset="fast",
        description=(
            "Fast 40-channel synthetic variant (4 train + 2 test days) for quick "
            "iteration. Generate with: python -m data.generate --preset fast"
        ),
    ),
}


def list_datasets() -> list[str]:
    """Return all registered dataset names."""
    return list(DATASETS.keys())


def get_dataset(name: str) -> DatasetSpec:
    """Look up a dataset spec by name."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {', '.join(DATASETS) or '(none)'}"
        )
    return DATASETS[name]
