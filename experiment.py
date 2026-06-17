"""Backwards-compatible experiment script.

The algorithms and orchestration now live in the ``models`` / ``data`` /
``engine`` packages. This script wires them together to reproduce the original
behaviour: run every model on the synthetic dataset, print a results table, and
draw the exploratory plots.

For new work prefer the CLI:

    python run.py --dataset synthetic_10min

Add new algorithms under ``models/`` and new datasets under ``data/registry.py``.
"""

from __future__ import annotations

from data.dataset import load_dataset
from data.registry import get_dataset
from engine.runner import Engine

DATASET = "synthetic_10min"


def main():
    data = load_dataset(get_dataset(DATASET))

    engine = Engine(data, cache_dir="cache_results", use_cache=True)
    results = engine.run()
    engine.print_results(results)

    _make_plots(data)


def _make_plots(data):
    """Reproduce the original exploratory plots."""
    try:
        from cors_plot import (
            plot_anomaly_window,
            plot_corr_heatmap,
            plot_drift_cluster,
            plot_fft_centroid_distribution,
            plot_pca_projection,
            plot_raw_signals,
        )
    except Exception as exc:  # plotting deps optional
        print(f"[INFO] Skipping plots ({exc})")
        return

    df_train, df_test = data.df_train, data.df_test
    feature_cols = data.feature_cols
    y_test = data.y_test

    plot_raw_signals(df_train, feature_cols, n=25)
    plot_raw_signals(df_test, feature_cols, n=25)

    cluster = df_train.columns[:10]
    plot_drift_cluster(df_train, cluster)

    if y_test is not None:
        plot_anomaly_window(df_test, feature_cols[0], y_test.astype(bool))

    plot_fft_centroid_distribution(df_train, feature_cols)
    plot_corr_heatmap(df_train, feature_cols, n=40)
    plot_pca_projection(df_train, feature_cols)


if __name__ == "__main__":
    main()
