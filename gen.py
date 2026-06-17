"""Deprecated shim — synthetic data generation moved to ``data/generate.py``.

Kept so existing workflows keep working. Prefer:

    python -m data.generate

or, for custom parameters:

    from data.generate import GenConfig, generate
    generate(GenConfig(num_channels=200, train_days=10, test_days=3))
"""

from __future__ import annotations

from data.generate import GenConfig, generate

if __name__ == "__main__":
    data = generate(GenConfig())
    print(
        f"Generated raw 10s data and {data.config.feature_bin_minutes}min "
        "aggregated features."
    )
