# Data Models

The shapes that flow through the system.

## Raw signals (synthetic)
Produced by `data/generate.py`, written to `signals_raw_10s.feather`.

| column        | type      | description                                  |
|---------------|-----------|----------------------------------------------|
| `timestamp`   | datetime  | regular cadence (default 10 s)               |
| `telem_000..` | float     | one column per telemetry channel (default 500)|

Injected structure (for evaluation):
- **Drift clusters** — groups of channels share a slow drift pattern.
- **Nominal behavioral noise** — daily product-usage activity layered across the
  whole timeline (train + test) on a fraction of channels, so a healthy detector
  learns it as *normal* (see below).
- **Hidden correlation channels** (1–10) — extreme scaling / FFT shifts / step
  changes during anomaly windows.
- **Primary anomaly** — channel `telem_000` gets an added oscillation + scaling
  during each test-day anomaly window.

### Nominal behavioral noise (`data/behaviors.py`)
Configured via `GenConfig.behaviors` (a list of `BehaviorSpec`). Each behavior is
applied to a random `fraction` of eligible channels. By default
`reserved_behavior_fraction=0.5`, so half the reserved anomaly-carrier channels
(0–10) also subscribe to the behavior signatures (they can be "shifty" too) while
the rest stay clean — set it to `0.0` to keep all anomaly channels clean, or
`1.0` to make every one of them shifty. Behaviors use a separate RNG stream so
they don't perturb base/anomaly reproducibility. `generate()` returns a
`behavior_assignments` map of label → channel indices.

Static-noise distributions (stationary noise floors):

| kind        | character | key params |
|-------------|-----------|------------|
| `gaussian`  | Normal / Gaussian noise (same thing) | `amplitude` (std) |
| `uniform`   | flat "random" noise ±amplitude | `amplitude` |
| `laplace`   | sharper peak, heavier tails than Gaussian | `amplitude` |
| `student_t` | heavy-tailed; occasional large spikes | `amplitude`, `df` |
| `pink`      | 1/f noise (analog electronics) | `amplitude` |

Behavioral activity (nominal but non-stationary — the "shifty" telemetry):

| kind             | models | key params |
|------------------|--------|------------|
| `voltage_shift`  | bus voltage settling at different operating points (random walk) | `amplitude`, `min_seconds`, `max_seconds` |
| `step_wave`      | random-length square waves between discrete levels (mode switching) | `amplitude`, `n_levels`, `min_seconds`, `max_seconds` |
| `thruster`       | sparse impulsive firings with exponential decay | `amplitude`, `rate_per_hour`, `pulse_seconds` |
| `reaction_wheel` | trapezoidal spin up / hold / spin down ramps | `amplitude`, `rate_per_hour`, `ramp_seconds`, `hold_seconds` |

Event timings are expressed in **seconds**, so they're independent of the raw
sample rate. Add or tune behaviors by editing `default_behaviors()` or passing a
custom `behaviors=[...]` list to `GenConfig`.

### Training window & anomaly target period (`GenConfig`)

| field                       | default | meaning                                                  |
|-----------------------------|---------|----------------------------------------------------------|
| `train_days`                | `30`    | length of the nominal **training window** (days)         |
| `test_days`                 | `5`     | test days following the training window                  |
| `anomaly_window_seconds`    | `3600`  | duration of each injected anomaly                        |
| `anomaly_offset_seconds`    | `0`     | offset from the start of each test day to the anomaly    |
| `anomaly_days`              | `None`  | which test-day indices get an anomaly (None => all)      |
| `reserved_behavior_fraction`| `0.5`   | fraction of anomaly channels that are also "shifty"      |

> **Keep labels in sync.** `DailyWindowLabels` mirrors these with
> `window_minutes` / `offset_minutes` / `days`. For the bundled dataset these are
> kept aligned via the `_SYNTH_*` constants in `data/registry.py`; if you change
> the generator's anomaly placement, update those constants (or your dataset's
> label strategy) to match.

### Presets

`data/generate.py` exposes named presets (`PRESETS`, via `get_preset(name)`):

| preset | channels | days (train+test) | raw res | use |
|--------|----------|-------------------|---------|-----|
| `full` | 500      | 30 + 5            | 10 s    | full-scale runs (`synthetic_10min`) |
| `fast` | 40       | 4 + 2             | 60 s    | agile iteration (`synthetic_fast`)  |

```bash
python -m data.generate --preset fast   # writes signals_fast_*.{feather,csv}
python run.py --dataset synthetic_fast
```

The `fast` preset keeps the full structure (clusters, behaviors, shifty anomaly
channels, injected anomalies) but generates in ~1s and runs every model in
seconds. Generated data files are git-ignored — regenerate them as needed.

## Feature matrix (aggregated)
Produced by `aggregate_signals()`, written to `signals_10min_features.csv`.
Each raw channel expands into per-bin features:

`<channel>_min`, `_max`, `_mean`, `_std`, `_median`, `_fft_centroid`

So 500 channels × 6 features = 3000 feature columns (+ `timestamp`).

## DatasetSpec
Declarative dataset description (`data/dataset.py`).

| field                | default       | meaning                                       |
|----------------------|---------------|-----------------------------------------------|
| `name`               | —             | registry key                                  |
| `path`               | —             | file path                                     |
| `loader`             | `"csv"`       | `"csv"` or `"feather"`                         |
| `timestamp_col`      | `"timestamp"` | time column name                              |
| `bin_minutes`        | `10`          | feature bin size (drives `bins_per_day`)      |
| `split_kind`         | `"days"`      | `"days"` \| `"fraction"` \| `"index"`          |
| `train_days`         | `30`          | for `split_kind="days"`                        |
| `train_fraction`     | `0.8`         | for `split_kind="fraction"`                    |
| `train_index`        | `None`        | for `split_kind="index"`                       |
| `test_days`          | `5`           | informational / used by label strategies      |
| `feature_cols`       | `None`        | explicit features, else all but timestamp/label|
| `label`              | `NoLabels()`  | a `LabelStrategy`                              |

## LoadedDataset
What `load_dataset(spec)` returns and the engine consumes.

| field          | type            | shape                          |
|----------------|-----------------|--------------------------------|
| `df`           | DataFrame       | full dataset                   |
| `df_train`     | DataFrame       | training rows                  |
| `df_test`      | DataFrame       | test rows                      |
| `feature_cols` | list[str]       | feature column names           |
| `X_train`      | ndarray         | `[n_train, n_features]`         |
| `X_test`       | ndarray         | `[n_test, n_features]`          |
| `y_test`       | ndarray \| None | `[n_test]` (or `None`)          |
| `bins_per_day` | int             | `24*60 / bin_minutes`           |

## Labels (`y_test`)
A `LabelStrategy.build(df_test, bins_per_day)` returns integer labels or `None`:

- `NoLabels` → `None` (unsupervised; metrics are `NaN`).
- `DailyWindowLabels(test_days, window_minutes, bin_minutes)` → 1s in a window
  at the start of each test day.
- `ColumnLabels(column)` → labels read from a column (excluded from features).

## Scores
Each model's `score(X)` returns a 1-D `float` array, one value per row of `X`,
where **higher means more anomalous**.
