# cors — Correlation-Detection Train/Test Engine

A modular harness for developing and comparing **correlation / anomaly
detection** algorithms on multivariate telemetry. Algorithms live in their own
`model_*.py` files behind a common interface, and a general **train/test
engine** drives them over **parametrized datasets** so you can swap algorithms
and data freely.

> **Long-term goal:** plug this structure into a live data environment and add
> new streaming datasets for train + test without touching the algorithms.

---

## Quickstart

```bash
# (one-time) install deps into your venv
pip install -r requirements.txt

# list registered datasets and models
python run.py --list

# FAST AGILE LOOP: auto-generates the small dataset on first run (~1s), then
# runs all models in seconds. Re-run anytime; add --regenerate to refresh data.
python run.py --dataset synthetic_fast

# run every model on the full bundled synthetic dataset
# (also auto-generated on first run — the full preset takes a while)
python run.py --dataset synthetic_10min

# run a subset, bypassing the score cache
python run.py --dataset synthetic_10min --models RollingCorr MahalanobisCorr --no-cache

# (re)generate the full synthetic dataset
python -m data.generate

# run the tests
pytest
```

> **Tip:** Use `synthetic_fast` (40 channels, 4 train + 2 test days) for
> iteration — it generates in ~1s and runs every model in seconds, while keeping
> the full structure (clusters, behaviors, shifty anomaly channels, injected
> anomalies). Switch to `synthetic_10min` for full-scale runs. Data files are
> git-ignored and auto-generated on first use (`--regenerate` forces a refresh;
> `--no-autogen` disables it). You can also pre-generate explicitly with
> `python -m data.generate --preset fast`.

`experiment.py` reproduces the original end-to-end run (all models + exploratory
plots) on the synthetic dataset and remains as a convenience entry point.

---

## Repository structure

```
cors/
├── run.py                  # CLI entry point for experiments
├── experiment.py           # legacy-compatible "run everything + plot" script
├── cors_plot.py            # exploratory plotting helpers
│
├── models/                 # one algorithm per model_*.py file
│   ├── base.py             # BaseModel: fit(X_train) / score(X) contract
│   ├── registry.py         # name -> model factory
│   ├── model_rolling_corr.py
│   ├── model_mahalanobis_corr.py
│   ├── model_pca_eigen_shift.py
│   ├── model_pca_reconstruction.py
│   ├── model_graph_corr_shift.py
│   ├── model_deepgraph_gnn.py
│   └── model_corals.py
│
├── data/                   # datasets: specs, loading, labels, generation
│   ├── dataset.py          # DatasetSpec + load_dataset() -> LoadedDataset
│   ├── labels.py           # ground-truth label strategies
│   ├── registry.py         # name -> DatasetSpec
│   └── generate.py         # parametrized synthetic generator
│
├── engine/                 # the train/test engine
│   ├── runner.py           # Engine: fit -> score -> evaluate -> results
│   ├── metrics.py          # corr / hit-rate evaluation
│   └── cache.py            # on-disk score cache
│
├── tests/                  # unit tests (pytest)
└── docs/                   # architecture & reference docs
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit
together, [docs/MODELS.md](docs/MODELS.md) for the algorithm reference (the
"API"), [docs/DATA_MODELS.md](docs/DATA_MODELS.md) for the data shapes, and
[docs/ENGINE.md](docs/ENGINE.md) for the engine + CLI.

---

## Adding an algorithm

1. Create `models/model_<name>.py` with a `BaseModel` subclass implementing
   `fit(X_train)` and `score(X)`.
2. Import it in `models/registry.py` and add an entry to `MODEL_FACTORIES`.
3. Add a unit test in `tests/test_models.py` (the contract test runs over every
   registered model automatically).

## Adding a dataset

1. Add a `DatasetSpec` to `data/registry.py` (path, split, label strategy).
2. Select it with `python run.py --dataset <name>`.

For data with a labelled column, use `ColumnLabels("your_label_col")`; for
unsupervised data use `NoLabels()`; for the synthetic injected-window scheme use
`DailyWindowLabels(...)`.

## Tuning the synthetic generator

`data/generate.py` injects realistic **nominal** activity (daily product usage)
on top of the base signals via `data/behaviors.py`: Gaussian/uniform/Laplace/
Student-t/pink noise floors plus satellite-style behaviors — `voltage_shift`,
`step_wave`, `thruster`, and `reaction_wheel`. Tune the mix per dataset:

```python
from data.generate import GenConfig, generate
from data.behaviors import BehaviorSpec

generate(GenConfig(
    num_channels=500,
    behaviors=[
        BehaviorSpec("gaussian", fraction=0.6, amplitude=0.3),
        BehaviorSpec("thruster", fraction=0.1, amplitude=4.0,
                     params={"rate_per_hour": 2.0}),
        BehaviorSpec("reaction_wheel", fraction=0.08, amplitude=3.0),
    ],
))
```

See [docs/DATA_MODELS.md](docs/DATA_MODELS.md) for the full behavior catalog.

---

## Interface (frontend / backend)

This is a research/experimentation project, so there is **no web frontend or
service backend**. The **interface is the `run.py` CLI** (and the Python API of
the `models` / `data` / `engine` packages). When this moves into a live data
environment, the data layer (`data/`) is the integration point — add a new
`DatasetSpec` (or a new loader) for the live source and the engine and models
are unchanged.
