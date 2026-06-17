# Architecture

The project is organised around one idea: **algorithms and data are
independent, and a general engine connects them.** That separation is what makes
it cheap to add algorithms, swap datasets, and eventually point the whole thing
at a live data environment.

```
            ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
  data/ ───▶│ LoadedDataset│───▶ │    Engine    │ ◀───│   models/    │
            │ X_train/X_test│    │ fit→score→eval│    │  BaseModel   │
            │   y_test      │    └──────┬───────┘     │  (registry)  │
            └─────────────┘            │             └──────────────┘
                                       ▼
                                 ModelResult[]
                              (metrics + timings)
```

## Layers

### `models/` — the algorithms
Every algorithm subclasses `BaseModel` and implements two methods:

- `fit(X_train)` — learn/calibrate a baseline from the training matrix.
- `score(X)` — return one anomaly score per row (higher = more anomalous).

Models are discovered through `models/registry.py` (`MODEL_FACTORIES`). The
engine never imports a concrete model directly — it asks the registry by name.
This is the extension point for "add more algorithms down the road."

Optional heavy dependencies (e.g. `torch_geometric` for `DeepGraph_GNN`) are
imported lazily; if they're missing the model raises `ImportError` and the
engine records it as `skipped` instead of crashing the run.

### `data/` — the datasets
- `DatasetSpec` is a declarative description: path, loader, split strategy,
  label strategy.
- `load_dataset(spec)` materialises a `LoadedDataset` (train/test frames,
  feature matrices, ground-truth labels).
- `labels.py` makes labelling pluggable (`NoLabels`, `DailyWindowLabels`,
  `ColumnLabels`) so synthetic injected windows and real labelled columns both
  flow through the same engine.
- `generate.py` is the parametrized synthetic data generator.

This is the integration point for the **live data environment**: a new live
source becomes a new `DatasetSpec` (and, if needed, a new loader).

### `engine/` — the harness
- `Engine` (in `runner.py`) loops over model names: instantiate → `fit` on
  `X_train` → `score` on `X_test` (cached) → `evaluate` vs `y_test` → collect a
  `ModelResult`.
- `metrics.py` computes correlation and top-K hit-rate.
- `cache.py` stores scores keyed on **model name + hyper-parameters + data
  hash**, so a cached result is only reused when both the algorithm and the data
  are identical.

## Control flow (a single run)

1. `run.py` resolves a `DatasetSpec` from the registry and calls `load_dataset`.
2. It constructs an `Engine` around the `LoadedDataset`.
3. `Engine.run(model_names)` produces a list of `ModelResult`.
4. Results are formatted into the familiar table.

## Why this shape

- **Add an algorithm** → one new `model_*.py` + one registry line. Nothing in
  the engine or data layer changes.
- **Add/swap a dataset** → one new `DatasetSpec`. Nothing in the engine or
  models changes.
- **Go live** → implement a loader/spec for the live source; the rest is reused.
