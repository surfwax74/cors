# Engine & CLI

## The engine (`engine/runner.py`)

```python
from data.registry import get_dataset
from data.dataset import load_dataset
from engine.runner import Engine

data = load_dataset(get_dataset("synthetic_10min"))
engine = Engine(data, cache_dir="cache_results", use_cache=True)

results = engine.run(
    model_names=["RollingCorr", "MahalanobisCorr"],
    param_overrides={"MahalanobisCorr": {"k": 30}},
)
engine.print_results(results)
```

For each model the engine:
1. instantiates it from the registry (with optional per-model overrides),
2. `fit`s it on `X_train` (timed),
3. `score`s `X_test`, going through the cache (timed),
4. `evaluate`s scores against `y_test`,
5. returns a `ModelResult`.

### `ModelResult`

| field           | meaning                                   |
|-----------------|-------------------------------------------|
| `name`          | model name                                |
| `status`        | `"ok"` \| `"skipped"` \| `"error"`         |
| `scores`        | the score array (when `ok`)               |
| `metrics`       | `{"corr": ..., "hit_rate": ...}`           |
| `fit_seconds`   | time spent in `fit`                       |
| `score_seconds` | time spent in `score`                     |
| `message`       | reason, when skipped/errored              |

A model whose optional deps are missing is `skipped`, not fatal — the rest of
the run continues.

## Metrics (`engine/metrics.py`)

- **`corr`** — Pearson correlation between scores and labels.
- **`hit_rate`** — fraction of the top-K scored points that are true anomalies,
  where K = number of positive labels.
- Unsupervised datasets (`y_test is None`) yield `NaN` for both.

## Caching (`engine/cache.py`)

Scores are cached under `cache_dir` keyed on **model name + hyper-parameters +
hash(X_train, X_test)**. Change the algorithm's params or the data and the key
changes, so you never silently reuse a stale result. Disable with
`use_cache=False` (CLI: `--no-cache`).

## CLI (`run.py`)

```
python run.py --list
python run.py --dataset synthetic_10min
python run.py --dataset synthetic_10min --models RollingCorr CorALS
python run.py --dataset synthetic_10min --no-cache --quiet
python run.py --dataset synthetic_10min --cache-dir /tmp/cors_cache
```

| flag          | meaning                                  |
|---------------|------------------------------------------|
| `--dataset`   | registered dataset name                  |
| `--models`    | subset of models (default: all)          |
| `--cache-dir` | score cache directory                    |
| `--no-cache`  | disable the score cache                  |
| `--list`      | print datasets + models and exit         |
| `--quiet`     | suppress per-model progress logs         |
| `--regenerate`| force-regenerate the dataset from its preset first |
| `--no-autogen`| error instead of auto-generating a missing dataset |

A dataset whose `DatasetSpec` declares a `generator_preset` is **auto-generated
on first use** if its file is missing (handy since data files are git-ignored).
`--regenerate` forces a refresh even when the file exists.
