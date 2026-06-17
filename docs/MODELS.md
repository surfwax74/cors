# Models (Algorithm Reference)

Every model implements the same contract (`models/base.py`):

```python
model.fit(X_train)   # -> self;  learn a baseline
model.score(X)       # -> np.ndarray[len(X)];  higher = more anomalous
model.fit_score(X_train, X)        # convenience
model.get_params() / set_params()  # hyper-parameters
```

Models are registered in `models/registry.py` and addressed by name.

## Registered models

| Name                 | File                          | Idea | Key params |
|----------------------|-------------------------------|------|------------|
| `RollingCorr`        | `model_rolling_corr.py`       | Frobenius norm of (window corr − baseline corr) | `window` |
| `MahalanobisCorr`    | `model_mahalanobis_corr.py`   | Mahalanobis distance in PCA-reduced correlation-eigenvalue space | `window, k, pca_dim, stride` |
| `PCA_Full`           | `model_pca_eigen_shift.py`    | L2 drift of explained-variance spectrum (full SVD) | `window, k` |
| `PCA_Randomized`     | `model_pca_eigen_shift.py`    | …same, randomized SVD | `window, k` |
| `PCA_Incremental`    | `model_pca_eigen_shift.py`    | …same, incremental PCA | `window, k` |
| `PCA_Reconstruction` | `model_pca_reconstruction.py` | MSE of reconstructing each window from trained PCA subspace | `window, k` |
| `GraphCorrShift`     | `model_graph_corr_shift.py`   | L1 change in \|corr\| adjacency vs baseline | `window` |
| `DeepGraph_GNN`      | `model_deepgraph_gnn.py`      | GraphConv autoencoder reconstruction error (optional: torch) | `window, epochs, corr_threshold, hidden_dim, train_stride, num_train_windows, use_cuda, lr` |
| `CorALS`             | `model_corals.py`             | Placeholder (returns zeros) for a future model | — |

## Conventions

- The first `window` rows of every `score()` output are `0.0` (not enough
  history to form a window).
- `fit()` must store everything needed by `score()` on `self` and return `self`.
- Heavy/optional dependencies are imported lazily so the rest of the project
  runs without them. `DeepGraphModel.is_available()` reports whether torch +
  torch_geometric are importable.

## Adding a model

```python
# models/model_my_algo.py
from .base import BaseModel
import numpy as np

class MyAlgoModel(BaseModel):
    name = "MyAlgo"

    def __init__(self, window: int = 12):
        super().__init__(window=window)
        self.window = window
        self.baseline = None

    def fit(self, X_train):
        self.baseline = X_train.mean(axis=0)
        self.fitted = True
        return self

    def score(self, X):
        return np.linalg.norm(X - self.baseline, axis=1)
```

Then register it:

```python
# models/registry.py
from .model_my_algo import MyAlgoModel
MODEL_FACTORIES["MyAlgo"] = lambda **p: MyAlgoModel(**p)
```

The parametrized contract test in `tests/test_models.py` will automatically
exercise the new model.
