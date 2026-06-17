"""Registry mapping model names to factories.

Add a new algorithm in three steps:
  1. Create ``models/model_<your_algo>.py`` with a ``BaseModel`` subclass.
  2. Import it here.
  3. Add an entry to ``MODEL_FACTORIES``.

The engine only ever talks to models through this registry, so nothing else
needs to change when you add an algorithm.
"""

from __future__ import annotations

from .base import BaseModel
from .model_corals import CoralsModel
from .model_deepgraph_gnn import DeepGraphModel
from .model_graph_corr_shift import GraphCorrShiftModel
from .model_mahalanobis_corr import MahalanobisCorrModel
from .model_pca_eigen_shift import PcaEigenShiftModel
from .model_pca_reconstruction import PcaReconstructionModel
from .model_rolling_corr import RollingCorrModel

# name -> factory(**overrides) -> BaseModel instance
MODEL_FACTORIES = {
    "RollingCorr": lambda **p: RollingCorrModel(**p),
    "MahalanobisCorr": lambda **p: MahalanobisCorrModel(**p),
    "PCA_Full": lambda **p: PcaEigenShiftModel(solver="full", **p),
    "PCA_Randomized": lambda **p: PcaEigenShiftModel(solver="randomized", **p),
    "PCA_Incremental": lambda **p: PcaEigenShiftModel(solver="incremental", **p),
    "PCA_Reconstruction": lambda **p: PcaReconstructionModel(**p),
    "GraphCorrShift": lambda **p: GraphCorrShiftModel(**p),
    "DeepGraph_GNN": lambda **p: DeepGraphModel(**p),
    "CorALS": lambda **p: CoralsModel(**p),
}

#: Default set of models the engine runs when none are specified explicitly.
DEFAULT_MODELS = list(MODEL_FACTORIES.keys())


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(MODEL_FACTORIES.keys())


def get_model(name: str, **overrides) -> BaseModel:
    """Instantiate a model by name, applying hyper-parameter overrides."""
    if name not in MODEL_FACTORIES:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(MODEL_FACTORIES)}"
        )
    return MODEL_FACTORIES[name](**overrides)
