"""Correlation / anomaly detection algorithms.

Each algorithm lives in its own ``model_*.py`` module and subclasses
``BaseModel``. Use the registry helpers to discover and instantiate them.
"""

from .base import BaseModel
from .registry import DEFAULT_MODELS, MODEL_FACTORIES, get_model, list_models

__all__ = [
    "BaseModel",
    "MODEL_FACTORIES",
    "DEFAULT_MODELS",
    "get_model",
    "list_models",
]
