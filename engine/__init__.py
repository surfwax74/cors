"""Train/test engine: caching, metrics, and the model runner."""

from .metrics import evaluate
from .runner import Engine, ModelResult

__all__ = ["Engine", "ModelResult", "evaluate"]
