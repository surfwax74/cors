"""The train/test engine.

The engine is the general harness the project is organised around. Given a
loaded dataset and a list of model names, it:

  1. instantiates each model from the registry (with optional param overrides),
  2. fits it on the training split,
  3. scores the test split (cached),
  4. evaluates the scores against ground truth, and
  5. collects timing + metrics into a results table.

Models that fail to load (e.g. missing optional dependencies) are recorded as
``status="skipped"`` rather than crashing the whole run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from data.dataset import LoadedDataset
from models.registry import DEFAULT_MODELS, get_model

from .cache import array_hash, cached_compute, make_key
from .metrics import evaluate


@dataclass
class ModelResult:
    name: str
    status: str  # "ok" | "skipped" | "error"
    scores: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    fit_seconds: float = 0.0
    score_seconds: float = 0.0
    message: str = ""


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[INFO] {msg}", flush=True)


class Engine:
    """Drives fit/score/evaluate over a dataset for a set of models."""

    def __init__(
        self,
        data: LoadedDataset,
        cache_dir: str = "cache_results",
        use_cache: bool = True,
        verbose: bool = True,
    ):
        self.data = data
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.verbose = verbose
        # Hash train+test together: a model's score depends on both splits.
        self._data_hash = array_hash(data.X_train) + array_hash(data.X_test)

    def run_model(self, name: str, **overrides) -> ModelResult:
        """Fit, score, and evaluate a single model by name."""
        try:
            model = get_model(name, **overrides)
        except KeyError as exc:
            return ModelResult(name=name, status="error", message=str(exc))

        X_train, X_test = self.data.X_train, self.data.X_test

        try:
            t0 = time.time()
            model.fit(X_train)
            fit_seconds = time.time() - t0

            key = make_key(model.name, model.get_params(), self._data_hash)

            t1 = time.time()
            scores = cached_compute(
                self.cache_dir,
                key,
                lambda: model.score(X_test),
                use_cache=self.use_cache,
            )
            score_seconds = time.time() - t1
        except ImportError as exc:
            _log(self.verbose, f"{name}: skipped ({exc})")
            return ModelResult(name=name, status="skipped", message=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            _log(self.verbose, f"{name}: error ({exc})")
            return ModelResult(name=name, status="error", message=str(exc))

        metrics = evaluate(scores, self.data.y_test)
        _log(
            self.verbose,
            f"{name}: done in {fit_seconds + score_seconds:.2f}s "
            f"(corr={metrics['corr']:.3f}, hit={metrics['hit_rate']:.3f})",
        )
        return ModelResult(
            name=name,
            status="ok",
            scores=scores,
            metrics=metrics,
            fit_seconds=fit_seconds,
            score_seconds=score_seconds,
        )

    def run(
        self,
        model_names: list[str] | None = None,
        param_overrides: dict[str, dict] | None = None,
    ) -> list[ModelResult]:
        """Run several models and return their results."""
        model_names = model_names or DEFAULT_MODELS
        param_overrides = param_overrides or {}

        _log(
            self.verbose,
            f"Running {len(model_names)} model(s) on dataset "
            f"'{self.data.spec.name}' "
            f"(train={len(self.data.X_train)}, test={len(self.data.X_test)})",
        )

        results = []
        for name in model_names:
            results.append(self.run_model(name, **param_overrides.get(name, {})))
        return results

    @staticmethod
    def format_results(results: list[ModelResult]) -> str:
        """Render a results table as a string."""
        lines = ["", "=== CORRELATION-BASED ANOMALY DETECTION RESULTS ===", ""]
        for r in results:
            if r.status == "ok":
                lines.append(
                    f"{r.name:20s}  Corr(y,score)={r.metrics['corr']:6.3f}   "
                    f"HitRate={r.metrics['hit_rate']:6.3f}   "
                    f"({r.fit_seconds + r.score_seconds:6.2f}s)"
                )
            else:
                lines.append(f"{r.name:20s}  [{r.status}] {r.message}")
        lines.append("")
        return "\n".join(lines)

    def print_results(self, results: list[ModelResult]) -> None:
        print(self.format_results(results))
