"""On-disk caching of model scores.

Cache keys combine the model name, its hyper-parameters, and a hash of the
train+test data, so a cached result is only reused when the algorithm *and* the
data are identical. This lets you re-run experiments cheaply while iterating on
new models or datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Callable

import numpy as np
import pandas as pd


def df_hash(df: pd.DataFrame) -> str:
    """Stable content hash of a DataFrame."""
    data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(data_bytes).hexdigest()


def array_hash(arr: np.ndarray) -> str:
    """Stable content hash of a numpy array."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def params_hash(params: dict) -> str:
    """Stable hash of a (JSON-serialisable) hyper-parameter dict."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def make_key(model_name: str, params: dict, data_hash: str) -> str:
    """Build a single cache key from model + params + data."""
    combined = f"{model_name}|{params_hash(params)}|{data_hash}"
    digest = hashlib.md5(combined.encode("utf-8")).hexdigest()
    return f"{model_name}_{digest}"


def cached_compute(
    cache_dir: str,
    key: str,
    compute_fn: Callable[[], object],
    use_cache: bool = True,
) -> object:
    """Return a cached result for ``key`` or compute, store, and return it."""
    if not use_cache:
        return compute_fn()

    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"{key}.pkl")

    if os.path.exists(fname):
        with open(fname, "rb") as fh:
            return pickle.load(fh)

    result = compute_fn()
    with open(fname, "wb") as fh:
        pickle.dump(result, fh)
    return result
