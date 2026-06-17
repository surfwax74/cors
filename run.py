"""Command-line entry point for train/test correlation-detection experiments.

Examples:
    # List what's available
    python run.py --list

    # Run all models on the bundled synthetic dataset
    python run.py --dataset synthetic_10min

    # Run a subset of models
    python run.py --dataset synthetic_10min --models RollingCorr MahalanobisCorr

    # Skip the score cache
    python run.py --dataset synthetic_10min --no-cache
"""

from __future__ import annotations

import argparse
import os

from data.registry import get_dataset, list_datasets
from data.dataset import DatasetSpec, load_dataset
from data.generate import generate, get_preset
from engine.runner import Engine
from models.registry import list_models


def ensure_dataset(spec: DatasetSpec, regenerate: bool, autogen: bool, verbose: bool) -> None:
    """Make sure the dataset file exists, (re)generating it from its preset.

    - If the file exists and ``regenerate`` is False, do nothing.
    - If it's missing (or ``regenerate``) and the spec declares a
      ``generator_preset``, generate it (when ``autogen``).
    - Otherwise raise a clear error explaining how to produce it.
    """
    exists = os.path.exists(spec.path)
    if exists and not regenerate:
        return

    reason = "regenerating" if exists else "missing"
    if spec.generator_preset is None:
        raise FileNotFoundError(
            f"Dataset file '{spec.path}' is {reason} and dataset '{spec.name}' has "
            "no generator_preset. Generate it manually (e.g. python -m data.generate)."
        )
    if not autogen:
        raise FileNotFoundError(
            f"Dataset file '{spec.path}' is {reason}. Run "
            f"`python -m data.generate --preset {spec.generator_preset}` "
            "or drop --no-autogen."
        )

    if verbose:
        note = " (full preset — this can take a while)" if spec.generator_preset == "full" else ""
        print(f"[INFO] Dataset '{spec.path}' {reason}; generating "
              f"preset '{spec.generator_preset}'{note}...", flush=True)
    generate(get_preset(spec.generator_preset))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Correlation-detection train/test engine")
    p.add_argument("--dataset", default="synthetic_10min", help="Registered dataset name")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to run (default: all registered models)",
    )
    p.add_argument("--cache-dir", default="cache_results", help="Score cache directory")
    p.add_argument("--no-cache", action="store_true", help="Disable the score cache")
    p.add_argument("--list", action="store_true", help="List datasets and models, then exit")
    p.add_argument("--quiet", action="store_true", help="Suppress per-model progress logs")
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of the dataset from its preset before running",
    )
    p.add_argument(
        "--no-autogen",
        action="store_true",
        help="Do not auto-generate a missing dataset; error instead",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print("Datasets:", ", ".join(list_datasets()) or "(none)")
        print("Models:  ", ", ".join(list_models()))
        return 0

    spec = get_dataset(args.dataset)
    ensure_dataset(
        spec,
        regenerate=args.regenerate,
        autogen=not args.no_autogen,
        verbose=not args.quiet,
    )
    data = load_dataset(spec)

    engine = Engine(
        data,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )
    results = engine.run(model_names=args.models)
    engine.print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
