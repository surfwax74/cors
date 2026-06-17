"""Dataset definitions, loading, labelling, and synthetic generation."""

from .dataset import DatasetSpec, LoadedDataset, load_dataset
from .labels import ColumnLabels, DailyWindowLabels, LabelStrategy, NoLabels
from .registry import DATASETS, get_dataset, list_datasets

__all__ = [
    "DatasetSpec",
    "LoadedDataset",
    "load_dataset",
    "LabelStrategy",
    "NoLabels",
    "DailyWindowLabels",
    "ColumnLabels",
    "DATASETS",
    "get_dataset",
    "list_datasets",
]
