"""Dataset loading utilities."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from src.config import DataConfig


def load_raw_dataset(cfg: DataConfig) -> dict[str, Dataset]:
    """Load Hugging Face dataset splits."""
    kwargs: dict[str, Any] = {"path": cfg.dataset_name}
    if cfg.dataset_config:
        kwargs["name"] = cfg.dataset_config

    ds = load_dataset(**kwargs)
    return {
        "train": ds[cfg.train_split],
        "eval": ds[cfg.eval_split],
    }


def subsample(dataset: Dataset, max_samples: int | None, seed: int = 42) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))
