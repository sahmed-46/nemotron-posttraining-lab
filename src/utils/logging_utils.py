"""Experiment logging helpers."""

from __future__ import annotations

import os

from src.config import ExperimentConfig


def init_wandb(cfg: ExperimentConfig, method: str) -> None:
    if not cfg.wandb.enabled:
        return
    import wandb

    wandb.init(
        project=os.getenv("WANDB_PROJECT", cfg.wandb.project),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=f"{method}-{cfg.model.name.split('/')[-1]}",
        config={
            "method": method,
            "model": cfg.model.name,
            "seed": cfg.seed,
            "learning_rate": cfg.training.learning_rate,
            "max_train_samples": cfg.data.max_train_samples,
        },
    )
