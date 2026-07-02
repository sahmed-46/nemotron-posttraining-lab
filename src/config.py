"""Configuration loading and merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    trust_remote_code: bool = False
    torch_dtype: str = "float16"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DataConfig:
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    eval_split: str = "test"
    max_train_samples: int = 512
    max_eval_samples: int = 128
    processed_dir: str = "data/processed"


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    bf16: bool = False
    fp16: bool = True
    beta: float = 0.1
    num_generations: int = 4
    max_completion_length: int = 256
    temperature: float = 0.7


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "nemotron-posttraining-lab"


@dataclass
class GrpoConfig:
    kl_coef: float = 0.04
    clip_range: float = 0.2


@dataclass
class ExperimentConfig:
    project_name: str = "nemotron-posttraining-lab"
    method: str = "sft"
    seed: int = 42
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    grpo: GrpoConfig = field(default_factory=GrpoConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _dict_to_config(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        project_name=data.get("project_name", "nemotron-posttraining-lab"),
        method=data.get("method", "sft"),
        seed=data.get("seed", 42),
        output_dir=data.get("output_dir", "outputs"),
        model=ModelConfig(**data.get("model", {})),
        data=DataConfig(**data.get("data", {})),
        training=TrainingConfig(**data.get("training", {})),
        wandb=WandbConfig(**data.get("wandb", {})),
        grpo=GrpoConfig(**data.get("grpo", {})),
    )


def load_config(*paths: str | Path) -> ExperimentConfig:
    merged: dict[str, Any] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})
    return _dict_to_config(merged)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
