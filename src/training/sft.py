"""Supervised Fine-Tuning (SFT) warm-start."""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from trl import SFTTrainer

from src.config import ExperimentConfig
from src.training.common import build_training_args, load_model_and_tokenizer
from src.utils.logging_utils import init_wandb
from src.utils.seed import set_seed


def run_sft(cfg: ExperimentConfig, processed_dir: str | Path) -> str:
    set_seed(cfg.seed)
    processed_dir = Path(processed_dir)
    train_ds = load_from_disk(str(processed_dir / "sft_train"))
    eval_ds = load_from_disk(str(processed_dir / "sft_eval"))

    run_dir = Path(cfg.output_dir) / "checkpoints" / "sft"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_wandb(cfg, "sft")
    model, tokenizer = load_model_and_tokenizer(cfg)
    training_args = build_training_args(cfg, str(run_dir), "sft")

    def formatting_func(example):
        return example["prompt"] + "\n" + example["completion"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=cfg.training.max_seq_length,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "final"))
    tokenizer.save_pretrained(str(run_dir / "final"))
    return str(run_dir / "final")
