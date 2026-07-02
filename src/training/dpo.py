"""Direct Preference Optimization (DPO)."""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer

from src.config import ExperimentConfig
from src.training.common import load_model_and_tokenizer
from src.utils.logging_utils import init_wandb
from src.utils.seed import set_seed


def run_dpo(cfg: ExperimentConfig, processed_dir: str | Path, model_path: str | None = None) -> str:
    set_seed(cfg.seed)
    processed_dir = Path(processed_dir)
    train_ds = load_from_disk(str(processed_dir / "dpo_train"))
    eval_ds = load_from_disk(str(processed_dir / "dpo_eval"))

    run_dir = Path(cfg.output_dir) / "checkpoints" / "dpo"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_wandb(cfg, "dpo")
    if model_path:
        cfg.model.name = model_path
    model, tokenizer = load_model_and_tokenizer(cfg)

    dpo_args = DPOConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        beta=cfg.training.beta,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        report_to="wandb" if cfg.wandb.enabled else "none",
        run_name=f"dpo-{Path(cfg.model.name).name}",
        remove_unused_columns=False,
        max_length=cfg.training.max_seq_length,
        max_prompt_length=cfg.training.max_seq_length // 2,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "final"))
    tokenizer.save_pretrained(str(run_dir / "final"))
    return str(run_dir / "final")
