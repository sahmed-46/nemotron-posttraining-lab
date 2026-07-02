"""Group Relative Policy Optimization (GRPO) with verifiable math rewards."""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from src.config import ExperimentConfig
from src.rewards.verifiable import combined_math_reward
from src.training.common import load_model_and_tokenizer
from src.utils.logging_utils import init_wandb
from src.utils.seed import set_seed


def _build_reward_func():
    def reward_func(completions, gold_answer=None, **kwargs):
        answers = gold_answer if gold_answer is not None else kwargs.get("gold_answer", [])
        scores = []
        for completion, gold in zip(completions, answers):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            scores.append(combined_math_reward(text, gold))
        return scores

    return reward_func


def run_grpo(cfg: ExperimentConfig, processed_dir: str | Path, model_path: str | None = None) -> str:
    set_seed(cfg.seed)
    processed_dir = Path(processed_dir)
    train_ds = load_from_disk(str(processed_dir / "grpo_train"))
    eval_ds = load_from_disk(str(processed_dir / "grpo_eval"))

    run_dir = Path(cfg.output_dir) / "checkpoints" / "grpo"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_wandb(cfg, "grpo")
    if model_path:
        cfg.model.name = model_path
    model, tokenizer = load_model_and_tokenizer(cfg)

    grpo_args = GRPOConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        report_to="wandb" if cfg.wandb.enabled else "none",
        run_name=f"grpo-{Path(cfg.model.name).name}",
        remove_unused_columns=False,
        num_generations=cfg.training.num_generations,
        max_completion_length=cfg.training.max_completion_length,
        temperature=cfg.training.temperature,
        beta=cfg.grpo.kl_coef,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_build_reward_func(),
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "final"))
    tokenizer.save_pretrained(str(run_dir / "final"))
    return str(run_dir / "final")
