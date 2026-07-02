"""Shared training utilities."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ExperimentConfig


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name.lower(), torch.float16)


def load_model_and_tokenizer(cfg: ExperimentConfig):
    dtype = resolve_torch_dtype(cfg.model.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.trust_remote_code,
        dtype=dtype,
        device_map="auto",
    )

    if cfg.model.use_lora:
        lora = LoraConfig(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)

    return model, tokenizer


def build_training_args(cfg: ExperimentConfig, run_dir: str, method: str):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_train_epochs,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        report_to="wandb" if cfg.wandb.enabled else "none",
        run_name=f"{method}-{cfg.model.name.split('/')[-1]}",
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=False,
    )
