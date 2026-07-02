"""Evaluation metrics for post-training comparison."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rewards.verifiable import combined_math_reward, math_answer_reward


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def evaluate_model(model_path: str, eval_dataset, max_samples: int | None = None) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    model.eval()

    rows = eval_dataset.select(range(min(len(eval_dataset), max_samples or len(eval_dataset))))
    correct = 0
    format_hits = 0
    total = len(rows)
    failures = []

    for row in tqdm(rows, desc="eval"):
        pred = generate_answer(model, tokenizer, row["prompt"])
        reward = math_answer_reward(pred, row["gold_answer"])
        full = combined_math_reward(pred, row["gold_answer"])
        if reward >= 1.0:
            correct += 1
        if full > reward:
            format_hits += 1
        if reward < 1.0:
            failures.append(
                {
                    "prompt": row["prompt"][:120],
                    "gold_answer": row["gold_answer"],
                    "prediction": pred[:200],
                }
            )

    return {
        "accuracy": correct / total if total else 0.0,
        "format_bonus_rate": format_hits / total if total else 0.0,
        "total": total,
        "failures": failures[:20],
    }


def save_eval_report(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
