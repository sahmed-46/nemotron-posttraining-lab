"""Convert raw datasets into SFT / DPO / GRPO formats."""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import Dataset

from src.rewards.verifiable import extract_final_number

ANSWER_RE = re.compile(r"####\s*(-?[\d.,]+)")


def _gsm8k_answer(text: str) -> str:
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).replace(",", "")
    return extract_final_number(text)


def format_sft_example(question: str, answer_text: str) -> dict[str, str]:
    gold = _gsm8k_answer(answer_text)
    prompt = (
        "Answer the math question. Show brief reasoning, then give the final numeric answer "
        "on the last line as: ANSWER: <number>\n\n"
        f"Question: {question}"
    )
    completion = f"{answer_text.strip()}\nANSWER: {gold}"
    return {"prompt": prompt, "completion": completion, "gold_answer": gold}


def format_grpo_example(question: str, answer_text: str) -> dict[str, str]:
    gold = _gsm8k_answer(answer_text)
    prompt = (
        "Answer the math question. Show brief reasoning, then give the final numeric answer "
        "on the last line as: ANSWER: <number>\n\n"
        f"Question: {question}"
    )
    return {"prompt": prompt, "gold_answer": gold}


def format_dpo_example(question: str, answer_text: str) -> dict[str, str]:
    gold = _gsm8k_answer(answer_text)
    prompt = (
        "Answer the math question. Show brief reasoning, then give the final numeric answer "
        "on the last line as: ANSWER: <number>\n\n"
        f"Question: {question}"
    )
    chosen = f"{answer_text.strip()}\nANSWER: {gold}"
    rejected = "I think the answer is 0.\nANSWER: 0"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "gold_answer": gold,
    }


def build_formatted_splits(raw_train: Dataset, raw_eval: Dataset) -> dict[str, Dataset]:
    def map_sft(batch):
        rows = [format_sft_example(q, a) for q, a in zip(batch["question"], batch["answer"])]
        return {k: [r[k] for r in rows] for k in rows[0]}

    def map_grpo(batch):
        rows = [format_grpo_example(q, a) for q, a in zip(batch["question"], batch["answer"])]
        return {k: [r[k] for r in rows] for k in rows[0]}

    def map_dpo(batch):
        rows = [format_dpo_example(q, a) for q, a in zip(batch["question"], batch["answer"])]
        return {k: [r[k] for r in rows] for k in rows[0]}

    return {
        "sft_train": raw_train.map(map_sft, batched=True, remove_columns=raw_train.column_names),
        "sft_eval": raw_eval.map(map_sft, batched=True, remove_columns=raw_eval.column_names),
        "grpo_train": raw_train.map(map_grpo, batched=True, remove_columns=raw_train.column_names),
        "grpo_eval": raw_eval.map(map_grpo, batched=True, remove_columns=raw_eval.column_names),
        "dpo_train": raw_train.map(map_dpo, batched=True, remove_columns=raw_train.column_names),
        "dpo_eval": raw_eval.map(map_dpo, batched=True, remove_columns=raw_eval.column_names),
    }


def save_processed_datasets(datasets: dict[str, Dataset], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, ds in datasets.items():
        ds.save_to_disk(str(out / name))

    # JSONL exports for NeMo-RL path
    rl_train = datasets["grpo_train"]
    rl_val = datasets["grpo_eval"]
    _write_jsonl(out / "rl_train.jsonl", rl_train)
    _write_jsonl(out / "rl_val.jsonl", rl_val)


def _write_jsonl(path: Path, dataset: Dataset) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps({"prompt": row["prompt"], "gold_answer": row["gold_answer"]}) + "\n")
