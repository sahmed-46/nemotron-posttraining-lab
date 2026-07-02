"""Verifiable rewards for RLVR-style training (math)."""

from __future__ import annotations

import re

ANSWER_LINE_RE = re.compile(r"ANSWER:\s*(-?[\d.,]+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?[\d.,]+")


def extract_final_number(text: str) -> str:
    match = ANSWER_LINE_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def normalize_number(value: str) -> str:
    value = value.strip().replace(",", "")
    if not value:
        return ""
    try:
        if "." in value:
            return str(float(value))
        return str(int(value))
    except ValueError:
        return value


def math_answer_reward(completion: str, gold_answer: str) -> float:
    """Return 1.0 for exact numeric match, else 0.0."""
    pred = normalize_number(extract_final_number(completion))
    gold = normalize_number(gold_answer)
    if not pred or not gold:
        return 0.0
    return 1.0 if pred == gold else 0.0


def format_reward(completion: str) -> float:
    """Small bonus for using required ANSWER: format."""
    return 0.1 if ANSWER_LINE_RE.search(completion) else 0.0


def combined_math_reward(completion: str, gold_answer: str) -> float:
    return min(1.0, math_answer_reward(completion, gold_answer) + format_reward(completion))
