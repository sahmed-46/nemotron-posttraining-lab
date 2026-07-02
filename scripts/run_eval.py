#!/usr/bin/env python3
"""Evaluate a checkpoint on held-out GRPO eval split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, project_root
from src.eval.metrics import evaluate_model, save_eval_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="outputs/eval/report.json")
    args = parser.parse_args()

    cfg = load_config(project_root() / args.config)
    eval_ds = load_from_disk(str(project_root() / cfg.data.processed_dir / "grpo_eval"))
    results = evaluate_model(args.model_path, eval_ds, max_samples=cfg.data.max_eval_samples)
    out = project_root() / args.output
    save_eval_report(results, out)
    print(f"Accuracy: {results['accuracy']:.2%} ({results['total']} samples)")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
