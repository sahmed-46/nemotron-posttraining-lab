#!/usr/bin/env python3
"""End-to-end ablation: SFT -> DPO/GRPO -> eval comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, project_root
from src.eval.metrics import evaluate_model, save_eval_report
from src.training.dpo import run_dpo
from src.training.grpo import run_grpo
from src.training.sft import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT + DPO + GRPO ablation pipeline")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config)
    processed = root / cfg.data.processed_dir
    eval_ds = load_from_disk(str(processed / "grpo_eval"))

    results = {}
    if not args.skip_train:
        sft_ckpt = run_sft(cfg, processed)
        dpo_cfg = load_config(root / args.config, root / "configs/dpo.yaml")
        grpo_cfg = load_config(root / args.config, root / "configs/grpo.yaml")
        dpo_ckpt = run_dpo(dpo_cfg, processed, model_path=sft_ckpt)
        grpo_ckpt = run_grpo(grpo_cfg, processed, model_path=sft_ckpt)
    else:
        sft_ckpt = str(root / cfg.output_dir / "checkpoints" / "sft" / "final")
        dpo_ckpt = str(root / cfg.output_dir / "checkpoints" / "dpo" / "final")
        grpo_ckpt = str(root / cfg.output_dir / "checkpoints" / "grpo" / "final")

    for name, path in [("base", cfg.model.name), ("sft", sft_ckpt), ("dpo", dpo_ckpt), ("grpo", grpo_ckpt)]:
        print(f"Evaluating {name} ...")
        report = evaluate_model(path, eval_ds, max_samples=cfg.data.max_eval_samples)
        results[name] = {"accuracy": report["accuracy"], "total": report["total"]}
        save_eval_report(report, root / cfg.output_dir / "eval" / f"{name}.json")

    summary_path = root / cfg.output_dir / "eval" / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
