#!/usr/bin/env python3
"""Download and prepare datasets for SFT / DPO / GRPO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, project_root
from src.data import build_formatted_splits, load_raw_dataset, save_processed_datasets, subsample


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare post-training datasets")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(project_root() / args.config)
    raw = load_raw_dataset(cfg.data)
    train = subsample(raw["train"], cfg.data.max_train_samples, cfg.seed)
    eval_ds = subsample(raw["eval"], cfg.data.max_eval_samples, cfg.seed)

    formatted = build_formatted_splits(train, eval_ds)
    out = project_root() / cfg.data.processed_dir
    save_processed_datasets(formatted, out)
    print(f"Saved processed datasets to {out}")
    for name, ds in formatted.items():
        print(f"  {name}: {len(ds)} rows")


if __name__ == "__main__":
    main()
