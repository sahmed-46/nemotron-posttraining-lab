#!/usr/bin/env python3
"""Run Direct Preference Optimization (DPO)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, project_root
from src.training.dpo import run_dpo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method-config", default="configs/dpo.yaml")
    parser.add_argument("--model-path", default=None, help="Optional SFT checkpoint path")
    args = parser.parse_args()

    cfg = load_config(project_root() / args.config, project_root() / args.method_config)
    processed = project_root() / cfg.data.processed_dir
    ckpt = run_dpo(cfg, processed, model_path=args.model_path)
    print(f"DPO checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
