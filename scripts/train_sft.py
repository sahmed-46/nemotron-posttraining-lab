#!/usr/bin/env python3
"""Run supervised fine-tuning warm-start."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, project_root
from src.training.sft import run_sft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method-config", default="configs/sft.yaml")
    args = parser.parse_args()

    cfg = load_config(project_root() / args.config, project_root() / args.method_config)
    processed = project_root() / cfg.data.processed_dir
    ckpt = run_sft(cfg, processed)
    print(f"SFT checkpoint saved to {ckpt}")


if __name__ == "__main__":
    main()
