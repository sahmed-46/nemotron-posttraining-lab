#!/usr/bin/env python3
"""Fix Colab dependency conflicts before training."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    # Colab ships an old torchao that breaks recent peft LoRA injection.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "accelerate",
            "peft",
            "transformers",
            "trl",
            "datasets",
            "wandb",
        ]
    )
    print("Colab dependencies ready.")


if __name__ == "__main__":
    main()
