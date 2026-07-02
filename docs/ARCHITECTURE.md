# Architecture

## Overview

This repo implements a **two-path post-training lab**:

1. **Local path (TRL + PyTorch)** — SFT, DPO, GRPO on 1 GPU for development and ablations
2. **Scale path (NeMo-RL)** — JSONL exports + config templates for Nemotron-style cluster training

```
Raw HF dataset (GSM8K or Nemotron RL blend)
        │
        ▼
scripts/prepare_datasets.py
        │
        ├── SFT datasets
        ├── DPO preference datasets
        └── GRPO / RLVR prompt datasets + JSONL
        │
        ▼
Training
        ├── scripts/train_sft.py
        ├── scripts/train_dpo.py
        └── scripts/train_grpo.py
        │
        ▼
scripts/run_eval.py / scripts/run_ablation.py
        │
        └── outputs/eval/*.json
```

## Modules

| Path | Responsibility |
|---|---|
| `src/config.py` | YAML config loading |
| `src/data/` | Dataset load + SFT/DPO/GRPO formatting |
| `src/rewards/` | Verifiable math rewards (RLVR) |
| `src/training/` | SFT, DPO, GRPO trainers |
| `src/eval/` | Accuracy eval + failure reports |
| `configs/nemo_rl/` | NeMo-RL cluster templates |

## Algorithms

- **SFT** — warm-start on gold answers
- **DPO** — preference pairs (chosen vs rejected)
- **GRPO** — group sampling + verifiable reward (RLVR)

## Outputs

```
outputs/
├── checkpoints/
│   ├── sft/final/
│   ├── dpo/final/
│   └── grpo/final/
└── eval/
    ├── base.json
    ├── sft.json
    ├── dpo.json
    ├── grpo.json
    └── ablation_summary.json
```
